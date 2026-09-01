"""
data_audit/04_phase5_mcnemar.py  (2026-08-18)

Paired significance test for Phase 5's Domain_939 finding, same protocol as
the report's own Section 4.4 McNemar analysis: continuity-corrected McNemar
on discordant cells, majority-class baseline is FIXED (always predicts the
negative class), not a refit model.

Three comparisons per target:
  1. Domain_939 vs Baseline_939 (direct, paired, same 939 sites) -- is the
     LOGO-cluster CV gain (+0.85pp Difficult, +0.64pp Easy) real or noise?
  2. Domain_939 vs fixed majority-class baseline -- does it clear
     significance where the original deployed model (N=733) did not?
  3. Baseline_939 vs fixed majority-class baseline -- sanity/reference point,
     same question for the plain 6-feature model on the larger N=939 set.

Reuses the best_configs already found by Phase 5's grid search (saved in
phase5_modeling_results.json) -- refits only the LOGO-cluster CV step to get
real per-site predictions (never saved the first time), no grid search rerun.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

log("Loading N=939 labeled catalog ...")
catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
master = pd.read_excel(os.path.join(BASE, "geosites_master_1667_with_accessibility.xlsx"))
domain_lookup = master[["Locality_ID", "Geological_Domain"]]

frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").merge(domain_lookup, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

dom_counts = merged["Geological_Domain"].value_counts()
rare_domains = dom_counts[dom_counts < 5].index
merged["Geological_Domain_grouped"] = merged["Geological_Domain"].where(
    ~merged["Geological_Domain"].isin(rare_domains), "Other")
merged["Geological_Domain_grouped"] = merged["Geological_Domain_grouped"].fillna("Unknown")
domain_dummies = pd.get_dummies(merged["Geological_Domain_grouped"], prefix="Domain").astype(float)
merged = pd.concat([merged, domain_dummies], axis=1)
FEATURES_DOMAIN = FEATURES_BASE + list(domain_dummies.columns)

def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cluster_of(sub):
    lat, lon = sub["Latitude_WGS84"].values, sub["Longitude_WGS84"].values
    n = len(sub)
    D = haversine_matrix(lat, lon)
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] <= 500:
                rx, ry = find(i), find(j)
                if rx != ry: parent[rx] = ry
    return np.array([find(i) for i in range(n)])

cluster_ids = cluster_of(merged)
log(f"clusters={len(np.unique(cluster_ids))}")

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def fold_proba(best_cfgs, X, y, tr, te):
    probs = []
    for kind, cfg in best_cfgs:
        m = make_model(kind, cfg)
        sw = combined_weight(y[tr])
        if kind == "XGB":
            pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
            if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te])[:, 1])
    return te, np.mean(probs, axis=0)

def logo_cluster_oof(cols, y, best_cfgs):
    X = merged[cols].values
    folds = list(LeaveOneGroupOut().split(X, y, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.full(N, np.nan)
    for te, p in results: proba[te] = p
    return proba

def mcnemar_test(correct_a, correct_b, valid):
    a, b = correct_a[valid], correct_b[valid]
    table = pd.crosstab(a, b).reindex(index=[False, True], columns=[False, True], fill_value=0).values
    res = mcnemar(table, exact=False, correction=True)
    n10 = int(((a == True) & (b == False)).sum())
    n01 = int(((a == False) & (b == True)).sum())
    return {"chi2": round(float(res.statistic), 4), "p": round(float(res.pvalue), 4),
            "n_a_right_b_wrong": n10, "n_a_wrong_b_right": n01, "n_valid": int(valid.sum())}

results = json.load(open(os.path.join(BASE, "results/json/training/phase5_modeling_results.json")))

out = {}
for target_name in ["difficult", "easy"]:
    y = (merged["Expert_Merged"] == ("Difficult" if target_name == "difficult" else "Easy")).astype(int).values
    baseline_cfgs = results[f"Baseline_939_{target_name}"]["best_configs"]
    domain_cfgs = results[f"Domain_939_{target_name}"]["best_configs"]

    log(f"[{target_name}] Refitting Baseline_939 LOGO-cluster CV (reused best_configs) ...")
    p_base = logo_cluster_oof(FEATURES_BASE, y, baseline_cfgs)
    acc_base = float(np.nanmean((p_base >= 0.5).astype(int) == y))
    log(f"  Baseline_939 acc={acc_base:.4f} (grid-search run reported {results[f'Baseline_939_{target_name}']['acc_logo_cluster']})")

    log(f"[{target_name}] Refitting Domain_939 LOGO-cluster CV (reused best_configs) ...")
    p_dom = logo_cluster_oof(FEATURES_DOMAIN, y, domain_cfgs)
    acc_dom = float(np.nanmean((p_dom >= 0.5).astype(int) == y))
    log(f"  Domain_939 acc={acc_dom:.4f} (grid-search run reported {results[f'Domain_939_{target_name}']['acc_logo_cluster']})")

    baseline_correct_fixed = (y == 0)
    base_correct = (p_base >= 0.5).astype(int) == y
    dom_correct = (p_dom >= 0.5).astype(int) == y
    valid_base = ~np.isnan(p_base)
    valid_dom = ~np.isnan(p_dom)

    out[target_name] = {
        "acc": {"baseline_939": round(acc_base, 4), "domain_939": round(acc_dom, 4)},
        "domain_vs_baseline": mcnemar_test(dom_correct, base_correct, valid_dom & valid_base),
        "domain_vs_majority_baseline": mcnemar_test(dom_correct, baseline_correct_fixed, valid_dom),
        "baseline_vs_majority_baseline": mcnemar_test(base_correct, baseline_correct_fixed, valid_base),
    }

out_path = os.path.join(BASE, "results", "json", "other", "phase5_mcnemar_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"Wrote {out_path}")

log("SUMMARY:")
for target_name in ["difficult", "easy"]:
    r = out[target_name]
    print(f"\n  {target_name}: acc baseline={r['acc']['baseline_939']} domain={r['acc']['domain_939']}")
    print(f"    Domain vs Baseline (direct):        {r['domain_vs_baseline']}")
    print(f"    Domain vs majority baseline:         {r['domain_vs_majority_baseline']}")
    print(f"    Baseline vs majority baseline:        {r['baseline_vs_majority_baseline']}")
