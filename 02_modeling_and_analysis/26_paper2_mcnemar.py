"""
data_audit/26_paper2_mcnemar.py  (2026-08-23)

Significance validation for Paper 2's core result: the per-region
best-feature-set table (data_audit/24). Neither the original G1 baseline
run (data_audit/05) nor the feature-selection run (data_audit/24) saved
per-site predictions -- only aggregate accuracy -- so this refits BOTH the
Baseline variant and the winning variant (when different) for all 14
region/target combos, using their already-found best_configs (no new grid
search), to get real paired per-site predictions. Two McNemar tests per
combo where the winner isn't Baseline:
  1. winning variant vs Baseline -- is switching feature sets a real gain?
  2. winning variant vs fixed local-majority baseline -- does the region
     have any real predictive skill at all (the question the old national
     report's "regional significance testing not possible" limitation was
     about)?
Where Baseline itself won, only test 2 applies (nothing to switch from).

Output: results/json/other/phase5_paper2_mcnemar_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
master = pd.read_excel(os.path.join(BASE, "geosites_master_1667_with_accessibility.xlsx"))
domain_lookup = master[["Locality_ID", "Geological_Domain"]]
infra = pd.read_csv(os.path.join(BASE, "data/final/infra_features.csv"))

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
    on="Locality_ID", how="inner").merge(domain_lookup, on="Locality_ID", how="left").merge(infra, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

SENTINEL_DIST_M = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
settle_dummies = pd.get_dummies(merged["nearest_settlement_type"], prefix="Settlement").astype(float)
merged = pd.concat([merged, settle_dummies], axis=1)
INFRA_COLS = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"] + list(settle_dummies.columns)

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

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr): return compute_sample_weight("balanced", y_tr)

def fit_predict_proba(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = combined_weight(y[tr])
    if kind == "XGB":
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict_proba(X[te])[:, 1]

def logo_cluster_oof(best_cfgs, X, y, groups):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    def fold(tr, te):
        probs = [fit_predict_proba(k, c, X, y, tr, te) for k, c in best_cfgs]
        return te, np.mean(probs, axis=0)
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fold)(tr, te) for tr, te in folds)
    proba = np.full(len(y), np.nan)
    for te, p in results: proba[te] = p
    return proba

def feature_cols(variant, dom_dummy_cols):
    if variant == "Baseline": return FEATURES_BASE
    if variant == "Domain": return FEATURES_BASE + dom_dummy_cols
    if variant == "Infra": return FEATURES_BASE + INFRA_COLS
    raise ValueError(variant)

def mcnemar_test(correct_a, correct_b):
    table = pd.crosstab(pd.Series(correct_a), pd.Series(correct_b)).reindex(
        index=[False, True], columns=[False, True], fill_value=0).values
    res = mcnemar(table, exact=False, correction=True)
    n10 = int(((correct_a == True) & (correct_b == False)).sum())
    n01 = int(((correct_a == False) & (correct_b == True)).sum())
    return {"chi2": round(float(res.statistic),4), "p": round(float(res.pvalue),4), "n10": n10, "n01": n01}

paper2_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_paper2_best_feature_results.json")))

out = []
for entry in paper2_results:
    reg, target = entry["region"], entry["target"]
    winner = entry["best_variant"]
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    groups = cluster_of(sub)
    dom_counts = sub["Geological_Domain"].value_counts()
    rare = dom_counts[dom_counts < 5].index
    sub["Domain_grouped"] = sub["Geological_Domain"].where(~sub["Geological_Domain"].isin(rare), "Other").fillna("Unknown")
    dom_dummies = pd.get_dummies(sub["Domain_grouped"], prefix="Domain").astype(float)
    sub = pd.concat([sub, dom_dummies], axis=1)
    yb = (sub["Expert_Merged"] == target).astype(int).values
    maj_class = int(pd.Series(yb).value_counts().idxmax())
    correct_majority = (yb == maj_class)

    log(f"{reg} / {target}: winner={winner}")
    variants_to_fit = {"Baseline"} | {winner}
    proba_by_variant = {}
    thr_by_variant = {}
    for variant in variants_to_fit:
        v_entry = entry["variants"][variant]
        cfgs = v_entry["best_configs"]
        thr = v_entry["tuned_threshold"] if v_entry["acc_tuned"] >= v_entry["acc_default"] else 0.5
        Xr = sub[feature_cols(variant, list(dom_dummies.columns))].values
        proba = logo_cluster_oof(cfgs, Xr, yb, groups)
        acc = float(np.nanmean((proba >= thr).astype(int) == yb))
        log(f"  {variant}: refit acc={acc:.4f} (thr={thr}) (script reported {v_entry['best_acc']})")
        proba_by_variant[variant] = proba
        thr_by_variant[variant] = thr

    correct_winner = (proba_by_variant[winner] >= thr_by_variant[winner]).astype(int) == yb
    valid_winner = ~np.isnan(proba_by_variant[winner])

    row = dict(region=reg, target=target, winner=winner, N=len(yb))
    row["winner_vs_majority"] = mcnemar_test(correct_winner[valid_winner], correct_majority[valid_winner])
    if winner != "Baseline":
        correct_baseline = (proba_by_variant["Baseline"] >= thr_by_variant["Baseline"]).astype(int) == yb
        valid_baseline = ~np.isnan(proba_by_variant["Baseline"])
        valid_both = valid_winner & valid_baseline
        row["winner_vs_baseline"] = mcnemar_test(correct_baseline[valid_both], correct_winner[valid_both])
    log(f"  -> {row}")
    out.append(row)

out_path = os.path.join(BASE, "results/json/other/phase5_paper2_mcnemar_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"\nWrote {out_path}")

log("\nSUMMARY:")
for r in out:
    wm = r["winner_vs_majority"]
    line = f"  {r['region']:28s} {r['target']:10s} winner={r['winner']:9s} vs_majority: p={wm['p']:.4f}"
    if "winner_vs_baseline" in r:
        wb = r["winner_vs_baseline"]
        line += f"  vs_baseline: p={wb['p']:.4f}"
    print(line)
