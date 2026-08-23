"""
data_audit/03_phase5_modeling.py  (2026-08-18)

Phase 5: modeling, informed by Phases 1-4's audit findings. Mirrors code/20's
exact methodology (same grid_search, same StratifiedGroupKFold(5x10) config
search, same 500m LOGO-cluster CV via LeaveOneGroupOut, same leave_region_out())
so results are directly comparable to the existing G0a_difficult (0.7408) /
G0a_easy (0.7326) headline numbers -- with two changes, both deliberate and
disclosed:
  1. N=939 (733 original + 206 el_ouali_2026 batch, both fully audited in
     Phases 1-2: typos fixed, regions geometrically corrected, all coordinate-
     conflicts resolved against real source evidence).
  2. Confidence-weighting DROPPED (class-balance weighting only). The new
     batch has no Confidence field, and per direct instruction this factor is
     being retired project-wide, not something to keep half-applying.

Three feature-set variants, per binary target (Difficult-vs-not, Easy-vs-not):
  Baseline_939       the same 6 features as G0a, on the expanded, audited N=939
  NoRuggedness_939    5 features (drop Ruggedness) -- Phase 3 found it collinear
                       with Slope_deg (Pearson r=0.87, Spearman r=0.93) and not
                       independently significant (ANOVA p=0.051)
  Domain_939          Baseline_939 + Geological_Domain (one-hot) -- Phase 3's
                       strongest lead (chi2=133, p<0.0001, 98.8% coverage), but
                       domain partially proxies Region, so tested under BOTH
                       LOGO-cluster CV and leave-region-out with equal weight,
                       exactly the discipline that already caught this same
                       failure mode for raw region-as-feature (G2/F4)

Output: results/json/training/phase5_modeling_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
FEATURES_NO_RUGGED = ["Dist_to_Highway_m", "Slope_deg", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

log("Loading N=939 labeled catalog (733 original + 206 el_ouali_2026, audited) ...")
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
log(f"N={N}")

# Geological_Domain: one-hot, rare domains (<5 labeled sites) folded into 'Other'
# to avoid single-domain dummy columns that are really just noise/overfitting bait.
dom_counts = merged["Geological_Domain"].value_counts()
rare_domains = dom_counts[dom_counts < 5].index
merged["Geological_Domain_grouped"] = merged["Geological_Domain"].where(
    ~merged["Geological_Domain"].isin(rare_domains), "Other")
merged["Geological_Domain_grouped"] = merged["Geological_Domain_grouped"].fillna("Unknown")
domain_dummies = pd.get_dummies(merged["Geological_Domain_grouped"], prefix="Domain").astype(float)
log(f"Geological_Domain groups used (after folding rare<5 into 'Other'): {domain_dummies.shape[1]}")
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
region = merged["Region"].values
log(f"clusters={len(np.unique(cluster_ids))}")

RF_GRID  = [dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl)
            for ne in [100, 200, 400] for md in [3,4,5,6,7,8,None] for msl in [1,2,3,5]]
XGB_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,150,250] for md in [3,4,5,6] for lr in [0.03,0.08,0.15]]
GBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,200] for md in [2,3,4] for lr in [0.05,0.1]]
LGBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr, num_leaves=nl)
             for ne in [100,200] for md in [3,5,-1] for lr in [0.05,0.1] for nl in [15,31]]
MODEL_KINDS = ["RF", "XGB", "GBM", "LGBM"]
GRIDS = {"RF": RF_GRID, "XGB": XGB_GRID, "GBM": GBM_GRID, "LGBM": LGBM_GRID}

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr):
    # Confidence-weighting dropped project-wide (see module docstring) -- class-balance only.
    return compute_sample_weight("balanced", y_tr)

def _fit_predict_proba(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = combined_weight(y[tr])
    if kind == "XGB":
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict(X[te]), m.predict_proba(X[te])

def _one_fit_acc(kind, cfg, X, y, tr, te):
    pred, _ = _fit_predict_proba(kind, cfg, X, y, tr, te)
    return accuracy_score(y[te], pred)

def grid_search(X, y, groups, n_repeats=5, n_splits=10, n_jobs=-1):
    tasks = []
    for kind in MODEL_KINDS:
        for cfg in GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    log(f"  grid_search: {len(tasks)} flat fits queued")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_fit_acc)(kind, cfg, X, y, tr, te) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v)} for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def best_cfgs_from_grid(df_grid):
    return [(k, json.loads(df_grid[df_grid.model == k].iloc[0].config)) for k in MODEL_KINDS]

def _logo_fold_proba(best_cfgs, X, y, tr, te):
    probs = []
    for kind, cfg in best_cfgs:
        pred, proba = _fit_predict_proba(kind, cfg, X, y, tr, te)
        probs.append(proba)
    return te, np.mean(probs, axis=0)

def logo_cluster_cv_proba(best_cfgs, X, y, groups, n_jobs=-1):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    log(f"  logo_cluster_cv: {len(folds)} folds")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.zeros((len(y), 2))
    for te, p in results: proba[te] = p
    return proba

def leave_region_out(X, y, region, best_cfgs):
    rows = []
    for reg in pd.unique(region):
        tr = region != reg; te = region == reg
        if te.sum() < 10:
            rows.append({"region": reg, "n_test": int(te.sum()), "acc": None, "local_majority": None})
            continue
        sw = combined_weight(y[tr])
        estimators = [(k.lower(), make_model(k, cfg)) for k, cfg in best_cfgs]
        m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        m.fit(X[tr], y[tr], sample_weight=sw)
        p = m.predict(X[te])
        acc = accuracy_score(y[te], p)
        maj = pd.Series(y[te]).value_counts(normalize=True).max()
        n_correct = int(round(acc * te.sum())); n_correct_maj = int(round(maj * te.sum()))
        rows.append({"region": reg, "n_test": int(te.sum()), "acc": round(acc,3),
                     "local_majority": round(maj,3), "gap_pp": round((acc-maj)*100,1),
                     "degenerate": n_correct == n_correct_maj})
    return pd.DataFrame(rows)

def run_full(X, y, groups, region, tag):
    df_grid = grid_search(X, y, groups)
    best = best_cfgs_from_grid(df_grid)
    log(f"  [{tag}] best={[(k,c) for k,c in best]}")
    proba = logo_cluster_cv_proba(best, X, y, groups)
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y, preds)
    log(f"  [{tag}] LOGO-cluster CV acc={acc:.4f}")
    df_lro = leave_region_out(X, y, region, best)
    print(df_lro.to_string(index=False), flush=True)
    return dict(acc_logo_cluster=round(acc,4), leave_region_out=df_lro.to_dict("records"), best_configs=best)

ALL = {}
targets = [
    ("difficult", (merged["Expert_Merged"]=="Difficult").astype(int).values),
    ("easy", (merged["Expert_Merged"]=="Easy").astype(int).values),
]
feature_sets = {
    "Baseline_939": FEATURES_BASE,
    "NoRuggedness_939": FEATURES_NO_RUGGED,
    "Domain_939": FEATURES_DOMAIN,
}

for fset_name, cols in feature_sets.items():
    X = merged[cols].values
    for target_name, y in targets:
        tag = f"{fset_name}_{target_name}"
        print("\n"+"="*70, flush=True); print(tag, flush=True); print("="*70, flush=True)
        ALL[tag] = run_full(X, y, cluster_ids, region, tag)

out_path = os.path.join(BASE, "results", "json", "training", "phase5_modeling_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(ALL, f, indent=2, default=str)
log(f"Wrote {out_path}")

log("Summary:")
for target_name, _ in targets:
    print(f"  {target_name}:")
    for fset_name in feature_sets:
        acc = ALL[f"{fset_name}_{target_name}"]["acc_logo_cluster"]
        print(f"    {fset_name}: {acc}")
