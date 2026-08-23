"""
code/27_lcp_feature_test.py  (2026-08-17)

THE key test of the day's top-priority research item: does the least-cost-
path "effective distance" feature (code/23, Tobler friction surface +
land-cover + wadi-crossing penalty, live-computed per site) close the
karst/wadi cross-region generalization gap documented in the report's
Limitations section -- or is it just a noisier version of the straight-line
Dist_to_Highway_m it's meant to improve on?

Mirrors code/20's G0a/G0b methodology EXACTLY (same grid_search, same
StratifiedGroupKFold(5x10) config search, same LOGO-cluster CV via
LeaveOneGroupOut, same leave_region_out(), same confidence+balance sample
weighting) so results are directly comparable to the existing G0a_difficult
(0.7408) / G0a_easy (0.7326) headline numbers and to G0b's straight-line
OSM-trail-distance ablation (0.764 / 0.7271 -- which did NOT close the
Eddakhla Easy leave-region-out gap, still -20.0pp).

Two variants tested per binary target (Difficult-vs-not, Easy-vs-not):
  LCP_replace  6 features, Dist_to_Highway_m REPLACED by LeastCost_Effective_m
  LCP_add      7 features, LeastCost_Effective_m ADDED alongside Dist_to_Highway_m

G0a itself is NOT rerun here (identical grid/protocol/data/seeds to code/20's
already-saved run -- rerunning would just reproduce it at real compute cost);
its saved numbers are loaded from results/json/training/final_v2_results_N733.json
for the comparison.

Output: results/json/training/lcp_feature_test_results.json
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
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

log("Loading labels + LCP feature ...")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    cols = ["Locality_ID", "Expert_Class"]
    if "Confidence" in labeled.columns:
        cols.append("Confidence")
    else:
        labeled["Confidence"] = "Medium"; cols.append("Confidence")
    frames.append(labeled[cols])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
lcp = pd.read_csv(os.path.join(BASE, "data/final/dist_to_road_leastcost_m.csv"))[["Locality_ID", "LeastCost_Effective_m"]]

merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").merge(lcp, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged["conf_weight"] = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7)
N = len(merged)
assert N == 733
log(f"N={N}, LeastCost_Effective_m missing: {merged['LeastCost_Effective_m'].isna().sum()}")

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
conf_w = merged["conf_weight"].values
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

def make_model(kind, cfg, n_classes):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr, conf_tr):
    return conf_tr * compute_sample_weight("balanced", y_tr)

def _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes):
    m = make_model(kind, cfg, n_classes)
    sw = combined_weight(y[tr], cw[tr])
    if kind == "XGB" and n_classes == 2:
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict(X[te]), (m.predict_proba(X[te]) if hasattr(m, "predict_proba") else None)

def _one_fit_acc(kind, cfg, X, y, cw, tr, te, n_classes):
    pred, _ = _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes)
    return accuracy_score(y[te], pred)

def grid_search(X, y, cw, groups, n_classes, n_repeats=5, n_splits=10, n_jobs=-1):
    tasks = []
    for kind in MODEL_KINDS:
        for cfg in GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    log(f"  grid_search: {len(tasks)} flat fits queued")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_fit_acc)(kind, cfg, X, y, cw, tr, te, n_classes) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v)} for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def best_cfgs_from_grid(df_grid):
    return [(k, json.loads(df_grid[df_grid.model == k].iloc[0].config)) for k in MODEL_KINDS]

def _logo_fold_proba(best_cfgs, X, y, cw, tr, te, n_classes):
    probs = []
    for kind, cfg in best_cfgs:
        pred, proba = _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes)
        if proba is None: proba = np.eye(n_classes)[pred]
        probs.append(proba)
    return te, np.mean(probs, axis=0)

def logo_cluster_cv_proba(best_cfgs, X, y, cw, groups, n_classes, n_jobs=-1):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    log(f"  logo_cluster_cv: {len(folds)} folds")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, cw, tr, te, n_classes) for tr, te in folds)
    proba = np.zeros((len(y), n_classes))
    for te, p in results: proba[te] = p
    return proba

def leave_region_out(X, y, cw, region, best_cfgs, n_classes):
    rows = []
    for reg in pd.unique(region):
        tr = region != reg; te = region == reg
        if te.sum() < 10:
            rows.append({"region": reg, "n_test": int(te.sum()), "acc": None, "local_majority": None})
            continue
        sw = combined_weight(y[tr], cw[tr])
        estimators = [(k.lower(), make_model(k, cfg, n_classes)) for k, cfg in best_cfgs]
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

def run_full(X, y, cw, groups, region, n_classes, tag, names):
    df_grid = grid_search(X, y, cw, groups, n_classes)
    best = best_cfgs_from_grid(df_grid)
    log(f"  [{tag}] best={[(k,c) for k,c in best]}")
    proba = logo_cluster_cv_proba(best, X, y, cw, groups, n_classes)
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y, preds)
    log(f"  [{tag}] LOGO-cluster CV acc={acc:.4f}")
    df_lro = leave_region_out(X, y, cw, region, best, n_classes)
    print(df_lro.to_string(index=False), flush=True)
    return dict(acc_logo_cluster=round(acc,4), leave_region_out=df_lro.to_dict("records"), best_configs=best)

ALL = {}
targets = [
    ("difficult", (merged["Expert_Merged"]=="Difficult").astype(int).values, ["Not-Difficult","Difficult"]),
    ("easy", (merged["Expert_Merged"]=="Easy").astype(int).values, ["Not-Easy","Easy"]),
]
feature_sets = {
    "LCP_replace": ["LeastCost_Effective_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"],
    "LCP_add": FEATURES_BASE + ["LeastCost_Effective_m"],
}

for fset_name, cols in feature_sets.items():
    X = merged[cols].values
    for target_name, y, names in targets:
        tag = f"{fset_name}_{target_name}"
        print("\n"+"="*70, flush=True); print(tag, flush=True); print("="*70, flush=True)
        ALL[tag] = run_full(X, y, conf_w, cluster_ids, region, 2, tag, names)

out_path = os.path.join(BASE, "results", "json", "training", "lcp_feature_test_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(ALL, f, indent=2, default=str)
log(f"Wrote {out_path}")

log("Summary vs G0a baseline:")
g0a = json.load(open(os.path.join(BASE, "results/json/training/final_v2_results_N733.json")))
for target_name in ["difficult", "easy"]:
    base_acc = g0a[f"G0a_{target_name}"]["acc_logo_cluster"]
    print(f"  {target_name}: G0a baseline={base_acc}")
    for fset_name in feature_sets:
        acc = ALL[f"{fset_name}_{target_name}"]["acc_logo_cluster"]
        print(f"    {fset_name}: {acc}  (delta {100*(acc-base_acc):+.2f}pp)")
