"""
data_audit/24_paper2_best_feature_per_region.py  (2026-08-23)

Paper 2 methodology: each region gets whichever feature set actually helps
it, not one feature set forced uniformly across all regions (user's explicit
call, 2026-08-22 -- infra features helped Marrakech-Safi/TTAH nationally but
hurt Souss-Massa/Rabat-Salé-Kénitra, so a single fixed set can't be optimal
everywhere). Tests three candidate feature sets per region/target, same G1
methodology as data_audit/05 (grid_search + 500m LOGO-cluster CV + threshold
tuning), and reports the best of the three per region/target:

  Baseline  6 terrain/road features (already run in data_audit/05, reused
            here rather than rerun)
  Domain    Baseline + one-hot Geological_Domain (rare domains folded to
            'Other', matches Phase 5 national methodology)
  Infra     Baseline + tourism-POI-density + nearest-settlement-type
            (the feature that drove today's national Easy-classifier gain)

Output: results/json/training/phase5_paper2_best_feature_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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
assert N == 939

SENTINEL_DIST_M = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
settle_dummies = pd.get_dummies(merged["nearest_settlement_type"], prefix="Settlement").astype(float)
merged = pd.concat([merged, settle_dummies], axis=1)
INFRA_COLS = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"] + list(settle_dummies.columns)

log(f"N={N}")

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

def combined_weight(y_tr): return compute_sample_weight("balanced", y_tr)

def fit_predict_proba(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = combined_weight(y[tr])
    if kind == "XGB":
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict(X[te]), m.predict_proba(X[te])

def one_fit_acc(kind, cfg, X, y, tr, te):
    pred, _ = fit_predict_proba(kind, cfg, X, y, tr, te)
    return accuracy_score(y[te], pred)

def grid_search(X, y, groups, n_repeats=5, n_splits=5, n_jobs=-1):
    tasks = []
    for kind in MODEL_KINDS:
        for cfg in GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_fit_acc)(kind, cfg, X, y, tr, te) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v)} for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def best_cfgs_from_grid(df_grid):
    return [(k, json.loads(df_grid[df_grid.model == k].iloc[0].config)) for k in MODEL_KINDS]

def logo_fold_proba(best_cfgs, X, y, tr, te):
    probs = [fit_predict_proba(k, c, X, y, tr, te)[1] for k, c in best_cfgs]
    return te, np.mean(probs, axis=0)

def logo_cluster_cv_proba(best_cfgs, X, y, groups, n_jobs=-1):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(logo_fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.zeros((len(y), 2))
    for te, p in results: proba[te] = p
    return proba

def best_threshold(y_true, proba_pos):
    best_t, best_ba = 0.5, balanced_accuracy_score(y_true, (proba_pos>=0.5).astype(int))
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba_pos>=t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba

def run_variant(Xr, yb, groups):
    df_grid = grid_search(Xr, yb, groups)
    best = best_cfgs_from_grid(df_grid)
    proba = logo_cluster_cv_proba(best, Xr, yb, groups)
    acc_default = accuracy_score(yb, (proba[:,1]>=0.5).astype(int))
    t_opt, _ = best_threshold(yb, proba[:,1])
    acc_tuned = accuracy_score(yb, (proba[:,1]>=t_opt).astype(int))
    return dict(acc_default=round(acc_default,4), acc_tuned=round(acc_tuned,4),
                tuned_threshold=round(t_opt,2), best_acc=round(max(acc_default,acc_tuned),4),
                best_configs=best)

REGIONS = ["Fés-Meknés", "Béni Mellal-Khénifra", "Tanger-Tétouan-Al Hoceima",
           "Drâa-Tafilalet", "Souss-Massa", "Marrakech-Safi", "Eddakhla-Oued Eddahab"]

baseline_results = {(r["region"], r["target"]): r for r in json.load(
    open(os.path.join(BASE, "results/json/training/phase5_regional_models_results.json"))) if not r.get("skipped")}

results = []
for reg in REGIONS:
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    groups = cluster_of(sub)
    dom_counts = sub["Geological_Domain"].value_counts()
    rare = dom_counts[dom_counts < 5].index
    sub["Domain_grouped"] = sub["Geological_Domain"].where(~sub["Geological_Domain"].isin(rare), "Other").fillna("Unknown")
    dom_dummies = pd.get_dummies(sub["Domain_grouped"], prefix="Domain").astype(float)
    sub = pd.concat([sub, dom_dummies], axis=1)

    for target in ["Difficult", "Easy"]:
        yb = (sub["Expert_Merged"] == target).astype(int).values
        n, n_pos = len(yb), int(yb.sum())
        log(f"{reg} / {target}: N={n} pos={n_pos}")
        if n_pos < 3 or n_pos > n - 3:
            log("  SKIPPED -- degenerate class balance")
            continue

        variants = {}
        base_r = baseline_results.get((reg, target))
        variants["Baseline"] = dict(best_acc=base_r["best_acc"], acc_default=base_r["acc_default"],
                                     acc_tuned=base_r["acc_tuned"], tuned_threshold=base_r["tuned_threshold"],
                                     best_configs=base_r["best_configs"]) if base_r else None

        Xr_domain = sub[FEATURES_BASE + list(dom_dummies.columns)].values
        log("  running Domain variant ...")
        variants["Domain"] = run_variant(Xr_domain, yb, groups)

        Xr_infra = sub[FEATURES_BASE + INFRA_COLS].values
        log("  running Infra variant ...")
        variants["Infra"] = run_variant(Xr_infra, yb, groups)

        best_variant = max([v for v in variants if variants[v] is not None], key=lambda v: variants[v]["best_acc"])
        log(f"  {reg}/{target}: Baseline={variants['Baseline']['best_acc'] if variants['Baseline'] else None} "
            f"Domain={variants['Domain']['best_acc']} Infra={variants['Infra']['best_acc']} -> BEST={best_variant}")

        results.append(dict(region=reg, target=target, N=n, n_pos=n_pos,
                             local_majority=base_r["local_majority"] if base_r else None,
                             variants=variants, best_variant=best_variant,
                             best_acc=variants[best_variant]["best_acc"]))

out_path = os.path.join(BASE, "results/json/training/phase5_paper2_best_feature_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
log(f"\nWrote {out_path}")

log("\nSUMMARY:")
for r in results:
    print(f"  {r['region']:28s} {r['target']:10s} best={r['best_variant']:9s} acc={r['best_acc']:.3f} local_maj={r['local_majority']}")
