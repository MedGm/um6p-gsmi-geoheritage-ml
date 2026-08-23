"""
data_audit/05_phase5_regional_models.py  (2026-08-18)

Per-region binary models on the expanded, audited N=939 set -- mirrors code/20's
G1 methodology exactly (grid_search, within-region 500m LOGO-cluster CV,
default vs balanced-accuracy-tuned threshold, report the better of the two),
minus confidence-weighting (dropped project-wide, see Phase 5 baseline script).

Expanded beyond G1's original 3 qualifying regions (Fés-Meknés, Béni
Mellal-Khénifra, Drâa-Tafilalet) to all 7 regions with enough total N to be
worth trying, since the el_ouali_2026 batch changed several regions'
viability substantially -- most notably Tanger-Tétouan-Al Hoceima
(52->128 sites, 6->21 Difficult), previously too thin for its own model.
Small/thin regions are still run and reported even when the result may be
degenerate (matches this project's established practice: report, don't hide).

Output: results/json/training/phase5_regional_models_results.json
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
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

log("Loading N=939 labeled catalog ...")
catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
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
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 939
log(f"N={N}")

REGIONS = ["Fés-Meknés", "Béni Mellal-Khénifra", "Tanger-Tétouan-Al Hoceima",
           "Drâa-Tafilalet", "Souss-Massa", "Marrakech-Safi", "Eddakhla-Oued Eddahab"]

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

def combined_weight(y_tr):
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

def grid_search(X, y, groups, n_repeats=5, n_splits=5, n_jobs=-1):
    # n_splits=5 (not 10) for regional fits -- smaller N per region makes 10-way
    # StratifiedGroupKFold splits too thin/unstable; 5-way matches G1's own scale.
    tasks = []
    for kind in MODEL_KINDS:
        for cfg in GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
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
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.zeros((len(y), 2))
    for te, p in results: proba[te] = p
    return proba, len(folds)

def best_threshold(y_true, proba_pos):
    best_t, best_ba = 0.5, balanced_accuracy_score(y_true, (proba_pos>=0.5).astype(int))
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba_pos>=t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba

results = []
for reg in REGIONS:
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    Xr = sub[FEATURES].values
    sub_clusters = cluster_of(sub)
    n_clusters = len(np.unique(sub_clusters))
    for target in ["Difficult", "Easy"]:
        yb = (sub["Expert_Merged"] == target).astype(int).values
        n, n_pos = len(yb), int(yb.sum())
        log(f"{reg} / {target}: N={n} pos={n_pos} clusters={n_clusters} -- running grid search ...")
        if n_pos < 3 or n_pos > n - 3:
            log(f"  SKIPPED -- degenerate class balance (pos={n_pos}/{n})")
            results.append(dict(region=reg, target=target, N=n, n_pos=n_pos, skipped="degenerate class balance"))
            continue
        df_grid = grid_search(Xr, yb, sub_clusters)
        best = best_cfgs_from_grid(df_grid)
        proba, n_folds = logo_cluster_cv_proba(best, Xr, yb, sub_clusters)
        preds_default = (proba[:,1] >= 0.5).astype(int)
        acc_default = accuracy_score(yb, preds_default)
        t_opt, _ = best_threshold(yb, proba[:,1])
        preds_tuned = (proba[:,1] >= t_opt).astype(int)
        acc_tuned = accuracy_score(yb, preds_tuned)
        maj = pd.Series(yb).value_counts(normalize=True).max()
        best_acc = max(acc_default, acc_tuned)
        log(f"  N={n} pos={n_pos} folds={n_folds} | default={acc_default:.3f} tuned({t_opt:.2f})={acc_tuned:.3f} "
            f"| BEST={best_acc:.3f} | local_majority={maj:.3f} | gap={100*(best_acc-maj):+.1f}pp")
        results.append(dict(region=reg, target=target, N=n, n_pos=n_pos, n_clusters=n_clusters,
                             acc_default=round(acc_default,3), acc_tuned=round(acc_tuned,3),
                             tuned_threshold=round(t_opt,2), best_acc=round(best_acc,3),
                             local_majority=round(maj,3), gap_pp=round((best_acc-maj)*100,1),
                             best_configs=best))

out_path = os.path.join(BASE, "results", "json", "training", "phase5_regional_models_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)
log(f"Wrote {out_path}")

log("\nSUMMARY:")
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
