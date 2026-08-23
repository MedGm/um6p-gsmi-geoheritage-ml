"""
data_audit/14_phase5_3class_rerun.py  (2026-08-22)

PV 2026-08-18, section 2: "the direct 3-class model does not give satisfactory
results -- why? needs deeper analysis." The only prior answer on record was
one line (tried early, abandoned, underperformed the two-binary approach) --
from BEFORE this session's full data audit, batch integration, and
confidence-weighting removal. This reruns the direct 3-class (Easy/Moderate/
Difficult) model from scratch with everything current:
  - N=939 (733 original + 206 el_ouali_2026, fully audited)
  - No confidence-weighting (class-balance only, per project-wide retirement)
  - Same grid_search / StratifiedGroupKFold(5x10) / 500m LOGO-cluster CV /
    leave_region_out methodology as every other Phase 5 script

Compares directly against the two-binary approach's effective 3-class
accuracy (via the ordinal Frank & Hall combination already tested in this
session: 0.5442) to give a real, current answer to "why does 3-class
underperform" instead of a stale one-liner.

Output: results/json/training/phase5_3class_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
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
CLASSES = ["Easy", "Moderate", "Difficult"]

log("Loading N=939 labeled catalog (733 original + 206 el_ouali_2026, audited) ...")
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
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 939
log(f"N={N}, class distribution: {merged['Expert_Merged'].value_counts().to_dict()}")

y = merged["Expert_Merged"].map({c: i for i, c in enumerate(CLASSES)}).values
X = merged[FEATURES_BASE].values
region = merged["Region"].values

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
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="mlogloss", num_class=3, **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, objective="multiclass", num_class=3, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def _fit_predict_proba(kind, cfg, X, y, tr, te):
    m = make_model(kind, cfg)
    sw = combined_weight(y[tr])
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
    proba = np.zeros((len(y), 3))
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

log("Running grid search (3-class direct) ...")
df_grid = grid_search(X, y, cluster_ids)
best = best_cfgs_from_grid(df_grid)
log(f"best={[(k,c) for k,c in best]}")

proba = logo_cluster_cv_proba(best, X, y, cluster_ids)
preds = proba.argmax(axis=1)
acc = accuracy_score(y, preds)
log(f"3-class direct LOGO-cluster CV acc={acc:.4f}")

cm = confusion_matrix(y, preds)
log("Confusion matrix (rows=true, cols=predicted), order Easy/Moderate/Difficult:")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

log("Leave-region-out ...")
df_lro = leave_region_out(X, y, region, best)
print(df_lro.to_string(index=False), flush=True)

majority_acc = pd.Series(y).value_counts(normalize=True).max()
out = dict(
    acc_logo_cluster=round(acc, 4),
    majority_baseline=round(majority_acc, 4),
    confusion_matrix=cm.tolist(),
    classes=CLASSES,
    leave_region_out=df_lro.to_dict("records"),
    best_configs=best,
)
out_path = os.path.join(BASE, "results", "json", "training", "phase5_3class_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
log(f"Wrote {out_path}")

log("\nSummary:")
log(f"  3-class direct (N=939, audited, no confidence-weighting): acc={acc:.4f}")
log(f"  majority-class baseline: {majority_acc:.4f}")
log(f"  ordinal Frank&Hall combination of the two binaries (already tested this session): 0.5442")
log(f"  naive independent-threshold combination of the two binaries: 0.5453")
