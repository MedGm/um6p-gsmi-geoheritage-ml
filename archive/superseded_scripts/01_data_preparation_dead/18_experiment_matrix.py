"""
Post-audit experiment matrix (2026-08-14). Answers the questions raised by the
N=747 forensic audit (see memory / the published audit report) with a properly
PARALLELIZED pipeline instead of code/17's n_jobs=1-everywhere version.

Why this exists: code/17 did ~25,000 sequential single-threaded model fits
(grid search x CV x LOGO), which is why it took 8222s on the Z8 workstation
despite its 28 cores -- n_jobs=1 was set everywhere to avoid a prior stall
caused by naive nested n_jobs=-1 (thousands of tiny process-pool spawns).
The fix used here is the standard one: parallelize ONCE at the outer
(config x fold) level with a single joblib.Parallel pool, keep n_jobs=1 on
each individual estimator. Validated in the audit: 629-fold LOGO for one
model type ran in 135s on an 8-core laptop with this structure.

CRITICAL PROCESS FIX from the audit: LOGO-cluster CV (629 clusters, 87%
singleton) is close to leave-one-site-out and substantially overstates true
cross-region generalization -- direct leave-region-out testing scored AT OR
BELOW the local majority-class baseline in 6 of 8 regions. Every experiment
below reports BOTH metrics. Do not compare experiments using LOGO-cluster CV
alone -- an experiment can "improve" that number while not helping (or even
hurting) real generalization.

Experiments:
  E0  Baseline: full 6 features, 3-class + binary Difficult + binary Easy.
  E1  Distance-ablation, targeted at the audit's real finding: does dropping
      Dist_to_Highway_m / Dist_to_Settlement_m improve CROSS-SOURCE transfer
      (train=non-book/test=book and reverse), not just same-region CV?
  E2  Region-as-feature: does giving the model explicit region context close
      the LOGO-cluster gap? (Leave-region-out is not meaningful for this cell
      by construction -- a held-out region's one-hot column is unseen at
      train time -- so E2 reports LOGO-cluster CV only, explicitly flagged.)
  E3  Per-region binary models (regions with N>=100: Fes-Meknes, BMK) vs.
      that region's OWN local majority-class baseline, not the national one.

Run (Windows/PowerShell or Linux, foreground is fine -- this should take
single-digit minutes now, not hours):
    cd geosite_project1
    python code\\18_experiment_matrix.py 2>&1 | Tee-Object -FilePath experiment_matrix.log
  (Linux/mac: python3 code/18_experiment_matrix.py 2>&1 | tee experiment_matrix.log)
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
MOUT = os.path.join(BASE, "data", "model_outputs")
os.makedirs(MOUT, exist_ok=True)

FEATURES_FULL = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
                  "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
FEATURES_NODIST = ["Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction"]

# ── 1. Load labels + features (same source of truth as code/17) ───────────────
log("Loading labels from regional_label_sources/*.csv")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    labeled["source_file"] = os.path.basename(f)
    reasoning = labeled["Expert_Reasoning"].astype(str) if "Expert_Reasoning" in labeled.columns else ""
    labeled["is_book"] = pd.Series(reasoning).str.contains("Geoheritage of the Middle Atlas", na=False).values
    frames.append(labeled[["Locality_ID", "Expert_Class", "source_file", "is_book"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_FULL],
    on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
N = len(merged)
log(f"N={N} (legacy-formula rows already excluded upstream by blanking them in ttah_expert_labels.csv)")
log(f"class balance: {merged['Expert_Merged'].value_counts().to_dict()}")

# ── 2. Cluster assignment (500m haversine union-find, identical to code/17) ───
lat, lon = merged["Latitude_WGS84"].values, merged["Longitude_WGS84"].values
n = len(merged)
def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
D = haversine_matrix(lat, lon)
parent = list(range(n))
def find(x):
    while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(x, y):
    rx, ry = find(x), find(y)
    if rx != ry: parent[rx] = ry
for i in range(n):
    for j in range(i+1, n):
        if D[i, j] <= 500: union(i, j)
cluster_ids = np.array([find(i) for i in range(n)])
n_clusters = len(np.unique(cluster_ids))
log(f"clusters={n_clusters} (singleton rate {(pd.Series(cluster_ids).value_counts()==1).mean()*100:.1f}%)")

y3 = merged["Expert_Merged"].map({"Easy": 0, "Moderate": 1, "Difficult": 2}).values
CLASS3 = ["Easy", "Moderate", "Difficult"]

# ── 3. Flat-parallel grid + CV (THE fix: one Parallel() pool for everything) ──
RF_GRID  = [dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl)
            for ne in [100, 200, 400] for md in [3,4,5,6,7,8,None] for msl in [1,2,3,5]]
XGB_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,150,250] for md in [3,4,5,6] for lr in [0.03,0.08,0.15]]
GBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,200] for md in [2,3,4] for lr in [0.05,0.1]]

def make_model(kind, cfg, n_classes):
    if kind == "RF":
        return RandomForestClassifier(random_state=42, n_jobs=1, class_weight="balanced", **cfg)
    if kind == "XGB":
        return XGBClassifier(random_state=42, n_jobs=1,
                              eval_metric="mlogloss" if n_classes > 2 else "logloss", **cfg)
    if kind == "GBM":
        return GradientBoostingClassifier(random_state=42, **cfg)

def _one_fit(kind, cfg, X, y, tr, te, n_classes):
    m = make_model(kind, cfg, n_classes)
    m.fit(X[tr], y[tr])
    return accuracy_score(y[te], m.predict(X[te]))

def grid_search(X, y, groups, n_classes, n_repeats=5, n_splits=10, n_jobs=-1):
    """Flat-parallel: every (model, config, repeat, fold) is one independent task."""
    tasks = []
    for kind, grid in [("RF", RF_GRID), ("XGB", XGB_GRID), ("GBM", GBM_GRID)]:
        for cfg in grid:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    log(f"  grid_search: {len(tasks)} flat fits queued (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_fit)(kind, cfg, X, y, tr, te, n_classes) for kind, cfg, tr, te in tasks)
    # aggregate back to (kind, cfg) -> mean accuracy
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v), "std_acc": np.std(v)}
            for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def _logo_fold(estimators, X, y, tr, te):
    m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
    m.fit(X[tr], y[tr])
    return te, m.predict(X[te])

def logo_cluster_cv(best_cfgs, X, y, groups, n_classes, n_jobs=-1):
    estimators = [(k.lower(), make_model(k, cfg, n_classes)) for k, cfg in best_cfgs]
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    log(f"  logo_cluster_cv: {len(folds)} folds (n_jobs={n_jobs})")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold)(estimators, X, y, tr, te) for tr, te in folds)
    preds = np.zeros(len(y), dtype=int)
    for te, p in results:
        preds[te] = p
    return preds

def best_cfgs_from_grid(df_grid):
    out = []
    for kind in ["RF", "XGB", "GBM"]:
        row = df_grid[df_grid.model == kind].iloc[0]
        out.append((kind, json.loads(row.config)))
    return out

def leave_region_out(X, y, region, best_cfgs, n_classes):
    """Simple, fixed-hyperparameter (no per-region grid search) region holdout,
    using the same best configs found in the LOGO-cluster grid search, for
    apples-to-apples comparability. This is intentionally the strictest
    possible spatial split -- no cluster from the held-out region's territory
    is ever in training."""
    rows = []
    for reg in pd.unique(region):
        tr = region != reg
        te = region == reg
        if te.sum() < 10:
            rows.append({"region": reg, "n_test": int(te.sum()), "acc": None,
                         "local_majority": None, "note": "too few (<10), skipped"})
            continue
        estimators = [(k.lower(), make_model(k, cfg, n_classes)) for k, cfg in best_cfgs]
        m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        acc = accuracy_score(y[te], p)
        maj = pd.Series(y[te]).value_counts(normalize=True).max()
        rows.append({"region": reg, "n_test": int(te.sum()), "acc": round(acc,3),
                     "local_majority": round(maj,3), "gap_pp": round((acc-maj)*100,1)})
    return pd.DataFrame(rows)

def adjacent_error_rate(y_true, y_pred, n_classes):
    if n_classes != 3:
        return None
    wrong = y_true != y_pred
    if wrong.sum() == 0:
        return 1.0
    dist = np.abs(y_true[wrong] - y_pred[wrong])
    return round((dist == 1).mean(), 3)

def run_target(X, y, groups, region, n_classes, tag, class_names):
    t1 = time.time()
    df_grid = grid_search(X, y, groups, n_classes)
    df_grid.to_csv(os.path.join(MOUT, f"grid_{tag}_N{len(y)}.csv"), index=False)
    best = best_cfgs_from_grid(df_grid)
    log(f"  [{tag}] grid done [{time.time()-t1:.0f}s] best={[(k, c) for k,c in best]}")

    preds = logo_cluster_cv(best, X, y, groups, n_classes)
    acc_logo = accuracy_score(y, preds)
    adj = adjacent_error_rate(y, preds, n_classes)
    log(f"  [{tag}] LOGO-cluster CV acc={acc_logo:.4f}  adjacent-error-rate={adj}  [{time.time()-t1:.0f}s]")
    print(confusion_matrix(y, preds), flush=True)
    print(classification_report(y, preds, target_names=class_names, zero_division=0), flush=True)

    df_lro = leave_region_out(X, y, region, best, n_classes)
    log(f"  [{tag}] leave-region-out:")
    print(df_lro.to_string(index=False), flush=True)

    return dict(tag=tag, N=len(y), n_clusters=len(np.unique(groups)),
                acc_logo_cluster=round(acc_logo,4), adjacent_error_rate=adj,
                leave_region_out=df_lro.to_dict("records"), best_configs=best)

ALL_RESULTS = {}

# ── E0: baseline, full features ────────────────────────────────────────────────
print("\n" + "="*70, flush=True)
print("E0 -- BASELINE (full 6 features)", flush=True)
print("="*70, flush=True)
X_full = merged[FEATURES_FULL].values
region = merged["Region"].values
r = run_target(X_full, y3, cluster_ids, region, 3, "E0_3class_full", CLASS3)
ALL_RESULTS["E0_3class"] = r
for tag, ybin, names in [("E0_difficult_full", (merged["Expert_Merged"]=="Difficult").astype(int).values, ["Not-Difficult","Difficult"]),
                          ("E0_easy_full", (merged["Expert_Merged"]=="Easy").astype(int).values, ["Not-Easy","Easy"])]:
    r = run_target(X_full, ybin, cluster_ids, region, 2, tag, names)
    ALL_RESULTS[tag] = r

# ── E1: distance ablation, focused on CROSS-SOURCE transfer ───────────────────
print("\n" + "="*70, flush=True)
print("E1 -- DISTANCE ABLATION (drop Dist_to_Highway_m, Dist_to_Settlement_m)", flush=True)
print("Focused test: does this help CROSS-SOURCE transfer, not just same-region CV?", flush=True)
print("="*70, flush=True)
X_nodist = merged[FEATURES_NODIST].values
is_book = merged["is_book"].values

def cross_source_holdout(X, y, is_book, label):
    tr, te = ~is_book, is_book
    m = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    m.fit(X[tr], y[tr]); p = m.predict(X[te])
    acc_b = accuracy_score(y[te], p); maj_b = pd.Series(y[te]).value_counts(normalize=True).max()
    m2 = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    m2.fit(X[te], y[te]); p2 = m2.predict(X[tr])
    acc_nb = accuracy_score(y[tr], p2); maj_nb = pd.Series(y[tr]).value_counts(normalize=True).max()
    print(f"  [{label}] train=nonbook->test=book: acc={acc_b:.3f} (local majority={maj_b:.3f})", flush=True)
    print(f"  [{label}] train=book->test=nonbook: acc={acc_nb:.3f} (local majority={maj_nb:.3f})", flush=True)
    return dict(label=label, train_nonbook_test_book_acc=round(acc_b,3), train_nonbook_test_book_majority=round(maj_b,3),
                train_book_test_nonbook_acc=round(acc_nb,3), train_book_test_nonbook_majority=round(maj_nb,3))

ALL_RESULTS["E1_cross_source_transfer"] = [
    cross_source_holdout(X_full, y3, is_book, "full_features"),
    cross_source_holdout(X_nodist, y3, is_book, "no_distance_features"),
]
r = run_target(X_nodist, y3, cluster_ids, region, 3, "E1_3class_nodist", CLASS3)
ALL_RESULTS["E1_3class_nodist_logo"] = r

# ── E2: region as an explicit feature (LOGO-cluster only -- see docstring) ────
print("\n" + "="*70, flush=True)
print("E2 -- REGION AS FEATURE (LOGO-cluster CV only; leave-region-out is not", flush=True)
print("meaningful here since a held-out region's dummy column is unseen at train time)", flush=True)
print("="*70, flush=True)
region_dummies = pd.get_dummies(merged["Region"], prefix="region")
X_region = np.hstack([X_full, region_dummies.values])
df_grid = grid_search(X_region, y3, cluster_ids, 3)
best = best_cfgs_from_grid(df_grid)
preds = logo_cluster_cv(best, X_region, y3, cluster_ids, 3)
acc = accuracy_score(y3, preds)
adj = adjacent_error_rate(y3, preds, 3)
print(f"E2 LOGO-cluster CV acc={acc:.4f}  adjacent-error-rate={adj}  (E0 baseline was {ALL_RESULTS['E0_3class']['acc_logo_cluster']})", flush=True)
ALL_RESULTS["E2_region_as_feature"] = dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                                            delta_vs_E0_pp=round((acc-ALL_RESULTS["E0_3class"]["acc_logo_cluster"])*100,2))

# ── E3: per-region binary models vs LOCAL majority baseline ───────────────────
print("\n" + "="*70, flush=True)
print("E3 -- PER-REGION BINARY MODELS (regions with N>=100), vs that region's", flush=True)
print("OWN local majority-class baseline -- not the national one", flush=True)
print("="*70, flush=True)
e3_results = []
for reg in merged["Region"].value_counts()[merged["Region"].value_counts() >= 100].index:
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    Xr = sub[FEATURES_FULL].values
    sub_lat, sub_lon = sub["Latitude_WGS84"].values, sub["Longitude_WGS84"].values
    Dr = haversine_matrix(sub_lat, sub_lon)
    nr = len(sub)
    parentr = list(range(nr))
    def findr(x, parentr=parentr):
        while parentr[x] != x: parentr[x] = parentr[parentr[x]]; x = parentr[x]
        return x
    for i in range(nr):
        for j in range(i+1, nr):
            if Dr[i, j] <= 500:
                rx, ry = findr(i), findr(j)
                if rx != ry: parentr[rx] = ry
    sub_clusters = np.array([findr(i) for i in range(nr)])
    for target in ["Difficult", "Easy"]:
        yb = (sub["Expert_Merged"] == target).astype(int).values
        if yb.sum() < 15 or (1-yb).sum() < 15:
            print(f"  {reg} / {target}: skipped, too few positive or negative examples", flush=True)
            continue
        df_grid = grid_search(Xr, yb, sub_clusters, 2)
        best = best_cfgs_from_grid(df_grid)
        preds = logo_cluster_cv(best, Xr, yb, sub_clusters, 2)
        acc = accuracy_score(yb, preds)
        maj = pd.Series(yb).value_counts(normalize=True).max()
        print(f"  {reg:25s} / {target:10s}  N={nr:4d}  acc={acc:.3f}  local_majority={maj:.3f}  gap={(acc-maj)*100:+.1f}pp", flush=True)
        e3_results.append(dict(region=reg, target=target, N=nr, acc=round(acc,3),
                                local_majority=round(maj,3), gap_pp=round((acc-maj)*100,1)))
ALL_RESULTS["E3_per_region_binary"] = e3_results

# ── Save everything ─────────────────────────────────────────────────────────
out_path = os.path.join(MOUT, f"experiment_matrix_results_N{N}.json")
json.dump(ALL_RESULTS, open(out_path, "w"), indent=2, default=str)
print("\n" + "="*70, flush=True)
print(f"ALL EXPERIMENTS DONE. Results saved to {out_path}", flush=True)
print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
print("="*70, flush=True)
