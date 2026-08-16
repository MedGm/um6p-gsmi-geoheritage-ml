"""
Comprehensive final test battery (2026-08-14), run on Z8 only. Covers every
item from the post-audit methodology discussion judged worth testing:

H0  Cluster-radius sensitivity: 250m/500m/1km/2km haversine clustering,
    3-class target, tree ensemble (RF/XGB/GBM/LightGBM/CatBoost -- adds
    CatBoost as a 5th member in the same pass). Answers directly: is the
    500m choice load-bearing, or does accuracy stay roughly stable across
    radii (which would mean the "leakage" concern is real and not just an
    artifact of one specific threshold)?
H1  k-NN, geographic (haversine) and feature-space (standardized 6 GIS
    features), several k values, evaluated THREE ways for the same k:
    naive random CV (no spatial grouping at all -- this is the leakage
    DEMONSTRATION, k-NN is maximally sensitive to near-duplicate leakage),
    cluster-aware LOGO CV (500m), and leave-region-out. The gap between
    the naive number and the other two IS the leakage effect, made visible
    on the algorithm most susceptible to it, instead of just argued about.
H2  Soil_Class / Geology_Class / LULC_Class_Name_WorldCover added as
    categorical features (missing filled with an explicit "missing"
    category, not dropped -- only 269/734 labeled rows have non-null
    values for these, ~37% coverage, since they were only ever computed
    for the original 324-site catalog; results should be read with that
    coverage caveat in mind, not treated as a clean full-coverage feature).
H3  Data-quality diagnostic: does accuracy differ for rows with
    N_Raster_Cells_Imputed > 0 (127+26+12+10+1 = 176 of 1154 catalog sites
    have SOME imputed raster cell) vs. 0? No new model, just a breakdown
    of H0's 500m baseline OOF predictions. (Coordinate_Precision dropped
    from this diagnostic -- checked, it's constant "point" for all 1154
    catalog rows, zero variance, not a usable signal.)
H4  Weighted-voting ensemble (soft-vote weights proportional to each base
    model's own grid-search CV accuracy) vs. the existing simple-average
    ensemble, same baseline features.
H5  Logistic regression baseline (multinomial, standardized features) --
    missing from every baseline table produced so far; a reviewer will
    ask for one.
H6  Gaussian Process classification (RBF kernel, standardized features,
    one-vs-rest via sklearn's GaussianProcessClassifier) -- the principled,
    explicit version of "nearby things are correlated," as opposed to
    letting it happen implicitly (and uncontrolled) via near-duplicate
    leakage.

NOT included (deliberately, not an oversight): kriging/geostatistical
interpolation (would need a from-scratch ordinal-kriging implementation,
too much unvalidated new code to bundle into one unattended long run) and
multi-scale terrain features (needs a new live Copernicus DEM extraction
at a bigger window -- real GIS work, better done and validated locally
first, not sight-unseen in this script). Both are documented as future
work, not silently dropped.

Requires (Z8, before running):
    pip install lightgbm catboost
Script degrades gracefully (skips whichever is missing) if either isn't
installed, but you lose that model's contribution to H0/H4 if you skip
the install.

Run (Z8, foreground fine):
    cd geosite_project1
    pip install lightgbm catboost
    python .\\code\\22_comprehensive_final.py 2>&1 | Tee-Object -FilePath comprehensive_final.log
Expect roughly 60-100 minutes -- this is deliberately the biggest single
run of the session (7 sub-experiments). Every stage flushes progress as it
goes and writes its own section to the output JSON incrementally in spirit
(all at the end in practice, but each H-block prints its own full results
to the log immediately, so partial progress is never lost even if a later
block fails or the run is interrupted -- paste back whatever's in the log
even if it didn't finish).
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log(f"LightGBM: {HAVE_LGBM}  CatBoost: {HAVE_CATBOOST}")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
MOUT = os.path.join(BASE, "data", "model_outputs")
os.makedirs(MOUT, exist_ok=True)

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}
SOLID_REGIONS = ["Fés-Meknés", "Béni Mellal-Khénifra", "Drâa-Tafilalet"]

# ── Load ─────────────────────────────────────────────────────────────────────
log("Loading labels")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    if "Confidence" not in labeled.columns:
        labeled["Confidence"] = "Medium"
    frames.append(labeled[["Locality_ID", "Expert_Class", "Confidence"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
extra_cols = ["Region", "Latitude_WGS84", "Longitude_WGS84", "N_Raster_Cells_Imputed",
              "Soil_Class", "Geology_Class", "LULC_Class_Name_WorldCover"] + FEATURES
merged = all_labels.merge(catalog[["Locality_ID"] + extra_cols], on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["conf_weight"] = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7)
N = len(merged)
log(f"N={N}, class balance: {merged['Expert_Merged'].value_counts().to_dict()}")
log(f"Soil_Class coverage: {merged['Soil_Class'].notna().sum()}/{N}  "
    f"Geology_Class: {merged['Geology_Class'].notna().sum()}/{N}  "
    f"LULC_Class_Name: {merged['LULC_Class_Name_WorldCover'].notna().sum()}/{N}")

y3 = merged["Expert_Merged"].map({"Easy": 0, "Moderate": 1, "Difficult": 2}).values
CLASS3 = ["Easy", "Moderate", "Difficult"]
region = merged["Region"].values
conf_w = merged["conf_weight"].values

def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

FULL_D = haversine_matrix(merged["Latitude_WGS84"].values, merged["Longitude_WGS84"].values)

def cluster_at_radius(D, radius_m):
    n = D.shape[0]
    parent = list(range(n))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] <= radius_m:
                rx, ry = find(i), find(j)
                if rx != ry: parent[rx] = ry
    return np.array([find(i) for i in range(n)])

# ── Tree-ensemble machinery (same pattern as code/18-21) ──────────────────────
RF_GRID  = [dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl)
            for ne in [100, 200, 400] for md in [3,4,5,6,7,8,None] for msl in [1,2,3,5]]
XGB_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,150,250] for md in [3,4,5,6] for lr in [0.03,0.08,0.15]]
GBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,200] for md in [2,3,4] for lr in [0.05,0.1]]
LGBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr, num_leaves=nl)
             for ne in [100,200] for md in [3,5,-1] for lr in [0.05,0.1] for nl in [15,31]] if HAVE_LGBM else []
CATBOOST_GRID = [dict(iterations=it, depth=d, learning_rate=lr)
                  for it in [100,200] for d in [3,5,7] for lr in [0.05,0.1]] if HAVE_CATBOOST else []

MODEL_KINDS = ["RF", "XGB", "GBM"] + (["LGBM"] if HAVE_LGBM else []) + (["CAT"] if HAVE_CATBOOST else [])
GRIDS = {"RF": RF_GRID, "XGB": XGB_GRID, "GBM": GBM_GRID, "LGBM": LGBM_GRID, "CAT": CATBOOST_GRID}

def make_model(kind, cfg, n_classes):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1,
                                            eval_metric="mlogloss" if n_classes>2 else "logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)
    if kind == "CAT": return CatBoostClassifier(random_state=42, verbose=False, thread_count=1,
                                                 allow_writing_files=False, **cfg)

def combined_weight(y_tr, conf_tr):
    return conf_tr * compute_sample_weight("balanced", y_tr)

def _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes):
    m = make_model(kind, cfg, n_classes)
    sw = combined_weight(y[tr], cw[tr])
    if kind == "XGB" and n_classes == 2:
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
    if kind == "CAT":
        m.fit(X[tr], y[tr], sample_weight=sw)
    else:
        m.fit(X[tr], y[tr], sample_weight=sw)
    return m.predict(X[te]).reshape(-1), (m.predict_proba(X[te]) if hasattr(m,"predict_proba") else None)

def _one_fit_acc(kind, cfg, X, y, cw, tr, te, n_classes):
    pred, _ = _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes)
    return accuracy_score(y[te], pred)

def grid_search(X, y, cw, groups, n_classes, n_repeats=5, n_splits=10, n_jobs=-1, kinds=None):
    kinds = kinds or MODEL_KINDS
    tasks = []
    for kind in kinds:
        for cfg in GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    log(f"    grid_search: {len(tasks)} flat fits queued ({kinds})")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_fit_acc)(kind, cfg, X, y, cw, tr, te, n_classes) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v)} for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def best_cfgs_from_grid(df_grid, kinds=None):
    kinds = kinds or MODEL_KINDS
    return [(k, json.loads(df_grid[df_grid.model == k].iloc[0].config),
             float(df_grid[df_grid.model == k].iloc[0].mean_acc)) for k in kinds]

def _logo_fold_proba(best_cfgs, X, y, cw, tr, te, n_classes, weighted):
    probs, weights = [], []
    for kind, cfg, cvacc in best_cfgs:
        pred, proba = _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes)
        if proba is None: proba = np.eye(n_classes)[pred]
        probs.append(proba); weights.append(cvacc if weighted else 1.0)
    weights = np.array(weights); weights = weights / weights.sum()
    return te, np.tensordot(weights, np.array(probs), axes=(0,0))

def logo_cluster_cv_proba(best_cfgs, X, y, cw, groups, n_classes, n_jobs=-1, weighted=False):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    log(f"    logo_cluster_cv: {len(folds)} folds (weighted_vote={weighted})")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, cw, tr, te, n_classes, weighted) for tr, te in folds)
    proba = np.zeros((len(y), n_classes))
    for te, p in results: proba[te] = p
    return proba

def leave_region_out_tree(X, y, cw, region_arr, best_cfgs, n_classes, weighted=False):
    rows = []
    for reg in SOLID_REGIONS:
        tr = region_arr != reg; te = region_arr == reg
        sw = combined_weight(y[tr], cw[tr])
        probs, weights = [], []
        for kind, cfg, cvacc in best_cfgs:
            m = make_model(kind, cfg, n_classes)
            m.fit(X[tr], y[tr], sample_weight=sw)
            p = m.predict_proba(X[te]) if hasattr(m, "predict_proba") else np.eye(n_classes)[m.predict(X[te])]
            probs.append(p); weights.append(cvacc if weighted else 1.0)
        weights = np.array(weights); weights = weights/weights.sum()
        proba = np.tensordot(weights, np.array(probs), axes=(0,0))
        pred = proba.argmax(axis=1)
        acc = accuracy_score(y[te], pred)
        maj = pd.Series(y[te]).value_counts(normalize=True).max()
        n = te.sum()
        rows.append({"region": reg, "n_test": int(n), "acc": round(acc,3),
                     "local_majority": round(maj,3), "gap_pp": round((acc-maj)*100,1)})
    return pd.DataFrame(rows)

def adjacent_error_rate(y_true, y_pred, n_classes):
    if n_classes != 3: return None
    wrong = y_true != y_pred
    if wrong.sum() == 0: return 1.0
    return round((np.abs(y_true[wrong]-y_pred[wrong])==1).mean(), 3)

ALL = {}
X_base = merged[FEATURES].values

# ── H0: cluster-radius sensitivity (+ CatBoost folded in) ─────────────────────
print("\n"+"="*70, flush=True); print("H0 -- CLUSTER-RADIUS SENSITIVITY (250/500/1000/2000m), 3-class", flush=True); print("="*70, flush=True)
h0_results = {}
for radius in [250, 500, 1000, 2000]:
    clusters_r = cluster_at_radius(FULL_D, radius)
    nclust = len(np.unique(clusters_r))
    singleton_rate = (pd.Series(clusters_r).value_counts()==1).mean()
    log(f"  radius={radius}m: clusters={nclust} singleton_rate={singleton_rate*100:.1f}%")
    df_grid = grid_search(X_base, y3, conf_w, clusters_r, 3)
    best = best_cfgs_from_grid(df_grid)
    proba = logo_cluster_cv_proba(best, X_base, y3, conf_w, clusters_r, 3)
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y3, preds)
    adj = adjacent_error_rate(y3, preds, 3)
    log(f"  [radius={radius}m] LOGO-cluster CV acc={acc:.4f} adjacent={adj} n_clusters={nclust}")
    df_lro = leave_region_out_tree(X_base, y3, conf_w, region, best, 3) if radius == 500 else None
    if df_lro is not None:
        print(df_lro.to_string(index=False), flush=True)
    h0_results[radius] = dict(n_clusters=int(nclust), singleton_rate=round(singleton_rate,3),
                               acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                               leave_region_out=(df_lro.to_dict("records") if df_lro is not None else None))
ALL["H0_cluster_radius_sensitivity"] = h0_results
log("H0 summary: " + "  ".join(f"{r}m={h0_results[r]['acc_logo_cluster']}" for r in [250,500,1000,2000]))

# ── H1: k-NN, naive vs cluster-aware vs leave-region-out ───────────────────────
print("\n"+"="*70, flush=True); print("H1 -- k-NN: naive CV vs cluster-aware CV vs leave-region-out (leakage demo)", flush=True); print("="*70, flush=True)
cluster_500 = cluster_at_radius(FULL_D, 500)
X_scaled = StandardScaler().fit_transform(X_base)
h1_results = []
for space, Xk in [("geographic_haversine", None), ("feature_space_standardized", X_scaled)]:
    for k in [3, 5, 10, 15, 25]:
        if space == "geographic_haversine":
            knn = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
            Dmat = FULL_D
        else:
            knn = KNeighborsClassifier(n_neighbors=k)
            Dmat = Xk

        # naive random CV (NO spatial grouping -- the leakage demonstration)
        naive_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, te in skf.split(Dmat if space!="geographic_haversine" else np.zeros((N,1)), y3):
            if space == "geographic_haversine":
                m = KNeighborsClassifier(n_neighbors=k, metric="precomputed")
                m.fit(FULL_D[np.ix_(tr,tr)], y3[tr])
                p = m.predict(FULL_D[np.ix_(te,tr)])
            else:
                m = KNeighborsClassifier(n_neighbors=k)
                m.fit(Xk[tr], y3[tr])
                p = m.predict(Xk[te])
            naive_scores.append(accuracy_score(y3[te], p))
        naive_acc = np.mean(naive_scores)

        # cluster-aware LOGO CV (500m)
        preds_cluster = np.zeros(N, dtype=int)
        for tr, te in LeaveOneGroupOut().split(Dmat if space!="geographic_haversine" else np.zeros((N,1)), y3, groups=cluster_500):
            if space == "geographic_haversine":
                m = KNeighborsClassifier(n_neighbors=min(k,len(tr)), metric="precomputed")
                m.fit(FULL_D[np.ix_(tr,tr)], y3[tr])
                preds_cluster[te] = m.predict(FULL_D[np.ix_(te,tr)])
            else:
                m = KNeighborsClassifier(n_neighbors=min(k,len(tr)))
                m.fit(Xk[tr], y3[tr])
                preds_cluster[te] = m.predict(Xk[te])
        cluster_acc = accuracy_score(y3, preds_cluster)

        # leave-region-out on the 3 solid regions
        lro_accs = {}
        for reg in SOLID_REGIONS:
            tr = (region != reg); te = (region == reg)
            if space == "geographic_haversine":
                m = KNeighborsClassifier(n_neighbors=min(k,tr.sum()), metric="precomputed")
                m.fit(FULL_D[np.ix_(np.where(tr)[0], np.where(tr)[0])], y3[tr])
                p = m.predict(FULL_D[np.ix_(np.where(te)[0], np.where(tr)[0])])
            else:
                m = KNeighborsClassifier(n_neighbors=min(k,tr.sum()))
                m.fit(Xk[tr], y3[tr])
                p = m.predict(Xk[te])
            lro_accs[reg] = round(accuracy_score(y3[te], p), 3)

        log(f"  [{space} k={k}] naive_random_cv={naive_acc:.3f}  cluster_aware_500m={cluster_acc:.3f}  "
            f"leave_region_out={lro_accs}")
        h1_results.append(dict(space=space, k=k, naive_random_cv_acc=round(naive_acc,3),
                                cluster_aware_500m_acc=round(cluster_acc,3),
                                leave_region_out=lro_accs,
                                leakage_gap_pp=round((naive_acc-cluster_acc)*100,1)))
ALL["H1_knn_leakage_demo"] = h1_results

# ── H2: categorical features (Soil/Geology/LULC), missing-as-category ─────────
print("\n"+"="*70, flush=True); print("H2 -- + Soil_Class/Geology_Class/LULC_Class_Name (missing-as-category)", flush=True); print("="*70, flush=True)
cat_df = merged[["Soil_Class", "Geology_Class", "LULC_Class_Name_WorldCover"]].copy()
cat_df["Soil_Class"] = cat_df["Soil_Class"].fillna(-1).astype(str)
cat_df["Geology_Class"] = cat_df["Geology_Class"].fillna(-1).astype(str)
cat_df["LULC_Class_Name_WorldCover"] = cat_df["LULC_Class_Name_WorldCover"].fillna("missing")
cat_dummies = pd.get_dummies(cat_df, prefix=["soil","geo","lulc"])
X_h2 = np.hstack([X_base, cat_dummies.values])
log(f"  H2 feature count: {X_h2.shape[1]} (base 6 + {cat_dummies.shape[1]} categorical dummies)")
df_grid = grid_search(X_h2, y3, conf_w, cluster_500, 3)
best = best_cfgs_from_grid(df_grid)
proba = logo_cluster_cv_proba(best, X_h2, y3, conf_w, cluster_500, 3)
preds = proba.argmax(axis=1)
acc = accuracy_score(y3, preds)
adj = adjacent_error_rate(y3, preds, 3)
log(f"  [H2] LOGO-cluster CV acc={acc:.4f} adjacent={adj} (H0 500m baseline was {h0_results[500]['acc_logo_cluster']})")
df_lro = leave_region_out_tree(X_h2, y3, conf_w, region, best, 3)
print(df_lro.to_string(index=False), flush=True)
ALL["H2_categorical_features"] = dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                                       delta_vs_h0_500m_pp=round((acc-h0_results[500]['acc_logo_cluster'])*100,2),
                                       leave_region_out=df_lro.to_dict("records"),
                                       coverage_caveat="Soil/Geology/LULC non-null for only "
                                       f"{merged['Soil_Class'].notna().sum()}/{N} rows (~37%)")

# ── H3: data-quality diagnostic (uses H0's 500m baseline OOF predictions) ─────
print("\n"+"="*70, flush=True); print("H3 -- DATA-QUALITY DIAGNOSTIC (N_Raster_Cells_Imputed vs accuracy)", flush=True); print("="*70, flush=True)
df_grid_500 = grid_search(X_base, y3, conf_w, cluster_500, 3)
best_500 = best_cfgs_from_grid(df_grid_500)
proba_500 = logo_cluster_cv_proba(best_500, X_base, y3, conf_w, cluster_500, 3)
preds_500 = proba_500.argmax(axis=1)
merged["correct_h0"] = (preds_500 == y3)
merged["imputed_flag"] = (merged["N_Raster_Cells_Imputed"] > 0)
diag = merged.groupby("imputed_flag")["correct_h0"].agg(["mean","count"])
log("  accuracy by imputed-cell flag:")
print(diag.to_string(), flush=True)
ALL["H3_data_quality_diagnostic"] = diag.reset_index().to_dict("records")

# ── H4: weighted-voting ensemble vs simple average ─────────────────────────────
print("\n"+"="*70, flush=True); print("H4 -- WEIGHTED-VOTE (by each model's own CV acc) vs SIMPLE AVERAGE", flush=True); print("="*70, flush=True)
proba_simple = logo_cluster_cv_proba(best_500, X_base, y3, conf_w, cluster_500, 3, weighted=False)
acc_simple = accuracy_score(y3, proba_simple.argmax(axis=1))
proba_weighted = logo_cluster_cv_proba(best_500, X_base, y3, conf_w, cluster_500, 3, weighted=True)
acc_weighted = accuracy_score(y3, proba_weighted.argmax(axis=1))
log(f"  simple_average={acc_simple:.4f}  weighted_by_cv_score={acc_weighted:.4f}")
ALL["H4_weighted_vs_simple_voting"] = dict(simple_average=round(acc_simple,4), weighted=round(acc_weighted,4))

# ── H5: logistic regression baseline ───────────────────────────────────────────
print("\n"+"="*70, flush=True); print("H5 -- LOGISTIC REGRESSION BASELINE", flush=True); print("="*70, flush=True)
preds_lr = np.zeros(N, dtype=int)
for tr, te in LeaveOneGroupOut().split(X_scaled, y3, groups=cluster_500):
    sw = combined_weight(y3[tr], conf_w[tr])
    lr = LogisticRegression(max_iter=2000, multi_class="multinomial")
    lr.fit(X_scaled[tr], y3[tr], sample_weight=sw)
    preds_lr[te] = lr.predict(X_scaled[te])
acc_lr = accuracy_score(y3, preds_lr)
log(f"  Logistic regression LOGO-cluster CV acc={acc_lr:.4f} (tree ensemble H0 500m was {h0_results[500]['acc_logo_cluster']})")
ALL["H5_logistic_regression_baseline"] = dict(acc_logo_cluster=round(acc_lr,4))

# ── H6: Gaussian Process classification ────────────────────────────────────────
# NOTE: GP fit cost is O(n^3) per fold (Laplace approximation, several Newton
# steps each doing a Cholesky decomposition), and one_vs_rest for 3 classes
# multiplies that by 3 fits per fold. Full 622-fold LOGO would plausibly take
# HOURS by itself. Using StratifiedGroupKFold(n_splits=10) instead -- still
# cluster-aware (same `cluster_500` groups, no cluster ever split across
# train/test), just far fewer/larger folds. This is a deliberate, disclosed
# deviation from the exact LOGO-cluster protocol used everywhere else in this
# session, made for GP's tractability specifically -- not a silent shortcut.
print("\n"+"="*70, flush=True); print("H6 -- GAUSSIAN PROCESS CLASSIFICATION (RBF kernel, 10-fold group CV -- NOT full LOGO, see comment)", flush=True); print("="*70, flush=True)
preds_gp = np.full(N, -1, dtype=int)
kernel = 1.0 * RBF(length_scale=1.0)
t_gp = time.time()
gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
n_gp_folds = 0
for tr, te in gkf.split(X_scaled, y3, groups=cluster_500):
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1, multi_class="one_vs_rest")
    gp.fit(X_scaled[tr], y3[tr])
    preds_gp[te] = gp.predict(X_scaled[te])
    n_gp_folds += 1
    log(f"    GP fold {n_gp_folds}/10 done  [{time.time()-t_gp:.0f}s]")
covered = preds_gp >= 0
acc_gp = accuracy_score(y3[covered], preds_gp[covered])
log(f"  Gaussian Process 10-fold group CV acc={acc_gp:.4f} (covered {covered.sum()}/{N})  [{time.time()-t_gp:.0f}s]  "
    f"(tree ensemble H0 500m was {h0_results[500]['acc_logo_cluster']})")
ALL["H6_gaussian_process"] = dict(acc_group_cv_10fold=round(acc_gp,4), n_covered=int(covered.sum()),
                                   note="10-fold StratifiedGroupKFold, not full LOGO -- see script docstring, GP is O(n^3)/fold")

out_path = os.path.join(MOUT, f"comprehensive_final_results_N{N}.json")
json.dump(ALL, open(out_path, "w"), indent=2, default=str)
print("\n"+"="*70, flush=True)
print(f"ALL DONE. Results saved to {out_path}", flush=True)
print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
print("="*70, flush=True)
