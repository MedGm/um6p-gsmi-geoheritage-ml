"""
Tiny-region training-exclusion ablation (2026-08-14). Standalone, focused,
run on Z8 -- answers one question: do the sub-30-example regions
(Eddakhla-Oued Eddahab 25, Marrakech-Safi 24, Rabat-Sale-Kenitra 21,
Laayoune-Sakia El Hamra 5, Guelmim-Oued Noun 5, Grand Casablanca-Settat 5 --
~90 rows combined, ~12% of N=733) hurt the pooled model's leave-region-out
performance on the 3 solid regions (Fes-Meknes, BMK, Drâa-Tafilalet) by
being IN the training pool, even though they were never the thing being
evaluated?

Method: two training pools --
  POOL_ALL   = everything (baseline, matches final_v2's G0a exactly: same
               features, same confidence+balance weighting, same 4-model
               ensemble RF/XGB/GBM/LightGBM)
  POOL_TRIM  = POOL_ALL with the 6 sub-30 regions removed entirely from
               TRAINING (not just skipped at evaluation time, which was
               already happening via the n_test<10 guard elsewhere --
               this removes them from every fold's training set too)
For both pools: LOGO-cluster CV (on whatever's left after trimming) AND
leave-region-out specifically on Fes-Meknes / BMK / Drâa-Tafilalet (the
only regions with enough test examples to mean anything), trained with
and without the tiny regions in the training side. Compare directly.

Run (Z8, foreground fine):
    cd geosite_project1
    python .\\code\\21_tiny_region_ablation.py 2>&1 | Tee-Object -FilePath tiny_region_ablation.log
Expect ~15-25 minutes (much smaller scope than code/18-20 -- 2 pools x
3 targets x grid+LOGO, no per-region binary, no trail feature, no G2).
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

log(f"LightGBM available: {HAVE_LGBM}")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
MOUT = os.path.join(BASE, "data", "model_outputs")
os.makedirs(MOUT, exist_ok=True)

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}
SUB30_REGIONS = ["Eddakhla-Oued Eddahab", "Marrakech-Safi", "Rabat-Salé-Kénitra",
                  "Laayoune-Sakia El Hamra", "Guelmim-Oued Noun", "Grand Casablanca-Settat"]
SOLID_REGIONS = ["Fés-Meknés", "Béni Mellal-Khénifra", "Drâa-Tafilalet"]

# ── Load (same as code/20) ──────────────────────────────────────────────────
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
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["conf_weight"] = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7)
log(f"N={len(merged)}, sub-30 regions total: {merged['Region'].isin(SUB30_REGIONS).sum()}")

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
             for ne in [100,200] for md in [3,5,-1] for lr in [0.05,0.1] for nl in [15,31]] if HAVE_LGBM else []
MODEL_KINDS = ["RF", "XGB", "GBM"] + (["LGBM"] if HAVE_LGBM else [])
GRIDS = {"RF": RF_GRID, "XGB": XGB_GRID, "GBM": GBM_GRID, "LGBM": LGBM_GRID}

def make_model(kind, cfg, n_classes):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1,
                                            eval_metric="mlogloss" if n_classes>2 else "logloss", **cfg)
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
    return m.predict(X[te]), (m.predict_proba(X[te]) if hasattr(m,"predict_proba") else None)

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
    log(f"    grid_search: {len(tasks)} flat fits queued")
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
    log(f"    logo_cluster_cv: {len(folds)} folds")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, cw, tr, te, n_classes) for tr, te in folds)
    proba = np.zeros((len(y), n_classes))
    for te, p in results: proba[te] = p
    return proba

def adjacent_error_rate(y_true, y_pred, n_classes):
    if n_classes != 3: return None
    wrong = y_true != y_pred
    if wrong.sum() == 0: return 1.0
    return round((np.abs(y_true[wrong]-y_pred[wrong])==1).mean(), 3)

def leave_solid_region_out(full_merged, feature_cols, best_cfgs, n_classes, y_col_fn, exclude_tiny):
    """Train on all regions except the held-out solid region (and, if
    exclude_tiny, also except the 6 sub-30 regions); test on the held-out
    solid region. Mirrors code/18-20's leave_region_out but scoped to only
    the 3 solid regions as test targets, with an explicit training-pool
    toggle for the tiny regions."""
    rows = []
    for reg in SOLID_REGIONS:
        if exclude_tiny:
            train_mask = (~full_merged["Region"].isin(SUB30_REGIONS)) & (full_merged["Region"] != reg)
        else:
            train_mask = full_merged["Region"] != reg
        test_mask = full_merged["Region"] == reg
        Xtr = full_merged.loc[train_mask, feature_cols].values
        Xte = full_merged.loc[test_mask, feature_cols].values
        y_all = y_col_fn(full_merged)
        ytr, yte = y_all[train_mask.values], y_all[test_mask.values]
        cwtr = full_merged.loc[train_mask, "conf_weight"].values
        sw = combined_weight(ytr, cwtr)
        estimators = [(k.lower(), make_model(k, cfg, n_classes)) for k, cfg in best_cfgs]
        m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        m.fit(Xtr, ytr, sample_weight=sw)
        p = m.predict(Xte)
        acc = accuracy_score(yte, p)
        maj = pd.Series(yte).value_counts(normalize=True).max()
        n = len(yte)
        nc = int(round(acc*n)); ncm = int(round(maj*n))
        rows.append({"region": reg, "n_train": int(train_mask.sum()), "n_test": n,
                     "acc": round(acc,3), "local_majority": round(maj,3),
                     "gap_pp": round((acc-maj)*100,1), "degenerate": nc==ncm})
    return pd.DataFrame(rows)

ALL = {}
y3_fn = lambda df: df["Expert_Merged"].map({"Easy":0,"Moderate":1,"Difficult":2}).values
ydiff_fn = lambda df: (df["Expert_Merged"]=="Difficult").astype(int).values
yeasy_fn = lambda df: (df["Expert_Merged"]=="Easy").astype(int).values

for pool_name, exclude_tiny in [("POOL_ALL", False), ("POOL_TRIM_sub30_excluded", True)]:
    print("\n"+"="*70, flush=True)
    print(f"{pool_name} (exclude_tiny_from_training={exclude_tiny})", flush=True)
    print("="*70, flush=True)

    if exclude_tiny:
        pool = merged[~merged["Region"].isin(SUB30_REGIONS)].reset_index(drop=True)
    else:
        pool = merged.reset_index(drop=True)
    pool_clusters = cluster_of(pool)
    log(f"  pool N={len(pool)}, clusters={len(np.unique(pool_clusters))}")
    Xp = pool[FEATURES].values
    cwp = pool["conf_weight"].values

    pool_results = {}
    for tag, y_fn, n_classes, names in [
        ("3class", y3_fn, 3, ["Easy","Moderate","Difficult"]),
        ("difficult", ydiff_fn, 2, ["Not-Difficult","Difficult"]),
        ("easy", yeasy_fn, 2, ["Not-Easy","Easy"]),
    ]:
        y = y_fn(pool)
        df_grid = grid_search(Xp, y, cwp, pool_clusters, n_classes)
        best = best_cfgs_from_grid(df_grid)
        proba = logo_cluster_cv_proba(best, Xp, y, cwp, pool_clusters, n_classes)
        preds = proba.argmax(axis=1)
        acc = accuracy_score(y, preds)
        adj = adjacent_error_rate(y, preds, n_classes)
        log(f"  [{pool_name}/{tag}] LOGO-cluster CV acc={acc:.4f} adjacent={adj}")
        if n_classes > 1:
            print(confusion_matrix(y, preds), flush=True)
            print(classification_report(y, preds, target_names=names, zero_division=0), flush=True)

        # leave-solid-region-out, using the FULL merged set as the candidate training
        # universe (so POOL_TRIM's "exclude_tiny" toggle actually removes the tiny
        # regions from the training side of this specific eval too)
        df_lro = leave_solid_region_out(merged, FEATURES, best, n_classes, y_fn, exclude_tiny)
        log(f"  [{pool_name}/{tag}] leave-solid-region-out:")
        print(df_lro.to_string(index=False), flush=True)

        pool_results[tag] = dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                                  leave_solid_region_out=df_lro.to_dict("records"), best_configs=best)
    ALL[pool_name] = pool_results

# ── direct comparison table ────────────────────────────────────────────────
print("\n"+"="*70, flush=True)
print("DIRECT COMPARISON: does excluding sub-30 regions from TRAINING move", flush=True)
print("the 3 solid regions' leave-region-out numbers?", flush=True)
print("="*70, flush=True)
for tag in ["3class","difficult","easy"]:
    a = pd.DataFrame(ALL["POOL_ALL"][tag]["leave_solid_region_out"]).set_index("region")["acc"]
    b = pd.DataFrame(ALL["POOL_TRIM_sub30_excluded"][tag]["leave_solid_region_out"]).set_index("region")["acc"]
    print(f"\n{tag}:", flush=True)
    for reg in SOLID_REGIONS:
        delta = b[reg]-a[reg]
        print(f"  {reg:25s} POOL_ALL={a[reg]:.3f}  POOL_TRIM={b[reg]:.3f}  delta={delta*100:+.1f}pp", flush=True)

out_path = os.path.join(MOUT, f"tiny_region_ablation_results.json")
json.dump(ALL, open(out_path, "w"), indent=2, default=str)
print(f"\nSaved to {out_path}. Total elapsed: {time.time()-t0:.0f}s", flush=True)
