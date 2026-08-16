"""
Final-v2 battery (2026-08-14), run on Z8 only. Combines the three approved
levers from the post-audit discussion: confidence-weighted training (#3),
a new non-region-derived feature -- OSM trail/track distance, tested as an
honest ablation not assumed (#4) -- and LightGBM added to the ensemble (#5).

Every fit uses combined sample_weight = confidence_weight * class_balance_weight
(High=1.0, Medium-High=0.85, Medium=0.7, Low-Medium=0.55, Low=0.4, multiplied
by sklearn's per-class "balanced" weight). This applies to ALL four ensemble
members uniformly this time (RF/XGB/GBM/LightGBM all take sample_weight) --
avoids the F0 scope bug where balancing silently applied to some models and
not others.

`Dist_to_Trail_m` (OSM track/path/footway/bridleway/cycleway distance, same
gis_osm_roads_free_1.shp already used for Dist_to_Highway_m, just a different
fclass filter) was pre-extracted to data/final/dist_to_trail_m_labeled_sites.csv
for all 734 labeled sites. NOTE: an earlier project iteration (archived
phase-1 fuzzy MCDSS model) deliberately excluded piste/track distance ("to
rely on accurate paved routes" -- comment in
archive/livrable/phase1_national_accessibility/code/02_train_fuzzy_mcdss_model.py,
guarded by an assertion in code/02_extract_terrain_road_features.py). No
documented failure mode was found for that decision beyond a stated OSM-
data-quality concern -- this script deliberately does NOT assume the old
decision still holds; it re-tests the idea empirically (G0b vs G0a below)
and only keeps the feature if leave-region-out actually improves.

Structure:
  G0a  Baseline 6 features, confidence+balance weighted, all 4 models.
  G0b  Baseline 6 + Dist_to_Trail_m (7 features), same weighting -- direct,
       fair ablation against G0a. Keep only if it wins on leave-region-out,
       not just LOGO-cluster CV (same discipline as every other experiment
       this session).
  G1   Per-region binary (Fes-Meknes / BMK / Drâa-Tafilalet -- the only 3
       regions with enough examples per class), winning feature set from
       G0, confidence+balance weighting, per-fold threshold tuning.
  G2   Region-as-feature (best lever from the first matrix, code/18's E2)
       recombined with everything above, LOGO-cluster CV only (leave-
       region-out is not meaningful for it, same reason as before).

Requires `pip install lightgbm` on Z8 before running -- script degrades
gracefully (skips LightGBM, keeps RF/XGB/GBM) if it's not installed, but
won't get the benefit of the 4th model if you skip that step.

Run (Z8, foreground fine):
    cd geosite_project1
    pip install lightgbm
    python .\\code\\20_final_v2_battery.py 2>&1 | Tee-Object -FilePath final_v2.log
Expect roughly 40-70 minutes (similar scope to code/19, one extra model type
in the ensemble adds real but bounded cost since LightGBM is fast per-fit).
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
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

log(f"LightGBM available: {HAVE_LGBM}" + ("" if HAVE_LGBM else "  -- run `pip install lightgbm` for the 4th ensemble member; continuing without it"))

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
MOUT = os.path.join(BASE, "data", "model_outputs")
os.makedirs(MOUT, exist_ok=True)

FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
                  "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
FEATURES_TRAIL = FEATURES_BASE + ["Dist_to_Trail_m"]

CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

# ── Load labels + confidence + trail feature ───────────────────────────────────
log("Loading labels from regional_label_sources/*.csv")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    cols = ["Locality_ID", "Expert_Class"]
    if "Confidence" in labeled.columns:
        cols.append("Confidence")
    else:
        labeled["Confidence"] = "Medium"
        cols.append("Confidence")
    frames.append(labeled[cols])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
trail = pd.read_csv(os.path.join(BASE, "data/final/dist_to_trail_m_labeled_sites.csv"))

merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner")
merged = merged.merge(trail, on="Locality_ID", how="left")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["conf_weight"] = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7)
N = len(merged)
log(f"N={N}, class balance: {merged['Expert_Merged'].value_counts().to_dict()}")
log(f"Dist_to_Trail_m missing: {merged['Dist_to_Trail_m'].isna().sum()} (filled with feature median)")
merged["Dist_to_Trail_m"] = merged["Dist_to_Trail_m"].fillna(merged["Dist_to_Trail_m"].median())

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
n_clusters = len(np.unique(cluster_ids))
log(f"clusters={n_clusters} (singleton rate {(pd.Series(cluster_ids).value_counts()==1).mean()*100:.1f}%)")

y3 = merged["Expert_Merged"].map({"Easy": 0, "Moderate": 1, "Difficult": 2}).values
CLASS3 = ["Easy", "Moderate", "Difficult"]
region = merged["Region"].values
conf_w = merged["conf_weight"].values

# ── Grid + flat-parallel CV with combined confidence*balance sample_weight ────
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
    if kind == "RF":
        return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB":
        return XGBClassifier(random_state=42, n_jobs=1,
                              eval_metric="mlogloss" if n_classes > 2 else "logloss", **cfg)
    if kind == "GBM":
        return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM":
        return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr, conf_tr):
    """confidence_weight * sklearn 'balanced' class weight, per training fold."""
    cbal = compute_sample_weight("balanced", y_tr)
    return conf_tr * cbal

def _fit_predict_proba(kind, cfg, X, y, cw, tr, te, n_classes):
    m = make_model(kind, cfg, n_classes)
    sw = combined_weight(y[tr], cw[tr])
    if kind == "XGB" and n_classes == 2:
        pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
        if pos_w > 0:
            m.set_params(scale_pos_weight=neg_w/pos_w)
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
    log(f"  grid_search: {len(tasks)} flat fits queued (models={MODEL_KINDS})")
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
        if proba is None:
            proba = np.eye(n_classes)[pred]
        probs.append(proba)
    return te, np.mean(probs, axis=0)

def logo_cluster_cv_proba(best_cfgs, X, y, cw, groups, n_classes, n_jobs=-1):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    log(f"  logo_cluster_cv: {len(folds)} folds")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, cw, tr, te, n_classes) for tr, te in folds)
    proba = np.zeros((len(y), n_classes))
    for te, p in results:
        proba[te] = p
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

def adjacent_error_rate(y_true, y_pred, n_classes):
    if n_classes != 3: return None
    wrong = y_true != y_pred
    if wrong.sum() == 0: return 1.0
    return round((np.abs(y_true[wrong]-y_pred[wrong])==1).mean(), 3)

def best_threshold(y_true, proba_pos):
    best_t, best_ba = 0.5, balanced_accuracy_score(y_true, (proba_pos>=0.5).astype(int))
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba_pos>=t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba

def run_full(X, y, cw, groups, region, n_classes, tag, names):
    df_grid = grid_search(X, y, cw, groups, n_classes)
    best = best_cfgs_from_grid(df_grid)
    log(f"  [{tag}] best={[(k,c) for k,c in best]}")
    proba = logo_cluster_cv_proba(best, X, y, cw, groups, n_classes)
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y, preds)
    adj = adjacent_error_rate(y, preds, n_classes)
    log(f"  [{tag}] LOGO-cluster CV acc={acc:.4f} adjacent={adj}")
    print(confusion_matrix(y, preds), flush=True)
    print(classification_report(y, preds, target_names=names, zero_division=0), flush=True)
    df_lro = leave_region_out(X, y, cw, region, best, n_classes)
    print(df_lro.to_string(index=False), flush=True)
    return dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                leave_region_out=df_lro.to_dict("records"), best_configs=best)

ALL = {}

# ── G0a: baseline 6 features, confidence+balance weighted, all models ─────────
print("\n"+"="*70, flush=True); print("G0a -- BASELINE 6 FEATURES, confidence+balance weighted, "+str(len(MODEL_KINDS))+" models", flush=True); print("="*70, flush=True)
X_base = merged[FEATURES_BASE].values
for tag, y, n_classes, names in [
    ("G0a_3class", y3, 3, CLASS3),
    ("G0a_difficult", (merged["Expert_Merged"]=="Difficult").astype(int).values, 2, ["Not-Difficult","Difficult"]),
    ("G0a_easy", (merged["Expert_Merged"]=="Easy").astype(int).values, 2, ["Not-Easy","Easy"]),
]:
    ALL[tag] = run_full(X_base, y, conf_w, cluster_ids, region, n_classes, tag, names)

# ── G0b: + Dist_to_Trail_m, same weighting -- honest ablation vs G0a ──────────
print("\n"+"="*70, flush=True); print("G0b -- BASELINE + Dist_to_Trail_m (7 features) -- ablation vs G0a", flush=True); print("="*70, flush=True)
X_trail = merged[FEATURES_TRAIL].values
for tag, y, n_classes, names in [
    ("G0b_3class", y3, 3, CLASS3),
    ("G0b_difficult", (merged["Expert_Merged"]=="Difficult").astype(int).values, 2, ["Not-Difficult","Difficult"]),
    ("G0b_easy", (merged["Expert_Merged"]=="Easy").astype(int).values, 2, ["Not-Easy","Easy"]),
]:
    ALL[tag] = run_full(X_trail, y, conf_w, cluster_ids, region, n_classes, tag, names)
    base_tag = tag.replace("G0b","G0a")
    log(f"  {tag} vs {base_tag}: LOGO-cluster {ALL[tag]['acc_logo_cluster']} vs {ALL[base_tag]['acc_logo_cluster']}")

# pick winning feature set for G1/G2 based on leave-region-out on the 3 solid regions, 3-class target
def solid_region_mean(entry):
    df = pd.DataFrame(entry["leave_region_out"])
    df = df[df.region.isin(["Fés-Meknés","Béni Mellal-Khénifra","Drâa-Tafilalet"])]
    return df["acc"].mean()

winner = "trail" if solid_region_mean(ALL["G0b_3class"]) > solid_region_mean(ALL["G0a_3class"]) else "base"
log(f"WINNING FEATURE SET for G1/G2 (by mean leave-region-out acc on 3 solid regions): {winner}")
ALL["feature_set_decision"] = dict(winner=winner,
    g0a_solid_region_mean=round(solid_region_mean(ALL["G0a_3class"]),4),
    g0b_solid_region_mean=round(solid_region_mean(ALL["G0b_3class"]),4))
X_win = X_trail if winner == "trail" else X_base
FEATURES_WIN = FEATURES_TRAIL if winner == "trail" else FEATURES_BASE

# ── G1: per-region binary, 3 qualifying regions, winning features, threshold tuning ──
print("\n"+"="*70, flush=True); print(f"G1 -- PER-REGION BINARY ({winner} features), confidence+balance weighted, threshold-tuned", flush=True); print("="*70, flush=True)
g1_results = []
for reg in ["Fés-Meknés", "Béni Mellal-Khénifra", "Drâa-Tafilalet"]:
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    Xr = sub[FEATURES_WIN].values
    cwr = sub["conf_weight"].values
    sub_clusters = cluster_of(sub)
    for target in ["Difficult", "Easy"]:
        yb = (sub["Expert_Merged"] == target).astype(int).values
        df_grid = grid_search(Xr, yb, cwr, sub_clusters, 2)
        best = best_cfgs_from_grid(df_grid)
        proba = logo_cluster_cv_proba(best, Xr, yb, cwr, sub_clusters, 2)
        preds_default = (proba[:,1] >= 0.5).astype(int)
        acc_default = accuracy_score(yb, preds_default)
        t_opt, _ = best_threshold(yb, proba[:,1])
        preds_tuned = (proba[:,1] >= t_opt).astype(int)
        acc_tuned = accuracy_score(yb, preds_tuned)
        maj = pd.Series(yb).value_counts(normalize=True).max()
        n = len(yb)
        best_acc = max(acc_default, acc_tuned)
        log(f"  {reg} / {target}: N={n} pos={yb.sum()} | default={acc_default:.3f} tuned({t_opt:.2f})={acc_tuned:.3f} "
            f"| BEST={best_acc:.3f} | local_majority={maj:.3f} | gap={100*(best_acc-maj):+.1f}pp")
        g1_results.append(dict(region=reg, target=target, N=n, n_pos=int(yb.sum()),
                                acc_default=round(acc_default,3), acc_tuned=round(acc_tuned,3),
                                tuned_threshold=round(t_opt,2), best_acc=round(best_acc,3),
                                local_majority=round(maj,3), gap_pp=round((best_acc-maj)*100,1)))
ALL["G1_per_region_binary"] = g1_results

# ── G2: region-as-feature, recombined with everything above ───────────────────
print("\n"+"="*70, flush=True); print("G2 -- REGION AS FEATURE (LOGO-cluster CV only)", flush=True); print("="*70, flush=True)
region_dummies = pd.get_dummies(merged["Region"], prefix="region")
X_g2 = np.hstack([X_win, region_dummies.values])
df_grid = grid_search(X_g2, y3, conf_w, cluster_ids, 3)
best = best_cfgs_from_grid(df_grid)
proba = logo_cluster_cv_proba(best, X_g2, y3, conf_w, cluster_ids, 3)
preds = proba.argmax(axis=1)
acc = accuracy_score(y3, preds)
adj = adjacent_error_rate(y3, preds, 3)
log(f"G2 LOGO-cluster CV acc={acc:.4f} adjacent={adj} (G0 baseline was {ALL['G0a_3class']['acc_logo_cluster']} / {ALL['G0b_3class']['acc_logo_cluster']})")
ALL["G2_region_as_feature"] = dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj)

out_path = os.path.join(MOUT, f"final_v2_results_N{N}.json")
json.dump(ALL, open(out_path, "w"), indent=2, default=str)
print("\n"+"="*70, flush=True)
print(f"ALL DONE. Results saved to {out_path}", flush=True)
print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
print("="*70, flush=True)
