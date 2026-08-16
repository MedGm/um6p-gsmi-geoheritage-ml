"""
Final validation battery (2026-08-14), run on Z8 only -- do not run this locally,
it uses the same full grid as code/18 and is meant for a many-core machine.

F0  Exact repeat of code/18's E0 (baseline, full features), to confirm
    reproducibility of the flat-parallel LOGO-cluster CV before trusting
    anything built on top of it.
F1  Per-region binary models on the three regions with enough examples per
    class (Fes-Meknes, BMK, Drâa-Tafilalet) -- WITH proper class-imbalance
    handling this time: scale_pos_weight for XGB, sample_weight for GBM
    (code/18's ensemble only balanced RF, which is almost certainly why
    BMK-Difficult collapsed to a degenerate majority-only classifier) --
    plus per-fold threshold tuning (best-balanced-accuracy threshold on the
    OOF soft-vote probability, not the default 0.5 cut).
F2  Ordinal-via-binary-decomposition vs. the flat nominal 3-class ensemble:
    train P(class >= Moderate) and P(class == Difficult) as two binary
    models, derive ordinal class probabilities, compare accuracy AND mean
    absolute class-distance error against F0's nominal 3-class result.
F3  Bootstrap 95% CI (2000 resamples) on the three solid (n>=87) leave-
    region-out accuracies from F0, to turn point estimates into citable
    intervals.
F4  Two engineered features (region-relative elevation z-score, Slope x
    Dist_to_Highway interaction) added to the full feature set, re-run
    through the same F0 protocol (LOGO-cluster CV + leave-region-out) to
    test whether they close any of the gap.

Run (Z8, foreground is fine):
    cd geosite_project1
    python .\\code\\19_final_battery.py 2>&1 | Tee-Object -FilePath final_battery.log
Expect roughly 45-90 minutes total (this is a strictly larger scope than
code/18's ~1533s run: F0 duplicates E0's 3 targets, F1 adds 6 grid-searched
region/target cells, F2/F4 add more grid-searched runs). Every stage flushes
progress as it goes, same as code/18 -- paste the log back when done, or in
sections if you want interim results sooner.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
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

# ── Load (identical to code/18) ────────────────────────────────────────────────
log("Loading labels from regional_label_sources/*.csv")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_FULL],
    on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
N = len(merged)
log(f"N={N}, class balance: {merged['Expert_Merged'].value_counts().to_dict()}")

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

# ── Flat-parallel grid + CV, extended with optional sample_weight/scale_pos_weight ──
RF_GRID  = [dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl)
            for ne in [100, 200, 400] for md in [3,4,5,6,7,8,None] for msl in [1,2,3,5]]
XGB_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,150,250] for md in [3,4,5,6] for lr in [0.03,0.08,0.15]]
GBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100,200] for md in [2,3,4] for lr in [0.05,0.1]]

def make_model(kind, cfg, n_classes, balanced=True):
    if kind == "RF":
        cw = "balanced" if balanced else None
        return RandomForestClassifier(random_state=42, n_jobs=1, class_weight=cw, **cfg)
    if kind == "XGB":
        return XGBClassifier(random_state=42, n_jobs=1,
                              eval_metric="mlogloss" if n_classes > 2 else "logloss", **cfg)
    if kind == "GBM":
        return GradientBoostingClassifier(random_state=42, **cfg)

def _fit_predict_proba(kind, cfg, X, y, tr, te, n_classes, balanced):
    m = make_model(kind, cfg, n_classes, balanced)
    sw = None
    if balanced and kind in ("XGB", "GBM"):
        sw = compute_sample_weight("balanced", y[tr])
    if kind == "XGB" and balanced and n_classes == 2:
        pos = (y[tr] == 1).sum(); neg = (y[tr] == 0).sum()
        m.set_params(scale_pos_weight=(neg / max(pos, 1)))
    if sw is not None:
        m.fit(X[tr], y[tr], sample_weight=sw)
    else:
        m.fit(X[tr], y[tr])
    return m.predict(X[te]), (m.predict_proba(X[te]) if hasattr(m, "predict_proba") else None)

def _one_fit_acc(kind, cfg, X, y, tr, te, n_classes, balanced):
    pred, _ = _fit_predict_proba(kind, cfg, X, y, tr, te, n_classes, balanced)
    return accuracy_score(y[te], pred)

def grid_search(X, y, groups, n_classes, balanced=True, n_repeats=5, n_splits=10, n_jobs=-1):
    tasks = []
    for kind, grid in [("RF", RF_GRID), ("XGB", XGB_GRID), ("GBM", GBM_GRID)]:
        for cfg in grid:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(X, y, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    log(f"  grid_search: {len(tasks)} flat fits queued (n_jobs={n_jobs}, balanced={balanced})")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_fit_acc)(kind, cfg, X, y, tr, te, n_classes, balanced) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [{"model": k, "config": c, "mean_acc": np.mean(v)} for (k, c), v in agg.items()]
    return pd.DataFrame(rows).sort_values("mean_acc", ascending=False)

def best_cfgs_from_grid(df_grid):
    return [(k, json.loads(df_grid[df_grid.model == k].iloc[0].config)) for k in ["RF", "XGB", "GBM"]]

def _logo_fold_proba(best_cfgs, X, y, tr, te, n_classes, balanced):
    probs = []
    for kind, cfg in best_cfgs:
        pred, proba = _fit_predict_proba(kind, cfg, X, y, tr, te, n_classes, balanced)
        if proba is None:
            proba = np.eye(n_classes)[pred]
        probs.append(proba)
    return te, np.mean(probs, axis=0)

def logo_cluster_cv_proba(best_cfgs, X, y, groups, n_classes, balanced=True, n_jobs=-1):
    folds = list(LeaveOneGroupOut().split(X, y, groups=groups))
    log(f"  logo_cluster_cv: {len(folds)} folds (n_jobs={n_jobs}, balanced={balanced})")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_logo_fold_proba)(best_cfgs, X, y, tr, te, n_classes, balanced) for tr, te in folds)
    proba = np.zeros((len(y), n_classes))
    for te, p in results:
        proba[te] = p
    return proba

def leave_region_out(X, y, region, best_cfgs, n_classes, balanced=True):
    rows = []
    for reg in pd.unique(region):
        tr = region != reg; te = region == reg
        if te.sum() < 10:
            rows.append({"region": reg, "n_test": int(te.sum()), "acc": None, "local_majority": None})
            continue
        estimators = [(k.lower(), make_model(k, cfg, n_classes, balanced)) for k, cfg in best_cfgs]
        m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)
        m.fit(X[tr], y[tr])
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
    """Sweep thresholds, return the one maximizing balanced accuracy."""
    best_t, best_ba = 0.5, balanced_accuracy_score(y_true, (proba_pos>=0.5).astype(int))
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba_pos>=t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba

ALL = {}

# ── F0: exact repeat of E0 ─────────────────────────────────────────────────────
print("\n"+"="*70, flush=True); print("F0 -- REPEAT OF E0 (reproducibility check)", flush=True); print("="*70, flush=True)
X_full = merged[FEATURES_FULL].values
for tag, y, n_classes, names in [
    ("F0_3class", y3, 3, CLASS3),
    ("F0_difficult", (merged["Expert_Merged"]=="Difficult").astype(int).values, 2, ["Not-Difficult","Difficult"]),
    ("F0_easy", (merged["Expert_Merged"]=="Easy").astype(int).values, 2, ["Not-Easy","Easy"]),
]:
    df_grid = grid_search(X_full, y, cluster_ids, n_classes, balanced=True)
    best = best_cfgs_from_grid(df_grid)
    proba = logo_cluster_cv_proba(best, X_full, y, cluster_ids, n_classes, balanced=True)
    preds = proba.argmax(axis=1)
    acc = accuracy_score(y, preds)
    adj = adjacent_error_rate(y, preds, n_classes)
    log(f"  [{tag}] LOGO-cluster CV acc={acc:.4f} adjacent={adj}")
    print(confusion_matrix(y, preds), flush=True)
    print(classification_report(y, preds, target_names=names, zero_division=0), flush=True)
    df_lro = leave_region_out(X_full, y, region, best, n_classes, balanced=True)
    print(df_lro.to_string(index=False), flush=True)
    ALL[tag] = dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                     leave_region_out=df_lro.to_dict("records"), best_configs=best,
                     proba=proba.tolist(), y_true=y.tolist())

# ── F1: per-region binary, 3 qualifying regions, WITH imbalance fix + threshold tuning ──
print("\n"+"="*70, flush=True)
print("F1 -- PER-REGION BINARY, imbalance-corrected (scale_pos_weight/sample_weight)", flush=True)
print("      + per-fold-OOF threshold tuning (not default 0.5)", flush=True)
print("="*70, flush=True)
f1_results = []
for reg in ["Fés-Meknés", "Béni Mellal-Khénifra", "Drâa-Tafilalet"]:
    sub = merged[merged["Region"] == reg].reset_index(drop=True)
    Xr = sub[FEATURES_FULL].values
    sub_clusters = cluster_of(sub)
    for target in ["Difficult", "Easy"]:
        yb = (sub["Expert_Merged"] == target).astype(int).values
        df_grid = grid_search(Xr, yb, sub_clusters, 2, balanced=True)
        best = best_cfgs_from_grid(df_grid)
        proba = logo_cluster_cv_proba(best, Xr, yb, sub_clusters, 2, balanced=True)
        preds_default = (proba[:,1] >= 0.5).astype(int)
        acc_default = accuracy_score(yb, preds_default)
        t_opt, ba_opt = best_threshold(yb, proba[:,1])
        preds_tuned = (proba[:,1] >= t_opt).astype(int)
        acc_tuned = accuracy_score(yb, preds_tuned)
        maj = pd.Series(yb).value_counts(normalize=True).max()
        n = len(yb)
        nc_def = int(round(acc_default*n)); nc_maj = int(round(maj*n))
        nc_tun = int(round(acc_tuned*n))
        log(f"  {reg} / {target}: N={n} pos={yb.sum()} | default_thr acc={acc_default:.3f} "
            f"(degenerate={nc_def==nc_maj}) | tuned_thr={t_opt:.2f} acc={acc_tuned:.3f} "
            f"(degenerate={nc_tun==nc_maj}) | local_majority={maj:.3f}")
        f1_results.append(dict(region=reg, target=target, N=n, n_pos=int(yb.sum()),
                                acc_default_threshold=round(acc_default,3),
                                acc_tuned_threshold=round(acc_tuned,3), tuned_threshold=round(t_opt,2),
                                local_majority=round(maj,3),
                                gap_default_pp=round((acc_default-maj)*100,1),
                                gap_tuned_pp=round((acc_tuned-maj)*100,1),
                                degenerate_default=nc_def==nc_maj, degenerate_tuned=nc_tun==nc_maj))
ALL["F1_per_region_binary_fixed"] = f1_results

# ── F2: ordinal-via-binary-decomposition vs nominal 3-class ───────────────────
print("\n"+"="*70, flush=True); print("F2 -- ORDINAL (binary decomposition) vs NOMINAL 3-class", flush=True); print("="*70, flush=True)
y_ge_mod = (y3 >= 1).astype(int)   # >= Moderate
y_is_diff = (y3 == 2).astype(int)  # == Difficult
df_grid_a = grid_search(X_full, y_ge_mod, cluster_ids, 2, balanced=True)
best_a = best_cfgs_from_grid(df_grid_a)
proba_a = logo_cluster_cv_proba(best_a, X_full, y_ge_mod, cluster_ids, 2, balanced=True)[:,1]
df_grid_b = grid_search(X_full, y_is_diff, cluster_ids, 2, balanced=True)
best_b = best_cfgs_from_grid(df_grid_b)
proba_b = logo_cluster_cv_proba(best_b, X_full, y_is_diff, cluster_ids, 2, balanced=True)[:,1]
# P(Easy)=1-P(>=Mod); P(Moderate)=P(>=Mod)*(1-P(Diff|>=Mod)) approx via P(>=Mod)-P(Diff); P(Difficult)=P(Diff)
p_easy = 1 - proba_a
p_diff = proba_b
p_mod = np.clip(1 - p_easy - p_diff, 0, None)
ordinal_proba = np.stack([p_easy, p_mod, p_diff], axis=1)
ordinal_proba = ordinal_proba / ordinal_proba.sum(axis=1, keepdims=True)
ordinal_preds = ordinal_proba.argmax(axis=1)
acc_ord = accuracy_score(y3, ordinal_preds)
mae_ord = np.abs(y3 - ordinal_preds).mean()
nominal_proba = np.array(ALL["F0_3class"]["proba"])
nominal_preds = nominal_proba.argmax(axis=1)
acc_nom = accuracy_score(y3, nominal_preds)
mae_nom = np.abs(y3 - nominal_preds).mean()
log(f"  Ordinal decomposition: acc={acc_ord:.4f} MAE={mae_ord:.4f}")
log(f"  Nominal 3-class (F0):   acc={acc_nom:.4f} MAE={mae_nom:.4f}")
ALL["F2_ordinal_vs_nominal"] = dict(ordinal_acc=round(acc_ord,4), ordinal_mae=round(mae_ord,4),
                                     nominal_acc=round(acc_nom,4), nominal_mae=round(mae_nom,4))

# ── F3: bootstrap 95% CI on the 3 solid leave-region-out numbers ──────────────
print("\n"+"="*70, flush=True); print("F3 -- BOOTSTRAP 95% CI (2000 resamples) on solid-N leave-region-out", flush=True); print("="*70, flush=True)
rng = np.random.default_rng(42)
f3_results = []
for reg in ["Fés-Meknés", "Béni Mellal-Khénifra", "Drâa-Tafilalet"]:
    tr = region != reg; te = region == reg
    best = ALL["F0_3class"]["best_configs"]
    estimators = [(k.lower(), make_model(k, cfg, 3, True)) for k, cfg in best]
    m = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    m.fit(X_full[tr], y3[tr])
    p = m.predict(X_full[te])
    correct = (p == y3[te]).astype(int)
    n = len(correct)
    boots = [rng.choice(correct, size=n, replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    acc = correct.mean()
    log(f"  {reg}: acc={acc:.3f}  95% CI=[{lo:.3f}, {hi:.3f}]  n={n}")
    f3_results.append(dict(region=reg, acc=round(acc,3), ci_lo=round(lo,3), ci_hi=round(hi,3), n=n))
ALL["F3_bootstrap_ci"] = f3_results

# ── F4: engineered features ────────────────────────────────────────────────────
print("\n"+"="*70, flush=True); print("F4 -- ENGINEERED FEATURES (region-relative elevation, slope x distance)", flush=True); print("="*70, flush=True)
merged["Elevation_region_z"] = merged.groupby("Region")["Elevation_m"].transform(
    lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0)
merged["Slope_x_Dist"] = merged["Slope_deg"] * merged["Dist_to_Highway_m"] / 1000.0
FEATURES_ENG = FEATURES_FULL + ["Elevation_region_z", "Slope_x_Dist"]
X_eng = merged[FEATURES_ENG].fillna(0).values
df_grid = grid_search(X_eng, y3, cluster_ids, 3, balanced=True)
best = best_cfgs_from_grid(df_grid)
proba = logo_cluster_cv_proba(best, X_eng, y3, cluster_ids, 3, balanced=True)
preds = proba.argmax(axis=1)
acc = accuracy_score(y3, preds)
adj = adjacent_error_rate(y3, preds, 3)
log(f"  [F4_engineered_3class] LOGO-cluster CV acc={acc:.4f} adjacent={adj} (F0 baseline was {ALL['F0_3class']['acc_logo_cluster']})")
df_lro = leave_region_out(X_eng, y3, region, best, 3, balanced=True)
print(df_lro.to_string(index=False), flush=True)
ALL["F4_engineered_features"] = dict(acc_logo_cluster=round(acc,4), adjacent_error_rate=adj,
                                      delta_vs_F0_pp=round((acc-ALL["F0_3class"]["acc_logo_cluster"])*100,2),
                                      leave_region_out=df_lro.to_dict("records"))

# strip large proba/y_true arrays before saving JSON (keep file readable)
for k in ["F0_3class","F0_difficult","F0_easy"]:
    ALL[k].pop("proba", None); ALL[k].pop("y_true", None)

out_path = os.path.join(MOUT, f"final_battery_results_N{N}.json")
json.dump(ALL, open(out_path, "w"), indent=2, default=str)
print("\n"+"="*70, flush=True)
print(f"ALL DONE. Results saved to {out_path}", flush=True)
print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)
print("="*70, flush=True)
