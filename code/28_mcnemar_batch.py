"""
code/28_mcnemar_batch.py  (2026-08-17)

Paired significance testing for today's two candidate findings, following the
exact protocol already used in the paper (Sec. 4.4): McNemar's test,
continuity-corrected, on discordant cells between paired predictions on the
identical 733 sites. Majority-class baseline is FIXED (always predicts the
majority/negative class -- 70.8% for Difficult-vs-not, 71.8% for
Easy-vs-not, per the paper's own confusion-matrix-derived numbers), not a
refit model.

Three questions, per binary target (Difficult-vs-not, Easy-vs-not):
  1. Mixed-effects (region-as-random-intercept logistic, code/24's
     C_random_intercept) vs. the fixed majority-class baseline -- does
     today's headline number (0.7749 Difficult) actually clear significance
     where the deployed ensemble (McNemar chi2=2.57, p=0.109) didn't?
  2. Mixed-effects vs. the deployed ensemble directly (paired model-vs-model,
     not vs. baseline) -- is the +3.4pp gap itself statistically supported,
     or within noise of two model families scored on the same 733 points?
  3. Favorability-augmented RF (code/26's 7-feature variant) vs. the fixed
     baseline -- is the +1.5pp Difficult gain (RF-only quick-screen) real or
     noise?

As a built-in sanity check, this script also reproduces the paper's own
already-published McNemar numbers (ensemble vs. baseline: chi2=2.57 p=0.109
Difficult, chi2=0.50 p=0.481 Easy) from results/json/other/oof_probabilities.csv
-- if this script's ensemble-vs-baseline result doesn't match those published
values closely, something diverged and the new results below should not be
trusted either.

Mixed-effects OOF predictions and favorability-ablation OOF predictions were
not saved by code/24 / code/26 (accuracy-only output) -- both are recomputed
here from scratch under the identical protocol, this time with per-site
predictions retained.

Output: results/json/other/mcnemar_batch_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd, joblib as jl
from joblib import Parallel, delayed
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

log("Loading N=733 labeled catalog ...")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn: continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    if "Confidence" not in labeled.columns: labeled["Confidence"] = "Medium"
    frames.append(labeled[["Locality_ID", "Expert_Class", "Confidence"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
fav = pd.read_csv(os.path.join(BASE, "data/final/favorability_score_labeled_sites.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner").merge(fav, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 733
conf_w = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7).values

def haversine_matrix(lat, lon):
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]; dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat/2)**2 + np.cos(lr[:,None])*np.cos(lr[None,:])*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def cluster_of(sub, radius_m=500):
    lat, lon = sub["Latitude_WGS84"].values, sub["Longitude_WGS84"].values
    n = len(sub)
    D = haversine_matrix(lat, lon)
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

cluster_ids = cluster_of(merged)
n_clusters = len(np.unique(cluster_ids))
log(f"clusters={n_clusters}")

# ============================================================ Mixed-effects OOF preds
Xraw = merged[FEATURES].copy()
Xs = (Xraw - Xraw.mean()) / Xraw.std()
data_me = Xs.copy()
data_me["Region"] = pd.Categorical(merged["Region"].values)
formula_fe = "y ~ " + " + ".join(FEATURES)

def me_fold(train_idx, test_idx, y):
    train = data_me.iloc[train_idx].copy()
    test = data_me.iloc[test_idx].copy()
    train["y"] = y[train_idx]
    if train["y"].nunique() < 2:
        return test_idx, np.full(len(test_idx), np.nan)
    try:
        model = BinomialBayesMixedGLM.from_formula(formula_fe, {"Region": "0 + C(Region)"}, train)
        res = model.fit_vb()
        fe = dict(zip(["Intercept"] + FEATURES, res.fe_mean))
        re = res.random_effects()["Mean"].to_dict()
        logit = fe["Intercept"] + sum(fe[f] * test[f].values for f in FEATURES)
        region_eff = np.array([re.get(f"C(Region)[{r}]", 0.0) for r in test["Region"]])
        p = 1.0 / (1.0 + np.exp(-(logit + region_eff)))
        return test_idx, p
    except Exception:
        return test_idx, np.full(len(test_idx), np.nan)

def me_oof(y):
    logo = LeaveOneGroupOut()
    folds = list(logo.split(data_me, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(me_fold)(tr, te, y) for tr, te in folds)
    preds = np.full(N, np.nan)
    for te, p in results:
        preds[te] = p
    return preds

# ============================================================ Favorability RF OOF preds
RF_CFG = {
    "difficult": dict(max_depth=None, min_samples_leaf=2, n_estimators=200, random_state=42, n_jobs=1),
    "easy": dict(max_depth=None, min_samples_leaf=1, n_estimators=400, random_state=42, n_jobs=1),
}

def rf_fold(cols, cfg, train_idx, test_idx, y):
    X = merged[cols].values
    y_tr = y[train_idx]
    if len(np.unique(y_tr)) < 2:
        return test_idx, np.full(len(test_idx), np.nan)
    sw = conf_w[train_idx] * compute_sample_weight("balanced", y_tr)
    m = RandomForestClassifier(**cfg)
    m.fit(X[train_idx], y_tr, sample_weight=sw)
    return test_idx, m.predict_proba(X[test_idx])[:, 1]

def rf_oof(cols, cfg, y):
    logo = LeaveOneGroupOut()
    folds = list(logo.split(merged[cols].values, y, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(rf_fold)(cols, cfg, tr, te, y) for tr, te in folds)
    preds = np.full(N, np.nan)
    for te, p in results:
        preds[te] = p
    return preds

# ============================================================ McNemar helper
def mcnemar_test(correct_a, correct_b, valid):
    a, b = correct_a[valid], correct_b[valid]
    table = pd.crosstab(a, b).reindex(index=[False, True], columns=[False, True], fill_value=0).values
    res = mcnemar(table, exact=False, correction=True)
    n10 = int(((a == True) & (b == False)).sum())
    n01 = int(((a == False) & (b == True)).sum())
    return {"chi2": round(float(res.statistic), 4), "p": round(float(res.pvalue), 4),
            "n_a_right_b_wrong": n10, "n_a_wrong_b_right": n01, "n_valid": int(valid.sum())}

ensemble_oof = pd.read_csv(os.path.join(BASE, "results/json/other/oof_probabilities.csv"))

results = {}
for target_name in ["difficult", "easy"]:
    y = (merged["Expert_Merged"] == ("Difficult" if target_name == "difficult" else "Easy")).astype(int).values

    log(f"[{target_name}] Recomputing mixed-effects OOF preds ...")
    p_me = me_oof(y)
    acc_me = float(np.nanmean((p_me >= 0.5).astype(int) == y))
    log(f"  mixed-effects acc={acc_me:.4f}")

    log(f"[{target_name}] Recomputing favorability-ablation OOF preds ...")
    p_base_rf = rf_oof(FEATURES, RF_CFG[target_name], y)
    p_fav_rf = rf_oof(FEATURES + ["Favorability_Score"], RF_CFG[target_name], y)
    acc_base_rf = float(np.nanmean((p_base_rf >= 0.5).astype(int) == y))
    acc_fav_rf = float(np.nanmean((p_fav_rf >= 0.5).astype(int) == y))
    log(f"  base RF acc={acc_base_rf:.4f}  +favorability RF acc={acc_fav_rf:.4f}")

    ens_row = ensemble_oof[ensemble_oof["target"] == target_name].set_index("Locality_ID")
    ens_row = ens_row.reindex(merged["Locality_ID"].values)
    p_ens = ens_row["oof_prob"].values.astype(float)
    y_true_ens = ens_row["y_true"].values
    assert np.array_equal(y_true_ens, y), "Locality_ID alignment mismatch between merged and ensemble OOF file"
    acc_ens = float(np.nanmean((p_ens >= 0.5).astype(int) == y))
    log(f"  deployed ensemble (reloaded) acc={acc_ens:.4f}")

    baseline_correct = (y == 0)  # fixed majority-class baseline: always predicts negative class
    me_correct = (p_me >= 0.5).astype(int) == y
    ens_correct = (p_ens >= 0.5).astype(int) == y
    fav_correct = (p_fav_rf >= 0.5).astype(int) == y
    base_rf_correct = (p_base_rf >= 0.5).astype(int) == y

    valid_me = ~np.isnan(p_me)
    valid_ens = ~np.isnan(p_ens)
    valid_fav = ~np.isnan(p_fav_rf) & ~np.isnan(p_base_rf)

    results[target_name] = {
        "acc": {"mixed_effects": round(acc_me,4), "deployed_ensemble": round(acc_ens,4),
                "base_rf_only": round(acc_base_rf,4), "favorability_rf": round(acc_fav_rf,4)},
        "mixed_effects_vs_baseline": mcnemar_test(me_correct, baseline_correct, valid_me),
        "ensemble_vs_baseline_SANITY_CHECK": mcnemar_test(ens_correct, baseline_correct, valid_ens),
        "mixed_effects_vs_ensemble": mcnemar_test(me_correct, ens_correct, valid_me & valid_ens),
        "favorability_rf_vs_baseline": mcnemar_test(fav_correct, baseline_correct, valid_fav),
        "favorability_rf_vs_base_rf": mcnemar_test(fav_correct, base_rf_correct, valid_fav),
    }

out_path = os.path.join(BASE, "results", "json", "other", "mcnemar_batch_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log(f"Wrote {out_path}")

log("SUMMARY:")
for target_name in ["difficult", "easy"]:
    r = results[target_name]
    print(f"\n  {target_name}:")
    print(f"    accuracies: {r['acc']}")
    print(f"    ensemble vs baseline (SANITY, should match paper's chi2=2.57/0.50 p=0.109/0.481): {r['ensemble_vs_baseline_SANITY_CHECK']}")
    print(f"    mixed-effects vs baseline: {r['mixed_effects_vs_baseline']}")
    print(f"    mixed-effects vs ensemble: {r['mixed_effects_vs_ensemble']}")
    print(f"    favorability-RF vs baseline: {r['favorability_rf_vs_baseline']}")
    print(f"    favorability-RF vs base-RF: {r['favorability_rf_vs_base_rf']}")
