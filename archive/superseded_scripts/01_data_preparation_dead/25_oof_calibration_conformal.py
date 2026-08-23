"""
code/25_oof_calibration_conformal.py  (2026-08-17)

Evaluation-layer work from the day's research plan: the deployed ensemble's
out-of-fold (OOF) predicted probabilities are not currently saved anywhere in
this repo -- code/20 computes LOGO-cluster CV accuracy directly without
persisting the per-site probabilities that produced it. This script re-fits
the EXACT deployed ensemble (same BEST_CONFIGS as report/scripts/make_maps.py,
RF+XGB+GBM+LightGBM, confidence+balance sample weighting) under the identical
500m LOGO-cluster CV protocol, saves the per-site OOF probability for both
binary targets, and uses those probabilities for two honest-uncertainty
checks that a bare accuracy number can't give:

  1. Calibration: reliability diagram bins + Brier score + expected
     calibration error (ECE), computed on truly out-of-fold probabilities
     (each site's probability comes from a model that never saw it) --
     then isotonic regression fit on those same OOF probabilities (a
     standard, defensible near-unbiased calibration procedure -- see e.g.
     Platt-scaling-on-CV-folds literature) to check how much calibration
     error is fixable post-hoc.
  2. Split-conformal prediction sets: the 733 OOF points are randomly split
     50/50 into a "conformal calibration" half (used only to pick the
     nonconformity quantile) and a "coverage validation" half (used only to
     check the resulting prediction sets actually hit their target coverage)
     -- so the coverage number reported is a genuine held-out check, not a
     self-consistency tautology. Nonconformity score s_i = 1 - p_i(true
     class) (the standard LAC/"least ambiguous set-valued classifier" score).

Sanity check: this script's own LOGO-cluster CV accuracy should closely
match code/20's G0a_difficult (0.7408) / G0a_easy (0.7326) -- if it doesn't,
something about the refit (data, weighting, configs) diverged from the
deployed model and the OOF probabilities below should not be trusted.

Output:
  results/json/other/oof_probabilities.csv       (Locality_ID, Region, target, y_true, oof_prob)
  results/json/other/calibration_conformal_results.json
"""
import glob, json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(BASE, "results", "json", "other")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

# Exact deployed configs -- report/scripts/make_maps.py BEST_CONFIGS
BEST_CONFIGS = {
    "difficult": {
        "RF": dict(max_depth=None, min_samples_leaf=2, n_estimators=200),
        "XGB": dict(learning_rate=0.08, max_depth=6, n_estimators=150),
        "GBM": dict(learning_rate=0.1, max_depth=4, n_estimators=200),
        "LGBM": dict(learning_rate=0.05, max_depth=-1, n_estimators=200, num_leaves=31),
    },
    "easy": {
        "RF": dict(max_depth=None, min_samples_leaf=1, n_estimators=400),
        "XGB": dict(learning_rate=0.08, max_depth=6, n_estimators=100),
        "GBM": dict(learning_rate=0.1, max_depth=4, n_estimators=200),
        "LGBM": dict(learning_rate=0.1, max_depth=-1, n_estimators=200, num_leaves=31),
    },
}

log("Loading N=733 labeled catalog ...")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    if "Confidence" not in labeled.columns:
        labeled["Confidence"] = "Medium"
    frames.append(labeled[["Locality_ID", "Expert_Class", "Confidence"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 733
conf_w = merged["Confidence"].map(CONF_WEIGHT).fillna(0.7).values
X = merged[FEATURES].values
log(f"N={N}")

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

def make_ensemble(cfg):
    return [
        ("RF", RandomForestClassifier(random_state=42, n_jobs=1, **cfg["RF"])),
        ("XGB", XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg["XGB"])),
        ("GBM", GradientBoostingClassifier(random_state=42, **cfg["GBM"])),
        ("LGBM", LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg["LGBM"])),
    ]

def oof_probabilities(y, cfg):
    logo = LeaveOneGroupOut()
    oof = np.full(N, np.nan)
    for train_idx, test_idx in logo.split(X, y, groups=cluster_ids):
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        sw = conf_w[train_idx] * compute_sample_weight("balanced", y_tr)
        probs = []
        for name, mdl in make_ensemble(cfg):
            mdl.fit(X[train_idx], y_tr, sample_weight=sw)
            probs.append(mdl.predict_proba(X[test_idx])[:, 1])
        oof[test_idx] = np.mean(probs, axis=0)
    return oof

def brier_ece(y, p, n_bins=10):
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    brier = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    reliability = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (p >= lo) & (p < hi) if b < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        conf = p[mask].mean()
        acc = y[mask].mean()
        w = mask.sum() / len(p)
        ece += w * abs(conf - acc)
        reliability.append({"bin": [round(lo,2), round(hi,2)], "n": int(mask.sum()),
                             "mean_predicted": round(float(conf), 4), "empirical_freq": round(float(acc), 4)})
    return brier, float(ece), reliability

def split_conformal(y, p, alpha=0.10, seed=42):
    """LAC nonconformity score s=1-p(true class); 50/50 split into
    calibration (picks quantile) and validation (checks coverage)."""
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    half = n // 2
    cal_idx, val_idx = idx[:half], idx[half:]
    p_true_cal = np.where(y[cal_idx] == 1, p[cal_idx], 1 - p[cal_idx])
    scores_cal = 1 - p_true_cal
    ncal = len(cal_idx)
    q_level = min(1.0, np.ceil((ncal + 1) * (1 - alpha)) / ncal)
    qhat = float(np.quantile(scores_cal, q_level))
    p_true_val = np.where(y[val_idx] == 1, p[val_idx], 1 - p[val_idx])
    covered = (1 - p_true_val) <= qhat
    coverage = float(covered.mean())
    set_sizes = []
    for pv in p[val_idx]:
        classes_in_set = 0
        for cls_p in [pv, 1 - pv]:
            if (1 - cls_p) <= qhat:
                classes_in_set += 1
        set_sizes.append(classes_in_set)
    return {
        "alpha": alpha, "target_coverage": 1 - alpha, "qhat": round(qhat, 4),
        "n_calibration": int(ncal), "n_validation": int(n - ncal),
        "empirical_coverage": round(coverage, 4),
        "mean_set_size": round(float(np.mean(set_sizes)), 4),
        "pct_singleton_sets": round(float(np.mean(np.array(set_sizes) == 1)), 4),
    }

results = {}
oof_rows = []
for target_name in ["difficult", "easy"]:
    y = (merged["Expert_Merged"] == ("Difficult" if target_name == "difficult" else "Easy")).astype(int).values
    log(f"Computing OOF probabilities for {target_name} ({n_clusters} folds) ...")
    p = oof_probabilities(y, BEST_CONFIGS[target_name])
    acc = float(np.nanmean((p >= 0.5).astype(int) == y))
    log(f"  {target_name}: LOGO-cluster acc={acc:.4f} (sanity check vs code/20 G0a: "
        f"{'0.7408' if target_name=='difficult' else '0.7326'})")

    brier, ece, reliability = brier_ece(y, p)
    iso = IsotonicRegression(out_of_bounds="clip")
    valid = ~np.isnan(p)
    iso.fit(p[valid], y[valid])
    p_cal = np.where(valid, iso.transform(np.nan_to_num(p)), np.nan)
    brier_cal, ece_cal, _ = brier_ece(y, p_cal)

    conformal = split_conformal(y, p)

    results[target_name] = {
        "acc_logo_cluster_sanity_check": round(acc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "reliability_diagram": reliability,
        "isotonic_calibrated": {"brier_score": round(brier_cal, 4), "ece": round(ece_cal, 4)},
        "split_conformal_90pct": conformal,
    }
    for i, r in merged.iterrows():
        oof_rows.append({"Locality_ID": r["Locality_ID"], "Region": r["Region"], "target": target_name,
                          "y_true": int(y[i]), "oof_prob": None if np.isnan(p[i]) else round(float(p[i]), 4)})

pd.DataFrame(oof_rows).to_csv(os.path.join(OUT_DIR, "oof_probabilities.csv"), index=False)
with open(os.path.join(OUT_DIR, "calibration_conformal_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log("Wrote oof_probabilities.csv + calibration_conformal_results.json")

log("Summary:")
for target_name in ["difficult", "easy"]:
    r = results[target_name]
    print(f"  {target_name}: acc={r['acc_logo_cluster_sanity_check']} brier={r['brier_score']} "
          f"ece={r['ece']} -> isotonic brier={r['isotonic_calibrated']['brier_score']} "
          f"ece={r['isotonic_calibrated']['ece']}")
    c = r["split_conformal_90pct"]
    print(f"    conformal: target={c['target_coverage']} empirical={c['empirical_coverage']} "
          f"mean_set_size={c['mean_set_size']} singleton_rate={c['pct_singleton_sets']}")
