"""
data_audit/07_regime_check.py  (2026-08-21)

Follow-up to 06_batch_origin_check.py. That script found el_ouali_2026's
Difficult sites sit in a different terrain regime than original_733's
(median Dist_to_Highway 4112m vs 1280m, Elevation 693m vs 1440m, both
p<0.0001) and original_733's Difficult class is 72% Fés-Meknés -- a
mountain-terrain signature standing in for "Difficult" nationally.

This script saves per-site OOF predictions (never saved before) to directly
test: is model failure driven by TERRAIN REGIME (mountain-near-road vs
remote-lowland) rather than by batch origin as such? Two direct tests:
  1. Split ALL Difficult sites (both origins pooled) into two regimes using
     el_ouali's own observed medians as thresholds (Elevation<693m OR
     Dist_to_Highway>4112m = "remote regime"); compare accuracy in each
     regime, regardless of which batch a site came from.
  2. Within original_733 alone, exclude Fés-Meknés and re-check accuracy on
     the remainder -- shows how much of the model's apparent original-batch
     skill is actually just "recognizes Fés-Meknés' own signature".
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
el_ouali_ids = set(pd.read_csv(
    os.path.join(BASE, "data/final/regional_label_sources/el_ouali_2026_expert_labels.csv"))["Locality_ID"])

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
merged["origin"] = np.where(merged["Locality_ID"].isin(el_ouali_ids), "el_ouali_2026", "original_733")
N = len(merged)
assert N == 939

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

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

def fold_proba(best_cfgs, X, y, tr, te):
    probs = []
    for kind, cfg in best_cfgs:
        m = make_model(kind, cfg)
        sw = combined_weight(y[tr])
        if kind == "XGB":
            pos_w = sw[y[tr]==1].sum(); neg_w = sw[y[tr]==0].sum()
            if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te])[:, 1])
    return te, np.mean(probs, axis=0)

def logo_cluster_oof(X, y, best_cfgs):
    folds = list(LeaveOneGroupOut().split(X, y, groups=cluster_ids))
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_proba)(best_cfgs, X, y, tr, te) for tr, te in folds)
    proba = np.full(len(y), np.nan)
    for te, p in results: proba[te] = p
    return proba

modeling_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_modeling_results.json")))
X = merged[FEATURES].values
y = (merged["Expert_Merged"] == "Difficult").astype(int).values
best_cfgs = modeling_results["Baseline_939_difficult"]["best_configs"]

log("Refitting Baseline_939 Difficult LOGO-cluster CV OOF (reused best_configs, saving per-site) ...")
proba = logo_cluster_oof(X, y, best_cfgs)
merged["proba"] = proba
merged["pred"] = (proba >= 0.5).astype(int)
merged["correct"] = merged["pred"] == y
merged["y"] = y

per_site_path = os.path.join(BASE, "results", "json", "other", "phase5_difficult_oof_per_site.csv")
merged.to_csv(per_site_path, index=False)
log(f"Saved per-site OOF to {per_site_path}")

# --- Test 1: terrain regime, pooled across both origins ---
diff = merged[merged["Expert_Merged"] == "Difficult"].copy()
ELEV_THRESH = 693.0   # el_ouali_2026 Difficult median
HWY_THRESH = 4112.0   # el_ouali_2026 Difficult median
diff["regime"] = np.where((diff["Elevation_m"] < ELEV_THRESH) | (diff["Dist_to_Highway_m"] > HWY_THRESH),
                            "remote_lowland", "mountain_nearroad")

log("\n=== TEST 1: accuracy by terrain regime, Difficult sites, BOTH origins pooled ===")
for regime in ["mountain_nearroad", "remote_lowland"]:
    sub = diff[diff.regime == regime]
    n = len(sub)
    acc = sub["correct"].mean()
    n_orig = (sub.origin == "original_733").sum()
    n_elo = (sub.origin == "el_ouali_2026").sum()
    log(f"  {regime:20s} N={n:4d} (orig={n_orig}, el_ouali={n_elo})  acc={acc:.4f}")

# --- Test 2: within original_733 alone, Fés-Meknés vs rest ---
log("\n=== TEST 2: within original_733 Difficult sites, Fés-Meknés vs rest ===")
orig_diff = diff[diff.origin == "original_733"]
for grp_name, mask in [("Fés-Meknés", orig_diff.Region == "Fés-Meknés"),
                        ("rest of original_733", orig_diff.Region != "Fés-Meknés")]:
    sub = orig_diff[mask]
    n = len(sub)
    acc = sub["correct"].mean() if n > 0 else float("nan")
    log(f"  {grp_name:22s} N={n:4d}  acc={acc:.4f}")

# --- Test 3: within original_733 alone, does the regime split ALSO predict failure? ---
log("\n=== TEST 3: within original_733 ONLY, terrain regime (isolates regime from origin) ===")
for regime in ["mountain_nearroad", "remote_lowland"]:
    sub = orig_diff[orig_diff.regime == regime]
    n = len(sub)
    acc = sub["correct"].mean() if n > 0 else float("nan")
    log(f"  {regime:20s} N={n:4d}  acc={acc:.4f}")

out = {
    "regime_pooled": diff.groupby("regime")["correct"].agg(["mean", "count"]).to_dict(orient="index"),
    "original733_fesmeknes_vs_rest": {
        "Fés-Meknés": {"n": int((orig_diff.Region == "Fés-Meknés").sum()),
                       "acc": float(orig_diff[orig_diff.Region == "Fés-Meknés"]["correct"].mean())},
        "rest": {"n": int((orig_diff.Region != "Fés-Meknés").sum()),
                 "acc": float(orig_diff[orig_diff.Region != "Fés-Meknés"]["correct"].mean())},
    },
    "original733_regime_only": orig_diff.groupby("regime")["correct"].agg(["mean", "count"]).to_dict(orient="index"),
}
out_path = os.path.join(BASE, "results", "json", "other", "regime_check_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
log(f"\nWrote {out_path}")
