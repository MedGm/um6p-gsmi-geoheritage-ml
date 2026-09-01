"""
02_modeling_and_analysis/32_deployed_model_confusion.py  (2026-09-01)

Precision/recall/F1 per class for the two models actually deployed in
Paper 1's headline table and national map: GP+Infra for Difficult
(StratifiedGroupKFold(10), matching 30/31's protocol) and Tree+Infra for
Easy (full 500m LOGO-cluster CV, reusing 17's best_configs). 30/31 only
reported aggregate accuracy; this fills in the per-class breakdown needed
for Paper 1's Table 8.

Output: results/json/other/deployed_model_confusion.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
infra = pd.read_csv(os.path.join(BASE, "data/final/infra_features.csv"))

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
    on="Locality_ID", how="inner").merge(infra, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

SENTINEL_DIST_M = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
settle_dummies = pd.get_dummies(merged["nearest_settlement_type"], prefix="Settlement").astype(float)
merged = pd.concat([merged, settle_dummies], axis=1)
INFRA_NUMERIC = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"]
FEATURES_INFRA = FEATURES_BASE + INFRA_NUMERIC + list(settle_dummies.columns)

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
log(f"N={N}, clusters={len(np.unique(cluster_ids))}")

X_infra = merged[FEATURES_INFRA].values
X_infra_scaled = StandardScaler().fit_transform(X_infra)

# --- Difficult: GP+Infra, StratifiedGroupKFold(10), matches 30/31's protocol ---
y_diff = (merged["Expert_Merged"] == "Difficult").astype(int).values
preds_gp = np.full(N, -1, dtype=int)
kernel = 1.0 * RBF(length_scale=1.0)
gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
n_fold = 0
for tr, te in gkf.split(X_infra_scaled, y_diff, groups=cluster_ids):
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
    gp.fit(X_infra_scaled[tr], y_diff[tr])
    preds_gp[te] = gp.predict(X_infra_scaled[te])
    n_fold += 1
    log(f"  GP+Infra fold {n_fold}/10 done")
acc_gp = accuracy_score(y_diff, preds_gp)
log(f"GP+Infra Difficult acc={acc_gp:.4f}")
prec_d, rec_d, f1_d, _ = precision_recall_fscore_support(y_diff, preds_gp, labels=[0,1])
log(f"  Difficult: not-Difficult P={prec_d[0]:.3f} R={rec_d[0]:.3f} F1={f1_d[0]:.3f} | Difficult P={prec_d[1]:.3f} R={rec_d[1]:.3f} F1={f1_d[1]:.3f}")

# --- Easy: Tree+Infra, full LOGO-cluster CV, reusing 17's best_configs ---
y_easy = (merged["Expert_Merged"] == "Easy").astype(int).values
infra_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_infra_feature_results.json")))
cfgs_easy = infra_results["InfraAdd_939_easy"]["best_configs"]

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def fold_pred(cfgs, X, y, tr, te):
    sw = compute_sample_weight("balanced", y[tr])
    probs = []
    for kind, cfg in cfgs:
        m = make_model(kind, cfg)
        m.fit(X[tr], y[tr], sample_weight=sw)
        probs.append(m.predict_proba(X[te])[:, 1])
    return te, (np.mean(probs, axis=0) >= 0.5).astype(int)

folds = list(LeaveOneGroupOut().split(X_infra, y_easy, groups=cluster_ids))
results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_pred)(cfgs_easy, X_infra, y_easy, tr, te) for tr, te in folds)
preds_easy = np.zeros(N, dtype=int)
for te, p in results: preds_easy[te] = p
acc_easy = accuracy_score(y_easy, preds_easy)
log(f"Tree+Infra Easy acc={acc_easy:.4f}")
prec_e, rec_e, f1_e, _ = precision_recall_fscore_support(y_easy, preds_easy, labels=[0,1])
log(f"  Easy: not-Easy P={prec_e[0]:.3f} R={rec_e[0]:.3f} F1={f1_e[0]:.3f} | Easy P={prec_e[1]:.3f} R={rec_e[1]:.3f} F1={f1_e[1]:.3f}")

result = {
    "difficult_gp_infra": {"acc": round(float(acc_gp),4),
        "not_difficult": {"precision": round(float(prec_d[0]),3), "recall": round(float(rec_d[0]),3), "f1": round(float(f1_d[0]),3)},
        "difficult": {"precision": round(float(prec_d[1]),3), "recall": round(float(rec_d[1]),3), "f1": round(float(f1_d[1]),3)}},
    "easy_tree_infra": {"acc": round(float(acc_easy),4),
        "not_easy": {"precision": round(float(prec_e[0]),3), "recall": round(float(rec_e[0]),3), "f1": round(float(f1_e[0]),3)},
        "easy": {"precision": round(float(prec_e[1]),3), "recall": round(float(rec_e[1]),3), "f1": round(float(f1_e[1]),3)}},
}
out_path = os.path.join(BASE, "results/json/other/deployed_model_confusion.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
log(f"Wrote {out_path}")
print(json.dumps(result, indent=2))
