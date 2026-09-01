"""
Check Oriental's 4 labeled sites against the deployed NATIONAL models
(GP+Infra Difficult, Tree+Infra Easy) -- same pipeline/config as
02_modeling_and_analysis/32_deployed_model_confusion.py, but only fitting
the specific CV folds whose test set intersects Oriental (Oriental's 4
sites are geographically far apart, ~100s of km, so each is its own
500m-cluster -- fitting all of GP's 10 folds / all ~1300+ LOGO folds
nationally is unnecessary just to score these 4).
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
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
oriental_idx = np.where(merged["Region"].values == "Oriental")[0]
log(f"N={N}, clusters={len(np.unique(cluster_ids))}, Oriental rows={oriental_idx.tolist()}")
log(f"Oriental cluster ids: {cluster_ids[oriental_idx].tolist()} (all distinct -> each its own cluster)")

X_infra = merged[FEATURES_INFRA].values
X_infra_scaled = StandardScaler().fit_transform(X_infra)

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

# --- Difficult: GP+Infra, StratifiedGroupKFold(10), same config/random_state as 30/31/32 ---
y_diff = (merged["Expert_Merged"] == "Difficult").astype(int).values
kernel = 1.0 * RBF(length_scale=1.0)
gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
preds_gp_oriental = {}
for fold_i, (tr, te) in enumerate(gkf.split(X_infra_scaled, y_diff, groups=cluster_ids)):
    te_set = set(te.tolist())
    hit = te_set.intersection(oriental_idx.tolist())
    if not hit:
        continue
    log(f"  GP+Infra: Oriental rows {sorted(hit)} land in fold {fold_i} -- fitting this fold only")
    gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
    gp.fit(X_infra_scaled[tr], y_diff[tr])
    p = gp.predict(X_infra_scaled[te])
    proba = gp.predict_proba(X_infra_scaled[te])[:, 1]
    idx_pos = {idx: k for k, idx in enumerate(te)}
    for idx in hit:
        preds_gp_oriental[idx] = (int(p[idx_pos[idx]]), float(proba[idx_pos[idx]]))

# --- Easy: Tree+Infra, full LOGO-cluster CV, reusing 17's best_configs, same as 32 ---
y_easy = (merged["Expert_Merged"] == "Easy").astype(int).values
infra_results = json.load(open(os.path.join(BASE, "results/json/training/phase5_infra_feature_results.json")))
cfgs_easy = infra_results["InfraAdd_939_easy"]["best_configs"]

logo = LeaveOneGroupOut()
preds_easy_oriental = {}
for fold_i, (tr, te) in enumerate(logo.split(X_infra, y_easy, groups=cluster_ids)):
    te_set = set(te.tolist())
    hit = te_set.intersection(oriental_idx.tolist())
    if not hit:
        continue
    log(f"  Tree+Infra: Oriental row {sorted(hit)} is its own LOGO fold {fold_i} -- fitting")
    sw = compute_sample_weight("balanced", y_easy[tr])
    probs = []
    for kind, cfg in cfgs_easy:
        m = make_model(kind, cfg)
        m.fit(X_infra[tr], y_easy[tr], sample_weight=sw)
        probs.append(m.predict_proba(X_infra[te])[:, 1])
    proba = np.mean(probs, axis=0)
    p = (proba >= 0.5).astype(int)
    idx_pos = {idx: k for k, idx in enumerate(te)}
    for idx in hit:
        preds_easy_oriental[idx] = (int(p[idx_pos[idx]]), float(proba[idx_pos[idx]]))

log("\n=== Oriental per-site results ===")
rows = []
for idx in oriental_idx:
    lid = merged.loc[idx, "Locality_ID"]
    true_cls = merged.loc[idx, "Expert_Merged"]
    d_pred, d_proba = preds_gp_oriental.get(idx, (None, None))
    e_pred, e_proba = preds_easy_oriental.get(idx, (None, None))
    # combined 3-class decision, same rule as 29_paper2_map_oof.py: Difficult wins if
    # predicted Difficult, else Easy if predicted Easy, else Moderate
    if d_pred == 1:
        combined = "Difficult"
    elif e_pred == 1:
        combined = "Easy"
    else:
        combined = "Moderate"
    correct = (combined == true_cls)
    row = dict(Locality_ID=lid, true_class=true_cls, gp_infra_difficult_pred=d_pred, gp_infra_difficult_proba=d_proba,
               tree_infra_easy_pred=e_pred, tree_infra_easy_proba=e_proba, combined_pred=combined, correct=correct,
               lat=merged.loc[idx, "Latitude_WGS84"], lon=merged.loc[idx, "Longitude_WGS84"])
    rows.append(row)
    log(f"{lid}: true={true_cls} | GP+Infra Difficult pred={d_pred} (p={d_proba:.3f}) | Tree+Infra Easy pred={e_pred} (p={e_proba:.3f}) | combined={combined} | correct={correct}")

out = pd.DataFrame(rows)
out_path = os.path.join(BASE, "results/json/other/oriental_national_model_check.csv")
out.to_csv(out_path, index=False)
log(f"\nSaved {out_path}")
log(f"Oriental accuracy under national model (combined 3-class): {out['correct'].mean():.3f} ({out['correct'].sum()}/{len(out)})")
