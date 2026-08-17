"""
code/26_favorability_feature_ablation.py  (2026-08-17)

Multi-task quick win from the day's research plan: scores all 733 labeled
sites with the existing geosite-location favorability model (models/final/
geosite_location_pilot_model_v3.joblib, AUC 0.927, trained to predict where
terrain/geology resembles known geosite locations -- a DIFFERENT target than
accessibility), then tests whether adding that score as a 7th feature to the
accessibility classifiers helps, under the same 500m LOGO-cluster CV protocol
used everywhere else in this repo. Single-config RF (not full grid search --
this is a quick screening test, not a claim of a new deployed model) with the
same confidence+class-balance sample weighting as G0a/G0b/G1/G2.

Output: results/json/other/favorability_feature_ablation.json
"""
import glob, json, os, time
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

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
log(f"N={N}, favorability missing: {merged['Favorability_Score'].isna().sum()}")

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

RF_CFG = {
    "difficult": dict(max_depth=None, min_samples_leaf=2, n_estimators=200, random_state=42, n_jobs=1),
    "easy": dict(max_depth=None, min_samples_leaf=1, n_estimators=400, random_state=42, n_jobs=1),
}

def logo_acc(cols, y, cfg):
    X = merged[cols].values
    logo = LeaveOneGroupOut()
    preds = np.full(N, np.nan)
    for tr, te in logo.split(X, y, groups=cluster_ids):
        if len(np.unique(y[tr])) < 2:
            continue
        sw = conf_w[tr] * compute_sample_weight("balanced", y[tr])
        m = RandomForestClassifier(**cfg)
        m.fit(X[tr], y[tr], sample_weight=sw)
        preds[te] = m.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(preds)
    return float(((preds[valid] >= 0.5).astype(int) == y[valid]).mean())

results = {}
for name, expr in [("difficult", merged["Expert_Merged"] == "Difficult"),
                     ("easy", merged["Expert_Merged"] == "Easy")]:
    y = expr.astype(int).values
    log(f"Running {name} (RF-only, single config, {n_clusters} folds x2 feature sets) ...")
    acc6 = logo_acc(FEATURES, y, RF_CFG[name])
    acc7 = logo_acc(FEATURES + ["Favorability_Score"], y, RF_CFG[name])
    results[name] = {"acc_6feat_RF_only": round(acc6, 4), "acc_7feat_with_favorability": round(acc7, 4),
                      "delta_pp": round(100 * (acc7 - acc6), 2)}
    log(f"  {name}: 6-feat={acc6:.4f}  +favorability={acc7:.4f}  delta={100*(acc7-acc6):+.2f}pp")

out_path = os.path.join(BASE, "results", "json", "other", "favorability_feature_ablation.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
log(f"Wrote {out_path}")
