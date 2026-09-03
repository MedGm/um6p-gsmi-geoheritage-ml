"""
Per-site OOF predictions for kNN(k=10) and Gaussian Process, both targets,
N=1662 baseline 6-feature set -- needed to compare against the already-saved
tree-ensemble per-site OOF (phase5_difficult_oof_per_site.csv /
phase5_easy_oof_per_site.csv) and find WHERE (which origin batch, which
region) kNN/GP now beat trees on Difficult and lose to trees on Easy.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

warnings.filterwarnings("ignore")
t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
    origin_tag = "batch3_2026" if "batch3" in bn else ("el_ouali_2026" if "el_ouali" in bn else "original_733")
    frames[-1]["__origin_file"] = origin_tag
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")

merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662, N

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

X = merged[FEATURES_BASE].values
X_scaled = StandardScaler().fit_transform(X)

results = {"Locality_ID": merged["Locality_ID"].tolist(), "Region": merged["Region"].tolist(),
           "origin": merged["__origin_file"].tolist()}

for target_name, target_class in [("difficult", "Difficult"), ("easy", "Easy")]:
    y = (merged["Expert_Merged"] == target_class).astype(int).values
    results[f"y_{target_name}"] = y.tolist()

    # kNN(k=10), full LOGO-cluster CV
    preds_knn = np.zeros(N, dtype=int)
    for tr, te in LeaveOneGroupOut().split(X_scaled, y, groups=cluster_ids):
        m = KNeighborsClassifier(n_neighbors=10)
        m.fit(X_scaled[tr], y[tr])
        preds_knn[te] = m.predict(X_scaled[te])
    acc_knn = (preds_knn == y).mean()
    log(f"{target_name}: kNN(k=10) full-LOGO acc={acc_knn:.4f}")
    results[f"knn_pred_{target_name}"] = preds_knn.tolist()

    # GP, StratifiedGroupKFold(10), matches official protocol
    preds_gp = np.full(N, -1, dtype=int)
    kernel = 1.0 * RBF(length_scale=1.0)
    gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    nf = 0
    for tr, te in gkf.split(X_scaled, y, groups=cluster_ids):
        gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
        gp.fit(X_scaled[tr], y[tr])
        preds_gp[te] = gp.predict(X_scaled[te])
        nf += 1
        log(f"  {target_name} GP fold {nf}/10 done")
    covered = preds_gp >= 0
    acc_gp = (preds_gp[covered] == y[covered]).mean()
    log(f"{target_name}: GP(10-fold) acc={acc_gp:.4f} covered={covered.sum()}/{N}")
    results[f"gp_pred_{target_name}"] = preds_gp.tolist()

out = pd.DataFrame(results)
out_path = os.path.join(BASE, "results/json/other/phase5_knn_gp_per_site.csv")
out.to_csv(out_path, index=False)
log(f"Saved {out_path}")
