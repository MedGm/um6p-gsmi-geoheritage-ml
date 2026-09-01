"""
data_audit/23_model_family_mcnemar.py  (2026-08-22)

McNemar check on data_audit/21's finding: kNN(k=10) and Gaussian Process both
numerically beat the deployed tree ensemble on N=939 (kNN: 0.7689 vs 0.7487
Difficult, 0.7380 vs 0.7167 Easy; GP: 0.7540 vs 0.7487 Difficult, 0.7487 vs
0.7167 Easy). Recomputes kNN/GP per-site predictions (kNN is nearly free;
GP ~4min/target) since script 21 only kept the aggregate accuracy, then
pairs against the already-saved Baseline_939 tree-ensemble OOF for a real
McNemar test, same standard as every other finding today.

Note: GP used StratifiedGroupKFold(10) (disclosed deviation, GP is O(n^3)
per fold), not full LOGO-cluster CV -- the tree-ensemble OOF being paired
against IS full LOGO-cluster. This is a real protocol mismatch, disclosed
here: the comparison is approximate, not a perfectly matched paired test.
kNN was run under full LOGO-cluster CV, so its comparison is clean.
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from statsmodels.stats.contingency_tables import mcnemar

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
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

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
X = merged[FEATURES_BASE].values
X_scaled = StandardScaler().fit_transform(X)
log(f"N={N}, clusters={len(np.unique(cluster_ids))}")

def mcnemar_test(correct_a, correct_b, label):
    table = pd.crosstab(pd.Series(correct_a), pd.Series(correct_b)).reindex(
        index=[False, True], columns=[False, True], fill_value=0).values
    res = mcnemar(table, exact=False, correction=True)
    n10 = int(((correct_a == True) & (correct_b == False)).sum())
    n01 = int(((correct_a == False) & (correct_b == True)).sum())
    log(f"  {label}: chi2={res.statistic:.4f} p={res.pvalue:.4f} (a_right_b_wrong={n10}, a_wrong_b_right={n01})")
    return {"chi2": round(float(res.statistic),4), "p": round(float(res.pvalue),4), "n10": n10, "n01": n01}

out = {}
for target_name, target_class in [("difficult", "Difficult"), ("easy", "Easy")]:
    y = (merged["Expert_Merged"] == target_class).astype(int).values
    log(f"\n=== {target_name} ===")

    baseline_oof = pd.read_csv(os.path.join(BASE, f"results/json/other/phase5_{target_name}_oof_per_site.csv"))
    correct_tree = ((baseline_oof["proba"] >= 0.5).astype(int) == baseline_oof["y"]).values

    # --- k-NN(k=10), full LOGO-cluster CV ---
    preds_knn = np.zeros(N, dtype=int)
    for tr, te in LeaveOneGroupOut().split(X_scaled, y, groups=cluster_ids):
        m = KNeighborsClassifier(n_neighbors=min(10, len(tr)))
        m.fit(X_scaled[tr], y[tr])
        preds_knn[te] = m.predict(X_scaled[te])
    correct_knn = (preds_knn == y)
    log(f"  kNN(k=10) acc={correct_knn.mean():.4f} (script 21 reported {'0.7689' if target_name=='difficult' else '0.7380'})")

    # --- Gaussian Process, StratifiedGroupKFold(10) -- disclosed protocol mismatch ---
    preds_gp = np.full(N, -1, dtype=int)
    kernel = 1.0 * RBF(length_scale=1.0)
    gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    for tr, te in gkf.split(X_scaled, y, groups=cluster_ids):
        gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
        gp.fit(X_scaled[tr], y[tr])
        preds_gp[te] = gp.predict(X_scaled[te])
    correct_gp = (preds_gp == y)
    log(f"  GP acc={correct_gp.mean():.4f} (script 21 reported {'0.7540' if target_name=='difficult' else '0.7487'})")

    out[target_name] = {
        "tree_vs_knn": mcnemar_test(correct_tree, correct_knn, "tree vs kNN(k=10)"),
        "tree_vs_gp": mcnemar_test(correct_tree, correct_gp, "tree vs GP (protocol mismatch, see docstring)"),
    }

out_path = os.path.join(BASE, "results/json/other/phase5_model_family_mcnemar_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log(f"\nWrote {out_path}")
