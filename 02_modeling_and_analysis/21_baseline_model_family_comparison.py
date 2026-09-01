"""
data_audit/21_baseline_model_family_comparison.py  (2026-08-22)

PV section 2: "justify rigorously the choice of the baseline model set, add
comparative results" for Logistic Regression, k-NN, and Gaussian Process,
alongside the deployed tree ensemble. Mirrors code/22_comprehensive_final.py's
H5/H6 methodology (same disclosed GP deviation: StratifiedGroupKFold(10) not
full LOGO, GP fit is O(n^3) per fold) but focused and rerun clean on the
current N=939 audited data, Baseline_939 6-feature set, no confidence
weighting -- the old comparison predates the el_ouali_2026 batch and the
confidence-weighting retirement.

Output: results/json/training/phase5_model_family_comparison_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

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

def combined_weight(y_tr):
    return compute_sample_weight("balanced", y_tr)

RESULTS = {}
for target_name, target_class in [("difficult", "Difficult"), ("easy", "Easy")]:
    y = (merged["Expert_Merged"] == target_class).astype(int).values
    log(f"\n=== {target_name} ===")

    # --- Logistic Regression: full LOGO-cluster CV, standardized features ---
    preds_lr = np.zeros(N, dtype=int)
    for tr, te in LeaveOneGroupOut().split(X_scaled, y, groups=cluster_ids):
        sw = combined_weight(y[tr])
        lr = LogisticRegression(max_iter=2000)
        lr.fit(X_scaled[tr], y[tr], sample_weight=sw)
        preds_lr[te] = lr.predict(X_scaled[te])
    acc_lr = accuracy_score(y, preds_lr)
    log(f"  Logistic Regression: LOGO-cluster CV acc={acc_lr:.4f}")

    # --- k-NN: feature-space, standardized, small k grid, full LOGO-cluster CV ---
    best_knn_acc, best_k = 0, None
    for k in [5, 10, 15, 25]:
        preds_knn = np.zeros(N, dtype=int)
        for tr, te in LeaveOneGroupOut().split(X_scaled, y, groups=cluster_ids):
            m = KNeighborsClassifier(n_neighbors=min(k, len(tr)))
            m.fit(X_scaled[tr], y[tr])
            preds_knn[te] = m.predict(X_scaled[te])
        acc_knn = accuracy_score(y, preds_knn)
        log(f"  k-NN (k={k}): LOGO-cluster CV acc={acc_knn:.4f}")
        if acc_knn > best_knn_acc:
            best_knn_acc, best_k = acc_knn, k

    # --- Gaussian Process: RBF kernel, StratifiedGroupKFold(10) -- disclosed
    #     deviation from full LOGO, GP fit is O(n^3) per fold (see docstring) ---
    preds_gp = np.full(N, -1, dtype=int)
    kernel = 1.0 * RBF(length_scale=1.0)
    gkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    t_gp = time.time()
    n_fold = 0
    for tr, te in gkf.split(X_scaled, y, groups=cluster_ids):
        gp = GaussianProcessClassifier(kernel=kernel, random_state=42, n_jobs=-1)
        gp.fit(X_scaled[tr], y[tr])
        preds_gp[te] = gp.predict(X_scaled[te])
        n_fold += 1
        log(f"    GP fold {n_fold}/10 done [{time.time()-t_gp:.0f}s]")
    covered = preds_gp >= 0
    acc_gp = accuracy_score(y[covered], preds_gp[covered])
    log(f"  Gaussian Process (10-fold group CV, not full LOGO): acc={acc_gp:.4f} (covered {covered.sum()}/{N})")

    RESULTS[target_name] = dict(
        logistic_regression=round(acc_lr, 4),
        knn_best=dict(k=best_k, acc=round(best_knn_acc, 4)),
        gaussian_process=dict(acc=round(acc_gp, 4), protocol="StratifiedGroupKFold(10), not full LOGO", covered=int(covered.sum())),
    )

out_path = os.path.join(BASE, "results", "json", "training", "phase5_model_family_comparison_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2)
log(f"\nWrote {out_path}")

log("\nSummary vs tree ensemble (Baseline_939: Difficult=0.7487, Easy=0.7167):")
for target_name in ["difficult", "easy"]:
    r = RESULTS[target_name]
    print(f"  {target_name}: LogReg={r['logistic_regression']}  kNN(k={r['knn_best']['k']})={r['knn_best']['acc']}  GP={r['gaussian_process']['acc']}")
