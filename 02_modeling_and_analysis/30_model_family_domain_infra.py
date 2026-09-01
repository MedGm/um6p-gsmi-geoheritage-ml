"""
02_modeling_and_analysis/30_model_family_domain_infra.py  (2026-09-01)

Follow-up to 21/23: at N=1662, kNN and Gaussian Process were found to
significantly beat the deployed tree ensemble on the Difficult target under
the plain 6-feature Baseline set (21: kNN 0.7972 vs tree 0.7665, p=0.0011;
GP 0.7924, p=0.0073 -- see 23_model_family_mcnemar.py). That comparison only
used Baseline features. Before deciding whether to redeploy kNN/GP for
Difficult, this checks whether their edge holds, grows, or evaporates once
the same Domain and Infra feature sets that helped the tree ensemble
(Domain 0.7858, Infra 0.7864) are also given to kNN and GP -- kNN in
particular can degrade with added dimensions unlike tree ensembles.

Also runs Easy for completeness, though 21/23 already found the tree
ensemble wins Easy under Baseline; useful to confirm that holds with richer
features too.

Same 500m LOGO-cluster CV protocol as 21 for kNN/LogReg; GP uses the same
disclosed StratifiedGroupKFold(10) deviation (O(n^3) per-fold cost).

Output: results/json/training/phase5_model_family_domain_infra_results.json
"""
import glob, json, os, time, warnings
import numpy as np, pandas as pd
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

catalog = pd.read_csv(os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv"))
master = pd.read_excel(os.path.join(BASE, "geosites_master_1667_with_accessibility.xlsx"))
domain_lookup = master[["Locality_ID", "Geological_Domain"]]
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
    on="Locality_ID", how="inner").merge(domain_lookup, on="Locality_ID", how="left").merge(infra, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 1662

dom_counts = merged["Geological_Domain"].value_counts()
rare_domains = dom_counts[dom_counts < 5].index
merged["Geological_Domain_grouped"] = merged["Geological_Domain"].where(
    ~merged["Geological_Domain"].isin(rare_domains), "Other").fillna("Unknown")
domain_dummies = pd.get_dummies(merged["Geological_Domain_grouped"], prefix="Domain").astype(float)
merged = pd.concat([merged, domain_dummies], axis=1)
FEATURES_DOMAIN = FEATURES_BASE + list(domain_dummies.columns)

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

RESULTS = {}
for variant_name, cols in [("Domain", FEATURES_DOMAIN), ("Infra", FEATURES_INFRA)]:
    X = merged[cols].values
    X_scaled = StandardScaler().fit_transform(X)
    RESULTS[variant_name] = {}
    for target_name, target_class in [("difficult", "Difficult"), ("easy", "Easy")]:
        y = (merged["Expert_Merged"] == target_class).astype(int).values
        log(f"\n=== {variant_name} / {target_name} ===")

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

        RESULTS[variant_name][target_name] = dict(
            knn_best=dict(k=best_k, acc=round(best_knn_acc, 4)),
            gaussian_process=dict(acc=round(acc_gp, 4), covered=int(covered.sum())),
        )

out_path = os.path.join(BASE, "results", "json", "training", "phase5_model_family_domain_infra_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(RESULTS, f, indent=2)
log(f"\nWrote {out_path}")

log("\nSummary vs tree ensemble (Domain: Difficult=0.7858 Easy=0.7202; Infra: Difficult=0.7864 Easy=0.7347):")
for variant_name in ["Domain", "Infra"]:
    for target_name in ["difficult", "easy"]:
        r = RESULTS[variant_name][target_name]
        print(f"  {variant_name}/{target_name}: kNN(k={r['knn_best']['k']})={r['knn_best']['acc']}  GP={r['gaussian_process']['acc']}")
