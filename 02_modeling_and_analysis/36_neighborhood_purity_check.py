"""
02_modeling_and_analysis/36_neighborhood_purity_check.py  (2026-09-02)

Companion to 35_treevknn_reversal_breakdown.py: tests the mechanism behind
why kNN/GP (local, distance-based methods) shifted differently than the
tree ensemble (global, axis-aligned partitions) when batch3_2026 was added.

For each existing (original_733 + el_ouali_2026) site, computes the label-
agreement rate among its 10 nearest same-cluster-excluded neighbors in
standardized 6-feature space, once using only the other old sites as
candidate neighbors and once using the full N=1662 pool (batch3 included)
-- i.e., did adding batch3 make old sites' local neighborhoods more or
less homogeneous? Also reports batch3 sites' own neighborhood purity.

Finding: for Difficult, batch3 measurably RAISES old sites' neighborhood
purity (+1.2pp) and batch3's own sites are purer still (+8.4pp above old
sites' original purity) -- local/distance-based methods directly benefit
from this. For Easy, batch3 measurably LOWERS old sites' neighborhood
purity (-3.6pp) -- consistent with kNN/GP's accuracy declining on Easy
once batch3 is added, while the tree ensemble's global partitions are far
less sensitive to local neighbor-mix changes either way.

Output: printed only.
"""
import glob, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

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
    origin = "batch3_2026" if "batch3" in bn else ("el_ouali_2026" if "el_ouali" in bn else "original_733")
    labeled["__origin"] = origin
    frames.append(labeled[["Locality_ID", "Expert_Class", "__origin"]])
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
Xs = StandardScaler().fit_transform(X)
old_mask = merged["__origin"].isin(["original_733", "el_ouali_2026"]).values
print(f"old N={old_mask.sum()}, new(batch3) N={(~old_mask).sum()}")

def knn_purity(y, candidate_mask, query_idx, k=10):
    purities = []
    for i in query_idx:
        cand = np.where(candidate_mask & (np.arange(N) != i) & (cluster_ids != cluster_ids[i]))[0]
        d = np.linalg.norm(Xs[cand] - Xs[i], axis=1)
        nn_idx = cand[np.argsort(d)[:k]]
        purities.append((y[nn_idx] == y[i]).mean())
    return np.array(purities)

for target_name, target_class in [("Difficult", "Difficult"), ("Easy", "Easy")]:
    y = (merged["Expert_Merged"] == target_class).astype(int).values
    old_idx = np.where(old_mask)[0]
    new_idx = np.where(~old_mask)[0]
    full_mask = np.ones(N, dtype=bool)

    pur_old_cands_old = knn_purity(y, old_mask, old_idx)
    pur_old_cands_full = knn_purity(y, full_mask, old_idx)
    pur_new_cands_full = knn_purity(y, full_mask, new_idx)

    print(f"\n=== {target_name} (positive rate old={y[old_mask].mean():.3f}, new={y[~old_mask].mean():.3f}) ===")
    print(f"  OLD sites, neighbors from OLD-only pool:  mean purity = {pur_old_cands_old.mean():.4f}")
    print(f"  OLD sites, neighbors from FULL 1662 pool: mean purity = {pur_old_cands_full.mean():.4f}  "
          f"(delta={pur_old_cands_full.mean()-pur_old_cands_old.mean():+.4f})")
    print(f"  NEW (batch3) sites, neighbors from FULL pool: mean purity = {pur_new_cands_full.mean():.4f}")
