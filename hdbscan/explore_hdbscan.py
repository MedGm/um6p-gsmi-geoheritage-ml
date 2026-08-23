"""
Exploratory HDBSCAN analysis (2026-08-18), prompted by a live question during the
closing meeting: does the leave-region-out generalization failure (report Section 5)
trace to Difficult sites forming tight local geographic pockets rather than being
smoothly distributed within each region? If so, a model trained without seeing a
region's own pockets would plausibly struggle exactly the way the leave-region-out
results already show.

Two separate questions, kept explicitly separate since they test different things:
  Q1. Geographic clustering (lat/lon, projected to meters): do HDBSCAN clusters
      have a skewed Difficult-rate relative to the national base rate? Tight,
      class-skewed clusters would support the "local pockets" story.
  Q2. Feature-space clustering (the 6 standardized terrain/infrastructure features,
      no coordinates): does unsupervised structure in the feature space align with
      the accessibility classes at all, independent of geography? A low alignment
      would be independent evidence for the already-documented "feature-set
      ceiling" (Section 4.4) -- the 6 features may just not carry enough separable
      structure for this task, regardless of model or geography.

Uses sklearn's built-in HDBSCAN (sklearn.cluster.HDBSCAN, no external package
needed -- confirmed available in the installed sklearn 1.9.0).

This is a same-day exploratory analysis, kept in its own folder, deliberately NOT
touching code/, report/, or presentation/ -- results here are not yet vetted for
inclusion anywhere.
"""
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
OUT = HERE

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

print("Loading N=733 labeled catalog ...")
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
N = len(merged)
assert N == 733
national_difficult_rate = (merged["Expert_Merged"] == "Difficult").mean()
print(f"N={N}, national Difficult rate={national_difficult_rate:.3f}")

# ============================================================ Q1: geographic clustering
print("\n=== Q1: HDBSCAN on geography (projected coords, meters) ===")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:26191", always_xy=True)
x, y = transformer.transform(merged["Longitude_WGS84"].values, merged["Latitude_WGS84"].values)
XY = np.column_stack([x, y])

# min_cluster_size=5: smaller clusters are too small to say anything about a
# "Difficult rate" with any statistical weight; kept deliberately modest since
# the whole point is to find LOCAL pockets, not big regional blobs.
geo_clusterer = HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean")
geo_labels = geo_clusterer.fit_predict(XY)
merged["geo_cluster"] = geo_labels

n_clusters = len(set(geo_labels)) - (1 if -1 in geo_labels else 0)
n_noise = (geo_labels == -1).sum()
print(f"Geographic clusters found: {n_clusters}, noise points (unclustered): {n_noise}/{N}")

rows = []
for c in sorted(set(geo_labels)):
    if c == -1:
        continue
    sub = merged[merged["geo_cluster"] == c]
    diff_rate = (sub["Expert_Merged"] == "Difficult").mean()
    rows.append({
        "cluster": c, "n": len(sub), "difficult_rate": round(diff_rate, 3),
        "vs_national": round(diff_rate - national_difficult_rate, 3),
        "regions": ", ".join(sorted(sub["Region"].unique())),
    })
geo_summary = pd.DataFrame(rows).sort_values("vs_national", ascending=False)
geo_summary.to_csv(os.path.join(OUT, "geo_clusters_difficult_rate.csv"), index=False)
print(geo_summary.to_string(index=False))

# Purity check: weighted average |deviation from national rate|, weighted by cluster size
weighted_dev = (geo_summary["n"] * geo_summary["vs_national"].abs()).sum() / geo_summary["n"].sum()
print(f"\nSize-weighted mean |deviation from national Difficult rate| across clusters: {weighted_dev:.3f}")
print(f"(national rate = {national_difficult_rate:.3f}; a large weighted deviation means clusters ARE skewed -- "
      f"supports the 'local pockets' hypothesis. A small one means Difficult sites are geographically diffuse "
      f"even at the HDBSCAN-cluster scale.)")

# Plot: national scatter, colored by geo_cluster, Difficult sites marked
fig, ax = plt.subplots(figsize=(7, 8))
noise_mask = geo_labels == -1
ax.scatter(merged.loc[noise_mask, "Longitude_WGS84"], merged.loc[noise_mask, "Latitude_WGS84"],
           c="#cccccc", s=12, label="Noise (unclustered)", zorder=1)
clustered = merged[~noise_mask]
sc = ax.scatter(clustered["Longitude_WGS84"], clustered["Latitude_WGS84"],
                 c=clustered["geo_cluster"], cmap="tab20", s=16, zorder=2)
difficult = merged[merged["Expert_Merged"] == "Difficult"]
ax.scatter(difficult["Longitude_WGS84"], difficult["Latitude_WGS84"],
           facecolors="none", edgecolors="red", s=60, linewidths=1.2, zorder=3, label="Difficult")
ax.set_title(f"HDBSCAN geographic clusters (n={n_clusters}, noise={n_noise}) -- red rings = Difficult sites")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "geo_clusters_map.png"), dpi=150)
plt.close(fig)
print("Saved geo_clusters_map.png")

# ============================================================ Q2: feature-space clustering
print("\n=== Q2: HDBSCAN on standardized terrain/infrastructure features (no geography) ===")
Xf = merged[FEATURES].values
Xf_std = (Xf - Xf.mean(axis=0)) / Xf.std(axis=0)
feat_clusterer = HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean")
feat_labels = feat_clusterer.fit_predict(Xf_std)
merged["feat_cluster"] = feat_labels

n_fclusters = len(set(feat_labels)) - (1 if -1 in feat_labels else 0)
n_fnoise = (feat_labels == -1).sum()
print(f"Feature-space clusters found: {n_fclusters}, noise points: {n_fnoise}/{N}")

frows = []
for c in sorted(set(feat_labels)):
    if c == -1:
        continue
    sub = merged[merged["feat_cluster"] == c]
    vc = sub["Expert_Merged"].value_counts(normalize=True).round(3).to_dict()
    frows.append({"cluster": c, "n": len(sub), **{f"pct_{k}": v for k, v in vc.items()}})
feat_summary = pd.DataFrame(frows)
feat_summary.to_csv(os.path.join(OUT, "feature_clusters_class_mix.csv"), index=False)
print(feat_summary.to_string(index=False))

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
class_codes = merged["Expert_Merged"].map({"Easy": 0, "Moderate": 1, "Difficult": 2}).values
valid = feat_labels != -1
ari = adjusted_rand_score(class_codes[valid], feat_labels[valid])
nmi = normalized_mutual_info_score(class_codes[valid], feat_labels[valid])
print(f"\nAlignment between feature-space clusters and true accessibility class "
      f"(clustered points only, n={valid.sum()}):")
print(f"  Adjusted Rand Index: {ari:.4f}  (0 = no better than random, 1 = perfect match)")
print(f"  Normalized Mutual Info: {nmi:.4f}  (0 = no shared information, 1 = perfect match)")

merged.to_csv(os.path.join(OUT, "hdbscan_full_results.csv"), index=False)
print("\nSaved hdbscan_full_results.csv, geo_clusters_difficult_rate.csv, feature_clusters_class_mix.csv, geo_clusters_map.png")
print("DONE.")
