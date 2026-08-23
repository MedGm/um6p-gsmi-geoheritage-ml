"""
Improved version of the HDBSCAN geographic-cluster figure. The first version
colored each site by raw cluster ID (a 57-color cycling colormap with only 20
distinct hues -- unreadable, and cluster ID itself carries no meaning). This
version colors each site by ITS CLUSTER'S Difficult-rate instead, which is the
actual finding ("pockets are skewed, not smoothly mixed") -- and adds Morocco's
outline + region borders for geographic context, using the same colorblind-safe
palette already established in report/scripts/make_maps_render.py.

Reuses hdbscan_full_results.csv (already computed by explore_hdbscan.py) -- no
reclustering needed.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))

df = pd.read_csv(os.path.join(HERE, "hdbscan_full_results.csv"))
national_rate = (df["Expert_Merged"] == "Difficult").mean()

cluster_rate = df[df["geo_cluster"] != -1].groupby("geo_cluster")["Expert_Merged"] \
    .apply(lambda s: (s == "Difficult").mean())
df["cluster_difficult_rate"] = df["geo_cluster"].map(cluster_rate)

# national.geojson stores Morocco and Western Sahara as two separate Natural Earth
# admin-0 polygons -- dissolved into one so no internal boundary line is drawn across
# the Sahara (not appropriate to show on a map from a Moroccan institution).
_morocco_raw = gpd.read_file(os.path.join(BASE, "data", "boundaries", "national.geojson")).to_crs("EPSG:4326")
morocco = gpd.GeoDataFrame(geometry=[_morocco_raw.union_all()], crs=_morocco_raw.crs)
regions = gpd.read_file(os.path.join(BASE, "data", "boundaries", "morocco_regions_admin12.geojson")).to_crs("EPSG:4326")

# Same colorblind-safe teal/orange pair used throughout the report's maps
TEAL = "#76A5AF"     # low Difficult-rate pockets
ORANGE = "#C1650A"   # high Difficult-rate pockets
cmap = LinearSegmentedColormap.from_list("difficult_rate", [TEAL, "#F2E9D8", ORANGE])

minx, miny, maxx, maxy = morocco.total_bounds
aspect = (maxy - miny) / (maxx - minx)
fig_w = 8.5
fig, ax = plt.subplots(figsize=(fig_w, fig_w * aspect * 0.92))

morocco.plot(ax=ax, facecolor="#FAFAF8", edgecolor="#4a4a4a", linewidth=0.9, zorder=0)
regions.boundary.plot(ax=ax, color="#c9c9c9", linewidth=0.5, zorder=1)

noise = df[df["geo_cluster"] == -1]
ax.scatter(noise["Longitude_WGS84"], noise["Latitude_WGS84"],
           c="#d9d9d9", s=14, linewidths=0, zorder=2, label="Non regroupé (isolé)")

clustered = df[df["geo_cluster"] != -1]
sc = ax.scatter(clustered["Longitude_WGS84"], clustered["Latitude_WGS84"],
                 c=clustered["cluster_difficult_rate"], cmap=cmap, vmin=0, vmax=1,
                 s=48, edgecolors="white", linewidths=0.4, zorder=3)

cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
cbar.set_label(f"Taux de sites Difficile dans le groupe local\n(moyenne nationale = {national_rate:.0%})", fontsize=9)

legend_elems = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor="#d9d9d9", markersize=7, label="Non regroupé (isolé)"),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=TEAL, markersize=8, label="Groupe majoritairement Facile/Modéré"),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=ORANGE, markersize=8, label="Groupe majoritairement Difficile"),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=8.5, frameon=True, facecolor="white", framealpha=0.9)

ax.set_title("Regroupement géographique (HDBSCAN) coloré par taux de sites Difficile du groupe",
             fontsize=12, pad=10)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")
fig.tight_layout()

out_path = os.path.join(HERE, "geo_clusters_map_v2.png")
fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(f"Saved {out_path}")
