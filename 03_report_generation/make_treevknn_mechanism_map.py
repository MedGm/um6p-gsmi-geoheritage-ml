"""
03_report_generation/make_treevknn_mechanism_map.py  (2026-09-02)

Explanatory figure for Paper 1's model-family section: WHERE the tree
ensemble's Difficult-class over-prediction actually happens on the third
labeling batch (batch3_2026), and why kNN/Gaussian Process avoid it there.

Source finding (02_modeling_and_analysis/35_treevknn_reversal_breakdown.py):
of batch3's 722 sites, the tree ensemble wrongly predicts Difficult for 53
that are genuinely not Difficult AND that kNN correctly calls not-Difficult
-- 42 of those 53 (79%) sit in just two mountain-terrain regions, Beni
Mellal-Khenifra (22) and Fes-Meknes (20), the same region whose locally
high Difficult rate is shown elsewhere (Paper 2, robustness section) to set
the tree ensemble's single global decision threshold. These sites have
real mountain terrain signatures (elevation, slope) that resemble
Fes-Meknes' Difficult profile, but are not actually Difficult -- the tree
ensemble's one global, class-balance-weighted threshold fires anyway,
while kNN/GP's local, distance-based reasoning does not generalize that
FM-specific rule onto BMK's genuinely different accessibility reality.

Output: report/figures/map_treevknn_mechanism.pdf
"""
import glob, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib
# Native PDF backend, not PGF/LaTeX -- admin12's detailed-coastline polygon
# boundaries have enough vertices that PGF (which renders every path as LaTeX
# drawing commands) overflows pdflatex's memory, the same failure the Morocco
# reference map and the Paper 2 mosaic map hit and were fixed the same way.
matplotlib.use("pdf")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9.5})
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "report", "figures")

BOUND_COL = "#2B5F72"
HIGHLIGHT_REGION_COL = "#F5E3C6"
CONTEXT_COL = "#B9C6C9"
TRUE_DIFFICULT_COL = "#C1650A"
FP_COL = "#B3131A"

catalog = pd.read_csv(os.path.join(FW, "data/final/geosites_mcdm_national.csv"))
knn_gp = pd.read_csv(os.path.join(FW, "results/json/other/phase5_knn_gp_per_site.csv"))
tree_d = pd.read_csv(os.path.join(FW, "results/json/other/phase5_difficult_oof_per_site.csv"))[
    ["Locality_ID", "pred"]].rename(columns={"pred": "tree_pred_difficult"})
df = knn_gp.merge(tree_d, on="Locality_ID").merge(
    catalog[["Locality_ID", "Latitude_WGS84", "Longitude_WGS84"]], on="Locality_ID", how="left")

new_mask = df["origin"] == "batch3_2026"
y = df["y_difficult"].values
tree_pred = df["tree_pred_difficult"].values
knn_pred = df["knn_pred_difficult"].values

# The 53 sites: batch3, tree wrongly says Difficult, kNN correctly says not-Difficult.
fp_mask = new_mask.values & (tree_pred == 1) & (y == 0) & (knn_pred == 0)
fp = df[fp_mask].copy()
batch3 = df[new_mask].copy()
batch3_true_difficult = batch3[batch3["y_difficult"] == 1]
print(f"batch3 N={len(batch3)}, tree-false-positive-corrected-by-kNN N={len(fp)}")
print(fp["Region"].value_counts())

admin12 = gpd.read_file(os.path.join(FW, "data/boundaries/morocco_regions_admin12.geojson"))
highlight_regions = admin12[admin12["nom_region"].isin(["Béni Mellal-Khénifra", "Fés-Meknés"])]

fig, ax = plt.subplots(figsize=(5.6, 6.6))
admin12.plot(ax=ax, facecolor="#FBFBFA", edgecolor=BOUND_COL, linewidth=0.8, zorder=1)
highlight_regions.plot(ax=ax, facecolor=HIGHLIGHT_REGION_COL, edgecolor=BOUND_COL, linewidth=1.1, zorder=2)

# All other (non-highlighted) batch3 sites: faint context dots.
ax.scatter(batch3["Longitude_WGS84"], batch3["Latitude_WGS84"], marker="o", s=10,
           facecolor=CONTEXT_COL, edgecolor="none", alpha=0.6, zorder=3, label="_nolegend_")

# True Difficult sites in batch3: small orange dots, for context on where real difficulty is.
ax.scatter(batch3_true_difficult["Longitude_WGS84"], batch3_true_difficult["Latitude_WGS84"],
           marker="o", s=16, facecolor=TRUE_DIFFICULT_COL, edgecolor="black", linewidth=0.3,
           zorder=4, label="_nolegend_")

# The 53 tree-false-positive sites: large red X, the actual error being explained.
ax.scatter(fp["Longitude_WGS84"], fp["Latitude_WGS84"], marker="x", s=70, linewidth=1.6,
           color=FP_COL, zorder=5, label="_nolegend_")

minx, miny, maxx, maxy = admin12.total_bounds
ax.set_xlim(minx - 0.3, maxx + 0.3)
ax.set_ylim(miny - 0.3, maxy + 0.3)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)

handles = [
    Patch(facecolor=HIGHLIGHT_REGION_COL, edgecolor=BOUND_COL, label="Béni Mellal-Khénifra / Fés-Meknés"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=CONTEXT_COL, markeredgecolor="none",
           markersize=6, label="Third-batch site (context)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=TRUE_DIFFICULT_COL, markeredgecolor="black",
           markeredgewidth=0.4, markersize=6, label="True Difficult site (third batch)"),
    Line2D([0], [0], marker="x", color=FP_COL, markersize=9, markeredgewidth=1.8, linestyle="none",
           label="Tree ensemble wrongly says Difficult (kNN/GP correct)"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=7.5,
          frameon=True, framealpha=0.92, edgecolor="#cccccc", borderpad=0.6, handletextpad=0.6, labelspacing=0.6)

plt.tight_layout()
out_path = os.path.join(OUT, "map_treevknn_mechanism.pdf")
plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print(f"Saved {out_path}")
