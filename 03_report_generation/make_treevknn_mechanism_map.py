r"""
03_report_generation/make_treevknn_mechanism_map.py  (2026-09-02, revised for
scope + zoom per user feedback)

Explanatory figure for Paper 1's model-family section: WHERE the tree
ensemble's Difficult-class over-prediction actually happens among the 722
sites added to complete the dataset (939->1662, internally tagged
"batch3_2026" in the label-source CSVs -- see data/final/regional_label_
sources/ -- but described in the papers as completing a single dataset,
not as a separate experimental batch), and why kNN/Gaussian Process avoid
it there.

Source finding (02_modeling_and_analysis/35_treevknn_reversal_breakdown.py):
of the 722 newly-completed sites, the tree ensemble wrongly predicts Difficult for 53
that are genuinely not Difficult AND that kNN correctly calls not-Difficult
-- 42 of those 53 (79%) sit in just two mountain-terrain regions, Beni
Mellal-Khenifra (22) and Fes-Meknes (20), the same region whose locally
high Difficult rate is shown elsewhere (Paper 2, robustness section) to set
the tree ensemble's single global decision threshold.

Two-panel layout (left: cropped map, right: zoom inset on the densest
cluster), each panel portrait-oriented so the combined figure is tall
rather than wide when placed at partial \linewidth in the report. Left
panel is cropped to mainland Morocco's northern/central regions (Tanger-
Tetouan-Al Hoceima down through Souss-Massa/Draa-Tafilalet) -- all but 3 of
the 53 highlighted sites fall inside this crop; the 3 in Guelmim-Oued Noun,
Laayoune-Sakia El Hamra and Eddakhla-Oued Eddahab (Western Sahara, far
south) are outside it and disclosed as such in the caption. Right panel
zooms into the single densest cluster of highlighted sites (a Beni
Mellal-Khenifra sub-area), shown as a bordered box on the left panel.

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
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "report", "figures")

BOUND_COL = "#2B5F72"
HIGHLIGHT_REGION_COL = "#F5E3C6"
CONTEXT_COL = "#B9C6C9"
TRUE_DIFFICULT_COL = "#C1650A"
FP_COL = "#B3131A"
ZOOM_BOX_COL = "#1A1A1A"

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

# Left-panel scope: mainland regions only, dropping the three Western Sahara
# regions (Guelmim-Oued Noun, Laayoune-Sakia El Hamra, Eddakhla-Oued Eddahab)
# -- they carry only 3 of the 53 highlighted sites between them and force a
# much larger, mostly-empty map if kept in view.
WESTERN_SAHARA = ["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Eddakhla-Oued Eddahab"]
mainland = admin12[~admin12["nom_region"].isin(WESTERN_SAHARA)]
mainland_highlight = mainland[mainland["nom_region"].isin(["Béni Mellal-Khénifra", "Fés-Meknés"])]

fp_in_view = fp[~fp["Region"].isin(WESTERN_SAHARA)]
fp_out_of_view = fp[fp["Region"].isin(WESTERN_SAHARA)]
batch3_in_view = batch3[~batch3["Region"].isin(WESTERN_SAHARA)]

# Crop to the highlighted sites' own bounding box (plus margin), not the full
# admin boundary -- the full mainland extent drags in Oriental's eastern
# panhandle and the empty Atlantic bulge, making the panel needlessly wide
# without adding any relevant sites (found 2026-09-02, user asked for a
# tighter, less horizontally-wide scope).
MARGIN_DEG = 0.5
mminx = fp_in_view["Longitude_WGS84"].min() - MARGIN_DEG
mmaxx = fp_in_view["Longitude_WGS84"].max() + MARGIN_DEG
mminy = fp_in_view["Latitude_WGS84"].min() - MARGIN_DEG
mmaxy = fp_in_view["Latitude_WGS84"].max() + MARGIN_DEG
batch3_true_difficult_in_view = batch3_true_difficult[~batch3_true_difficult["Region"].isin(WESTERN_SAHARA)]
print(f"In cropped view: {len(fp_in_view)}/{len(fp)} highlighted sites "
      f"({len(fp_out_of_view)} outside, in {sorted(fp_out_of_view['Region'].unique())})")

# Zoom box: the single densest cluster of highlighted sites (a Beni Mellal-
# Khenifra sub-area), picked by inspection of the coordinate spread.
ZOOM = dict(lon_min=-6.75, lon_max=-5.80, lat_min=31.90, lat_max=32.50)

# Explicit, computed axes positions rather than aspect="equal" + guessed
# figure-fraction offsets: with aspect="equal", matplotlib shrinks each map
# axes to its true data aspect ratio WITHIN its allocated cell, and neither
# a fixed-offset legend nor a post-draw get_position() reliably captures
# where that shrunk box actually ends up on the pdf backend (found
# 2026-09-02, both approaches left a large wrong blank gap). Instead, size
# each axes box in figure-fraction units to already match its own data
# aspect ratio exactly -- no shrinking occurs, so there is no ambiguity
# about where an axes' visible content ends.
xlimL = (mminx - 0.25, mmaxx + 0.25)
ylimL = (mminy - 0.25, mmaxy + 0.25)
xlimR = (ZOOM["lon_min"], ZOOM["lon_max"])
ylimR = (ZOOM["lat_min"], ZOOM["lat_max"])
aspect_L = (ylimL[1] - ylimL[0]) / (xlimL[1] - xlimL[0])
aspect_R = (ylimR[1] - ylimR[0]) / (xlimR[1] - xlimR[0])

FIG_W_IN = 6.6
MARGIN = 0.03      # figure-fraction outer margin
GAP = 0.04          # figure-fraction gap between the two panels
TITLE_H = 0.05      # figure-fraction reserved for each panel's title
LEGEND_H = 0.24     # figure-fraction reserved for the legend row

panel_w = (1 - 2 * MARGIN - GAP) / 2

# Solve for a figure height such that the left (taller) panel's fraction-height,
# combined with margins/title/legend fractions, is self-consistent.
FIG_H_IN = (panel_w * FIG_W_IN * aspect_L) / (1 - 2 * MARGIN - TITLE_H - LEGEND_H)

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
maps_bottom = 1 - MARGIN - TITLE_H - (panel_w * FIG_W_IN * aspect_L) / FIG_H_IN
axL = fig.add_axes([MARGIN, maps_bottom, panel_w, (panel_w * FIG_W_IN * aspect_L) / FIG_H_IN])
axR = fig.add_axes([MARGIN + panel_w + GAP, maps_bottom, panel_w, (panel_w * FIG_W_IN * aspect_R) / FIG_H_IN])

def draw_panel(ax, xlim, ylim, point_s_scale=1.0):
    mainland.plot(ax=ax, facecolor="#FBFBFA", edgecolor=BOUND_COL, linewidth=0.8, zorder=1)
    mainland_highlight.plot(ax=ax, facecolor=HIGHLIGHT_REGION_COL, edgecolor=BOUND_COL, linewidth=1.1, zorder=2)
    ax.scatter(batch3_in_view["Longitude_WGS84"], batch3_in_view["Latitude_WGS84"], marker="o",
               s=10 * point_s_scale, facecolor=CONTEXT_COL, edgecolor="none", alpha=0.6, zorder=3)
    ax.scatter(batch3_true_difficult_in_view["Longitude_WGS84"], batch3_true_difficult_in_view["Latitude_WGS84"],
               marker="o", s=16 * point_s_scale, facecolor=TRUE_DIFFICULT_COL, edgecolor="black",
               linewidth=0.3, zorder=4)
    ax.scatter(fp_in_view["Longitude_WGS84"], fp_in_view["Latitude_WGS84"], marker="x",
               s=70 * point_s_scale, linewidth=1.6 * point_s_scale, color=FP_COL, zorder=5)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)

# Left: cropped mainland scope.
draw_panel(axL, (mminx - 0.25, mmaxx + 0.25), (mminy - 0.25, mmaxy + 0.25))
axL.add_patch(Rectangle((ZOOM["lon_min"], ZOOM["lat_min"]), ZOOM["lon_max"] - ZOOM["lon_min"],
                         ZOOM["lat_max"] - ZOOM["lat_min"], fill=False, edgecolor=ZOOM_BOX_COL,
                         linewidth=1.3, zorder=6))
axL.set_title("Mainland scope", fontsize=9)

# Right: zoom on the densest cluster.
draw_panel(axR, (ZOOM["lon_min"], ZOOM["lon_max"]), (ZOOM["lat_min"], ZOOM["lat_max"]), point_s_scale=2.2)
for spine in axR.spines.values():
    spine.set_edgecolor(ZOOM_BOX_COL); spine.set_linewidth(1.3)
axR.set_title("Zoom: densest cluster", fontsize=9)

handles = [
    Patch(facecolor=HIGHLIGHT_REGION_COL, edgecolor=BOUND_COL, label="Béni Mellal-Khénifra / Fés-Meknés"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=CONTEXT_COL, markeredgecolor="none",
           markersize=6, label="Newly-completed site (context)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor=TRUE_DIFFICULT_COL, markeredgecolor="black",
           markeredgewidth=0.4, markersize=6, label="True Difficult site (newly-completed)"),
    Line2D([0], [0], marker="x", color=FP_COL, markersize=9, markeredgewidth=1.8, linestyle="none",
           label="Tree ensemble wrongly says Difficult (kNN/GP correct)"),
]
# Legend sits directly below the maps' own computed bottom (maps_bottom),
# in the LEGEND_H fraction already reserved for it above -- no guessing,
# no post-draw measurement needed since maps_bottom was computed exactly.
legend_ax = fig.add_axes([MARGIN, 0.005, 1 - 2 * MARGIN, maps_bottom - 0.01])
legend_ax.axis("off")
legend_ax.legend(handles=handles, loc="center", ncol=1, fontsize=7.5,
                  frameon=True, framealpha=0.92, edgecolor="#cccccc", borderpad=0.6,
                  handletextpad=0.6, labelspacing=0.6)

out_path = os.path.join(OUT, "map_treevknn_mechanism.pdf")
plt.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
plt.close(fig)
print(f"Saved {out_path}")
