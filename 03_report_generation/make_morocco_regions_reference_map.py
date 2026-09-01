"""
03_report_generation/make_morocco_regions_reference_map.py  (2026-08-23)

Paper 2's opening figure: Morocco's 12 official administrative regions,
numbered per the standard 1-12 indexing (Indice field in the boundaries
file, verified to match: 1=Tanger-Tétouan-Al Hoceima ... 12=Eddakhla-Oued
Eddahab / Dakhla-Oued Ed-Dahab), each labeled with its number and name.
Purely a reference/context map -- no accessibility data plotted.

Output: report/figures/map_regions_reference.pdf
"""
import os
import geopandas as gpd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "report", "figures")

# Native PDF backend, not PGF/LaTeX text -- the regions polygons have enough
# vertices that PGF's path-as-LaTeX-commands compilation stalls for minutes.
# Serif font family alone keeps it visually close to the report's look
# without paying that compile cost.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
})

ACCENT = "#2B5F72"
FILL = "#EDF2F3"
FILL_ALT = "#DCE6E8"

regions = gpd.read_file(os.path.join(FW, "data/boundaries/morocco_regions_admin12.geojson"))
regions = regions.sort_values("Indice").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(5.2, 6.4))
ax.set_facecolor("white")

for i, row in regions.iterrows():
    color = FILL if row["Indice"] % 2 == 1 else FILL_ALT
    gpd.GeoSeries([row.geometry]).plot(ax=ax, facecolor=color, edgecolor=ACCENT, linewidth=0.9)
    c = row.geometry.representative_point()
    ax.annotate(str(int(row["Indice"])), (c.x, c.y), ha="center", va="center",
                fontsize=9, fontweight="bold", color=ACCENT)

ax.set_xlim(regions.total_bounds[0] - 0.3, regions.total_bounds[2] + 0.3)
ax.set_ylim(regions.total_bounds[1] - 0.3, regions.total_bounds[3] + 0.3)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

legend_lines = [f"{int(r['Indice'])}. {r['nom_region']}" for _, r in regions.iterrows()]
legend_text = "\n".join(legend_lines)
ax.text(1.02, 0.98, legend_text, transform=ax.transAxes, fontsize=7.5,
         va="top", ha="left", linespacing=1.6)

plt.tight_layout()
out_path = os.path.join(OUT, "map_regions_reference.pdf")
plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
print(f"Saved {out_path}")
