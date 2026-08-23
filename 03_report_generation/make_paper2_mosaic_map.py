"""
03_report_generation/make_paper2_mosaic_map.py  (2026-08-23)

Paper 2's final figure: one national-extent 3-class accessibility map,
assembled from the already-computed per-unit rasters in
results/grids/paper2_region_grids.pkl (no refitting -- reuses that pickle's
`cls`/`lon2d`/`lat2d` per unit directly).

Zone-composition decisions (per user's explicit call, 2026-08-23: "for the
south, use the best one out of duo+eddakhla independent or full trio, same
for casa+rabat" -- a single choice per zone, not per-target):

  South (Guelmim-Oued Noun + Laâyoune-Sakia El Hamra + Eddakhla-Oued Eddahab):
    Recomputed after the 2026-08-23 OOF ring-truth fix (02_modeling_and_analysis/29) -- the
    original comparison below used the buggy nearest-grid-cell misclass count
    and wrongly showed independent as clearly ahead; the corrected numbers
    are an EXACT tie:
    trio 3-class accuracy (true LOGO-CV OOF) = 42/55 = 0.764
    independent (duo + eddakhla) weighted accuracy = (18+24)/55 = 0.764
      duo: 22 sites, 4 misclassified -> 18/22 = 0.818
      eddakhla: 33 sites, 9 misclassified -> 24/33 = 0.727
    -> DECISION: independent kept (duo's own map for Guelmim+Laâyoune,
       eddakhla's own map for Eddakhla) -- not because it is more accurate
       (it is not, the two are tied), but because it was already the built
       pipeline and gives finer per-territory modeling at no accuracy cost.
       Disclosed as a tie, not a win, in the paper text.

  Rabat-Salé-Kénitra + Grand Casablanca-Settat:
    No standalone alternative exists for the WHOLE zone: Casablanca-Settat
    was never fit standalone (N=7, too thin -- the reason this merge exists
    at all) and Rabat's own Difficult target is degenerate standalone
    (n_pos<3). Rabat-standalone/merged Easy comparison restricted to Rabat's
    own sites came back an exact tie (0.8095 both ways, see
    02_modeling_and_analysis/28_paper2_rabatcasa_zone_compare.py).
    -> DECISION: merged (rabatcasa) for the whole zone -- only config with
       full coverage, and no accuracy sacrificed on the one target where an
       alternative existed at all.

Oriental has zero labeled sites in the whole N=939 catalog (never modeled)
-- shown greyed out/hatched with a caption note, not filled from any
national-level fallback model, per user's explicit choice.

Revised 2026-08-23 alongside make_paper2_region_maps.py after user review:
same three fixes ported here (this script had the identical bugs, since it
was written from the same first-draft pattern): (1) misclassification ring
truth now comes from 02_modeling_and_analysis/29's exact per-site LOGO-cluster CV OOF
predictions, not a coarse nearest-grid-cell raster lookup; (2) each unit's
raster is clipped to its own exact polygon boundary so no color bleeds past
the true region shape; (3) all 939 points at national scale are far denser
than any single region map, so declutter_points/auto_min_dist_km (ported
from make_maps_render.py, Paper 1's own established fix for this) is applied
here too, per-unit so a crowded corridor in one region doesn't starve
thinning budget from a sparser region elsewhere.

Output: report/figures/map_national_mosaic.pdf
"""
import math, os, glob, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod

import matplotlib
# Native PDF backend, not PGF/LaTeX -- this figure clips 9 units' rasters to
# their own real (detailed-coastline) polygon boundaries, and PGF renders
# every clip path as LaTeX drawing commands: with this many high-vertex-count
# paths in one figure, pdflatex overflows ("TeX capacity exceeded"), the same
# class of failure the Morocco reference map hit and was fixed the same way.
matplotlib.use("pdf")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 9})
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures")
GRID_DIR = os.path.join(FW, "results", "grids")

EASY_COL, MODERATE_COL, DIFFICULT_COL = "#76A5AF", "#E5D3A7", "#C1650A"
BOUND_COL = "#2B5F72"
NODATA_COL = "#D9D9D6"
MARKER, MARKER_EDGE = "o", "black"
POINT_COLORS = {"Easy": EASY_COL, "Moderate": MODERATE_COL, "Difficult": DIFFICULT_COL}
POINT_S = 13  # was 5 -- at national scale the old size was nearly invisible
CLASS_TO_INT = {"Easy": 0, "Moderate": 1, "Difficult": 2}
GEOD = Geod(ellps="WGS84")

def boundary_clip_patch(gdf_row_geom):
    # Exterior rings ONLY -- see make_paper2_region_maps.py's boundary_clip_patch
    # docstring: union_all() on a merged unit's two source polygons leaves
    # degenerate sliver "holes" along their shared seam, which rendered as a
    # blank gap over a whole sub-region when included as clip-path holes.
    union = gdf_row_geom
    geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
    verts, codes = [], []
    for geom in geoms:
        xy = np.array(geom.exterior.coords)
        verts.extend(xy.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)

def declutter_points(lat, lon, min_dist_km):
    lat, lon = np.asarray(lat), np.asarray(lon)
    kept_lat, kept_lon = [], []
    keep_mask = np.zeros(len(lat), dtype=bool)
    for i in range(len(lat)):
        if not kept_lat:
            keep_mask[i] = True
            kept_lat.append(lat[i]); kept_lon.append(lon[i])
            continue
        d = np.asarray(GEOD.inv(np.full(len(kept_lon), lon[i]), np.full(len(kept_lat), lat[i]),
                                 kept_lon, kept_lat)[2]) / 1000.0
        if d.min() >= min_dist_km:
            keep_mask[i] = True
            kept_lat.append(lat[i]); kept_lon.append(lon[i])
    return keep_mask

def auto_min_dist_km(bounds, fig_width_in, marker_s, safety=1.3):
    minx, miny, maxx, maxy = bounds
    clat = (miny + maxy) / 2
    _, _, span_m = GEOD.inv(minx, clat, maxx, clat)
    km_per_point = (span_m / 1000 / fig_width_in) / 72.0
    marker_diam_pts = 2 * math.sqrt(marker_s / math.pi)
    return marker_diam_pts * km_per_point * safety

DECLUTTER_N_THRESHOLD = 60

all_grids = pickle.load(open(os.path.join(GRID_DIR, "paper2_region_grids.pkl"), "rb"))

MOSAIC_UNITS = ["fesmeknes", "bmk", "ttah", "draa", "soussmassa", "marrakech", "south_duo", "eddakhla", "rabatcasa"]
# south_trio deliberately excluded -- lost the zone decision above.

REGION_TO_UNIT = {
    "Fés-Meknés": "fesmeknes", "Béni Mellal-Khénifra": "bmk", "Tanger-Tétouan-Al Hoceima": "ttah",
    "Drâa-Tafilalet": "draa", "Souss-Massa": "soussmassa", "Marrakech-Safi": "marrakech",
    "Eddakhla-Oued Eddahab": "eddakhla", "Guelmim-Oued Noun": "south_duo", "Laayoune-Sakia El Hamra": "south_duo",
    "Rabat-Salé-Kénitra": "rabatcasa", "Grand Casablanca-Settat": "rabatcasa",
}

admin12 = gpd.read_file(os.path.join(FW, "data/boundaries/morocco_regions_admin12.geojson"))
oriental = admin12[admin12["nom_region"] == "Oriental"]

# --- labeled sites for the overlay (same convention as every other map) ---
catalog = pd.read_csv(os.path.join(FW, "data/final/geosites_mcdm_national.csv"))
frames = []
for f in sorted(glob.glob(os.path.join(FW, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
sites = all_labels.merge(catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"]], on="Locality_ID", how="inner")
sites = sites.dropna(subset=["Region"]).reset_index(drop=True)
sites["Expert_Merged"] = sites["Expert_Class"].replace("Very Difficult", "Difficult")
sites["unit"] = sites["Region"].map(REGION_TO_UNIT)
sites = sites[sites["unit"].notna()].reset_index(drop=True)

# No per-site misclassification overlay at national scale, by design (user
# decision, 2026-08-23): at 939 points across the whole country, per-site
# correctness rings read as a visual accuracy VERDICT that doesn't match the
# real per-region numbers (dense red clusters make good models look bad at a
# glance), and the exact figures are already reported per region in the
# tables/text. This map's job is where sites are and their predicted zone;
# the per-region maps (which show correctness at a legible scale, thinned to
# match true accuracy) carry the correctness detail.
print(f"National mosaic: {len(sites)} sites overlaid (true class only, no misclassification overlay)")

# Declutter per unit (each unit's own scale/density), then combine -- avoids a
# crowded corridor in one region starving the thinning budget of a sparser
# region elsewhere, and matches this figure's own true render scale (7in wide).
FIG_WIDTH_IN = 7.0
draw_frames = []
for key in MOSAIC_UNITS:
    u = all_grids[key]
    unit_sites = sites[sites["unit"] == key].copy()
    if len(unit_sites) > DECLUTTER_N_THRESHOLD:
        min_dist_km = auto_min_dist_km(u["bounds"], FIG_WIDTH_IN, POINT_S)
        keep = declutter_points(unit_sites["Latitude_WGS84"].values, unit_sites["Longitude_WGS84"].values, min_dist_km)
        unit_sites = unit_sites[keep]
    draw_frames.append(unit_sites)
sites_draw = pd.concat(draw_frames, ignore_index=True) if draw_frames else sites.iloc[0:0]
print(f"  declutter: showing {len(sites_draw)}/{len(sites)} markers total")

# --- render ---
minx, miny, maxx, maxy = admin12.total_bounds
fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, 8.6))
ax.set_facecolor("white")

cmap = ListedColormap([EASY_COL, MODERATE_COL, DIFFICULT_COL])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

unit_images = []  # (im, clip_path) pairs -- clip applied after tight_layout, see below
for key in MOSAIC_UNITS:
    u = all_grids[key]
    ux0, uy0, ux1, uy1 = u["bounds"]
    cls_masked = np.ma.masked_where(u["cls"] < 0, u["cls"])
    cls_hi = np.repeat(np.repeat(cls_masked, 4, axis=0), 4, axis=1)  # nearest-upsample -- see make_paper2_region_maps.py's upsample_nearest for why
    im = ax.imshow(cls_hi, extent=(ux0, ux1, uy0, uy1), origin="lower", cmap=cmap, norm=norm,
                    alpha=0.85, zorder=2, interpolation="nearest")
    unit_images.append((im, boundary_clip_patch(u["gdf"].union_all())))

# Oriental: greyed out, hatched, no data
if len(oriental):
    oriental.plot(ax=ax, facecolor=NODATA_COL, edgecolor=BOUND_COL, linewidth=1.0, hatch="///", zorder=1)
    c = oriental.geometry.iloc[0].representative_point()
    ax.annotate("Oriental\n(no data)", (c.x, c.y), ha="center", va="center", fontsize=7, color="#555555", style="italic")

admin12.boundary.plot(ax=ax, edgecolor=BOUND_COL, linewidth=1.1, zorder=3)

for cls_name in ["Moderate", "Easy", "Difficult"]:
    pts = sites_draw[sites_draw["Expert_Merged"] == cls_name]
    if len(pts) == 0:
        continue
    ax.scatter(pts["Longitude_WGS84"], pts["Latitude_WGS84"], marker=MARKER, s=POINT_S,
               facecolor=POINT_COLORS[cls_name], edgecolor=MARKER_EDGE, linewidth=0.5, zorder=5)

ax.set_xlim(minx - 0.3, maxx + 0.3)
ax.set_ylim(miny - 0.3, maxy + 0.3)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)

class_handles = [Patch(facecolor=EASY_COL, label="Predicted Easy"), Patch(facecolor=MODERATE_COL, label="Predicted Moderate"),
                 Patch(facecolor=DIFFICULT_COL, label="Predicted Difficult"),
                 Patch(facecolor=NODATA_COL, edgecolor=BOUND_COL, hatch="///", label="No data (Oriental)")]
point_handles = [Line2D([0], [0], marker=MARKER, color="none", markerfacecolor=POINT_COLORS[c],
                         markeredgecolor=MARKER_EDGE, markeredgewidth=0.5, markersize=8,
                         label=f"True {c.lower()} site") for c in ["Easy", "Moderate", "Difficult"]]
ax.legend(handles=class_handles + point_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=3, fontsize=7, frameon=True, framealpha=0.92, edgecolor="#cccccc", borderpad=0.6,
          handletextpad=0.5, labelspacing=0.45, columnspacing=1.0)

plt.tight_layout()
# Clip applied AFTER tight_layout -- see make_paper2_region_maps.py for the full
# explanation. tight_layout repositions the axes for the below-anchored legend;
# a clip_path set before that (per unit, in the loop above) clips at the stale
# pre-tight_layout axes position at save time, silently blanking part of a
# unit's true shape -- confirmed by bisecting the render with intermediate
# savefig() checkpoints on the per-region map script that shares this pattern.
for im, clip_path in unit_images:
    im.set_clip_path(PathPatch(clip_path, transform=ax.transData))
out_path = os.path.join(OUT, "map_national_mosaic.pdf")
plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print(f"Saved {out_path}")
