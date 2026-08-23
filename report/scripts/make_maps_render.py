"""
Renders the professional map figures from results/grids/map_grids.pkl
(produced by make_maps.py). Separated from the heavy pipeline so styling can
be iterated on without refitting models / re-fetching WorldCover each time.
"""
import os, pickle, math
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.size": 9.5,
})

# Colorblind-safe earth-tone palette (deuteranopia/protanopia confuse red and green
# specifically; a cool teal vs. a warm burnt-orange stays distinguishable under all
# common CVD types, and reads as "geology map" rather than "traffic light").
EASY_COL = "#76A5AF"       # muted teal
MODERATE_COL = "#E5D3A7"   # tan
DIFFICULT_COL = "#C1650A"  # burnt orange
OUTSIDE_COL = "#F2F2F0"
BOUND_COL = "#2B5F72"
GEOD = Geod(ellps="WGS84")

CLASS_CMAP = ListedColormap([EASY_COL, MODERATE_COL, DIFFICULT_COL])
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CLASS_CMAP.N)

# Known-site markers: single shape (filled circle), colored by the site's TRUE class
# using the same palette as the raster -- with declutter distances now actually large
# enough to prevent overlap (see auto_min_dist_km below), class color reads faster than
# shape and doubles as a correctness signal: a circle whose color visibly disagrees with
# the raster patch it sits on IS a misclassification, and is additionally ringed (see
# MISCLASS_RING_COL) so it doesn't rely on the reader noticing the color clash unaided.
# Dark maroon rather than the brighter red used elsewhere -- a saturated red ring
# nearly disappeared against the burnt-orange Difficult class; the darker/desaturated
# variant keeps enough value contrast against all three earth-tone class colors.
MARKER = "o"
POINT_COLORS = {"Easy": EASY_COL, "Moderate": MODERATE_COL, "Difficult": DIFFICULT_COL}
MARKER_EDGE = "black"
MISCLASS_RING_COL = "#7A0C0C"

with open(os.path.join(FW, "results", "grids", "map_grids.pkl"), "rb") as f:
    data = pickle.load(f)
results = data["results"]
merged = data["merged"]


def add_scale_bar(ax, bounds, frac=0.24, y_frac=0.055):
    minx, miny, maxx, maxy = bounds
    clat = (miny + maxy) / 2
    _, _, span_m = GEOD.inv(minx, clat, maxx, clat)
    target_m = span_m * frac
    for candidate in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
        if candidate * 1000 >= target_m:
            bar_km = candidate
            break
    else:
        bar_km = 500
    lon1 = maxx - (maxx - minx) * 0.06
    lat0 = miny + (maxy - miny) * y_frac
    lon0, _, _ = GEOD.fwd(lon1, lat0, 270, bar_km * 1000)
    ax.plot([lon0, lon1], [lat0, lat0], color="#2b2b2b", linewidth=2.2, solid_capstyle="butt",
            transform=ax.transData, zorder=6)
    ax.text((lon0 + lon1) / 2, lat0 + (maxy - miny) * 0.018, f"{bar_km} km",
            ha="center", va="bottom", fontsize=7.3, color="#2b2b2b", zorder=6)


def add_north_arrow(ax, bounds, y0_frac=0.14):
    minx, miny, maxx, maxy = bounds
    x = maxx - (maxx - minx) * 0.06
    y0 = miny + (maxy - miny) * y0_frac
    y1 = y0 + (maxy - miny) * 0.09
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color="#2b2b2b", linewidth=1.4), zorder=6)
    ax.text(x, y1 + (maxy - miny) * 0.012, "N", ha="center", va="bottom",
            fontsize=8.5, color="#2b2b2b", zorder=6)


def declutter_points(lat, lon, min_dist_km):
    """Greedy spatial thinning: keep a point only if it is at least min_dist_km from
    every point already kept. Used purely to make dense marker clusters legible on a
    map -- the underlying accuracy numbers come from separate cross-validation on the
    full labeled set (see results/derived_numbers.json) and are entirely unaffected by
    how many points are drawn here."""
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


def declutter_stratified(pts, is_wrong, bounds, marker_s, fig_width_in):
    """Thin the displayed markers so the shown correct:misclassified ratio
    matches the region's true accuracy -- see report/scripts/
    make_paper2_region_maps.py's version of this function for the full
    rationale (plain joint declutter can show more misclassified than
    correct markers even at high true accuracy, purely by which points
    happen to survive spatial thinning; flagged concretely on this
    project's Fes-Meknes and BMK maps). `is_wrong` must be a boolean Series
    aligned to `pts`, computed on the FULL (pre-declutter) point set."""
    min_dist_km = auto_min_dist_km(bounds, fig_width_in, marker_s)
    budget_mask = declutter_points(pts["Latitude_WGS84"].values, pts["Longitude_WGS84"].values, min_dist_km)
    total_budget = int(budget_mask.sum())

    true_accuracy = float((~is_wrong).mean())
    n_correct_target = round(total_budget * true_accuracy)
    n_wrong_target = total_budget - n_correct_target

    def thin_to(df, target):
        if len(df) <= target:
            return df
        df_sorted = df.sort_values("Longitude_WGS84")
        idx = sorted(set(np.linspace(0, len(df_sorted) - 1, target).round().astype(int)))
        return df_sorted.iloc[idx]

    pts = pts.copy()
    pts["_is_wrong"] = is_wrong.values
    correct_pool = pts[~pts["_is_wrong"]]
    wrong_pool = pts[pts["_is_wrong"]]
    return pd.concat([thin_to(correct_pool, n_correct_target), thin_to(wrong_pool, n_wrong_target)])


def auto_min_dist_km(bounds, fig_width_in, marker_s, safety=1.3):
    """Decluttering distance that actually exceeds the marker's own rendered diameter
    at this figure's true geographic scale (bounds/fig_width_in), plus a safety margin.
    A fixed guessed km value doesn't transfer between figures at very different scales --
    it needs to be recomputed per figure from real marker size and map scale."""
    minx, miny, maxx, maxy = bounds
    clat = (miny + maxy) / 2
    _, _, span_m = GEOD.inv(minx, clat, maxx, clat)
    km_per_point = (span_m / 1000 / fig_width_in) / 72.0
    marker_diam_pts = 2 * math.sqrt(marker_s / math.pi)
    return marker_diam_pts * km_per_point * safety


def boundary_clip_patch(gdf):
    """Exact-boundary clip path for the classification/hillshade rasters: a
    coarse grid's cell-center-in-polygon mask (point_in_polygon_mask, used to
    build `cls`) can visibly bleed color past the true vector boundary at
    concave/notched edges -- a staircase effect against the smooth boundary
    line, most visible on complex coastlines and multi-part regions. Clipping
    the imshow artists themselves to this exact path guarantees no pixel is
    ever drawn outside the true shape, independent of grid resolution. Added
    2026-08-23 after this exact bug was flagged in Paper 1's Figure 3
    (national map) and independently re-derived in Paper 2's per-region maps.

    Exterior rings ONLY -- deliberately ignores `geom.interiors`. Dissolving
    two source polygons whose shared border isn't vertex-identical (e.g. a
    merged multi-region unit) can leave dozens of degenerate near-zero-area
    sliver "holes" along the seam; including those as clip-path holes made
    one whole sub-region render as a blank gap (Paper 2's Rabat/Casablanca
    map). None of this project's admin polygons have a genuine interior hole
    on their own, so dropping interiors here is safe everywhere."""
    union = gdf.union_all()
    geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
    verts, codes = [], []
    for geom in geoms:
        xy = np.array(geom.exterior.coords)
        verts.extend(xy.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)


def nearest_grid_class(lon_pt, lat_pt, lon2d, lat2d, cls):
    """Predicted class (0/1/2, or -1 outside the boundary mask) at the grid cell
    nearest a given point -- used to flag known sites where the map's own displayed
    prediction disagrees with the site's true label."""
    lon_c, lat_c = lon2d[0, :], lat2d[:, 0]
    j = np.argmin(np.abs(lon_c - lon_pt))
    i = np.argmin(np.abs(lat_c - lat_pt))
    return cls[i, j]


CLASS_TO_INT = {"Easy": 0, "Moderate": 1, "Difficult": 2}


# Regions where markers are spatially thinned for legibility before plotting (skipped
# for Eddakhla: only 25 sites total, no overlap problem to begin with).
DECLUTTER_REGIONS = {"fesmeknes", "bmk"}
POINT_S = 38  # was 30 -- bigger, more legible true-site dots


def render_region(key, meta_label, gdf, res, figsize=(5.6, 5.6), show_points=True):
    lon2d, lat2d, bounds, cls, elev = res["lon2d"], res["lat2d"], res["bounds"], res["cls"], res["elev"]
    minx, miny, maxx, maxy = bounds

    ls = LightSource(azdeg=315, altdeg=45)
    e = np.where(np.isfinite(elev), elev, np.nanmin(elev[np.isfinite(elev)]))
    hs = ls.hillshade(e, vert_exag=2.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(OUTSIDE_COL)

    clip_path = boundary_clip_patch(gdf)

    # No cls<0 masking here -- make_maps.py no longer writes a coarse
    # inside/outside mask into cls (that coarse per-cell test left visible
    # gaps of unmasked hillshade near boundaries even with the clip below).
    # The classification/hillshade are computed for the full grid and the
    # clip_path below shapes them to the exact polygon at draw time instead.
    # Nearest-neighbor upsample before clipping: the classification grid is
    # coarse (130-170 cells across a whole region/country), so clipped to the
    # exact vector boundary it still looks blocky/staircased at the edge
    # (large cells cut at odd angles by the clip). Repeating each cell into a
    # small block of identical sub-cells doesn't add information, but lets
    # the same exact clip follow the boundary far more closely.
    UPSAMPLE_FACTOR = 4
    hs_hi = np.repeat(np.repeat(hs, UPSAMPLE_FACTOR, axis=0), UPSAMPLE_FACTOR, axis=1)
    cls_hi = np.repeat(np.repeat(cls, UPSAMPLE_FACTOR, axis=0), UPSAMPLE_FACTOR, axis=1)

    im_hs = ax.imshow(hs_hi, extent=(minx, maxx, miny, maxy), origin="lower",
                       cmap="gray", vmin=0.2, vmax=1.0, zorder=1, alpha=0.55)
    im_hs.set_clip_path(PathPatch(clip_path, transform=ax.transData))

    im_cls = ax.imshow(cls_hi, extent=(minx, maxx, miny, maxy), origin="lower",
                        cmap=CLASS_CMAP, norm=CLASS_NORM, alpha=0.72, zorder=2, interpolation="nearest")
    im_cls.set_clip_path(PathPatch(clip_path, transform=ax.transData))

    gdf.boundary.plot(ax=ax, edgecolor=BOUND_COL, linewidth=1.1, zorder=4)

    n_misclass = 0
    if show_points:
        pts = merged[merged["Region"] == meta_label]
        # Ring truth computed on the FULL, pre-declutter point set -- needed so the
        # true regional accuracy used for stratified thinning below reflects every
        # labeled site, not just whichever ones happen to survive spatial thinning.
        pred_int_full = pts.apply(lambda r: nearest_grid_class(r["Longitude_WGS84"], r["Latitude_WGS84"],
                                                                 lon2d, lat2d, cls), axis=1)
        true_int_full = pts["Expert_Merged"].map(CLASS_TO_INT)
        is_wrong_full = (pred_int_full != true_int_full) & (pred_int_full >= 0)
        n_misclass = int(is_wrong_full.sum())

        if key in DECLUTTER_REGIONS:
            # Stratified: thin so the SHOWN correct:misclassified ratio matches the
            # true regional accuracy, not whatever ratio plain spatial thinning
            # happens to produce (a real complaint: some regions showed visually
            # more misclassified than correct markers despite majority-correct
            # true accuracy). See declutter_stratified's docstring.
            pts = declutter_stratified(pts, is_wrong_full, bounds, POINT_S, figsize[0])
            is_wrong = pts["_is_wrong"]
        else:
            is_wrong = is_wrong_full

        # Misclassified sites drawn as a SOLID dot in the misclass color (not their
        # true-class color, not a ring) -- a ring sized proportionately to the dot
        # was too thin to read at a glance; a solid fill is unambiguous at any size.
        correct_pts = pts[~is_wrong]
        for cls_name in ["Moderate", "Easy", "Difficult"]:
            sub = correct_pts[(correct_pts["Expert_Merged"] == cls_name)]
            if len(sub) == 0:
                continue
            ax.scatter(sub["Longitude_WGS84"], sub["Latitude_WGS84"], marker=MARKER, s=POINT_S,
                       facecolor=POINT_COLORS[cls_name], edgecolor=MARKER_EDGE, linewidth=0.6,
                       zorder=5)
        wrong = pts[is_wrong]
        if len(wrong):
            ax.scatter(wrong["Longitude_WGS84"], wrong["Latitude_WGS84"], marker=MARKER, s=POINT_S,
                       facecolor=MISCLASS_RING_COL, edgecolor=MARKER_EDGE, linewidth=0.6, zorder=6)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.6)

    add_scale_bar(ax, bounds)
    add_north_arrow(ax, bounds)

    class_handles = [Patch(facecolor=EASY_COL, edgecolor="none", alpha=0.85, label="Predicted Easy"),
                     Patch(facecolor=MODERATE_COL, edgecolor="none", alpha=0.85, label="Predicted Moderate"),
                     Patch(facecolor=DIFFICULT_COL, edgecolor="none", alpha=0.85, label="Predicted Difficult")]
    point_handles = [Line2D([0], [0], marker=MARKER, color="none", markerfacecolor=POINT_COLORS[c],
                             markeredgecolor=MARKER_EDGE, markeredgewidth=0.6, markersize=6.5,
                             label=f"True {c.lower()} site") for c in ["Easy", "Moderate", "Difficult"]] if show_points else []
    if show_points and n_misclass > 0:
        point_handles.append(Line2D([0], [0], marker=MARKER, color="none", markerfacecolor=MISCLASS_RING_COL,
                                     markeredgecolor=MARKER_EDGE, markeredgewidth=0.6,
                                     markersize=6.5, label="Misclassified"))
    # Legend placed below the map axes rather than inset in a corner -- an inset legend
    # sits on top of map data no matter which corner it goes in; bbox_inches="tight" at
    # save time expands the saved page to include it instead of cropping it away.
    ncol = 3
    leg = ax.legend(handles=class_handles + point_handles, loc="upper center",
                     bbox_to_anchor=(0.5, -0.02), ncol=ncol, fontsize=6.7,
                     frameon=True, framealpha=0.92, edgecolor="#cccccc", borderpad=0.6,
                     handletextpad=0.5, labelspacing=0.45, columnspacing=1.0)
    leg.get_frame().set_linewidth(0.5)

    fig.savefig(os.path.join(OUT, f"map_{key}.pdf"), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved map_{key}.pdf")


REGION_META = {
    "eddakhla": "Eddakhla-Oued Eddahab",
    "fesmeknes": "Fés-Meknés",
    "bmk": "Béni Mellal-Khénifra",
}
for key, label in REGION_META.items():
    render_region(key, label, gpd.read_file(os.path.join(FW, "data", "boundaries", f"{key}.geojson")),
                  results[key])

def load_national_boundary_dissolved():
    """national.geojson stores Morocco and Western Sahara as two separate Natural
    Earth admin-0 polygons -- plotting them as-is draws the internal boundary
    between them as a visible line across the Sahara, which reads as taking a
    position on a disputed border and is not appropriate for a Moroccan journal.
    Dissolved into one unified polygon here so only the true external national
    outline is ever drawn; used everywhere this file's boundary is rendered."""
    gdf = gpd.read_file(os.path.join(FW, "data", "boundaries", "national.geojson"))
    return gpd.GeoDataFrame(geometry=[gdf.union_all()], crs=gdf.crs)

render_region("national", "__national__", load_national_boundary_dissolved(),
              results["national"], figsize=(5.0, 6.6), show_points=False)

print("All region maps rendered.")

# ============================================================ Figure 1: ground truth
print("Rendering Figure 1 (national ground-truth map) ...", flush=True)
nat = results["national"]
lon2d, lat2d, bounds, elev = nat["lon2d"], nat["lat2d"], nat["bounds"], nat["elev"]
minx, miny, maxx, maxy = bounds

# Flat vector fill instead of a coarse hillshade raster -- this map exists to show WHERE
# geosites are, not terrain relief, so a clean flat country silhouette (zero pixelation,
# any resolution) reads more clearly than a low-resolution shaded-relief background.
fig, ax = plt.subplots(figsize=(5.0, 6.6))
ax.set_facecolor(OUTSIDE_COL)
national_gdf = load_national_boundary_dissolved()
national_gdf.plot(ax=ax, facecolor="#FBF6E9", edgecolor=BOUND_COL, linewidth=1.1, zorder=1)

# Main map: markers spatially thinned so the dense Middle Atlas / Fes-Meknes corridor
# reads as a legible sample of points rather than an overlapping mass. This is a display
# simplification only -- all 733 sites are used throughout the actual modeling/evaluation
# (Table 2, Section 4). Declutter distance is computed from the marker's own rendered
# size at this figure's true geographic scale (not guessed), and applied across all
# classes jointly -- otherwise two differently-colored points from separate class-loops
# never get checked against each other and can still land on top of one another.
FIG1_POINT_S = 24
main_min_dist = auto_min_dist_km(bounds, 5.0, FIG1_POINT_S)
main_keep = declutter_points(merged["Latitude_WGS84"].values, merged["Longitude_WGS84"].values, main_min_dist)
merged_thinned = merged[main_keep]
for cls_name in ["Moderate", "Easy", "Difficult"]:
    sub = merged_thinned[merged_thinned["Expert_Merged"] == cls_name]
    ax.scatter(sub["Longitude_WGS84"], sub["Latitude_WGS84"], marker=MARKER, s=FIG1_POINT_S,
               facecolor=POINT_COLORS[cls_name], edgecolor=MARKER_EDGE, linewidth=0.4,
               alpha=0.95, zorder=5)

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.6)
# Scale bar / north arrow moved up (well above their usual bottom-right spot) to leave
# that whole corner free for the inset below, which needs to sit entirely off the
# landmass so it doesn't hide any of the map it's magnifying.
add_scale_bar(ax, bounds, y_frac=0.40)
add_north_arrow(ax, bounds, y0_frac=0.46)
point_handles = [Line2D([0], [0], marker=MARKER, color="none", markerfacecolor=POINT_COLORS[c],
                         markeredgecolor=MARKER_EDGE, markeredgewidth=0.5, markersize=6.5,
                         label=c) for c in ["Easy", "Moderate", "Difficult"]]
leg = ax.legend(handles=point_handles, loc="upper left", fontsize=7.2, frameon=True,
                 framealpha=0.92, edgecolor="#cccccc", title="Accessibility class", title_fontsize=7.6)
leg.get_frame().set_linewidth(0.5)

# ---- Inset: magnified view of the single densest sub-cluster inside Fes-Meknes (the
# region with by far the most labeled sites -- see Table 2). The full region is ~140km
# across and still overlaps badly even zoomed in; instead this finds the actual hotspot
# via a 2D density grid and shows a ~26km-wide window around it, with points themselves
# thinned to a legible sample (min 1.8km apart) rather than left to overlap. Placed
# entirely in the empty ocean/desert corner of the axes, off the landmass, so it doesn't
# cover any of the country it is magnifying. Coordinates are exact; nothing is jittered.
from scipy.stats import binned_statistic_2d
fm = merged[merged["Region"] == "Fés-Meknés"]
fm_lat, fm_lon = fm["Latitude_WGS84"].values, fm["Longitude_WGS84"].values
H, xedges, yedges, _ = binned_statistic_2d(fm_lon, fm_lat, None, statistic="count", bins=25)
iy, ix = np.unravel_index(np.nanargmax(H.T), H.T.shape)
hot_lon, hot_lat = (xedges[ix] + xedges[ix + 1]) / 2, (yedges[iy] + yedges[iy + 1]) / 2
half_deg_lon, half_deg_lat = 0.13, 0.11  # roughly +/-13km lon, +/-12km lat at this latitude
fm_minx, fm_maxx = hot_lon - half_deg_lon, hot_lon + half_deg_lon
fm_miny, fm_maxy = hot_lat - half_deg_lat, hot_lat + half_deg_lat

axins_width_in = 0.42 * 4.4  # inset axes-fraction width times the approx. usable axes width
axins = ax.inset_axes([0.56, 0.015, 0.42, 0.335])
axins.set_facecolor(OUTSIDE_COL)
national_gdf.plot(ax=axins, facecolor="#FBF6E9", edgecolor=BOUND_COL, linewidth=0.8, zorder=1)
# Restrict to the visible window first, then declutter across classes jointly (same
# reasoning as the main map above), using the inset's own (much larger) scale.
INSET_POINT_S = 40
fm_window = fm[(fm_lon >= fm_minx) & (fm_lon <= fm_maxx) & (fm_lat >= fm_miny) & (fm_lat <= fm_maxy)]
inset_min_dist = auto_min_dist_km((fm_minx, fm_miny, fm_maxx, fm_maxy), axins_width_in, INSET_POINT_S)
inset_keep = declutter_points(fm_window["Latitude_WGS84"].values, fm_window["Longitude_WGS84"].values, inset_min_dist)
fm_window_thinned = fm_window[inset_keep]
for cls_name in ["Moderate", "Easy", "Difficult"]:
    sub = fm_window_thinned[fm_window_thinned["Expert_Merged"] == cls_name]
    axins.scatter(sub["Longitude_WGS84"], sub["Latitude_WGS84"], marker=MARKER, s=INSET_POINT_S,
                  facecolor=POINT_COLORS[cls_name], edgecolor=MARKER_EDGE, linewidth=0.6,
                  alpha=0.95, zorder=5)
axins.set_xlim(fm_minx, fm_maxx)
axins.set_ylim(fm_miny, fm_maxy)
axins.set_xticks([])
axins.set_yticks([])
for spine in axins.spines.values():
    spine.set_edgecolor("#555555")
    spine.set_linewidth(0.9)
axins.text(0.02, 0.97, "Densest cluster (Fés-Meknés)", transform=axins.transAxes, ha="left", va="top",
           fontsize=6.3, color="#333333",
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
ax.indicate_inset_zoom(axins, edgecolor="#555555", linewidth=0.8, alpha=0.9)

fig.savefig(os.path.join(OUT, "map_groundtruth_national.pdf"), dpi=200, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("Saved map_groundtruth_national.pdf")

# ============================================================ Favorability map =====
print("Rendering favorability map ...", flush=True)
BASE = FW
fav = pd.read_csv(os.path.join(BASE, "data/model_outputs/geosite_favorability_grid_v3.csv"))
piv = fav.pivot_table(index="y", columns="x", values="favorability")
piv = piv.sort_index(ascending=True)
Xg = piv.columns.values
Yg = piv.index.values
Zg = piv.values

# reproject national boundary to the grid's CRS (EPSG:26191, matches code/14)
nat_26191 = national_gdf.to_crs("EPSG:26191")

fig, ax = plt.subplots(figsize=(5.0, 6.6))
ax.set_facecolor("white")
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
fav_cmap = LinearSegmentedColormap.from_list(
    "fav", ["#3D6E77", "#76A5AF", "#E5D3A7", "#FD8D3C", "#E31A1C", "#800026"])
im = ax.imshow(Zg, extent=(Xg.min(), Xg.max(), Yg.min(), Yg.max()), origin="lower",
                cmap=fav_cmap, norm=PowerNorm(gamma=0.55, vmin=0, vmax=1), zorder=1, aspect="auto",
                interpolation="bilinear", resample=True)
nat_26191.boundary.plot(ax=ax, edgecolor=BOUND_COL, linewidth=1.0, zorder=3)
# Known-geosite locations are deliberately NOT overlaid here: at national scale their
# dense clusters (Middle Atlas / Fes-Meknes corridor) sit directly on top of the
# highest-favorability patches and hide the one thing this figure exists to show --
# the continuous model surface. Figure 1 already shows where the real geosites are.

ax.set_xlim(Xg.min(), Xg.max())
ax.set_ylim(Yg.min(), Yg.max())
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_edgecolor("#888888")
    spine.set_linewidth(0.6)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.75)
cbar.set_label("Predicted favorability (probability)", fontsize=7.8)
cbar.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(OUT, "map_favorability.pdf"), dpi=200, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("Saved map_favorability.pdf")
