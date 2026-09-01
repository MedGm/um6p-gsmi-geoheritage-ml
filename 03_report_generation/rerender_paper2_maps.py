"""
03_report_generation/rerender_paper2_maps.py  (2026-08-23)

Re-renders the 10 Paper 2 unit maps from the ALREADY-COMPUTED
results/grids/paper2_region_grids.pkl (cls/lon2d/lat2d/elev/gdf per unit) --
no model refitting, no OSM/WorldCover re-extraction. Used specifically to
apply the boundary_clip_patch fix (dropping degenerate interior-ring
"holes" left by union_all() on merged units, which rendered a whole
sub-region -- e.g. Grand Casablanca-Settat -- as a blank gap) without
paying for another full ~35min pipeline run, since that bug is purely a
render-time clip-path issue, not a modeling one.

Output: report/figures/map_region_<key>.pdf for each of the 10 units
(overwritten in place; results/grids/paper2_region_grids.pkl unchanged).
"""
import os, glob, json, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D
from pyproj import Geod

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "report", "figures")
GRID_DIR = os.path.join(FW, "results", "grids")

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}", "font.size": 9.5,
})
EASY_COL, MODERATE_COL, DIFFICULT_COL, BOUND_COL = "#76A5AF", "#E5D3A7", "#C1650A", "#2B5F72"
MARKER, MARKER_EDGE, MISCLASS_RING_COL = "o", "black", "#7A0C0C"
POINT_COLORS = {"Easy": EASY_COL, "Moderate": MODERATE_COL, "Difficult": DIFFICULT_COL}
POINT_S = 34  # true-site dots: bigger, so they read clearly (was 26)
GEOD = Geod(ellps="WGS84")
DECLUTTER_N_THRESHOLD = 60

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
    import math
    minx, miny, maxx, maxy = bounds
    clat = (miny + maxy) / 2
    _, _, span_m = GEOD.inv(minx, clat, maxx, clat)
    km_per_point = (span_m / 1000 / fig_width_in) / 72.0
    marker_diam_pts = 2 * math.sqrt(marker_s / math.pi)
    return marker_diam_pts * km_per_point * safety

# Canvas is figsize=(5.0, 6.0) inches (below), but the LaTeX report includes
# this figure at \includegraphics[width=0.5\linewidth] on a 160mm (6.30in)
# text width -- a PRINTED width of ~3.15in, not 5.0in. Using the canvas size
# understates the true on-page marker spacing by ~37%, leaving points
# visibly overlapping (flagged 2026-09-01; same fix applied in
# make_paper2_region_maps.py, which this script's cached-grid path bypasses).
PRINTED_FIG_WIDTH_IN = 160 / 25.4 * 0.5

def declutter_stratified(sub_draw, bounds, point_s, fig_width_in=PRINTED_FIG_WIDTH_IN):
    """Thin the displayed markers so the shown correct:misclassified ratio
    matches the region's TRUE accuracy, instead of whatever ratio plain
    joint spatial thinning happens to produce. Plain declutter_points keeps
    points in original row order regardless of correctness, so a region at
    e.g. 80% true accuracy could easily end up SHOWING more misclassified
    than correct markers just because the wrong ones happened to be spaced
    further apart (and so survive thinning more often) -- exactly the
    misleading pattern flagged in Fes-Meknes and BMK's maps.

    1. Run the ordinary joint declutter once to get a total marker BUDGET
       (keeps overall map density the same as before).
    2. Split that budget into a correct/wrong target count matching the
       region's true accuracy exactly.
    3. Within each pool (correct, wrong) independently, sort by longitude
       as a simple spatial-spread proxy and take an evenly-spaced
       subsample down to that pool's target count -- keeps a reasonable
       geographic spread without needing a second distance search.
    """
    min_dist_km = auto_min_dist_km(bounds, fig_width_in, point_s)
    budget_mask = declutter_points(sub_draw["Latitude_WGS84"].values, sub_draw["Longitude_WGS84"].values, min_dist_km)
    total_budget = int(budget_mask.sum())

    true_accuracy = float((~sub_draw["_is_wrong"]).mean())
    n_correct_target = round(total_budget * true_accuracy)
    n_wrong_target = total_budget - n_correct_target

    def thin_to(df, target):
        if len(df) <= target:
            return df
        df_sorted = df.sort_values("Longitude_WGS84")
        idx = sorted(set(np.linspace(0, len(df_sorted) - 1, target).round().astype(int)))
        return df_sorted.iloc[idx]

    correct_pool = sub_draw[~sub_draw["_is_wrong"]]
    wrong_pool = sub_draw[sub_draw["_is_wrong"]]
    return pd.concat([thin_to(correct_pool, n_correct_target), thin_to(wrong_pool, n_wrong_target)])

def boundary_clip_patch(gdf):
    """Exterior rings ONLY -- see make_paper2_region_maps.py's version of this
    function for the full explanation (union_all() dissolve-seam sliver holes)."""
    union = gdf.union_all()
    geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
    verts, codes = [], []
    for geom in geoms:
        xy = np.array(geom.exterior.coords)
        verts.extend(xy.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)

def hillshade(elev, dx=1183.4, dy=1183.4, azdeg=315, altdeg=45):
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    e = np.where(np.isfinite(elev), elev, np.nanmin(elev))
    return ls.hillshade(e, vert_exag=1.5, dx=dx, dy=dy)

UPSAMPLE_FACTOR = 4

def upsample_nearest(arr, factor=UPSAMPLE_FACTOR):
    """Nearest-neighbor upsample purely for display: the underlying
    classification grid is coarse (130-220 cells across a whole region), so
    each cell is large relative to the boundary's curvature -- clipping that
    coarse raster to the exact vector boundary still leaves a visibly blocky,
    staircased edge (large square cells cut at odd angles) even though the
    clip itself is geometrically exact. Repeating each cell into a small
    block of identical sub-cells before clipping doesn't add any new
    information (no new predictions), but it lets the SAME exact clip follow
    the boundary far more closely, since each individual cell is now tiny
    relative to the curve -- this is what a human review meant by asking for
    the raster to actually "follow the border as accurately as the raster
    resolution allows"."""
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)

FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]

catalog = pd.read_csv(os.path.join(FW, "data/final/geosites_mcdm_national.csv"))
frames = []
for f in sorted(glob.glob(os.path.join(FW, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn: continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
merged = all_labels.merge(catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
                           on="Locality_ID", how="inner")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
assert len(merged) == 1662

map_oof = json.load(open(os.path.join(FW, "results/json/other/phase5_paper2_map_oof.json")))

UNITS_REGIONS = {
    "fesmeknes": ["Fés-Meknés"], "bmk": ["Béni Mellal-Khénifra"], "ttah": ["Tanger-Tétouan-Al Hoceima"],
    "draa": ["Drâa-Tafilalet"], "soussmassa": ["Souss-Massa"], "marrakech": ["Marrakech-Safi"],
    "eddakhla": ["Eddakhla-Oued Eddahab"],
    "south_duo": ["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra"],
    "south_trio": ["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Eddakhla-Oued Eddahab"],
    "rabatcasa": ["Rabat-Salé-Kénitra", "Grand Casablanca-Settat"],
}

with open(os.path.join(GRID_DIR, "paper2_region_grids.pkl"), "rb") as f:
    all_grids = pickle.load(f)

for key, regions in UNITS_REGIONS.items():
    u = all_grids[key]
    lon2d, lat2d, bounds, cls, elev, gdf = u["lon2d"], u["lat2d"], u["bounds"], u["cls"], u["elev"], u["gdf"]
    minx, miny, maxx, maxy = bounds
    sub = merged[merged["Region"].isin(regions)].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    hs = hillshade(elev)
    hs_hi = upsample_nearest(hs)
    im_hs = ax.imshow(hs_hi, extent=(minx, maxx, miny, maxy), origin="lower", cmap="gray", vmin=0.2, vmax=1.0, zorder=1, alpha=0.55)
    cmap = ListedColormap([EASY_COL, MODERATE_COL, DIFFICULT_COL])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    cls_hi = upsample_nearest(cls)
    im_cls = ax.imshow(cls_hi, extent=(minx, maxx, miny, maxy), origin="lower", cmap=cmap, norm=norm,
                        alpha=0.72, zorder=2, interpolation="nearest")
    gdf.boundary.plot(ax=ax, edgecolor=BOUND_COL, linewidth=1.0, zorder=3)

    oof_combined = map_oof[key]["combined"]
    oof_lookup = dict(zip(oof_combined["locality_ids"], zip(oof_combined["true_int"], oof_combined["pred_int"])))
    true_int = sub["Locality_ID"].map(lambda lid: oof_lookup[lid][0])
    pred_int = sub["Locality_ID"].map(lambda lid: oof_lookup[lid][1])
    is_wrong = (pred_int != true_int)
    n_misclass = int(is_wrong.sum())

    sub_draw = sub.copy()
    sub_draw["_is_wrong"] = is_wrong.values
    if len(sub_draw) > DECLUTTER_N_THRESHOLD:
        sub_draw = declutter_stratified(sub_draw, bounds, POINT_S)

    # Misclassified sites are drawn as a SOLID dot in the misclass color (not their
    # true-class color, not a thin ring on top of it) -- a ring at a size proportionate
    # to the dot was too thin to read at a glance; a solid fill is unambiguous at any size.
    correct_draw = sub_draw[~sub_draw["_is_wrong"]]
    for cls_name in ["Moderate", "Easy", "Difficult"]:
        pts = correct_draw[correct_draw["Expert_Merged"] == cls_name]
        if len(pts) == 0: continue
        ax.scatter(pts["Longitude_WGS84"], pts["Latitude_WGS84"], marker=MARKER, s=POINT_S,
                   facecolor=POINT_COLORS[cls_name], edgecolor=MARKER_EDGE, linewidth=0.6, zorder=5)
    wrong = sub_draw[sub_draw["_is_wrong"]]
    n_misclass_shown = len(wrong)
    if n_misclass_shown:
        ax.scatter(wrong["Longitude_WGS84"], wrong["Latitude_WGS84"], marker=MARKER, s=POINT_S,
                   facecolor=MISCLASS_RING_COL, edgecolor=MARKER_EDGE, linewidth=0.6, zorder=6)

    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_edgecolor("#888888"); spine.set_linewidth(0.6)
    class_handles = [Patch(facecolor=EASY_COL, label="Predicted Easy"), Patch(facecolor=MODERATE_COL, label="Predicted Moderate"),
                     Patch(facecolor=DIFFICULT_COL, label="Predicted Difficult")]
    point_handles = [Line2D([0], [0], marker=MARKER, color="none", markerfacecolor=POINT_COLORS[c],
                             markeredgecolor=MARKER_EDGE, markeredgewidth=0.6, markersize=6.5,
                             label=f"True {c.lower()} site") for c in ["Easy", "Moderate", "Difficult"]]
    if n_misclass_shown > 0:
        point_handles.append(Line2D([0], [0], marker=MARKER, color="none", markerfacecolor=MISCLASS_RING_COL,
                                     markeredgecolor=MARKER_EDGE, markeredgewidth=0.6, markersize=6.5,
                                     label="Misclassified"))
    ax.legend(handles=class_handles + point_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=6.5, frameon=True, framealpha=0.92, edgecolor="#cccccc", borderpad=0.6,
              handletextpad=0.5, labelspacing=0.45, columnspacing=1.0)
    plt.tight_layout()
    # Clip applied AFTER tight_layout -- see make_paper2_region_maps.py for the
    # full explanation (tight_layout repositions the axes for the below-anchored
    # legend; a clip_path set before that, under the PGF backend, silently
    # clips at the stale pre-tight_layout position at save time).
    clip_path = boundary_clip_patch(gdf)
    im_cls.set_clip_path(PathPatch(clip_path, transform=ax.transData))
    # Hillshade clipped to the SAME exact boundary too -- previously left
    # unclipped as deliberate "context outside the region", but the raster's
    # clip edge is antialiased (a soft-blended, not hard-cut, transition), so
    # the unclipped grey hillshade sitting directly behind it showed through
    # as a visible grey fringe/halo hugging the true boundary line. Clipping
    # both layers identically removes that fringe entirely -- nothing grey
    # is left outside the polygon for the antialiased edge to blend against.
    im_hs.set_clip_path(PathPatch(clip_path, transform=ax.transData))
    out_path = os.path.join(OUT, f"map_region_{key}.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved {out_path} ({len(sub)} sites, {n_misclass} misclassified)")

print("Done.")
