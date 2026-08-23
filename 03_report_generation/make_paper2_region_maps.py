"""
03_report_generation/make_paper2_region_maps.py  (2026-08-23, revised same day
after user review flagged the ring-marker/resolution/boundary bugs below)

Renders Paper 2's per-region and per-merged-group accessibility maps: for
each of the 10 units (7 individual regions + 3 merged groups), fits that
unit's OWN winning feature-set model per target (from
phase5_paper2_best_feature_results.json / phase5_paper2_merged_regions_results.json
/ phase5_paper2_rabat_standalone_results.json) refit once on that unit's
full labeled data, projects across a real geographic grid built from local
terrain rasters + WorldCover friction + settlement distance (Baseline
features, reused from make_maps.py) and, where the winning variant is
+Infra, real grid-level tourism-POI/settlement features (region_infra_grid.py,
extracted once per unit's territory). Where the winning variant is +Domain,
substitutes Baseline for the MAP specifically (geological domain has no
continuous spatial layer to sample -- disclosed, not silent) while the
accuracy numbers reported elsewhere still reflect the true Domain-based
result.

Revision fixes three real bugs a human review caught in the first version:
  1. Misclassification rings were determined by looking up the NEAREST GRID
     CELL's class in a coarse (130x130 over the whole region bbox) in-sample
     raster -- far coarser than the 500m LOGO-cluster radius used everywhere
     else, so nearby sites of different true classes routinely collapsed
     onto one cell. Rings now come from 02_modeling_and_analysis/29_paper2_map_oof.py's
     actual per-site LOGO-cluster CV out-of-fold predictions (the exact same
     methodology and numbers as the reported accuracies), independent of
     raster resolution entirely.
  2. The classification raster could bleed past the true polygon boundary at
     a coarse grid resolution (cell-center-in-polygon test staircases against
     the smooth vector boundary line). Fixed with an explicit clip path on
     the imshow artist, so no colored pixel is ever drawn outside the true
     region shape regardless of grid coarseness; grid resolution was also
     raised (130->220) for a visibly crisper raster.
  3. The +Infra feature composition used for the map's training refit and
     grid projection was missing the nearest-settlement-type one-hot columns
     that are actually part of the winning +Infra feature set (only the 3
     numeric infra columns were included) -- fixed to match the true +Infra
     composition exactly for the 3 units whose map genuinely renders +Infra
     (fesmeknes Difficult, bmk Easy, eddakhla Difficult; substituted units
     were unaffected since they render Baseline instead).
  4. Dense regions (Fes-Meknes N=337, BMK N=174, TTAH N=128) plotted every
     known-site marker unconditionally, so overlapping points piled into a
     solid mass that reads as "almost everything is wrong" regardless of true
     accuracy. Paper 1's OWN established script (make_maps_render.py) already
     solved this for its own dense regions with declutter_points/
     auto_min_dist_km (greedy spatial thinning to the marker's own rendered
     size at true map scale) -- ported here rather than re-invented, since it
     was missed when this script was written from scratch.

Output: report/figures/map_region_<key>.pdf for each of the 10 units, plus
        results/grids/paper2_region_grids.pkl (cached grids + predictions,
        reused by the national mosaic assembly script)
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "s3.eu-central-1.amazonaws.com"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

import glob, pickle, subprocess, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter
from pyproj import Transformer, Geod

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "..", "figures")
GRID_DIR = os.path.join(FW, "results", "grids")
os.makedirs(OUT, exist_ok=True)
os.makedirs(GRID_DIR, exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "font.family": "serif", "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}", "font.size": 9.5,
})
EASY_COL, MODERATE_COL, DIFFICULT_COL, OUTSIDE_COL, BOUND_COL = "#76A5AF", "#E5D3A7", "#C1650A", "#F2F2F0", "#2B5F72"
MARKER, MARKER_EDGE, MISCLASS_RING_COL = "o", "black", "#7A0C0C"
POINT_COLORS = {"Easy": EASY_COL, "Moderate": MODERATE_COL, "Difficult": DIFFICULT_COL}
CLASS_TO_INT = {"Easy": 0, "Moderate": 1, "Difficult": 2}
POINT_S = 34  # true-site dots: bigger, so they read clearly (was 26)
GEOD = Geod(ellps="WGS84")
# Above this many labeled sites, markers are spatially thinned for legibility
# before plotting -- see declutter_points/auto_min_dist_km below, ported from
# make_maps_render.py where the same problem was already solved for Paper 1's
# own dense regions (Fes-Meknes, BMK).
DECLUTTER_N_THRESHOLD = 60
UPSAMPLE_FACTOR = 4

def upsample_nearest(arr, factor=UPSAMPLE_FACTOR):
    """Nearest-neighbor upsample purely for display -- see rerender_paper2_maps.py's
    version of this function for the full explanation. A coarse classification
    grid clipped to the exact vector boundary still looks blocky/staircased at
    the edge (large cells cut at odd angles); repeating each cell into a small
    block of identical sub-cells before clipping lets the same exact clip
    follow the boundary far more closely."""
    return np.repeat(np.repeat(arr, factor, axis=0), factor, axis=1)

def declutter_points(lat, lon, min_dist_km):
    """Greedy spatial thinning: keep a point only if it is at least
    min_dist_km from every point already kept. Display-only -- the underlying
    accuracy numbers and misclassification ring truth come from 02_modeling_and_analysis/29's
    full-sample OOF predictions and are entirely unaffected by how many points
    are drawn here."""
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
    """Decluttering distance that actually exceeds the marker's own rendered
    diameter at this figure's true geographic scale, plus a safety margin."""
    import math
    minx, miny, maxx, maxy = bounds
    clat = (miny + maxy) / 2
    _, _, span_m = GEOD.inv(minx, clat, maxx, clat)
    km_per_point = (span_m / 1000 / fig_width_in) / 72.0
    marker_diam_pts = 2 * math.sqrt(marker_s / math.pi)
    return marker_diam_pts * km_per_point * safety

def declutter_stratified(sub_draw, bounds, point_s, fig_width_in=5.0):
    """Thin the displayed markers so the shown correct:misclassified ratio
    matches the region's TRUE accuracy, instead of whatever ratio plain
    joint spatial thinning happens to produce -- see rerender_paper2_maps.py's
    version of this function for the full explanation (a region at e.g. 80%
    true accuracy could show more misclassified than correct markers just
    because the wrong ones happened to survive ordinary thinning more often;
    flagged concretely in Fes-Meknes and BMK's maps)."""
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

FEATURES_BASE = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness", "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
WC_FRICTION = {10: 0.55, 20: 0.35, 30: 0.15, 40: 0.20, 50: 0.05, 60: 0.10, 70: 0.70, 80: 0.90, 90: 0.75, 95: 0.75, 100: 0.60}
PBF_PATH = os.path.join(FW, "data/osm/morocco-latest.osm.pbf")

# ============================================================ 1. Labeled data =====
catalog = pd.read_csv(os.path.join(FW, "data/final/geosites_mcdm_national.csv"))
infra_geosite = pd.read_csv(os.path.join(FW, "data/final/infra_features.csv"))
frames = []
for f in sorted(glob.glob(os.path.join(FW, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn: continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
merged = all_labels.merge(catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES_BASE],
                           on="Locality_ID", how="inner").merge(infra_geosite, on="Locality_ID", how="left")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
SENTINEL = 60000.0
merged["dist_nearest_tourism_poi_m"] = merged["dist_nearest_tourism_poi_m"].fillna(SENTINEL)
merged["dist_nearest_settlement_town_m"] = merged["dist_nearest_settlement_town_m"].fillna(SENTINEL)
merged["nearest_settlement_type"] = merged["nearest_settlement_type"].fillna("None")
INFRA_NUMERIC = ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"]
CODE_TO_SETTLE_CAT = {0: "None", 1: "hamlet", 2: "village", 3: "town", 4: "city"}
assert len(merged) == 939
log(f"N={len(merged)}")

# Ring-marker ground truth: exact per-site LOGO-cluster CV out-of-fold
# predictions from the true winning feature set (02_modeling_and_analysis/29), NOT derived
# from this script's own (coarser, in-sample) raster -- see module docstring.
map_oof = json.load(open(os.path.join(FW, "results/json/other/phase5_paper2_map_oof.json")))

# ============================================================ 2. Model results ====
best_feature = {(r["region"], r["target"]): r for r in json.load(open(os.path.join(FW, "results/json/training/phase5_paper2_best_feature_results.json")))}
merged_groups_res = {(r["group"], r["target"]): r for r in json.load(open(os.path.join(FW, "results/json/training/phase5_paper2_merged_regions_results.json")))}
rabat_standalone = {(r["region"], r["target"]): r for r in json.load(open(os.path.join(FW, "results/json/training/phase5_paper2_rabat_standalone_results.json")))}

UNITS = {
    "fesmeknes":  dict(label="Fés-Meknés", regions=["Fés-Meknés"], source="individual"),
    "bmk":        dict(label="Béni Mellal-Khénifra", regions=["Béni Mellal-Khénifra"], source="individual"),
    "ttah":       dict(label="Tanger-Tétouan-Al Hoceima", regions=["Tanger-Tétouan-Al Hoceima"], source="individual"),
    "draa":       dict(label="Drâa-Tafilalet", regions=["Drâa-Tafilalet"], source="individual"),
    "soussmassa": dict(label="Souss-Massa", regions=["Souss-Massa"], source="individual"),
    "marrakech":  dict(label="Marrakech-Safi", regions=["Marrakech-Safi"], source="individual"),
    "eddakhla":   dict(label="Eddakhla-Oued Eddahab", regions=["Eddakhla-Oued Eddahab"], source="individual"),
    "south_duo":  dict(label="Guelmim-Oued Noun + Laâyoune-Sakia El Hamra", regions=["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra"], source="merged", group_key="South_GuelmimLaayoune"),
    "south_trio": dict(label="Guelmim-Oued Noun + Laâyoune-Sakia El Hamra + Eddakhla-Oued Eddahab", regions=["Guelmim-Oued Noun", "Laayoune-Sakia El Hamra", "Eddakhla-Oued Eddahab"], source="merged", group_key="South_GuelmimLaayouneEddakhla"),
    "rabatcasa":  dict(label="Rabat-Salé-Kénitra + Casablanca-Settat", regions=["Rabat-Salé-Kénitra", "Grand Casablanca-Settat"], source="merged", group_key="RabatCasablanca"),
}

def get_winner_and_configs(unit_key, meta, target):
    if meta["source"] == "individual":
        entry = best_feature[(meta["regions"][0], target)]
    else:
        entry = merged_groups_res[(meta["group_key"], target)]
    winner = entry["best_variant"]
    cfgs = entry["variants"][winner]["best_configs"]
    return winner, cfgs

def make_model(kind, cfg):
    if kind == "RF": return RandomForestClassifier(random_state=42, n_jobs=-1, **cfg)
    if kind == "XGB": return XGBClassifier(random_state=42, eval_metric="logloss", **cfg)
    if kind == "GBM": return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM": return LGBMClassifier(random_state=42, verbosity=-1, **cfg)

def fit_final(cfgs, X, y):
    sw = compute_sample_weight("balanced", y)
    fitted = []
    for kind, cfg in cfgs:
        m = make_model(kind, cfg)
        if kind == "XGB":
            pos_w = sw[y==1].sum(); neg_w = sw[y==0].sum()
            if pos_w > 0: m.set_params(scale_pos_weight=neg_w/pos_w)
        m.fit(X, y, sample_weight=sw)
        fitted.append(m)
    return fitted

def predict_proba_ensemble(fitted, Xg):
    return np.mean([m.predict_proba(Xg)[:, 1] for m in fitted], axis=0)

# ============================================================ 3. Rasters ==========
# Elevation/Slope/Ruggedness come from archive/gis_data/physical_task2_corrected/,
# NOT archive/gis_data/physical/ -- confirmed by a 2026-08-23 human review that the
# map's terrain content didn't look geographically coherent. code/02_extract_
# terrain_road_features.py (which built the CATALOG's own Elevation_m/Slope_deg/
# Ruggedness -- the features the models were actually TRAINED and EVALUATED on)
# reads from physical_task2_corrected/, documented there as "registration-corrected
# via coastline calibration, already reviewed". Every map-rendering script had been
# reading the OLD, uncorrected physical/ directory instead -- verified concretely:
# sampling physical/elevation_meters.tif at real catalog site coordinates missed
# entirely (raw -9999 nodata) 33% of the time and was off by 500-1100m even where
# valid; physical_task2_corrected/ + nodata-fill (below) brings that to a ~130-200m
# median bias, consistent with ordinary coarse-DEM-vs-ground-truth variation at
# 1.2km resolution, not a registration bug.
#
# Dist_to_Highway_m has no corrected raster (physical_task2_corrected/ only has a
# "_PRE_CALIBRATION" highway file) because the catalog computes it a different way
# entirely -- true point-to-nearest-road-line vector distance via geopandas
# sjoin_nearest, not raster sampling at all. physical/distance_to_highways_meters.tif
# checked out fine against the catalog (differences consistent with ordinary raster
# quantization at pixel-width scale, not a registration bug), so it's kept as the
# map's Dist_to_Highway_m source -- the closest cheap proxy available for a
# continuous grid without redoing true vector distance at every one of ~50k cells.
PHYS = os.path.join(FW, "archive/gis_data/physical_task2_corrected")
PHYS_OLD = os.path.join(FW, "archive/gis_data/physical")
raster_arrays = {}
raster_transforms = {}  # PER-KEY -- the corrected rasters use a different affine
                         # origin than the old ones (confirmed: ~24.85km Y-shift,
                         # exactly the "coastline calibration" the corrected
                         # directory's name promises), so elevation/slope/
                         # ruggedness and dist-to-highway are NOT interchangeable
                         # under one shared transform -- each raster must be
                         # sampled with its own.
raster_crs = None
for key, fname, phys_dir in [("Elevation_m", "elevation_meters.tif", PHYS),
                              ("Slope_deg", "slope_degrees.tif", PHYS),
                              ("Ruggedness", "ruggedness.tif", PHYS),
                              ("Dist_to_Highway_m", "distance_to_highways_meters.tif", PHYS_OLD)]:
    with rasterio.open(os.path.join(phys_dir, fname)) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is not None and (arr == nodata).any():
            from scipy.ndimage import distance_transform_edt as _edt
            invalid = (arr == nodata)
            idx = _edt(invalid, return_distances=False, return_indices=True)
            arr = arr[tuple(idx)]
        raster_arrays[key] = arr
        raster_transforms[key] = src.transform
        raster_crs = src.crs  # same CRS (EPSG:26191) for both directories, safe to share
to_26191 = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
inv_transforms = {k: ~t for k, t in raster_transforms.items()}

def sample_local_stack(lon, lat):
    x, y = to_26191.transform(lon, lat)
    out = {}
    for k in ["Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Highway_m"]:
        col, row = inv_transforms[k] * (x, y)
        row_i = np.clip(np.round(row).astype(int), 0, raster_arrays[k].shape[0] - 1)
        col_i = np.clip(np.round(col).astype(int), 0, raster_arrays[k].shape[1] - 1)
        out[k] = raster_arrays[k][row_i, col_i]
    return out

cities = pd.read_csv(os.path.join(FW, "data/archive/pipeline_intermediates/morocco_reference_cities_geocoded.csv"))
clat, clon = cities["Latitude"].values, cities["Longitude"].values

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat/2)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def dist_to_settlement(lon, lat):
    d = np.stack([haversine_m(lat, lon, la, lo) for la, lo in zip(clat, clon)], axis=-1)
    return d.min(axis=-1)

def tile_key(lat, lon): return int(np.floor(lat/3)*3), int(np.floor(lon/3)*3)
def tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    return f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map.tif"

def worldcover_friction_grid(lon_min, lat_min, lon_max, lat_max, ny, nx):
    lat_tiles = sorted(set(int(np.floor(v/3)*3) for v in [lat_min, lat_max]))
    lon_tiles = sorted(set(int(np.floor(v/3)*3) for v in [lon_min, lon_max]))
    full = np.full((ny, nx), 60, dtype=np.uint8)
    lon_edges = np.linspace(lon_min, lon_max, nx+1)
    lat_edges = np.linspace(lat_max, lat_min, ny+1)
    for lt in lat_tiles:
        for ln in lon_tiles:
            try:
                with rasterio.open(tile_url(lt, ln)) as src:
                    tb = src.bounds
                    ov_lon_min, ov_lon_max = max(lon_min, tb.left), min(lon_max, tb.right)
                    ov_lat_min, ov_lat_max = max(lat_min, tb.bottom), min(lat_max, tb.top)
                    if ov_lon_min >= ov_lon_max or ov_lat_min >= ov_lat_max: continue
                    win = from_bounds(ov_lon_min, ov_lat_min, ov_lon_max, ov_lat_max, src.transform)
                    i0 = max(0, np.searchsorted(-lon_edges, -ov_lon_min) - 1); i1 = min(nx, np.searchsorted(-lon_edges, -ov_lon_max))
                    j0 = max(0, np.searchsorted(-lat_edges, -ov_lat_max) - 1); j1 = min(ny, np.searchsorted(-lat_edges, -ov_lat_min))
                    sub_h, sub_w = max(1, j1-j0), max(1, i1-i0)
                    arr = src.read(1, window=win, out_shape=(sub_h, sub_w), resampling=Resampling.mode)
                    full[j0:j0+sub_h, i0:i0+sub_w] = arr
            except Exception as e:
                log(f"    WorldCover tile ({lt},{ln}) failed: {e}")
    return np.vectorize(lambda c: WC_FRICTION.get(int(c), 0.3))(full).astype(float)

def hillshade(elev, dx=1183.4, dy=1183.4, azdeg=315, altdeg=45):
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    e = np.where(np.isfinite(elev), elev, np.nanmin(elev))
    return ls.hillshade(e, vert_exag=1.5, dx=dx, dy=dy)

def build_region_grid(gdf, n_grid):
    minx, miny, maxx, maxy = gdf.total_bounds
    aspect = (maxx-minx) / max(1e-6, (maxy-miny))
    nx, ny = (n_grid, max(20, int(n_grid/aspect))) if aspect >= 1 else (max(20, int(n_grid*aspect)), n_grid)
    lon_edges = np.linspace(minx, maxx, nx+1); lat_edges = np.linspace(miny, maxy, ny+1)
    lon_c = (lon_edges[:-1]+lon_edges[1:])/2; lat_c = (lat_edges[:-1]+lat_edges[1:])/2
    lon2d, lat2d = np.meshgrid(lon_c, lat_c)
    return lon2d, lat2d, (minx, miny, maxx, maxy), nx, ny

def nearest_grid_class(lon_pt, lat_pt, lon2d, lat2d, cls):
    lon_c, lat_c = lon2d[0, :], lat2d[:, 0]
    j = np.argmin(np.abs(lon_c - lon_pt))
    i = np.argmin(np.abs(lat_c - lat_pt))
    return cls[i, j]

def point_in_polygon_mask(lon2d, lat2d, gdf):
    union = gdf.union_all()
    geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
    mask = np.zeros(lon2d.shape, dtype=bool)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    for geom in geoms:
        xx, yy = geom.exterior.xy
        mask |= MplPath(np.column_stack([xx, yy])).contains_points(pts).reshape(lon2d.shape)
    return mask

def boundary_clip_patch(gdf):
    """Exact-boundary clip path for the classification raster: a coarse grid's
    cell-center-in-polygon mask can visibly bleed color past the true vector
    boundary at concave/notched edges (staircase effect against the smooth
    line) -- clipping the imshow artist itself to this path guarantees no
    pixel is ever drawn outside the true region shape, independent of grid
    resolution.

    Exterior rings ONLY -- deliberately ignores `geom.interiors`. For a
    merged unit (e.g. Rabat-Salé-Kénitra + Grand Casablanca-Settat),
    `gdf.union_all()` on two adjacent-but-not-vertex-identical polygons
    produces dozens of degenerate near-zero-area sliver "holes" along the
    shared seam (float-precision mismatch between the two source polygons'
    boundaries, not real geography) -- verified none of the 12 admin regions
    in this project's boundary file has a genuine interior hole on its own.
    Including those slivers as clip-path holes rendered one whole merged
    sub-region as a big blank/unclipped gap (matplotlib's path fill rule
    tripping over ~110 degenerate interior rings), which is exactly the bug
    a human review caught by zooming into the Rabat/Casablanca map."""
    union = gdf.union_all()
    geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
    verts, codes = [], []
    for geom in geoms:
        xy = np.array(geom.exterior.coords)
        verts.extend(xy.tolist())
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(xy) - 2) + [MplPath.CLOSEPOLY])
    return MplPath(verts, codes)

# ============================================================ 4. Region boundaries
admin12 = gpd.read_file(os.path.join(FW, "data/boundaries/morocco_regions_admin12.geojson"))

# ============================================================ 5. Per-unit pipeline
_grid_pkl_path = os.path.join(GRID_DIR, "paper2_region_grids.pkl")
if os.path.exists(_grid_pkl_path):
    with open(_grid_pkl_path, "rb") as f:
        all_grids = pickle.load(f)
    log(f"Resuming: loaded {len(all_grids)} previously-completed units from {_grid_pkl_path}")
else:
    all_grids = {}
os.makedirs("/tmp/paper2_infra_grid", exist_ok=True)
SENTINEL = 60000.0

for key, meta in UNITS.items():
    if key in all_grids:
        log(f"\n=== {meta['label']} ({key}) -- already in grid cache, skipping ===")
        continue
    log(f"\n=== {meta['label']} ({key}) ===")
    try:
        gdf = admin12[admin12["nom_region"].isin(meta["regions"])]
        sub = merged[merged["Region"].isin(meta["regions"])].reset_index(drop=True)
        log(f"  N={len(sub)}, building grid ...")
        lon2d, lat2d, bounds, nx, ny = build_region_grid(gdf, 220)
        minx, miny, maxx, maxy = bounds
        local = sample_local_stack(lon2d, lat2d)
        friction = worldcover_friction_grid(minx, miny, maxx, maxy, ny, nx)
        dsettle = dist_to_settlement(lon2d, lat2d)

        winners = {}
        for target in ["Difficult", "Easy"]:
            w, cfgs = get_winner_and_configs(key, meta, target)
            winners[target] = (w, cfgs)
        need_infra = any(w == "Infra" for w, _ in winners.values())

        infra_grid = None
        infra_ok = False
        if need_infra:
            log("  extracting infra grid ...")
            np.save("/tmp/paper2_infra_grid/lon2d.npy", lon2d)
            np.save("/tmp/paper2_infra_grid/lat2d.npy", lat2d)
            out_npz = f"/tmp/paper2_infra_grid/{key}.npz"
            r = subprocess.run([sys.executable, os.path.join(HERE, "region_infra_grid.py"),
                                 str(minx-0.1), str(miny-0.1), str(maxx+0.1), str(maxy+0.1),
                                 "/tmp/paper2_infra_grid/lon2d.npy", "/tmp/paper2_infra_grid/lat2d.npy",
                                 out_npz, PBF_PATH], capture_output=True, text=True, timeout=900)
            log(f"    worker: {r.stdout.strip()} {r.stderr.strip()[-200:] if r.returncode else ''}")
            if r.returncode == 0 and os.path.exists(out_npz):
                infra_grid = np.load(out_npz)
                # extraction can "succeed" but still be all-NaN (e.g. MemoryError caught
                # internally, or genuinely nothing found) -- validate before trusting it,
                # same discipline as 02_modeling_and_analysis/17's NaN crash earlier this session.
                dist_col = infra_grid["dist_nearest_tourism_poi_m"]
                infra_ok = np.isfinite(dist_col).sum() > 0.5 * dist_col.size
            if not infra_ok:
                log("  INFRA GRID INVALID/FAILED -- falling back to Baseline for any Infra-winning target here")

        # Settlement-type one-hot for this unit's own sample -- part of the TRUE
        # +Infra feature composition (02_modeling_and_analysis/24-28), previously missing from
        # this script's Infra map rendering entirely.
        settle_dummies = pd.get_dummies(sub["nearest_settlement_type"], prefix="Settlement").astype(float)
        sub_ext = pd.concat([sub, settle_dummies], axis=1)
        INFRA_COLS = INFRA_NUMERIC + list(settle_dummies.columns)

        preds = {}
        for target in ["Difficult", "Easy"]:
            winner, cfgs = winners[target]
            map_variant = "Baseline" if winner == "Domain" else winner  # Domain has no continuous layer -- substitute, disclosed
            if map_variant == "Infra" and not infra_ok:
                map_variant = "Baseline"  # graceful degrade, disclosed in the saved grid dict below
            y = (sub["Expert_Merged"] == target).astype(int).values
            X_train = sub[FEATURES_BASE].values if map_variant == "Baseline" else sub_ext[FEATURES_BASE + INFRA_COLS].values
            if map_variant != winner:
                # Winner used a feature set the map can't render (Domain, or Infra that
                # failed extraction) -- use Baseline's OWN best_configs, not the winner's
                # (different feature count/shape).
                src = best_feature[(meta["regions"][0], target)] if meta["source"] == "individual" else merged_groups_res[(meta["group_key"], target)]
                cfgs_use = src["variants"]["Baseline"]["best_configs"]
            else:
                cfgs_use = cfgs
            fitted = fit_final(cfgs_use, X_train, y)

            if map_variant == "Baseline":
                Xg = np.column_stack([local["Dist_to_Highway_m"].ravel(), local["Slope_deg"].ravel(), local["Ruggedness"].ravel(),
                                       local["Elevation_m"].ravel(), friction.ravel(), dsettle.ravel()])
            else:  # Infra, validated
                n_poi = np.nan_to_num(infra_grid["n_tourism_poi_10km"].ravel(), nan=0.0)
                d_poi = np.nan_to_num(infra_grid["dist_nearest_tourism_poi_m"].ravel(), nan=SENTINEL)
                d_settle_infra = np.nan_to_num(infra_grid["dist_nearest_settlement_town_m"].ravel(), nan=SENTINEL)
                settle_code_grid = infra_grid["settlement_type_code"].ravel().astype(int)
                settle_cat_grid = np.vectorize(CODE_TO_SETTLE_CAT.get)(settle_code_grid)
                settle_onehot_grid = np.column_stack([
                    (settle_cat_grid == cat.split("Settlement_", 1)[1]).astype(float) for cat in settle_dummies.columns
                ])
                Xg = np.column_stack([local["Dist_to_Highway_m"].ravel(), local["Slope_deg"].ravel(), local["Ruggedness"].ravel(),
                                       local["Elevation_m"].ravel(), friction.ravel(), dsettle.ravel(),
                                       n_poi, d_poi, d_settle_infra, settle_onehot_grid])
            p = predict_proba_ensemble(fitted, Xg).reshape(lon2d.shape)
            preds[target] = dict(proba=p, winner=winner, map_variant=map_variant)
            log(f"  {target}: winner={winner} map_uses={map_variant} mean_proba={p.mean():.3f}")

        p_diff = gaussian_filter(preds["Difficult"]["proba"], sigma=0.6)
        p_easy = gaussian_filter(preds["Easy"]["proba"], sigma=0.6)
        cls = np.full(lon2d.shape, 1, dtype=int)
        cls[p_diff >= 0.5] = 2
        cls[(p_diff < 0.5) & (p_easy >= 0.5)] = 0
        # No inside/outside masking (previously `cls = np.where(inside, cls, -1)`) --
        # that coarse per-cell-center-in-polygon test left visible gaps of unmasked
        # hillshade near boundaries even with the clip_path fix below, since the mask
        # and the clip use different notions of "inside" at this grid's resolution.
        # Classification is computed for the full grid; im_cls.set_clip_path below
        # does the exact boundary shaping instead, with no gap and no bleed.
    except Exception as e:
        log(f"  UNIT FAILED: {type(e).__name__}: {e} -- skipping this unit, continuing with the rest")
        import traceback; traceback.print_exc()
        continue

    all_grids[key] = dict(lon2d=lon2d, lat2d=lat2d, bounds=bounds, cls=cls, elev=local["Elevation_m"], gdf=gdf,
                           winners={t: preds[t]["winner"] for t in ["Difficult","Easy"]},
                           map_variants={t: preds[t]["map_variant"] for t in ["Difficult","Easy"]})

    # --- render ---
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    hs = hillshade(all_grids[key]["elev"])
    hs_hi = upsample_nearest(hs)
    im_hs = ax.imshow(hs_hi, extent=(minx,maxx,miny,maxy), origin="lower", cmap="gray", vmin=0.2, vmax=1.0, zorder=1, alpha=0.55)
    cmap = ListedColormap([EASY_COL, MODERATE_COL, DIFFICULT_COL])
    norm = BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
    cls_masked = np.ma.masked_where(cls < 0, cls)
    cls_hi = upsample_nearest(cls_masked)
    im_cls = ax.imshow(cls_hi, extent=(minx,maxx,miny,maxy), origin="lower", cmap=cmap, norm=norm, alpha=0.72, zorder=2, interpolation="nearest")
    gdf.boundary.plot(ax=ax, edgecolor=BOUND_COL, linewidth=1.0, zorder=3)

    # Known-geosite overlay: true class, ringed where the site is misclassified.
    # Correctness comes from 02_modeling_and_analysis/29's actual LOGO-cluster CV out-of-fold
    # prediction (same methodology/numbers as the reported accuracy), looked up
    # by Locality_ID -- NOT from a nearest-grid-cell lookup on this raster,
    # which at this grid's resolution can put several nearby, differently-
    # labeled sites in the same cell and misrepresent well-performing models.
    oof_combined = map_oof[key]["combined"]
    oof_lookup = dict(zip(oof_combined["locality_ids"], zip(oof_combined["true_int"], oof_combined["pred_int"])))
    true_int = sub["Locality_ID"].map(lambda lid: oof_lookup[lid][0])
    pred_int = sub["Locality_ID"].map(lambda lid: oof_lookup[lid][1])
    is_wrong = (pred_int != true_int)
    n_misclass = int(is_wrong.sum())  # TRUE full-sample count, matches the accuracy reported in text/captions

    # Dense regions pile overlapping markers into a solid mass that visually
    # reads as "almost everything is wrong" regardless of true accuracy --
    # thin the DISPLAYED points only (jointly across correct/misclassified,
    # so thinning doesn't selectively hide or over-represent either), same
    # declutter_points/auto_min_dist_km approach already used for Paper 1's
    # own dense regions. n_misclass above (used in the caption/log text) is
    # computed on the full sample and is unaffected by this.
    sub_draw = sub.copy()
    sub_draw["_is_wrong"] = is_wrong.values
    if len(sub_draw) > DECLUTTER_N_THRESHOLD:
        sub_draw = declutter_stratified(sub_draw, bounds, POINT_S)
        log(f"  declutter: showing {len(sub_draw)}/{len(sub)} markers, accuracy-proportional")

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
    # Clip applied AFTER tight_layout, not before: tight_layout repositions the axes
    # to make room for the legend anchored below it, and a clip_path set on an
    # earlier ax.transData (PGF backend specifically) does not track that
    # repositioning -- the clip then applies at the WRONG, stale axes position,
    # silently cutting away part of the true polygon at save time. This produced
    # exactly the blank-gap bug a human review caught by zooming into the
    # Rabat/Casablanca map: everything looked correct at every earlier checkpoint
    # in this function, and broke specifically between tight_layout() and the
    # final savefig. Confirmed by bisecting the render with intermediate
    # savefig() checkpoints until the exact break point was found.
    clip_path = boundary_clip_patch(gdf)
    im_cls.set_clip_path(PathPatch(clip_path, transform=ax.transData))
    # Hillshade clipped to the SAME exact boundary too -- previously left
    # unclipped as deliberate "context outside the region", but the raster's
    # clip edge is antialiased (soft-blended, not hard-cut), so the unclipped
    # grey hillshade sitting directly behind it showed through as a visible
    # grey fringe/halo hugging the true boundary line. Clipping both layers
    # identically removes that fringe -- nothing grey left outside the
    # polygon for the antialiased edge to blend against.
    im_hs.set_clip_path(PathPatch(clip_path, transform=ax.transData))
    out_path = os.path.join(OUT, f"map_region_{key}.pdf")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    log(f"  Saved {out_path} ({len(sub)} sites, {n_misclass} misclassified)")

with open(os.path.join(GRID_DIR, "paper2_region_grids.pkl"), "wb") as f:
    pickle.dump(all_grids, f)
log(f"\nWrote {os.path.join(GRID_DIR, 'paper2_region_grids.pkl')}")
log("All region maps done.")
