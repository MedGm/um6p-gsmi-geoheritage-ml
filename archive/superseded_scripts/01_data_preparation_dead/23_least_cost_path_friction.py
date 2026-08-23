"""
code/23_least_cost_path_friction.py  (2026-08-17)

Least-cost-path "effective distance" from each of the 733 labeled geosites to
the nearest paved road, as a replacement candidate for the straight-line
Dist_to_Highway_m feature -- built from a Tobler-hiking-function + land-cover
+ wadi-crossing friction surface.

Motivation (report Limitations, Sec. 7, and the day's research plan): a site
can sit meters from a paved road in straight-line terms while requiring a
detour around a cliff or an unbridged wadi crossing -- straight-line distance
can't see either. code/20's G0b already tested straight-line OSM-trail
distance as an ablation (national Difficult acc_logo_cluster 0.7408->0.764,
but Eddakhla leave-region-out Easy gap unchanged at -20.0pp) -- still
Euclidean-in-spirit. This tests whether a friction-weighted PATH distance
closes that gap. Not yet tried in this repo.

Resolution note (important, found during development): a first version of
this script built ONE national friction grid at the existing 1183m physical-
raster resolution (matching archive/gis_data/physical/*.tif) and ran the
least-cost search on that. Sanity-checking on Eddakhla's 25 sites showed most
of them snapping to ~0 cost even where Dist_to_Highway_m was several hundred
metres, because at 1183m/cell with `all_touched=True` road rasterization, a
site up to ~1.6km from a road can land in the same pixel as a touched road
segment. That result would have been dominated by grid-snap noise, not signal.
This version instead builds a per-site LOCAL high-resolution window (90m,
~13x finer) fetched live from Copernicus GLO-30 (elevation, native ~30m,
decimated to 90m) and ESA WorldCover (land cover, native 10m), with roads/
waterways rasterized onto that same local grid -- giving genuine sub-km
differentiation.

Method (per site):
  1. A local window in WGS84 degrees, half-width = min(max(3x the site's
     Dist_to_Highway_m, 12km), 60km), expanded (x3, up to 3 tries) if no
     paved road is reachable in the window.
  2. Elevation fetched from Copernicus GLO-30 COGs (/vsicurl, tiled 1x1
     degree), decimated to ~90m via averaged windowed read; slope computed
     from that via np.gradient with latitude-adjusted pixel spacing (same
     approach as code/06_replace_terrain_with_copernicus_dem.py).
  3. Tobler's hiking function (Tobler 1993): W = 6*exp(-3.5*|S+0.05|) km/h,
     S = rise/run, applied isotropically (unsigned slope) -- a documented
     simplification, not full anisotropic/direction-of-travel Tobler.
  4. Land-cover friction multiplier from ESA WorldCover (/vsicurl, tiled 3x3
     degree), same WC_FRICTION mapping used elsewhere in this repo
     (report/scripts/make_maps.py).
  5. Waterways (OSM, Geofabrik) rasterized onto the local grid; a crossing
     cell adds a fixed additive time penalty. No bridge-detection at this
     resolution/timeframe -- a documented simplification.
  6. Paved roads (OSM, fclass in {motorway, trunk, primary, secondary,
     tertiary} -- identical filter to the existing Dist_to_Highway_m
     methodology) rasterized as near-zero-cost cells.
  7. skimage.graph.MCP_Geometric finds least-cost path from the site's pixel
     to the nearest paved-road pixel in the window.

Output: data/final/dist_to_road_leastcost_m.csv
  (Locality_ID, Dist_to_Highway_m [straight-line, for comparison],
   LeastCost_Hours, LeastCost_Effective_m [hours converted to a flat-ground-
   equivalent metre distance via Tobler's flat-ground speed, 5.037 km/h, so
   it's on the same scale as Dist_to_Highway_m]).

Run:
    cd geosite_project1
    python code/23_least_cost_path_friction.py --test eddakhla   # 25 sites, sanity check
    python code/23_least_cost_path_friction.py                   # all 733 sites
"""
import argparse, glob, os, time, warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, rowcol
import geopandas as gpd
from skimage.graph import MCP_Geometric

warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
ROADS_DIR = os.path.join(BASE, "archive", "gis_data", "roads")
OUT_CSV = os.path.join(BASE, "data", "final", "dist_to_road_leastcost_m.csv")

PAVED_FCLASS = ["motorway", "trunk", "primary", "secondary", "tertiary"]
WC_FRICTION = {
    10: 0.55, 20: 0.35, 30: 0.15, 40: 0.20, 50: 0.05, 60: 0.10,
    70: 0.70, 80: 0.90, 90: 0.75, 95: 0.75, 100: 0.60,
}
LANDCOVER_K = 3.0
WADI_PENALTY_HOURS = 0.5
ROAD_CELL_COST = 0.001
RES_M = 90.0
FLAT_SPEED_KMH = 6 * np.exp(-3.5 * abs(0.0 + 0.05))  # ~5.037 km/h

# ============================================================ Tile fetch helpers (cached)
_dem_cache = {}
def dem_tile(lat_tile, lon_tile):
    key = (lat_tile, lon_tile)
    if key not in _dem_cache:
        ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
        ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
        name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
        url = f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
        try:
            _dem_cache[key] = rasterio.open(url)
        except Exception:
            _dem_cache[key] = None
    return _dem_cache[key]

_wc_cache = {}
def wc_tile(lat_tile, lon_tile):
    key = (lat_tile, lon_tile)
    if key not in _wc_cache:
        ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
        ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
        name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"
        url = f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif"
        try:
            _wc_cache[key] = rasterio.open(url)
        except Exception:
            _wc_cache[key] = None
    return _wc_cache[key]

def mosaic_read(open_tile_fn, tile_size_deg, lon_min, lat_min, lon_max, lat_max, ny, nx,
                 resampling, fill_value):
    lat_tiles = sorted(set(int(np.floor(v / tile_size_deg) * tile_size_deg) for v in [lat_min, lat_max]))
    lon_tiles = sorted(set(int(np.floor(v / tile_size_deg) * tile_size_deg) for v in [lon_min, lon_max]))
    full = np.full((ny, nx), fill_value, dtype=np.float64)
    lon_edges = np.linspace(lon_min, lon_max, nx + 1)
    lat_edges = np.linspace(lat_max, lat_min, ny + 1)
    got_any = False
    for lt in lat_tiles:
        for ln in lon_tiles:
            src = open_tile_fn(lt, ln)
            if src is None:
                continue
            tb = src.bounds
            ov_lon_min, ov_lon_max = max(lon_min, tb.left), min(lon_max, tb.right)
            ov_lat_min, ov_lat_max = max(lat_min, tb.bottom), min(lat_max, tb.top)
            if ov_lon_min >= ov_lon_max or ov_lat_min >= ov_lat_max:
                continue
            win = from_bounds(ov_lon_min, ov_lat_min, ov_lon_max, ov_lat_max, src.transform)
            i0 = np.searchsorted(-lon_edges, -ov_lon_min) - 1
            i1 = np.searchsorted(-lon_edges, -ov_lon_max)
            j0 = np.searchsorted(-lat_edges, -ov_lat_max) - 1
            j1 = np.searchsorted(-lat_edges, -ov_lat_min)
            i0, i1 = max(0, i0), min(nx, i1)
            j0, j1 = max(0, j0), min(ny, j1)
            sub_h, sub_w = max(1, j1 - j0), max(1, i1 - i0)
            try:
                arr = src.read(1, window=win, out_shape=(sub_h, sub_w), resampling=resampling).astype(np.float64)
                nodata = src.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                full[j0:j0 + sub_h, i0:i0 + sub_w] = arr
                got_any = True
            except Exception as e:
                print(f"    tile read failed ({lt},{ln}): {e}", flush=True)
    return full, got_any

# ============================================================ Per-site local grid + cost
def build_local_cost(lon, lat, half_km, roads_gdf, waterways_gdf):
    dlat = half_km / 111.0
    dlon = half_km / (111.0 * max(np.cos(np.radians(lat)), 0.15))
    lon_min, lon_max = lon - dlon, lon + dlon
    lat_min, lat_max = lat - dlat, lat + dlat

    px_deg_lat = RES_M / 111000.0
    px_deg_lon = RES_M / (111000.0 * max(np.cos(np.radians(lat)), 0.15))
    ny = max(3, int(round((lat_max - lat_min) / px_deg_lat)))
    nx = max(3, int(round((lon_max - lon_min) / px_deg_lon)))

    elev, ok = mosaic_read(dem_tile, 1, lon_min, lat_min, lon_max, lat_max, ny, nx,
                            Resampling.average, np.nan)
    if not ok:
        return None
    wc, _ = mosaic_read(wc_tile, 3, lon_min, lat_min, lon_max, lat_max, ny, nx,
                         Resampling.mode, 60)

    px_km_y = (lat_max - lat_min) * 111.0 / ny
    px_km_x = (lon_max - lon_min) * 111.0 * np.cos(np.radians(lat)) / nx
    gy, gx = np.gradient(elev, px_km_y * 1000.0, px_km_x * 1000.0)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    slope_deg = np.nan_to_num(slope_deg, nan=0.0)

    S = np.tan(np.radians(np.abs(slope_deg)))
    W_kmh = np.clip(6 * np.exp(-3.5 * np.abs(S + 0.05)), 0.3, None)
    px_km_avg = (px_km_x + px_km_y) / 2.0
    tobler_hours = px_km_avg / W_kmh

    lc_friction = np.vectorize(lambda c: WC_FRICTION.get(int(c), 0.3))(wc)
    cost = tobler_hours * (1.0 + LANDCOVER_K * lc_friction)

    transform = Affine((lon_max - lon_min) / nx, 0, lon_min,
                        0, -(lat_max - lat_min) / ny, lat_max)

    def rasterize_lines(gdf, bounds_pad=0.02):
        bbox = (lon_min - bounds_pad, lat_min - bounds_pad, lon_max + bounds_pad, lat_max + bounds_pad)
        sub = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        if len(sub) == 0:
            return np.zeros((ny, nx), dtype=np.uint8)
        return rasterize([(g, 1) for g in sub.geometry if g is not None and not g.is_empty],
                          out_shape=(ny, nx), transform=transform, fill=0, dtype=np.uint8, all_touched=True)

    wadi_mask = rasterize_lines(waterways_gdf)
    road_mask = rasterize_lines(roads_gdf)
    cost = cost + wadi_mask * WADI_PENALTY_HOURS
    cost = np.where(road_mask == 1, ROAD_CELL_COST, cost)
    cost = np.where(np.isfinite(cost), cost, np.nanmax(cost[np.isfinite(cost)]) if np.isfinite(cost).any() else 10.0)

    if road_mask.sum() == 0:
        return None

    row, col = rowcol(transform, lon, lat)
    row = int(np.clip(row, 0, ny - 1))
    col = int(np.clip(col, 0, nx - 1))

    return cost, road_mask, (row, col)

def site_least_cost_hours(lon, lat, straight_dist_m, roads_gdf, waterways_gdf):
    half_km = min(max(3 * straight_dist_m / 1000.0, 12.0), 60.0)
    for _ in range(4):
        built = build_local_cost(lon, lat, half_km, roads_gdf, waterways_gdf)
        if built is not None:
            cost, road_mask, (row, col) = built
            mcp = MCP_Geometric(cost, fully_connected=True)
            costs, _ = mcp.find_costs([(row, col)])
            road_costs = costs[road_mask == 1]
            road_costs = road_costs[np.isfinite(road_costs)]
            if len(road_costs) > 0:
                return float(road_costs.min())
        half_km = min(half_km * 2.5, 150.0)
    return None

# ============================================================ Main
def compute_for_sites(df, roads_gdf, waterways_gdf):
    results = []
    for i, r in df.reset_index(drop=True).iterrows():
        lon, lat = r["Longitude_WGS84"], r["Latitude_WGS84"]
        straight_m = r["Dist_to_Highway_m"]
        hours = site_least_cost_hours(lon, lat, straight_m, roads_gdf, waterways_gdf)
        eff_m = hours * FLAT_SPEED_KMH * 1000.0 if hours is not None else np.nan
        results.append({
            "Locality_ID": r["Locality_ID"],
            "Dist_to_Highway_m": straight_m,
            "LeastCost_Hours": hours,
            "LeastCost_Effective_m": eff_m,
        })
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            log(f"  {i+1}/{len(df)} sites done")
    return pd.DataFrame(results)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["eddakhla"], default=None,
                     help="Run on a small subset first (Eddakhla, 25 sites) for a sanity check")
    args = ap.parse_args()

    log("Loading paved roads + waterways (WGS84, no reprojection needed for local windows) ...")
    roads_gdf = gpd.read_file(os.path.join(ROADS_DIR, "gis_osm_roads_free_1.shp"),
                               where=f"fclass IN ({','.join(repr(f) for f in PAVED_FCLASS)})",
                               engine="pyogrio")
    waterways_gdf = gpd.read_file(os.path.join(ROADS_DIR, "gis_osm_waterways_free_1.shp"), engine="pyogrio")
    log(f"  {len(roads_gdf)} paved segments, {len(waterways_gdf)} waterway segments")

    log("Loading labeled catalog (733 sites) ...")
    frames = []
    for f in sorted(glob.glob(os.path.join(BASE, "data/final/regional_label_sources/*.csv"))):
        bn = os.path.basename(f)
        if "expert_labels" not in bn or "combined" in bn:
            continue
        dfl = pd.read_csv(f)
        labeled = dfl[dfl["Expert_Class"].notna() & (dfl["Expert_Class"] != "")].copy()
        frames.append(labeled[["Locality_ID"]])
    all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
    catalog = pd.read_csv(os.path.join(BASE, "data/final/geosites_mcdm_national.csv"))
    merged = all_labels.merge(
        catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84", "Dist_to_Highway_m"]],
        on="Locality_ID", how="inner").dropna(subset=["Region"]).reset_index(drop=True)
    log(f"N={len(merged)} labeled sites loaded")

    if args.test == "eddakhla":
        merged = merged[merged["Region"] == "Eddakhla-Oued Eddahab"].reset_index(drop=True)
        log(f"TEST MODE: subset to Eddakhla, N={len(merged)}")

    out = compute_for_sites(merged, roads_gdf, waterways_gdf)
    log(f"Done. hours: min={out['LeastCost_Hours'].min():.3f} "
        f"max={out['LeastCost_Hours'].max():.3f} mean={out['LeastCost_Hours'].mean():.3f} "
        f"nan={out['LeastCost_Hours'].isna().sum()}")
    log("Straight-line vs least-cost-effective (m):")
    print(out.sort_values("Dist_to_Highway_m").to_string(index=False))

    if args.test:
        out.to_csv(os.path.join(BASE, "data", "final", f"dist_to_road_leastcost_TEST_{args.test}.csv"), index=False)
        log("TEST output written (not the production file).")
    else:
        out.to_csv(OUT_CSV, index=False)
        log(f"Wrote {OUT_CSV}")
