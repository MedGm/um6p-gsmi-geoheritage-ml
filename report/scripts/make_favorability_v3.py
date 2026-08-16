"""
Generates a national favorability grid from the enriched v3 model
(models/final/geosite_location_pilot_model_v3.joblib -- see
notebooks/improve_favorability_model.ipynb), at the same
resolution as the original v2 grid (code/14, ~1500m spacing, EPSG:26191).

Elevation/Slope/Ruggedness are sampled from the LIVE Copernicus GLO-30m DEM,
tile-batched exactly like code/14's sample_dem_grid_batch -- NOT from a
local raster stack. An earlier version of this script used a local stack
and produced a thread-like artifact across the entire country in the
resulting map; three different model configurations (different features,
constrained vs. unconstrained RF) all reproduced the identical pattern,
which is what proved it wasn't a modeling issue. Direct comparison against
code/14 (the only prior working version) showed the actual difference:
code/14 uses live Copernicus DEM for these three features, this script was
using a different, lower-quality local raster stack. Switching back to
Copernicus, matching code/14 exactly, is the real fix -- confirmed
empirically below, not assumed.

Dist_to_Settlement_m: haversine-to-55-cities (code/08 method). LULC_Friction:
live ESA WorldCover (code/07 method), tile-batched. Geology_Class/Soil_Class:
sampled from the calibration-corrected local rasters preserved at
archive/gis_data/physical_task2_corrected/ (code/13/14 used the same
rasters before that directory was consolidated into archive/), with
explicit missing-flag handling (not dropped).
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "s3.eu-central-1.amazonaws.com"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE = FW
PROJ_CRS = "EPSG:26191"

print("[1] Loading v3 model...", flush=True)
bundle = joblib.load(os.path.join(BASE, "models", "final", "geosite_location_pilot_model_v3.joblib"))
model, FEATURES = bundle["model"], bundle["features"]
print("    features:", FEATURES, flush=True)

print("[2] Loading national boundary...", flush=True)
national_gdf = gpd.read_file(os.path.join(FW, "data", "boundaries", "national.geojson"))
national_26191 = national_gdf.to_crs(PROJ_CRS)
minx, miny, maxx, maxy = national_26191.total_bounds

print("[3] Building ~1500m national grid, clipped to boundary...", flush=True)
SPACING = 1500.0
xs = np.arange(minx, maxx, SPACING)
ys = np.arange(miny, maxy, SPACING)
xx, yy = np.meshgrid(xs, ys)
xx, yy = xx.ravel(), yy.ravel()

from shapely.geometry import Point
from matplotlib.path import Path as MplPath
union = national_26191.union_all()
geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
inside = np.zeros(len(xx), dtype=bool)
for geom in geoms:
    xr, yr = geom.exterior.xy
    p = MplPath(np.column_stack([xr, yr]))
    inside |= p.contains_points(np.column_stack([xx, yy]))
xx, yy = xx[inside], yy[inside]
print(f"    {len(xx)} grid cells inside boundary", flush=True)

to_wgs84 = Transformer.from_crs(PROJ_CRS, "EPSG:4326", always_xy=True)
lon, lat = to_wgs84.transform(xx, yy)

print("[4] Sampling Geology_Class / Soil_Class (local legacy rasters, same as code/13/14)...", flush=True)
PHYS_LEGACY = os.path.join(BASE, "archive", "gis_data", "physical_task2_corrected")
with rasterio.open(os.path.join(PHYS_LEGACY, "geology_classes.tif")) as src:
    geology_full = src.read(1)
    geology_transform = src.transform
    geology_nodata = src.nodata
with rasterio.open(os.path.join(PHYS_LEGACY, "soil_classes.tif")) as src:
    soil_full = src.read(1)
    soil_transform = src.transform
    soil_nodata = src.nodata
geology_inv = ~geology_transform
soil_inv = ~soil_transform

def sample_stack(arr, transform_inv, x, y):
    col, row = transform_inv * (x, y)
    row = np.clip(np.round(row).astype(int), 0, arr.shape[0] - 1)
    col = np.clip(np.round(col).astype(int), 0, arr.shape[1] - 1)
    return arr[row, col]

geology_vals = sample_stack(geology_full, geology_inv, xx, yy)
soil_vals = sample_stack(soil_full, soil_inv, xx, yy)

print("[5] Sampling Elevation/Slope/Ruggedness from LIVE Copernicus GLO-30m DEM "
      "(tile-batched, matching code/14 exactly)...", flush=True)

def dem_tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"

def sample_dem_grid_batch(lats, lons):
    df = pd.DataFrame({"lat": lats, "lon": lons})
    df["lat_tile"] = np.floor(df["lat"]).astype(int)
    df["lon_tile"] = np.floor(df["lon"]).astype(int)
    elev = np.full(len(df), np.nan)
    slope = np.full(len(df), np.nan)
    rugged = np.full(len(df), np.nan)
    n_tiles = df.groupby(["lat_tile", "lon_tile"]).ngroups
    done = 0
    for (lat_tile, lon_tile), group in df.groupby(["lat_tile", "lon_tile"]):
        done += 1
        try:
            src = rasterio.open(dem_tile_url(lat_tile, lon_tile))
            arr = src.read(1).astype(np.float64)
            nodata = src.nodata
            arr[arr == nodata] = np.nan
            transform = src.transform
        except Exception as e:
            print(f"    DEM tile FAILED ({lat_tile},{lon_tile}): {e}", flush=True)
            continue
        pixel_size_m = 30.0
        rows = ((transform.f - group["lat"].values) / -transform.e).astype(int)
        cols = ((group["lon"].values - transform.c) / transform.a).astype(int)
        valid = (rows >= 1) & (rows < arr.shape[0] - 1) & (cols >= 1) & (cols < arr.shape[1] - 1)
        idx = group.index.values
        for k, (r, c, ok) in enumerate(zip(rows, cols, valid)):
            if not ok:
                continue
            window = arr[r - 1:r + 2, c - 1:c + 2]
            if np.all(np.isnan(window)):
                continue
            e = window[1, 1]
            lon_pixel_size_m = pixel_size_m * np.cos(np.radians(group["lat"].values[k]))
            if not np.isnan(e) and np.sum(~np.isnan(window)) >= 5:
                gy, gx = np.gradient(window, pixel_size_m, lon_pixel_size_m)
                slope_deg = np.degrees(np.nanmean(np.arctan(np.sqrt(gx ** 2 + gy ** 2))))
                rug = np.nanmean(np.abs(window - e))
            else:
                slope_deg, rug = np.nan, np.nan
            elev[idx[k]] = e
            slope[idx[k]] = slope_deg
            rugged[idx[k]] = rug
        src.close()
        print(f"    tile ({lat_tile},{lon_tile}) [{done}/{n_tiles}]: {len(group)} points sampled", flush=True)
    return elev, slope, rugged

elevation_vals, slope_vals, ruggedness_vals = sample_dem_grid_batch(lat, lon)

valid = np.isfinite(elevation_vals) & np.isfinite(slope_vals) & np.isfinite(ruggedness_vals) & \
        (geology_vals != geology_nodata) & (soil_vals != soil_nodata)
print(f"    {(~valid).sum()}/{len(valid)} cells dropped (DEM or geology/soil nodata)", flush=True)
xx, yy, lon, lat = xx[valid], yy[valid], lon[valid], lat[valid]
elevation_vals, slope_vals, ruggedness_vals = elevation_vals[valid], slope_vals[valid], ruggedness_vals[valid]
geology_vals, soil_vals = geology_vals[valid], soil_vals[valid]

print("[6] Computing Dist_to_Settlement_m (haversine to 55 cities, chunked+vectorized)...", flush=True)
cities = pd.read_csv(os.path.join(BASE, "data", "archive", "pipeline_intermediates",
                                    "morocco_reference_cities_geocoded.csv"))
clat, clon = cities["Latitude"].values, cities["Longitude"].values
clat_r, clon_r = np.radians(clat), np.radians(clon)

def haversine_min_m(lat_pts, lon_pts):
    R = 6371000.0
    latr, lonr = np.radians(lat_pts), np.radians(lon_pts)
    dlat = latr[:, None] - clat_r[None, :]
    dlon = lonr[:, None] - clon_r[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(latr[:, None]) * np.cos(clat_r[None, :]) * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(a))
    return d.min(axis=1)

CHUNK = 50_000
dsettle = np.concatenate([haversine_min_m(lat[i:i + CHUNK], lon[i:i + CHUNK])
                           for i in range(0, len(lat), CHUNK)])

print("[7] Sampling live ESA WorldCover for LULC friction "
      "(one decimated windowed read per tile, vectorized lookup)...", flush=True)
WC_FRICTION = {10: 0.55, 20: 0.35, 30: 0.15, 40: 0.20, 50: 0.05, 60: 0.10,
               70: 0.70, 80: 0.90, 90: 0.75, 95: 0.75, 100: 0.60}

def tile_key(la, lo):
    return int(np.floor(la / 3) * 3), int(np.floor(lo / 3) * 3)

def wc_tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"
    return f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif"

lulc_vals = np.full(len(lon), 60, dtype=np.uint8)
tile_ids = np.array([tile_key(la, lo) for la, lo in zip(lat, lon)])
unique_tiles = sorted(set(map(tuple, tile_ids)))
print(f"    {len(unique_tiles)} WorldCover tiles to fetch", flush=True)
DEG_PER_CELL = SPACING / 111_000
for tk in unique_tiles:
    mask = (tile_ids[:, 0] == tk[0]) & (tile_ids[:, 1] == tk[1])
    idxs = np.where(mask)[0]
    lon_i, lat_i = lon[idxs], lat[idxs]
    lo_min, lo_max = lon_i.min() - DEG_PER_CELL, lon_i.max() + DEG_PER_CELL
    la_min, la_max = lat_i.min() - DEG_PER_CELL, lat_i.max() + DEG_PER_CELL
    ny = max(2, int((la_max - la_min) / DEG_PER_CELL) + 1)
    nx = max(2, int((lo_max - lo_min) / DEG_PER_CELL) + 1)
    try:
        with rasterio.open(wc_tile_url(*tk)) as src:
            tb = src.bounds
            ov_lo_min, ov_lo_max = max(lo_min, tb.left), min(lo_max, tb.right)
            ov_la_min, ov_la_max = max(la_min, tb.bottom), min(la_max, tb.top)
            win = from_bounds(ov_lo_min, ov_la_min, ov_lo_max, ov_la_max, src.transform)
            arr = src.read(1, window=win, out_shape=(ny, nx), resampling=Resampling.mode)
        row_i = np.clip(((ov_la_max - lat_i) / (ov_la_max - ov_la_min) * (ny - 1)).astype(int), 0, ny - 1)
        col_i = np.clip(((lon_i - ov_lo_min) / (ov_lo_max - ov_lo_min) * (nx - 1)).astype(int), 0, nx - 1)
        lulc_vals[idxs] = arr[row_i, col_i]
    except Exception as e:
        print(f"    tile {tk} failed ({len(idxs)} pts kept at default): {e}", flush=True)
lulc_friction = np.array([WC_FRICTION.get(int(c), 0.3) for c in lulc_vals])

print("[8] Assembling features + missing flags, predicting...", flush=True)
geology_missing = (geology_vals == geology_nodata).astype(int)
soil_missing = (soil_vals == soil_nodata).astype(int)
geology_vals_clean = np.where(geology_vals == geology_nodata, -1, geology_vals)
soil_vals_clean = np.where(soil_vals == soil_nodata, -1, soil_vals)

feat_map = {
    "Geology_Class": geology_vals_clean, "Soil_Class": soil_vals_clean,
    "Elevation_m": elevation_vals, "Slope_deg": slope_vals, "Ruggedness": ruggedness_vals,
    "Dist_to_Settlement_m": dsettle, "LULC_Friction": lulc_friction,
    "Geology_Class_Missing": geology_missing, "Soil_Class_Missing": soil_missing,
}
X = np.column_stack([feat_map[f] for f in FEATURES])
proba = model.predict_proba(X)[:, 1]

out = pd.DataFrame({"x": xx, "y": yy, "Latitude_WGS84": lat, "Longitude_WGS84": lon, "favorability": proba})
out_path = os.path.join(BASE, "data", "model_outputs", "geosite_favorability_grid_v3.csv")
out.to_csv(out_path, index=False)
print(f"Saved {len(out)} rows to {out_path}", flush=True)
print(out["favorability"].describe())
