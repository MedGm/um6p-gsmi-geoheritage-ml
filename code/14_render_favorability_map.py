"""
Renders a clean national favorability map from the retrained presence-background pilot
model (code/13, AUC=0.860), replacing the original report figure
(report/figures/geosite_favorability_map.png), which had faint stray watermark/basemap
text artifacts bleeding through at low opacity -- this version is a plain, clean render.

Grid: ~1.5km spacing in EPSG:26191 over Morocco+Western Sahara's real boundary (not a
bounding box -- masked to the actual polygon), giving a comparable point count to the
report's stated "350,105 valid land pixels" for the original map.

Features (same 5 as the presence-background model, code/13): Geology_Class, Soil_Class
sampled directly from the local physical/*.tif rasters (already EPSG:26191, fast);
Elevation_m/Slope_deg/Ruggedness from live Copernicus GLO-30 DEM, tile-batched as
whole-array reads (one HTTP fetch per ~1x1deg tile, then vectorized local sampling for
every grid point in that tile) rather than the original per-point windowed-read approach
code/06/12/13 use -- necessary at this point count (per-point live reads would mean
hundreds of thousands of remote requests).
"""
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import joblib
import importlib.util
from pyproj import Transformer
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
PHYS_DIR = os.path.join(BASE, "archive", "gis_data", "physical")
GEOLOGY_TIF = os.path.join(PHYS_DIR, "geology_classes.tif")
SOIL_TIF = os.path.join(PHYS_DIR, "soil_classes.tif")
MODEL_PATH = os.path.join(BASE, "models", "final", "geosite_location_pilot_model_v3.joblib")
CATALOG_CSV = os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv")
OUT_PNG = os.path.join(BASE, "report", "figures", "geosite_favorability_map_v3.png")

PROJ_CRS = "EPSG:26191"
GRID_SPACING_M = 1500
FEATURES = ["Geology_Class", "Soil_Class", "Elevation_m", "Slope_deg", "Ruggedness"]


def dem_tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def sample_dem_grid_batch(lats, lons):
    """Whole-tile-array batched sampling: one HTTP fetch per needed tile, then
    vectorized local (numpy) sampling for every point in that tile -- required at
    grid scale (per-point live windowed reads, fine for hundreds of points, would mean
    hundreds of thousands of remote requests here)."""
    df = pd.DataFrame({"lat": lats, "lon": lons})
    df["lat_tile"] = np.floor(df["lat"]).astype(int)
    df["lon_tile"] = np.floor(df["lon"]).astype(int)
    elev = np.full(len(df), np.nan)
    slope = np.full(len(df), np.nan)
    rugged = np.full(len(df), np.nan)

    for (lat_tile, lon_tile), group in df.groupby(["lat_tile", "lon_tile"]):
        try:
            src = rasterio.open(dem_tile_url(lat_tile, lon_tile))
            arr = src.read(1).astype(np.float64)
            nodata = src.nodata
            arr[arr == nodata] = np.nan
            transform = src.transform
        except Exception as e:
            print(f"  DEM tile FAILED ({lat_tile},{lon_tile}): {e}", flush=True)
            continue
        pixel_size_m = 30.0
        rows = ((transform.f - group["lat"].values) / -transform.e).astype(int)
        cols = ((group["lon"].values - transform.c) / transform.a).astype(int)
        valid = (rows >= 1) & (rows < arr.shape[0] - 1) & (cols >= 1) & (cols < arr.shape[1] - 1)
        idx = group.index.values
        for k, (r, c, ok) in enumerate(zip(rows, cols, valid)):
            if not ok:
                continue
            window = arr[r-1:r+2, c-1:c+2]
            if np.all(np.isnan(window)):
                continue
            e = window[1, 1]
            lon_pixel_size_m = pixel_size_m * np.cos(np.radians(group["lat"].values[k]))
            if not np.isnan(e) and np.sum(~np.isnan(window)) >= 5:
                gy, gx = np.gradient(window, pixel_size_m, lon_pixel_size_m)
                slope_deg = np.degrees(np.nanmean(np.arctan(np.sqrt(gx**2 + gy**2))))
                rug = np.nanmean(np.abs(window - e))
            else:
                slope_deg, rug = np.nan, np.nan
            elev[idx[k]] = e
            slope[idx[k]] = slope_deg
            rugged[idx[k]] = rug
        src.close()
        print(f"  tile ({lat_tile},{lon_tile}): {len(group)} points sampled", flush=True)
    return elev, slope, rugged


def main():
    spec = importlib.util.spec_from_file_location("c01", os.path.join(HERE, "01_consolidate_geosite_catalog.py"))
    c01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c01)

    print("=== Step 1: build grid over Morocco+WS boundary ===", flush=True)
    morocco_wgs84 = c01.fetch_morocco_boundary()
    morocco_proj = morocco_wgs84.to_crs(PROJ_CRS).union_all()
    minx, miny, maxx, maxy = morocco_proj.bounds
    xs = np.arange(minx, maxx, GRID_SPACING_M)
    ys = np.arange(miny, maxy, GRID_SPACING_M)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
    print(f"Bounding-box grid: {len(grid)} points, filtering to real boundary...", flush=True)

    gdf = gpd.GeoDataFrame(grid, geometry=gpd.points_from_xy(grid.x, grid.y), crs=PROJ_CRS)
    within = gdf.geometry.within(morocco_proj)
    grid = grid[within].reset_index(drop=True)
    print(f"Grid points within Morocco+WS: {len(grid)}", flush=True)

    print("\n=== Step 2: sample Geology_Class / Soil_Class (local rasters) ===", flush=True)
    coords = list(zip(grid["x"].values, grid["y"].values))
    with rasterio.open(GEOLOGY_TIF) as src:
        grid["Geology_Class"] = [v[0] for v in src.sample(coords)]
        geo_nodata = src.nodata
    with rasterio.open(SOIL_TIF) as src:
        grid["Soil_Class"] = [v[0] for v in src.sample(coords)]
        soil_nodata = src.nodata

    to_wgs84 = Transformer.from_crs(PROJ_CRS, "EPSG:4326", always_xy=True)
    lon, lat = to_wgs84.transform(grid["x"].values, grid["y"].values)
    grid["Latitude_WGS84"], grid["Longitude_WGS84"] = lat, lon

    print("\n=== Step 3: sample Elevation/Slope/Ruggedness (Copernicus DEM, tile-batched) ===", flush=True)
    elev, slope, rugged = sample_dem_grid_batch(grid["Latitude_WGS84"].values, grid["Longitude_WGS84"].values)
    grid["Elevation_m"], grid["Slope_deg"], grid["Ruggedness"] = elev, slope, rugged

    n_before = len(grid)
    grid = grid[
        (grid["Geology_Class"] != geo_nodata) & (grid["Soil_Class"] != soil_nodata) &
        grid[["Elevation_m", "Slope_deg", "Ruggedness"]].notna().all(axis=1)
    ].reset_index(drop=True)
    print(f"\nDropped {n_before - len(grid)} nodata points -> {len(grid)} valid land pixels", flush=True)

    print("\n=== Step 4: predict favorability ===", flush=True)
    model = joblib.load(MODEL_PATH)
    grid["favorability"] = model.predict_proba(grid[FEATURES].values)[:, 1]

    grid.to_csv(os.path.join(BASE, "data", "model_outputs", "geosite_favorability_grid_v3.csv"), index=False)
    print(f"Saved grid predictions ({len(grid)} pixels)", flush=True)

    print("\n=== Step 5: render map ===", flush=True)
    catalog = pd.read_csv(CATALOG_CSV)
    to_proj = Transformer.from_crs("EPSG:4326", PROJ_CRS, always_xy=True)
    cat_x, cat_y = to_proj.transform(catalog["Longitude_WGS84"].values, catalog["Latitude_WGS84"].values)

    fig, ax = plt.subplots(figsize=(14, 16))
    sc = ax.scatter(grid["x"], grid["y"], c=grid["favorability"], cmap="YlOrRd",
                     s=(GRID_SPACING_M / 1000) ** 2 * 1.1, marker="s", vmin=0, vmax=1, linewidths=0)
    ax.scatter(cat_x, cat_y, s=6, c="cyan", edgecolors="black", linewidths=0.3,
               label=f"Known geosites (n={len(catalog)})", zorder=5)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Predicted geosite favorability (probability)")
    ax.set_xlabel("Easting (EPSG:26191, m)")
    ax.set_ylabel("Northing (EPSG:26191, m)")
    auc_text = "0.860"
    ax.set_title(f"Geosite Geological Favorability -- Exploratory Pilot Map\n"
                 f"(presence-background RF, spatial-block-CV AUC={auc_text}, retrained with expanded N={len(catalog)} catalog)")
    ax.legend(loc="lower left")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved map to {OUT_PNG}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
