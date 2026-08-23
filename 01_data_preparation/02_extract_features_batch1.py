"""
Computes the same 6 production features (Elevation_m, Slope_deg, Ruggedness,
Dist_to_Highway_m, LULC_Friction, Dist_to_Settlement_m) for the new localities found in
data/newdb_v2/ (the expanded 2026-08-09 geoheritage source), using the exact same
methodology already used for the 324-site production catalog:
  - Elevation/Slope/Ruggedness: live Copernicus GLO-30 DEM (code/06's method)
  - LULC_Friction: live ESA WorldCover v200 (code/07's method)
  - Dist_to_Highway_m: point-to-nearest-line distance to OSM paved roads (code/02's method)
  - Dist_to_Settlement_m: haversine to nearest of 55 reference cities (code/08's method)

Does NOT assign accessibility labels (Easy/Moderate/Difficult) -- these new sites have
no expert label, and this project has a hard-won, repeatedly-documented lesson against
formula-derived/circular labels (the original 91.9%-accuracy label bug; AHP_Class flagged
a "ghost result" in the README). They're added to the raw catalog as unlabeled entries,
same as any of the 73 already-unlabeled sites in the current 324-site catalog.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import warnings
warnings.filterwarnings("ignore")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
NEWDB_V2 = os.path.join(BASE, "data", "newdb_v2")
# MISSING as of the 2026-08-15 directory cleanup: see code/02_extract_terrain_road_features.py
# for the same issue and the re-fetch URL.
ROADS_SHP = os.path.join(BASE, "archive", "gis_data", "roads", "gis_osm_roads_free_1.shp")
CITIES_CSV = os.path.join(BASE, "data", "archive", "pipeline_intermediates", "morocco_reference_cities_geocoded.csv")
OUT_CSV = os.path.join(NEWDB_V2, "geosites_new_localities_features.csv")

HIGHWAY_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary"]
PROJ_CRS = "EPSG:26191"

WC_FRICTION = {
    10: 0.55, 20: 0.35, 30: 0.15, 40: 0.20, 50: 0.05, 60: 0.10,
    70: 0.70, 80: 0.90, 90: 0.75, 95: 0.75, 100: 0.60,
}


def dem_tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def sample_terrain(lat, lon, src, window_px=2):
    row, col = src.index(lon, lat)
    size = 2 * window_px + 1
    r0, c0 = max(0, row - window_px), max(0, col - window_px)
    window = ((r0, min(src.height, row + window_px + 1)), (c0, min(src.width, col + window_px + 1)))
    arr = src.read(1, window=window).astype(np.float64)
    nodata = src.nodata
    arr[arr == nodata] = np.nan
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan, np.nan, np.nan
    center_r, center_c = row - r0, col - c0
    center_r = min(max(center_r, 0), arr.shape[0] - 1)
    center_c = min(max(center_c, 0), arr.shape[1] - 1)
    elevation = arr[center_r, center_c]
    pixel_size_m = 30.0
    lon_pixel_size_m = pixel_size_m * np.cos(np.radians(lat))
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        gy, gx = np.gradient(arr, pixel_size_m, lon_pixel_size_m)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(np.nanmean(slope_rad))
        ruggedness = np.nanmean(np.abs(arr - elevation))
    else:
        slope_deg, ruggedness = np.nan, np.nan
    return elevation, slope_deg, ruggedness


def wc_tile_key(lat, lon):
    return int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)


def wc_tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"
    return f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif"


def sample_class(lat, lon, src, window_px=1):
    row, col = src.index(lon, lat)
    r0, c0 = max(0, row - window_px), max(0, col - window_px)
    window = ((r0, min(src.height, row + window_px + 1)), (c0, min(src.width, col + window_px + 1)))
    arr = src.read(1, window=window)
    if arr.size == 0:
        return np.nan
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    matched = pd.read_csv(os.path.join(NEWDB_V2, "geosites_localities_vs_final_catalog.csv"))
    df = matched[matched["match_class"] == "new"].copy().reset_index(drop=True)
    print(f"New localities to featurize: {len(df)}", flush=True)
    assert df["Locality_ID"].is_unique

    # --- Elevation/Slope/Ruggedness via Copernicus DEM ---
    df["lat_tile"] = np.floor(df["Latitude_WGS84"]).astype(int)
    df["lon_tile"] = np.floor(df["Longitude_WGS84"]).astype(int)
    elev, slope, rugged = [np.nan] * len(df), [np.nan] * len(df), [np.nan] * len(df)
    n_failed = 0
    for (lat_tile, lon_tile), group in df.groupby(["lat_tile", "lon_tile"]):
        try:
            src = rasterio.open(dem_tile_url(lat_tile, lon_tile))
        except Exception as e:
            print(f"  DEM tile FAILED ({lat_tile},{lon_tile}): {e}", flush=True)
            n_failed += len(group)
            continue
        for idx, row in group.iterrows():
            try:
                e, s, r = sample_terrain(row["Latitude_WGS84"], row["Longitude_WGS84"], src)
                elev[idx], slope[idx], rugged[idx] = e, s, r
            except Exception:
                n_failed += 1
        src.close()
    df["Elevation_m"], df["Slope_deg"], df["Ruggedness"] = elev, slope, rugged
    print(f"DEM sampling done, {n_failed} failures", flush=True)

    # --- LULC_Friction via WorldCover ---
    df["_lat_tile"], df["_lon_tile"] = zip(*df.apply(lambda r: wc_tile_key(r["Latitude_WGS84"], r["Longitude_WGS84"]), axis=1))
    wc_class = [np.nan] * len(df)
    n_failed_wc = 0
    for (lat_tile, lon_tile), group in df.groupby(["_lat_tile", "_lon_tile"]):
        try:
            src = rasterio.open(wc_tile_url(lat_tile, lon_tile))
        except Exception as e:
            print(f"  WorldCover tile FAILED ({lat_tile},{lon_tile}): {e}", flush=True)
            n_failed_wc += len(group)
            continue
        for idx, row in group.iterrows():
            try:
                wc_class[idx] = sample_class(row["Latitude_WGS84"], row["Longitude_WGS84"], src)
            except Exception:
                n_failed_wc += 1
        src.close()
    df["LULC_Friction"] = pd.Series(wc_class).map(WC_FRICTION)
    df = df.drop(columns=["lat_tile", "lon_tile", "_lat_tile", "_lon_tile"])
    print(f"WorldCover sampling done, {n_failed_wc} failures", flush=True)

    # --- Dist_to_Highway_m via OSM roads ---
    print("Loading OSM roads...", flush=True)
    roads = gpd.read_file(ROADS_SHP)
    roads = roads[roads["fclass"].isin(HIGHWAY_CLASSES)].to_crs(PROJ_CRS)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude_WGS84, df.Latitude_WGS84), crs="EPSG:4326").to_crs(PROJ_CRS)
    nearest = gpd.sjoin_nearest(gdf, roads[["geometry"]], distance_col="dist_m")
    nearest = nearest.sort_values("dist_m").groupby(level=0).first()
    df["Dist_to_Highway_m"] = nearest["dist_m"].reindex(df.index).values
    print("Highway distance done", flush=True)

    # --- Dist_to_Settlement_m via reference cities ---
    cities = pd.read_csv(CITIES_CSV)
    clat, clon = cities["Latitude"].values, cities["Longitude"].values
    df["Dist_to_Settlement_m"] = df.apply(
        lambda r: haversine_m(r["Latitude_WGS84"], r["Longitude_WGS84"], clat, clon).min(), axis=1)
    print("Settlement distance done", flush=True)

    for col in ["Elevation_m", "Slope_deg", "Ruggedness", "LULC_Friction", "Dist_to_Highway_m", "Dist_to_Settlement_m"]:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  filled {n_nan} NaN in {col} with median", flush=True)

    out_cols = ["Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Region",
                "Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Highway_m", "LULC_Friction", "Dist_to_Settlement_m"]
    df[out_cols].to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(df)} feature rows to {OUT_CSV}", flush=True)
    print(df[["Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Highway_m", "LULC_Friction", "Dist_to_Settlement_m"]].describe(), flush=True)


if __name__ == "__main__":
    main()
