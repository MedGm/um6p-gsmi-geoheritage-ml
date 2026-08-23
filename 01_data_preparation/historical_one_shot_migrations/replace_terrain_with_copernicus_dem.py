"""
Replace the old scanned-map-derived elevation/slope/ruggedness with real, modern,
30m-resolution Copernicus GLO-30 DEM data, sampled directly at each geosite (and its
immediate neighborhood for slope/ruggedness), via remote windowed reads (no full-tile
download needed -- COG + /vsicurl/).

This directly addresses the real DEM-quality problems found during this project's
investigation (Toubkal summit nodata gap, ~20-25km registration error in the old scans,
Jbel Tissouka-style local imprecision) by switching to an independently-produced,
properly-georeferenced, much higher-resolution source.
"""
import os
import numpy as np
import pandas as pd
import rasterio
import warnings
warnings.filterwarnings("ignore")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "training")
IN_CSV = os.path.join(DATA_DIR, "geosites_accessibility_dataset_v1.csv")

def tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"

def sample_terrain(lat, lon, src, window_px=2):
    """Sample elevation at the point, and slope/ruggedness from a small (2*window_px+1)
    pixel window around it, using real-world meter spacing (30m pixels, but longitude
    spacing narrows with latitude -- account for that when converting to slope degrees)."""
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

    # Pixel size ~30m in latitude; longitude pixel size shrinks by cos(latitude)
    pixel_size_m = 30.0
    lon_pixel_size_m = pixel_size_m * np.cos(np.radians(lat))
    if arr.shape[0] >= 3 and arr.shape[1] >= 3:
        gy, gx = np.gradient(arr, pixel_size_m, lon_pixel_size_m)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(np.nanmean(slope_rad))
        ruggedness = np.nanmean(np.abs(arr - elevation))  # mean absolute deviation from center = simple TRI proxy
    else:
        slope_deg, ruggedness = np.nan, np.nan

    return elevation, slope_deg, ruggedness

def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} sites")

    df["lat_tile"] = np.floor(df["Latitude_WGS84"]).astype(int)
    df["lon_tile"] = np.floor(df["Longitude_WGS84"]).astype(int)

    new_elev, new_slope, new_rugged = [np.nan] * len(df), [np.nan] * len(df), [np.nan] * len(df)
    open_srcs = {}
    n_failed = 0

    for tile_key, group in df.groupby(["lat_tile", "lon_tile"]):
        lat_tile, lon_tile = tile_key
        url = tile_url(lat_tile, lon_tile)
        try:
            src = rasterio.open(url)
        except Exception as e:
            print(f"  FAILED to open tile ({lat_tile},{lon_tile}): {e}")
            n_failed += len(group)
            continue
        print(f"  Tile N{lat_tile}/W{-lon_tile}: {len(group)} sites")
        for idx, row in group.iterrows():
            try:
                e, s, r = sample_terrain(row["Latitude_WGS84"], row["Longitude_WGS84"], src)
                new_elev[df.index.get_loc(idx)] = e
                new_slope[df.index.get_loc(idx)] = s
                new_rugged[df.index.get_loc(idx)] = r
            except Exception as ex:
                n_failed += 1
        src.close()

    df["Elevation_m_Copernicus"] = new_elev
    df["Slope_deg_Copernicus"] = new_slope
    df["Ruggedness_Copernicus"] = new_rugged
    df = df.drop(columns=["lat_tile", "lon_tile"])

    print(f"\nFailed samples: {n_failed}")
    print(f"\nOld vs new elevation comparison:")
    comp = df[["Elevation_m", "Elevation_m_Copernicus"]].dropna()
    print(f"  Mean abs difference: {(comp['Elevation_m'] - comp['Elevation_m_Copernicus']).abs().mean():.1f}m")
    print(f"  Correlation: {comp['Elevation_m'].corr(comp['Elevation_m_Copernicus']):.3f}")

    df.to_csv(IN_CSV, index=False)
    print(f"\nSaved with new Copernicus columns to {IN_CSV}")

if __name__ == "__main__":
    main()
