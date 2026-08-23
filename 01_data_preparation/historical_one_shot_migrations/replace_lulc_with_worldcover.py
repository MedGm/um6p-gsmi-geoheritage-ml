"""
Replace the old scanned-map-decoded LULC classification with real, modern ESA
WorldCover v200 (2021) 10m land-cover data, sampled directly at each geosite via
remote windowed reads (COG + /vsicurl/, no full-tile download).

This replaces the LULC_Class_Name_Verified column (recovered via color-swatch
matching against a scanned map PNG, with one documented unresolvable color
collision) with an independently-produced, properly-georeferenced source.

ESA WorldCover classes (v200, 2021):
  10 Tree cover, 20 Shrubland, 30 Grassland, 40 Cropland, 50 Built-up,
  60 Bare/sparse vegetation, 70 Snow/ice, 80 Water bodies,
  90 Herbaceous wetland, 95 Mangroves, 100 Moss/lichen
"""
import os
import numpy as np
import pandas as pd
import rasterio
import warnings
warnings.filterwarnings("ignore")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["AWS_S3_ENDPOINT"] = "s3.eu-central-1.amazonaws.com"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "data", "training")
IN_CSV = os.path.join(DATA_DIR, "geosites_accessibility_dataset_v1.csv")

WC_CLASS_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse vegetation", 70: "Snow/ice",
    80: "Water bodies", 90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss/lichen",
}

# Friction: how much this land cover impedes physical approach on foot/vehicle.
# Built-up/Grassland/Cropland easiest; Tree cover and wetland/water hardest.
WC_FRICTION = {
    10: 0.55,  # Tree cover -- dense vegetation impedes approach
    20: 0.35,  # Shrubland
    30: 0.15,  # Grassland
    40: 0.20,  # Cropland -- usually has field access tracks
    50: 0.05,  # Built-up -- easiest, roads/infrastructure present
    60: 0.10,  # Bare/sparse vegetation -- open ground, easy to cross
    70: 0.70,  # Snow/ice
    80: 0.90,  # Water bodies -- not approachable directly
    90: 0.75,  # Herbaceous wetland
    95: 0.75,  # Mangroves
    100: 0.60, # Moss/lichen (high-altitude terrain in Morocco context)
}


def tile_key(lat, lon):
    return int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)


def tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"
    return f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif"


def sample_class(lat, lon, src, window_px=1):
    """Majority class in a small window around the point (10m pixels -- a single
    pixel can land on a road/building sliver; a small window is more representative)."""
    row, col = src.index(lon, lat)
    r0, c0 = max(0, row - window_px), max(0, col - window_px)
    window = ((r0, min(src.height, row + window_px + 1)), (c0, min(src.width, col + window_px + 1)))
    arr = src.read(1, window=window)
    if arr.size == 0:
        return np.nan
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def main():
    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} sites")

    df["_lat_tile"], df["_lon_tile"] = zip(*df.apply(
        lambda r: tile_key(r["Latitude_WGS84"], r["Longitude_WGS84"]), axis=1))

    new_class = [np.nan] * len(df)
    n_failed = 0

    for (lat_tile, lon_tile), group in df.groupby(["_lat_tile", "_lon_tile"]):
        url = tile_url(lat_tile, lon_tile)
        try:
            src = rasterio.open(url)
        except Exception as e:
            print(f"  FAILED to open tile ({lat_tile},{lon_tile}): {e}")
            n_failed += len(group)
            continue
        print(f"  Tile {lat_tile}/{lon_tile}: {len(group)} sites")
        for idx, row in group.iterrows():
            try:
                c = sample_class(row["Latitude_WGS84"], row["Longitude_WGS84"], src)
                new_class[df.index.get_loc(idx)] = c
            except Exception:
                n_failed += 1
        src.close()

    df["LULC_Class_WorldCover"] = new_class
    df["LULC_Class_Name_WorldCover"] = df["LULC_Class_WorldCover"].map(WC_CLASS_NAMES)
    df["LULC_Friction_WorldCover"] = df["LULC_Class_WorldCover"].map(WC_FRICTION)
    df = df.drop(columns=["_lat_tile", "_lon_tile"])

    print(f"\nFailed samples: {n_failed}")
    print(f"\nWorldCover class distribution:\n{df['LULC_Class_Name_WorldCover'].value_counts()}")

    df.to_csv(IN_CSV, index=False)
    print(f"\nSaved with new WorldCover columns to {IN_CSV}")


if __name__ == "__main__":
    main()
