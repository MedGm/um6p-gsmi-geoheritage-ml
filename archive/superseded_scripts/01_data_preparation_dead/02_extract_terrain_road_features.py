"""
Terrain + road-distance feature extraction for the geosite_v2 locality catalog
(Task 4).

Reuses the methodology already built, independently reviewed, and confirmed
correct in the earlier phase1-foundations-fix work
(livrable/phase1_national_accessibility/code/01_extract_geosite_features.py):

1. Load locality coordinates as a GeoDataFrame in EPSG:4326, reproject to
   EPSG:26191 (Sahara Lambert).
2. Sample each physical raster at every point via rasterio.sample().
3. Compute TRUE point-to-nearest-line vector distance to OSM roads (filtered
   to paved/official classes: motorway, trunk, primary, secondary, tertiary
   -- no track/path/piste classes) via geopandas.sjoin_nearest, in the
   projected CRS. This is a continuous vector distance, NOT a rasterized/
   quantized distance-transform value.
4. Fill any raster nodata gaps with the column median (classes: mode).

Adapted for the new inputs:
  - geosites_localities_master.csv (Locality_ID / Geosite_Name /
    Latitude_WGS84 / Longitude_WGS84 / Region columns)
  - collected_data/worktree_task2_corrected/gis_data/physical/ rasters
    (registration-corrected via coastline calibration, already reviewed)

Output: geosites_features.csv, one row per locality with valid coordinates.
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

GIS_DIR = os.path.join(PROJECT_ROOT, "archive", "gis_data", "physical_task2_corrected")
LOCALITIES_CSV = os.path.join(PROJECT_ROOT, "geosite_v2_work", "data", "geosites_localities_master.csv")
# MISSING as of the 2026-08-15 directory cleanup: this Geofabrik Morocco OSM extract
# (roads layer, gis_osm_roads_free_1.shp) lived under collected_data/, which was deleted
# as redundant without noticing this script depended on it. Re-fetch from
# https://download.geofabrik.de/africa/morocco-latest-free.shp.zip before rerunning.
ROADS_SHP = os.path.join(PROJECT_ROOT, "archive", "gis_data", "roads", "gis_osm_roads_free_1.shp")
OUT_CSV = os.path.join(PROJECT_ROOT, "geosite_v2_work", "data", "geosites_features.csv")

RASTER_FILES = {
    "Elevation_m": "elevation_meters.tif",
    "Slope_deg": "slope_degrees.tif",
    "Ruggedness": "ruggedness.tif",
    "Dist_to_Dam_m": "distance_to_dams_meters.tif",
    "Dist_to_River_m": "distance_to_rivers_meters.tif",
    "LULC_Class": "lulc_classes.tif",
    "Soil_Class": "soil_classes.tif",
    "Geology_Class": "geology_classes.tif",
}
CLASS_COLS = ["LULC_Class", "Soil_Class", "Geology_Class"]
CONTINUOUS_COLS = ["Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Dam_m", "Dist_to_River_m"]

HIGHWAY_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary"]
PROJ_CRS = "EPSG:26191"

OUTPUT_COLUMNS = [
    "Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84", "Region",
    "Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Dam_m", "Dist_to_River_m",
    "LULC_Class", "Soil_Class", "Geology_Class", "Dist_to_Highway_m",
]


def sample_rasters(gdf_proj):
    xs = gdf_proj.geometry.x.values
    ys = gdf_proj.geometry.y.values
    coords = list(zip(xs, ys))
    out = {}
    for feat, fname in RASTER_FILES.items():
        path = os.path.join(GIS_DIR, fname)
        with rasterio.open(path) as src:
            assert str(src.crs) == PROJ_CRS, f"{fname} CRS {src.crs} != {PROJ_CRS}"
            vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
            nodata = src.nodata
            if nodata is not None:
                vals[vals == np.float32(nodata)] = np.nan
        out[feat] = vals
    return pd.DataFrame(out, index=gdf_proj.index)


def compute_highway_distance(sites_gdf, roads_gdf):
    """True point-to-nearest-highway-line Euclidean distance (m), via sjoin_nearest.

    sites_gdf must be indexed by a genuinely unique key (Locality_ID here).
    """
    assert sites_gdf.index.is_unique, "sites_gdf index must be unique for correct realignment"
    nearest = gpd.sjoin_nearest(sites_gdf, roads_gdf[["geometry"]], distance_col="dist_m")
    nearest = nearest.sort_values("dist_m").groupby(level=0).first()
    nearest = nearest.reindex(sites_gdf.index)
    assert len(nearest) == len(sites_gdf), "distance realignment lost/duplicated rows"
    assert nearest["dist_m"].isna().sum() == 0, "some sites failed to match a nearest road"
    return nearest["dist_m"].values


def main():
    df = pd.read_csv(LOCALITIES_CSV)
    n_raw = len(df)
    print(f"Loaded {n_raw} rows from {LOCALITIES_CSV}")

    # --- Drop rows with missing coordinates (explicit, counted, no silent garbage) ---
    n_nan_coords = df[["Latitude_WGS84", "Longitude_WGS84"]].isna().any(axis=1).sum()
    df = df.dropna(subset=["Latitude_WGS84", "Longitude_WGS84"]).copy()
    print(f"Excluded {n_nan_coords} localities with missing coordinates -> {len(df)} geolocated localities")

    assert df["Locality_ID"].is_unique, "Locality_ID expected unique"
    df = df.set_index("Locality_ID", drop=False)
    df.index.name = None  # avoid collision with sjoin_nearest's internal reset_index()

    # --- Build projected GeoDataFrame ---
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.Longitude_WGS84, df.Latitude_WGS84),
        crs="EPSG:4326",
    ).to_crs(PROJ_CRS)
    assert gdf.index.is_unique

    # --- Sample terrain / thematic rasters ---
    terrain_df = sample_rasters(gdf)
    for col in terrain_df.columns:
        df[col] = terrain_df[col].values

    # --- Vector point-to-line highway distance ---
    print("Loading OSM roads, filtering to paved highway classes "
          "(motorway/trunk/primary/secondary/tertiary)...")
    roads = gpd.read_file(ROADS_SHP)
    n_roads_all = len(roads)
    roads = roads[roads["fclass"].isin(HIGHWAY_CLASSES)].to_crs(PROJ_CRS)
    print(f"Highway segments retained: {len(roads)} / {n_roads_all} total OSM road segments")

    dist_m = compute_highway_distance(gdf, roads)
    df["Dist_to_Highway_m"] = dist_m

    # --- Sanity check: distance should be (near-)continuous, not quantized ---
    n_unique_dist = df["Dist_to_Highway_m"].nunique()
    n_unique_coords = df.drop_duplicates(subset=["Latitude_WGS84", "Longitude_WGS84"]).shape[0]
    print(f"Dist_to_Highway_m unique values: {n_unique_dist} / {len(df)} sites "
          f"({n_unique_coords} unique coordinate pairs)")
    assert n_unique_dist > n_unique_coords * 0.9, (
        "Distance values look quantized relative to the number of distinct site locations "
        "-- vector distance computation likely broken"
    )

    # --- Fill sporadic raster-edge NaNs ---
    for col in CONTINUOUS_COLS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"  filled {n_nan} NaN in {col} with median")
    for col in CLASS_COLS:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            mode = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode)
            print(f"  filled {n_nan} NaN in {col} with mode ({mode})")

    out_df = df[OUTPUT_COLUMNS].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(out_df)} locality feature rows to {OUT_CSV}")

    # --- Final assertions ---
    assert out_df["Geosite_Name"].notna().all()
    assert out_df["Dist_to_Highway_m"].min() >= 0
    assert out_df["Elevation_m"].between(-100, 4500).mean() > 0.95, "Elevation values look wrong"
    assert "Dist_to_Piste_m" not in out_df.columns, "Pistes must stay excluded from the feature set"
    print("All sanity assertions passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"ASSERTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
