"""
code/34_retrain_favorability_v4.py  (2026-08-17)

Retrains the geosite-location favorability model on the expanded catalog
(1,667 sites after code/33's merge of the 2026-08-16 source, up from 1,154),
reconstructing the CURRENTLY DEPLOYED v3 methodology exactly (models/final/
geosite_location_pilot_model_v3.joblib) rather than v2's (code/13,
AUC=0.815, 5 features, nodata rows dropped): v3's training script was never
saved as a numbered pipeline script (the same undocumented-pilot gap already
flagged for v2 in code/13's own docstring, apparently never closed for v3
either) -- reconstructed here from the loaded model bundle's own confirmed
contents (features list and RandomForestClassifier hyperparameters read
directly off `models/final/geosite_location_pilot_model_v3.joblib`, not
guessed) plus report/scripts/make_favorability_v3.py's grid-scoring code
(which shows the exact imputation convention used at inference time and
therefore, necessarily, at training time too):

  - 9 features: Geology_Class, Soil_Class, Elevation_m, Slope_deg, Ruggedness,
    Dist_to_Settlement_m, LULC_Friction, Geology_Class_Missing, Soil_Class_Missing
  - Missing Geology_Class/Soil_Class imputed as -1 with a companion binary
    missing-flag (v3's improvement over v2's "drop nodata rows" -- keeps
    every presence point instead of losing ~20% of them to nodata)
  - RandomForestClassifier(max_depth=14, min_samples_leaf=8, n_estimators=300,
    n_jobs=-1, random_state=42) -- confirmed hyperparameters, not reconstructed guesses
  - Background: 5x presence count, uniform random within the geology raster's
    valid-data extent intersected with the real Morocco+WS boundary (same
    0.1deg buffer convention as code/01's border_check) -- same policy as v2
  - Evaluation: 0.5x0.5deg spatial block GroupKFold CV (same as v2/v3)

Output: models/final/geosite_location_pilot_model_v4.joblib
        data/model_outputs/geosite_presence_background_pilot_v4.csv
"""
import importlib.util
import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import warnings
warnings.filterwarnings("ignore")

os.environ["AWS_NO_SIGN_REQUEST"] = "YES"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
PHYS_DIR = os.path.join(BASE, "archive", "gis_data", "physical_task2_corrected")
GEOLOGY_TIF = os.path.join(PHYS_DIR, "geology_classes.tif")
SOIL_TIF = os.path.join(PHYS_DIR, "soil_classes.tif")
CATALOG_CSV = os.path.join(BASE, "data", "final", "geosites_mcdm_national.csv")
CITIES_CSV = os.path.join(BASE, "data", "archive", "pipeline_intermediates", "morocco_reference_cities_geocoded.csv")
OUT_MODEL = os.path.join(BASE, "models", "final", "geosite_location_pilot_model_v4.joblib")
OUT_CSV = os.path.join(BASE, "data", "model_outputs", "geosite_presence_background_pilot_v4.csv")

V3_MODEL = os.path.join(BASE, "models", "final", "geosite_location_pilot_model_v3.joblib")

PROJ_CRS = "EPSG:26191"
BACKGROUND_RATIO = 5
BLOCK_DEG = 0.5
N_SPLITS = 5
RANDOM_STATE = 42
WC_FRICTION = {10: 0.55, 20: 0.35, 30: 0.15, 40: 0.20, 50: 0.05, 60: 0.10,
               70: 0.70, 80: 0.90, 90: 0.75, 95: 0.75, 100: 0.60}


def dem_tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
    return f"/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def sample_terrain(lat, lon, src, window_px=2):
    row, col = src.index(lon, lat)
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


def sample_dem_batch(lats, lons):
    df = pd.DataFrame({"lat": lats, "lon": lons})
    df["lat_tile"] = np.floor(df["lat"]).astype(int)
    df["lon_tile"] = np.floor(df["lon"]).astype(int)
    elev, slope, rugged = [np.nan] * len(df), [np.nan] * len(df), [np.nan] * len(df)
    for (lat_tile, lon_tile), group in df.groupby(["lat_tile", "lon_tile"]):
        try:
            src = rasterio.open(dem_tile_url(lat_tile, lon_tile))
        except Exception as e:
            print(f"  DEM tile FAILED ({lat_tile},{lon_tile}): {e}", flush=True)
            continue
        for idx, row in group.iterrows():
            try:
                e, s, r = sample_terrain(row["lat"], row["lon"], src)
                elev[idx], slope[idx], rugged[idx] = e, s, r
            except Exception:
                pass
        src.close()
    return np.array(elev), np.array(slope), np.array(rugged)


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


def sample_lulc_batch(lats, lons):
    df = pd.DataFrame({"lat": lats, "lon": lons})
    df["lat_tile"], df["lon_tile"] = zip(*df.apply(lambda r: wc_tile_key(r["lat"], r["lon"]), axis=1))
    wc_class = [np.nan] * len(df)
    for (lat_tile, lon_tile), group in df.groupby(["lat_tile", "lon_tile"]):
        try:
            src = rasterio.open(wc_tile_url(lat_tile, lon_tile))
        except Exception as e:
            print(f"  WorldCover tile FAILED ({lat_tile},{lon_tile}): {e}", flush=True)
            continue
        for idx, row in group.iterrows():
            try:
                wc_class[idx] = sample_class(row["lat"], row["lon"], src)
            except Exception:
                pass
        src.close()
    return pd.Series(wc_class).map(WC_FRICTION).values


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    v3 = joblib.load(V3_MODEL)
    v3_model, v3_features = v3["model"], v3["features"]
    print(f"v3 reference: features={v3_features}", flush=True)
    print(f"v3 reference: hyperparams={v3_model.get_params()}", flush=True)
    rf_params = dict(max_depth=v3_model.max_depth, min_samples_leaf=v3_model.min_samples_leaf,
                      n_estimators=v3_model.n_estimators, n_jobs=-1, random_state=RANDOM_STATE)

    print("\n=== Step 1: presence points (expanded catalog, N=1667 after 2026-08-16 merge) ===", flush=True)
    catalog = pd.read_csv(CATALOG_CSV)
    presence = catalog[["Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84",
                         "Elevation_m", "Slope_deg", "Ruggedness", "LULC_Friction", "Dist_to_Settlement_m",
                         "Geology_Class", "Soil_Class"]].copy()
    presence["presence"] = 1
    print(f"Presence points: {len(presence)}", flush=True)

    print("\n=== Step 2: background points (5x presence, geology valid-extent x Morocco+WS boundary) ===", flush=True)
    spec = importlib.util.spec_from_file_location("c01", os.path.join(HERE, "01_consolidate_geosite_catalog.py"))
    c01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c01)
    print("Fetching Morocco+WS boundary to constrain background sampling...", flush=True)
    boundary_path = os.path.join(BASE, "data", "boundaries", "national.geojson")
    morocco = gpd.read_file(boundary_path).to_crs("EPSG:4326").union_all().buffer(0.1)

    with rasterio.open(GEOLOGY_TIF) as src:
        geology_nodata = src.nodata
        geology_transform = src.transform
        geology_full = src.read(1)

    valid_mask = geology_full != geology_nodata
    valid_rows, valid_cols = np.where(valid_mask)
    to_wgs84 = Transformer.from_crs(PROJ_CRS, "EPSG:4326", always_xy=True)

    n_background = BACKGROUND_RATIO * len(presence)
    rng = np.random.default_rng(RANDOM_STATE)
    bg_lon_list, bg_lat_list = [], []
    batch_size = n_background * 2
    attempts = 0
    while len(bg_lon_list) < n_background and attempts < 15:
        sel = rng.choice(len(valid_rows), size=batch_size, replace=False)
        sel_rows, sel_cols = valid_rows[sel], valid_cols[sel]
        cx, cy = rasterio.transform.xy(geology_transform, sel_rows, sel_cols)
        clon, clat = to_wgs84.transform(np.array(cx), np.array(cy))
        for lo, la in zip(clon, clat):
            if len(bg_lon_list) >= n_background:
                break
            if morocco.contains(Point(lo, la)):
                bg_lon_list.append(lo)
                bg_lat_list.append(la)
        attempts += 1
    bg_lon, bg_lat = np.array(bg_lon_list), np.array(bg_lat_list)
    print(f"Background points within Morocco+WS boundary: {len(bg_lon)} (target {n_background}, {attempts} rounds)", flush=True)
    to_proj = Transformer.from_crs("EPSG:4326", PROJ_CRS, always_xy=True)
    bg_x, bg_y = to_proj.transform(bg_lon, bg_lat)

    n_bg = len(bg_lon)
    background = pd.DataFrame({
        "Locality_ID": [f"bg_{i:05d}" for i in range(n_bg)],
        "Geosite_Name": "background",
        "Latitude_WGS84": bg_lat, "Longitude_WGS84": bg_lon,
        "presence": 0,
    })
    with rasterio.open(GEOLOGY_TIF) as src:
        background["Geology_Class"] = [v[0] for v in src.sample(list(zip(bg_x, bg_y)))]
    with rasterio.open(SOIL_TIF) as src:
        background["Soil_Class"] = [v[0] for v in src.sample(list(zip(bg_x, bg_y)))]

    print(f"Sampling Copernicus DEM for {n_bg} background points (tile-batched)...", flush=True)
    bg_elev, bg_slope, bg_rugged = sample_dem_batch(background["Latitude_WGS84"].values, background["Longitude_WGS84"].values)
    background["Elevation_m"], background["Slope_deg"], background["Ruggedness"] = bg_elev, bg_slope, bg_rugged

    print(f"Sampling ESA WorldCover for {n_bg} background points (tile-batched)...", flush=True)
    background["LULC_Friction"] = sample_lulc_batch(background["Latitude_WGS84"].values, background["Longitude_WGS84"].values)

    print(f"Computing Dist_to_Settlement_m for {n_bg} background points...", flush=True)
    cities = pd.read_csv(CITIES_CSV)
    clat, clon = cities["Latitude"].values, cities["Longitude"].values
    background["Dist_to_Settlement_m"] = background.apply(
        lambda r: haversine_m(r["Latitude_WGS84"], r["Longitude_WGS84"], clat, clon).min(), axis=1)

    geology_nodata_val = geology_nodata
    with rasterio.open(SOIL_TIF) as src:
        soil_nodata_val = src.nodata

    print("\n=== Step 3: combine, impute missing Geology/Soil as -1 + flag (v3 convention, NOT dropped) ===", flush=True)
    cols = ["Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84",
            "Geology_Class", "Soil_Class", "Elevation_m", "Slope_deg", "Ruggedness",
            "LULC_Friction", "Dist_to_Settlement_m", "presence"]
    combined = pd.concat([presence[cols], background[cols]], ignore_index=True)

    n_before = len(combined)
    combined = combined[combined[["Elevation_m", "Slope_deg", "Ruggedness", "LULC_Friction", "Dist_to_Settlement_m"]].notna().all(axis=1)].reset_index(drop=True)
    print(f"Dropped {n_before - len(combined)} rows with missing DEM/WorldCover/settlement features "
          f"(genuine sampling failures, not geology/soil nodata) -> {len(combined)} remain", flush=True)

    combined["Geology_Class_Missing"] = ((combined["Geology_Class"].isna()) | (combined["Geology_Class"] == geology_nodata_val)).astype(int)
    combined["Soil_Class_Missing"] = ((combined["Soil_Class"].isna()) | (combined["Soil_Class"] == soil_nodata_val)).astype(int)
    combined["Geology_Class"] = combined["Geology_Class"].where(combined["Geology_Class_Missing"] == 0, -1)
    combined["Soil_Class"] = combined["Soil_Class"].where(combined["Soil_Class_Missing"] == 0, -1)
    print(f"Geology_Class_Missing: {combined['Geology_Class_Missing'].sum()}, "
          f"Soil_Class_Missing: {combined['Soil_Class_Missing'].sum()} "
          f"(presence={( (combined['presence']==1) & (combined['Geology_Class_Missing']==1) ).sum()} geo / "
          f"{( (combined['presence']==1) & (combined['Soil_Class_Missing']==1) ).sum()} soil)", flush=True)
    print(f"presence={( combined['presence']==1 ).sum()}, background={( combined['presence']==0 ).sum()}", flush=True)

    combined.to_csv(OUT_CSV, index=False)

    print("\n=== Step 4: Spatial Block CV (0.5x0.5 deg, GroupKFold) ===", flush=True)
    combined["block_lat"] = np.floor(combined["Latitude_WGS84"] / BLOCK_DEG).astype(int)
    combined["block_lon"] = np.floor(combined["Longitude_WGS84"] / BLOCK_DEG).astype(int)
    combined["block_id"] = combined["block_lat"].astype(str) + "_" + combined["block_lon"].astype(str)
    n_blocks = combined["block_id"].nunique()
    print(f"Spatial blocks: {n_blocks}", flush=True)

    X = combined[v3_features].values
    y = combined["presence"].values
    groups = combined["block_id"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    probs = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups=groups):
        m = RandomForestClassifier(**rf_params)
        m.fit(X[tr], y[tr])
        probs[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, probs)
    preds = (probs >= 0.5).astype(int)
    print(f"\nSpatial Block CV AUC: {auc:.4f}  (v3 on 1,154-site catalog: AUC=0.927)", flush=True)
    print(classification_report(y, preds, target_names=["Background", "Presence"]), flush=True)

    print("\n=== Step 5: fit final model on all data, save ===", flush=True)
    final_model = RandomForestClassifier(**rf_params)
    final_model.fit(X, y)
    joblib.dump({"model": final_model, "features": v3_features}, OUT_MODEL)
    print(f"Saved model to {OUT_MODEL}", flush=True)

    fi = pd.Series(final_model.feature_importances_, index=v3_features).sort_values(ascending=False)
    print("\nFeature importances:")
    print(fi.to_string())

    print(f"\nDONE. v4 AUC={auc:.4f} vs v3 AUC=0.927 (N=1154->1667 presence, "
          f"{BACKGROUND_RATIO}x background)", flush=True)


if __name__ == "__main__":
    main()
