"""
Reconstructs and retrains the geosite geological-favorability pilot (report Section
"Exploratory Pilot: Geosite Geological Favorability Mapping", AUC=0.815), which had no
saved script (README/independent-review-documented reproducibility gap -- built as
inline analysis). Methodology reconstructed directly from the report text
(report/geosite_accessibility_report.tex, "sec:location" section) and verified against
its stated numbers before trusting it:

- Presence class: catalogued geosites (report: 324; this run: all 1154 in the current
  catalog, since the 830 new localities added 2026-08-09 are real geosite locations
  even without accessibility labels -- this model only needs "is this a real geosite",
  which the new sites qualify for).
- Background class: random points across Morocco sampled from the national geology
  raster's valid-data extent, at 5x the presence count.
- 5 covariates: Geology_Class, Soil_Class (sampled from
  collected_data/worktree_task2_corrected/gis_data/physical/{geology,soil}_classes.tif
  -- confirmed the correct raster set by reproducing the report's exact stated nodata
  dropout counts, 94/324 geology and 67/324 soil, before using it for anything else),
  Elevation_m, Slope_deg, Ruggedness (reused directly from the catalog's existing
  Copernicus-sourced columns for presence points; live-sampled the same way for
  background points, for full presence/background consistency).
- Points with nodata (-9999) in either Geology_Class or Soil_Class are dropped
  (matches the report's stated procedure).
- Random Forest classifier, Spatial Block CV: 0.5x0.5 degree blocks via GroupKFold
  (block size explicitly stated in the report; n_splits and RF hyperparameters were NOT
  stated in the report text, so reasonable, documented defaults are used here -- this
  is disclosed as a reconstruction, not a bit-exact reproduction, appropriate for an
  already-exploratory pilot).
"""
import os
import importlib.util
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
OUT_MODEL = os.path.join(BASE, "models", "geosite_location_pilot_model_v2.joblib")
OUT_CSV = os.path.join(BASE, "data", "model_outputs", "geosite_presence_background_pilot_v2.csv")

PROJ_CRS = "EPSG:26191"
BACKGROUND_RATIO = 5
BLOCK_DEG = 0.5
N_SPLITS = 5
RANDOM_STATE = 42


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


def main():
    print("=== Step 1: presence points (current catalog, N includes the 2026-08-09 additions) ===", flush=True)
    catalog = pd.read_csv(CATALOG_CSV)
    presence = catalog[["Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84",
                         "Elevation_m", "Slope_deg", "Ruggedness"]].copy()
    presence["presence"] = 1
    print(f"Presence points: {len(presence)}", flush=True)

    transformer = Transformer.from_crs("EPSG:4326", PROJ_CRS, always_xy=True)
    px, py = transformer.transform(presence["Longitude_WGS84"].values, presence["Latitude_WGS84"].values)

    with rasterio.open(GEOLOGY_TIF) as src:
        presence["Geology_Class"] = [v[0] for v in src.sample(list(zip(px, py)))]
        geology_nodata = src.nodata
        geology_bounds = src.bounds
        geology_transform = src.transform
        geology_shape = (src.height, src.width)
        geology_full = src.read(1)
    with rasterio.open(SOIL_TIF) as src:
        presence["Soil_Class"] = [v[0] for v in src.sample(list(zip(px, py)))]
        soil_nodata = src.nodata

    n_geo_nodata = (presence["Geology_Class"] == geology_nodata).sum()
    n_soil_nodata = (presence["Soil_Class"] == soil_nodata).sum()
    print(f"Presence nodata: Geology {n_geo_nodata}/{len(presence)}, Soil {n_soil_nodata}/{len(presence)}", flush=True)

    print("\n=== Step 2: background points (5x presence, from geology's valid-data extent) ===", flush=True)
    # The geology raster's own non-nodata mask extends to ~lon -21.7 -- well past any real
    # Moroccan coastline (a data artifact in this old scanned/georeferenced source, confirmed
    # by direct inspection: valid pixels found ~800km out in the Atlantic). Constrain to the
    # real Morocco+Western-Sahara boundary too, reusing 01's already-validated fetch, same
    # 0.1deg buffer convention as its border_check -- otherwise background points land in the
    # ocean, where Copernicus DEM has no coverage (explains the wall of tile 404s on the first
    # run) and which isn't valid "background terrain" for this model regardless.
    spec = importlib.util.spec_from_file_location("c01", os.path.join(HERE, "01_consolidate_geosite_catalog.py"))
    c01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c01)
    print("Fetching Morocco+WS boundary to constrain background sampling...", flush=True)
    morocco = c01.fetch_morocco_boundary().union_all().buffer(0.1)

    valid_mask = geology_full != geology_nodata
    valid_rows, valid_cols = np.where(valid_mask)
    to_wgs84 = Transformer.from_crs(PROJ_CRS, "EPSG:4326", always_xy=True)

    n_background = BACKGROUND_RATIO * len(presence)
    rng = np.random.default_rng(RANDOM_STATE)
    bg_lon_list, bg_lat_list = [], []
    # oversample in batches and filter to Morocco's real boundary until we have enough
    batch_size = n_background * 2
    attempts = 0
    while len(bg_lon_list) < n_background and attempts < 10:
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
    print(f"Background points within Morocco+WS boundary: {len(bg_lon)} (target {n_background}, {attempts} sampling rounds)", flush=True)
    from pyproj import Transformer as _T
    to_proj = _T.from_crs("EPSG:4326", PROJ_CRS, always_xy=True)
    bg_x, bg_y = to_proj.transform(bg_lon, bg_lat)

    n_bg_actual = len(bg_lon)
    background = pd.DataFrame({
        "Locality_ID": [f"bg_{i:05d}" for i in range(n_bg_actual)],
        "Geosite_Name": "background",
        "Latitude_WGS84": bg_lat, "Longitude_WGS84": bg_lon,
        "presence": 0,
    })
    with rasterio.open(GEOLOGY_TIF) as src:
        background["Geology_Class"] = [v[0] for v in src.sample(list(zip(bg_x, bg_y)))]
    with rasterio.open(SOIL_TIF) as src:
        background["Soil_Class"] = [v[0] for v in src.sample(list(zip(bg_x, bg_y)))]

    print(f"Sampling Copernicus DEM for {n_bg_actual} background points (tile-batched)...", flush=True)
    bg_elev, bg_slope, bg_rugged = sample_dem_batch(background["Latitude_WGS84"].values, background["Longitude_WGS84"].values)
    background["Elevation_m"], background["Slope_deg"], background["Ruggedness"] = bg_elev, bg_slope, bg_rugged

    n_geo_nodata_bg = (background["Geology_Class"] == geology_nodata).sum()
    n_soil_nodata_bg = (background["Soil_Class"] == soil_nodata).sum()
    print(f"Background nodata: Geology {n_geo_nodata_bg}/{len(background)}, Soil {n_soil_nodata_bg}/{len(background)}", flush=True)

    print("\n=== Step 3: combine, drop nodata ===", flush=True)
    cols = ["Locality_ID", "Geosite_Name", "Latitude_WGS84", "Longitude_WGS84",
            "Geology_Class", "Soil_Class", "Elevation_m", "Slope_deg", "Ruggedness", "presence"]
    combined = pd.concat([presence[cols], background[cols]], ignore_index=True)
    n_before = len(combined)
    combined = combined[
        (combined["Geology_Class"] != geology_nodata) & (combined["Soil_Class"] != soil_nodata) &
        combined[["Elevation_m", "Slope_deg", "Ruggedness"]].notna().all(axis=1)
    ].reset_index(drop=True)
    print(f"Dropped {n_before - len(combined)} rows with nodata -> {len(combined)} remain "
          f"({(combined['presence']==1).sum()} presence, {(combined['presence']==0).sum()} background)", flush=True)

    combined.to_csv(OUT_CSV, index=False)

    print("\n=== Step 4: Spatial Block CV (0.5x0.5 deg, GroupKFold) ===", flush=True)
    combined["block_lat"] = np.floor(combined["Latitude_WGS84"] / BLOCK_DEG).astype(int)
    combined["block_lon"] = np.floor(combined["Longitude_WGS84"] / BLOCK_DEG).astype(int)
    combined["block_id"] = combined["block_lat"].astype(str) + "_" + combined["block_lon"].astype(str)
    n_blocks = combined["block_id"].nunique()
    print(f"Spatial blocks: {n_blocks}", flush=True)

    FEATURES = ["Geology_Class", "Soil_Class", "Elevation_m", "Slope_deg", "Ruggedness"]
    X = combined[FEATURES].values
    y = combined["presence"].values
    groups = combined["block_id"].values

    gkf = GroupKFold(n_splits=N_SPLITS)
    probs = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups=groups):
        m = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X[tr], y[tr])
        probs[te] = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y, probs)
    preds = (probs >= 0.5).astype(int)
    print(f"\nSpatial Block CV AUC: {auc:.4f}", flush=True)
    print(classification_report(y, preds, target_names=["Background", "Presence"]), flush=True)

    print("\n=== Step 5: fit final model on all data, save ===", flush=True)
    final_model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    final_model.fit(X, y)
    joblib.dump(final_model, OUT_MODEL)
    print(f"Saved model to {OUT_MODEL}", flush=True)

    fi = pd.Series(final_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importances:")
    print(fi.to_string())

    print(f"\nDONE. AUC={auc:.4f} (report's original pilot, N=324 presence: AUC=0.815)", flush=True)


if __name__ == "__main__":
    main()
