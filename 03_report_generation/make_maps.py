"""
Professional GIS accessibility maps for geosite_ai_section.tex.

Fits the production Difficult/Easy models (exact hyperparameters already
selected via grid search -- refit once on the full labeled set, not a new
search). Easy is the four-member tree ensemble (Baseline features); Difficult
is GP+Infra (Gaussian Process, Infra feature set) -- 02_modeling_and_analysis/
30-31 found this significantly beats the best tree-ensemble option for
Difficult at N=1662 (McNemar p=0.020). Applies both over a real spatial grid
per region, built from:
  - local terrain rasters (archive/gis_data/physical/):
    elevation, slope, ruggedness, distance-to-highway, all EPSG:26191
  - live ESA WorldCover (same source/method as code/07, decimated windowed
    reads -- fast, no per-point fetches) for LULC friction
  - vectorized haversine distance to the 55 reference settlements (code/08's
    exact list) for Dist_to_Settlement_m
  - region boundaries fetched from OSM Nominatim (Fes-Meknes, Beni Mellal-
    Khenifra, Dakhla-Oued Ed-Dahab) and Natural Earth (national Morocco +
    Western Sahara, same source as code/01's fetch_morocco_boundary)

Run once locally (explicitly authorized for this task): fits are single
fits at known hyperparameters, not a search; only the grid feature
extraction is new engineering. All boundaries are cached to
data/boundaries/ on first fetch.
"""
import os, json, time, warnings
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
from scipy.ndimage import gaussian_filter
from rasterio.transform import xy as transform_xy
from shapely.geometry import Point
from pyproj import Transformer
import requests

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource, ListedColormap

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
import subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FW = os.path.abspath(os.path.join(HERE, ".."))
BASE = FW
OUT = os.path.join(HERE, "..", "report", "figures")
BOUND_DIR = os.path.join(FW, "data", "boundaries")
os.makedirs(OUT, exist_ok=True)
os.makedirs(BOUND_DIR, exist_ok=True)

plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}",
    "font.size": 9.5,
})

ACCENT = "#2B5F72"
HIGHLIGHT = "#C9782E"
MODERATE_COL = "#D9CBB0"
NEUTRAL = "#A6A6A6"
OUTSIDE_COL = "#EFEFEF"

FEATURES = ["Dist_to_Highway_m", "Slope_deg", "Ruggedness",
            "Elevation_m", "LULC_Friction", "Dist_to_Settlement_m"]
CONF_WEIGHT = {"High": 1.0, "Medium-High": 0.85, "Medium": 0.7, "Low-Medium": 0.55, "Low": 0.4}

WC_FRICTION = {
    10: 0.55, 20: 0.35, 30: 0.15, 40: 0.20, 50: 0.05, 60: 0.10,
    70: 0.70, 80: 0.90, 90: 0.75, 95: 0.75, 100: 0.60,
}

# Updated 2026-08-22 to Baseline_939's best_configs (results/json/training/
# phase5_modeling_results.json) -- N=733-era configs replaced for the N=939
# audited/batch-integrated dataset, confidence weighting dropped (see below).
BEST_CONFIGS = {
    "difficult": {
        "RF": dict(max_depth=None, min_samples_leaf=1, n_estimators=400),
        "XGB": dict(learning_rate=0.15, max_depth=5, n_estimators=250),
        "GBM": dict(learning_rate=0.1, max_depth=4, n_estimators=200),
        "LGBM": dict(learning_rate=0.1, max_depth=-1, n_estimators=200, num_leaves=31),
    },
    "easy": {
        "RF": dict(max_depth=None, min_samples_leaf=1, n_estimators=400),
        "XGB": dict(learning_rate=0.08, max_depth=6, n_estimators=250),
        "GBM": dict(learning_rate=0.05, max_depth=4, n_estimators=200),
        "LGBM": dict(learning_rate=0.1, max_depth=-1, n_estimators=200, num_leaves=31),
    },
}

# ============================================================ 1. Load data & fit ==
print("[1] Loading N=939 labels (733 original + 206 el_ouali_2026, audited) + fitting production ensembles ...", flush=True)
import glob
frames = []
for f in sorted(glob.glob(os.path.join(FW, "data/final/regional_label_sources/*.csv"))):
    bn = os.path.basename(f)
    if "expert_labels" not in bn or "combined" in bn:
        continue
    df = pd.read_csv(f)
    labeled = df[df["Expert_Class"].notna() & (df["Expert_Class"] != "")].copy()
    frames.append(labeled[["Locality_ID", "Expert_Class"]])
all_labels = pd.concat(frames, ignore_index=True).drop_duplicates("Locality_ID")
catalog = pd.read_csv(os.path.join(FW, "data/final/geosites_mcdm_national.csv"))
merged = all_labels.merge(
    catalog[["Locality_ID", "Region", "Latitude_WGS84", "Longitude_WGS84"] + FEATURES],
    on="Locality_ID", how="inner")
merged["Expert_Merged"] = merged["Expert_Class"].replace("Very Difficult", "Difficult")
merged = merged.dropna(subset=["Region"]).reset_index(drop=True)
assert len(merged) == 1662

# Confidence-weighting retired project-wide 2026-08-21 (class-balance weighting only) --
# see [[project-geosite-accessibility-status]] memory / 02_modeling_and_analysis/03_phase5_modeling.py.
X = merged[FEATURES].values

def make_ensemble(target_cfg):
    models = [
        ("RF", RandomForestClassifier(random_state=42, n_jobs=-1, **target_cfg["RF"])),
        ("XGB", XGBClassifier(random_state=42, eval_metric="logloss", **target_cfg["XGB"])),
        ("GBM", GradientBoostingClassifier(random_state=42, **target_cfg["GBM"])),
        ("LGBM", LGBMClassifier(random_state=42, verbosity=-1, **target_cfg["LGBM"])),
    ]
    return models

def fit_binary(y_binary, cfg):
    sw = compute_sample_weight("balanced", y_binary)
    fitted = []
    for name, mdl in make_ensemble(cfg):
        mdl.fit(X, y_binary, sample_weight=sw)
        fitted.append(mdl)
    return fitted

def predict_proba_ensemble(fitted, Xg):
    probs = np.mean([m.predict_proba(Xg)[:, 1] for m in fitted], axis=0)
    return probs

y_difficult = (merged["Expert_Merged"] == "Difficult").astype(int).values
y_easy = (merged["Expert_Merged"] == "Easy").astype(int).values
fitted_difficult = fit_binary(y_difficult, BEST_CONFIGS["difficult"])
fitted_easy = fit_binary(y_easy, BEST_CONFIGS["easy"])
print("    done.", flush=True)

# ============================================================ 1b. Deployed Difficult model
# 02_modeling_and_analysis/30-31 (2026-09-01): at N=1662, GP+Infra (0.8081)
# significantly beats the best tree-ensemble option, Tree+Infra (0.7864),
# McNemar p=0.020 -- GP+Domain scored marginally higher (0.8087) but Domain
# has no usable spatial raster (archive/cards_and_rasters/Domaines.tif has no
# georeferencing) so can't drive a grid map; GP+Infra is the practical choice
# since Infra features ARE grid-computable (region_infra_grid.py). Easy stays
# the tree ensemble (Baseline) -- it wins there under every feature set tested.
print("[1b] Fitting deployed GP+Infra Difficult model (mirrors 02_modeling_and_analysis/30-31) ...", flush=True)
infra = pd.read_csv(os.path.join(FW, "data/final/infra_features.csv"))
merged_infra = merged.merge(infra, on="Locality_ID", how="left")
SENTINEL_DIST_M = 60000.0
merged_infra["dist_nearest_tourism_poi_m"] = merged_infra["dist_nearest_tourism_poi_m"].fillna(SENTINEL_DIST_M)
merged_infra["dist_nearest_settlement_town_m"] = merged_infra["dist_nearest_settlement_town_m"].fillna(SENTINEL_DIST_M)
merged_infra["nearest_settlement_type"] = merged_infra["nearest_settlement_type"].fillna("None")
SETTLEMENT_CATS = ["None", "city", "hamlet", "town", "village"]  # matches training's alphabetical pd.get_dummies order exactly
for cat in SETTLEMENT_CATS:
    merged_infra[f"Settlement_{cat}"] = (merged_infra["nearest_settlement_type"] == cat).astype(float)
FEATURES_INFRA = FEATURES + ["n_tourism_poi_10km", "dist_nearest_tourism_poi_m", "dist_nearest_settlement_town_m"] + [f"Settlement_{c}" for c in SETTLEMENT_CATS]
infra_scaler = StandardScaler().fit(merged_infra[FEATURES_INFRA].values)
X_infra_train_scaled = infra_scaler.transform(merged_infra[FEATURES_INFRA].values)
gp_difficult = GaussianProcessClassifier(kernel=1.0 * RBF(length_scale=1.0), random_state=42, n_jobs=-1)
gp_difficult.fit(X_infra_train_scaled, y_difficult)
print("    done.", flush=True)

# ============================================================ 1b. Region-specific models
# The pooled national ensemble above is the correct model ONLY for the national map.
# Eddakhla, Fes-Meknes, and BMK are each discussed in the report text via their OWN
# region-specific/leave-region-out result (96.0%, 78.6%, 77.7%) -- this section mirrors
# code/20_final_v2_battery.py's leave_region_out() and G1 per-region methodology exactly
# (same grids, same 5x10 StratifiedGroupKFold CV, same threshold tuning) so each map is
# generated by the actual model the adjacent text describes, not a crop of the pooled
# national one. code/20 does not persist the G1 per-region hyperparameters it finds
# (only the accuracy numbers), so Fes-Meknes/BMK's grid search is re-run here; the
# resulting accuracies are printed against results/final_v2_results_N733.json as a
# reproducibility check.
print("[1b] Fitting region-specific models (mirrors code/20 G1 / leave-region-out) ...", flush=True)

with open(os.path.join(FW, "results", "json", "training", "final_v2_results_N733.json"), encoding="utf-8") as _f:
    _g1_json = json.load(_f)
_g1_lookup = {(r["region"], r["target"]): r["best_acc"] for r in _g1_json["G1_per_region_binary"]}

RF_GRID = [dict(n_estimators=ne, max_depth=md, min_samples_leaf=msl)
           for ne in [100, 200, 400] for md in [3, 4, 5, 6, 7, 8, None] for msl in [1, 2, 3, 5]]
XGB_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100, 150, 250] for md in [3, 4, 5, 6] for lr in [0.03, 0.08, 0.15]]
GBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr)
            for ne in [100, 200] for md in [2, 3, 4] for lr in [0.05, 0.1]]
LGBM_GRID = [dict(n_estimators=ne, max_depth=md, learning_rate=lr, num_leaves=nl)
             for ne in [100, 200] for md in [3, 5, -1] for lr in [0.05, 0.1] for nl in [15, 31]]
GS_MODEL_KINDS = ["RF", "XGB", "GBM", "LGBM"]
GS_GRIDS = {"RF": RF_GRID, "XGB": XGB_GRID, "GBM": GBM_GRID, "LGBM": LGBM_GRID}

def gs_make_model(kind, cfg):
    if kind == "RF":
        return RandomForestClassifier(random_state=42, n_jobs=1, **cfg)
    if kind == "XGB":
        return XGBClassifier(random_state=42, n_jobs=1, eval_metric="logloss", **cfg)
    if kind == "GBM":
        return GradientBoostingClassifier(random_state=42, **cfg)
    if kind == "LGBM":
        return LGBMClassifier(random_state=42, n_jobs=1, verbosity=-1, **cfg)

def cluster_of(sub):
    """500m haversine union-find clustering, identical to code/20's cluster_of."""
    lat, lon = sub["Latitude_WGS84"].values, sub["Longitude_WGS84"].values
    n = len(sub)
    R = 6371000
    lr, lo = np.radians(lat), np.radians(lon)
    dlat = lr[:, None] - lr[None, :]
    dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lr[:, None]) * np.cos(lr[None, :]) * np.sin(dlon / 2) ** 2
    D = 2 * R * np.arcsin(np.sqrt(a))
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= 500:
                rx, ry = find(i), find(j)
                if rx != ry:
                    parent[rx] = ry
    return np.array([find(i) for i in range(n)])

def gs_combined_weight(y_tr, conf_tr):
    return conf_tr * compute_sample_weight("balanced", y_tr)

def gs_fit_predict_proba(kind, cfg, Xr, yr, cwr, tr, te):
    m = gs_make_model(kind, cfg)
    sw = gs_combined_weight(yr[tr], cwr[tr])
    if kind == "XGB":
        pos_w, neg_w = sw[yr[tr] == 1].sum(), sw[yr[tr] == 0].sum()
        if pos_w > 0:
            m.set_params(scale_pos_weight=neg_w / pos_w)
    m.fit(Xr[tr], yr[tr], sample_weight=sw)
    return m.predict(Xr[te]), m.predict_proba(Xr[te])

def gs_one_fit_acc(kind, cfg, Xr, yr, cwr, tr, te):
    pred, _ = gs_fit_predict_proba(kind, cfg, Xr, yr, cwr, tr, te)
    return accuracy_score(yr[te], pred)

def region_grid_search(Xr, yr, cwr, groups, n_repeats=5, n_splits=10):
    tasks = []
    for kind in GS_MODEL_KINDS:
        for cfg in GS_GRIDS[kind]:
            for rep in range(n_repeats):
                skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=rep)
                for tr, te in skf.split(Xr, yr, groups=groups):
                    tasks.append((kind, cfg, tr, te))
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(gs_one_fit_acc)(kind, cfg, Xr, yr, cwr, tr, te) for kind, cfg, tr, te in tasks)
    from collections import defaultdict
    agg = defaultdict(list)
    for (kind, cfg, tr, te), acc in zip(tasks, results):
        agg[(kind, json.dumps(cfg, sort_keys=True))].append(acc)
    rows = [(k, json.loads(c), np.mean(v)) for (k, c), v in agg.items()]
    rows.sort(key=lambda r: -r[2])
    best = {}
    for kind, cfg, acc in rows:
        if kind not in best:
            best[kind] = cfg
    return [(k, best[k]) for k in GS_MODEL_KINDS]

def region_logo_cluster_cv_proba(best_cfgs, Xr, yr, cwr, groups):
    folds = list(LeaveOneGroupOut().split(Xr, yr, groups=groups))
    def fold_proba(tr, te):
        probs = [gs_fit_predict_proba(k, c, Xr, yr, cwr, tr, te)[1] for k, c in best_cfgs]
        return te, np.mean(probs, axis=0)
    results = Parallel(n_jobs=-1, backend="loky")(delayed(fold_proba)(tr, te) for tr, te in folds)
    proba = np.zeros((len(yr), 2))
    for te, p in results:
        proba[te] = p
    return proba

def region_best_threshold(y_true, proba_pos):
    best_t, best_ba = 0.5, balanced_accuracy_score(y_true, (proba_pos >= 0.5).astype(int))
    for t in np.arange(0.05, 0.96, 0.01):
        ba = balanced_accuracy_score(y_true, (proba_pos >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t, best_ba

def fit_binary_subset(Xs, ys, cws, cfg):
    sw = cws * compute_sample_weight("balanced", ys)
    fitted = []
    for name, mdl in make_ensemble(cfg):
        mdl.fit(Xs, ys, sample_weight=sw)
        fitted.append(mdl)
    return fitted

def fit_region_specific(region_name, target):
    """Own-region grid search + threshold tuning + refit-once-on-full-region-data,
    exactly mirroring code/20's G1 block. Returns (fitted_ensemble, threshold, best_acc)."""
    sub = merged[merged["Region"] == region_name].reset_index(drop=True)
    Xr = sub[FEATURES].values
    # Confidence-weighting retired project-wide (see line ~125) -- this function
    # predates that and was left referencing a "Confidence" column that no longer
    # exists in `merged`, never caught because the cache-reuse shortcut below
    # always skipped this path in practice. Neutral weight matches every other
    # script's post-retirement behavior (class-balance weighting only, applied
    # inside fit_binary_subset).
    cwr = np.ones(len(sub))
    yr = (sub["Expert_Merged"] == target).astype(int).values
    groups = cluster_of(sub)
    best = region_grid_search(Xr, yr, cwr, groups)
    proba = region_logo_cluster_cv_proba(best, Xr, yr, cwr, groups)
    acc_default = accuracy_score(yr, (proba[:, 1] >= 0.5).astype(int))
    t_opt, _ = region_best_threshold(yr, proba[:, 1])
    acc_tuned = accuracy_score(yr, (proba[:, 1] >= t_opt).astype(int))
    threshold = t_opt if acc_tuned > acc_default else 0.5
    best_acc = max(acc_default, acc_tuned)
    reported = _g1_lookup.get((region_name, target))
    print(f"    {region_name}/{target}: N={len(yr)} best_acc={best_acc:.3f} (threshold={threshold:.2f}) "
          f"-- reported in final_v2_results_N733.json: {reported}", flush=True)
    fitted = fit_binary_subset(Xr, yr, cwr, dict(best))
    return fitted, threshold, best_acc

# 2026-08-22: Paper 1 (national) update only -- this region-specific fitting
# (Eddakhla leave-region-out, Fés-Meknés/BMK own-region grid search) feeds
# only the regional maps, which are Paper 2's deferred scope. Skip the
# expensive refit when a cached map_grids.pkl already exists; the national
# block below never touches these variables.
_map_grids_cache = os.path.join(FW, "results", "grids", "map_grids.pkl")
_reuse_regional_cache = os.path.exists(_map_grids_cache)
if not _reuse_regional_cache:
    print("  Eddakhla-Oued Eddahab (leave-region-out, national hyperparameters) ...", flush=True)
    _mask_not_eddakhla = (merged["Region"] != "Eddakhla-Oued Eddahab").values
    fitted_eddakhla_diff = fit_binary_subset(X[_mask_not_eddakhla], y_difficult[_mask_not_eddakhla],
                                              compute_sample_weight("balanced", y_difficult[_mask_not_eddakhla]),
                                              BEST_CONFIGS["difficult"])
    fitted_eddakhla_easy = fit_binary_subset(X[_mask_not_eddakhla], y_easy[_mask_not_eddakhla],
                                              compute_sample_weight("balanced", y_easy[_mask_not_eddakhla]),
                                              BEST_CONFIGS["easy"])

    print("  Fes-Meknes (own-region grid search + threshold tuning) ...", flush=True)
    fitted_fesmeknes_diff, thresh_fesmeknes_diff, _ = fit_region_specific("Fés-Meknés", "Difficult")
    fitted_fesmeknes_easy, thresh_fesmeknes_easy, _ = fit_region_specific("Fés-Meknés", "Easy")

    print("  Beni Mellal-Khenifra (own-region grid search + threshold tuning) ...", flush=True)
    fitted_bmk_diff, thresh_bmk_diff, _ = fit_region_specific("Béni Mellal-Khénifra", "Difficult")
    fitted_bmk_easy, thresh_bmk_easy, _ = fit_region_specific("Béni Mellal-Khénifra", "Easy")
    print("    done.", flush=True)
else:
    print("  Reusing cached regional models (Paper 1/national update only) -- skipping region-specific refit.", flush=True)

# ============================================================ 2. Boundaries =======
def fetch_boundary(cache_name, query):
    path = os.path.join(BOUND_DIR, f"{cache_name}.geojson")
    if os.path.exists(path):
        return gpd.read_file(path)
    r = requests.get("https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "geojson", "polygon_geojson": 1, "limit": 1},
        headers={"User-Agent": "geosite-research-project/1.0 (academic use)"})
    r.raise_for_status()
    d = r.json()
    gdf = gpd.GeoDataFrame.from_features(d["features"], crs="EPSG:4326")
    gdf.to_file(path, driver="GeoJSON")
    time.sleep(1.1)
    return gdf

def fetch_national_boundary():
    path = os.path.join(BOUND_DIR, "national.geojson")
    if os.path.exists(path):
        return gpd.read_file(path)
    url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    morocco = world[world["ADMIN"].isin(["Morocco", "Western Sahara"])].to_crs("EPSG:4326")
    morocco.to_file(path, driver="GeoJSON")
    return morocco

print("[2] Fetching region boundaries ...", flush=True)
REGIONS = {
    "eddakhla": dict(label="Eddakhla-Oued Eddahab", query="Dakhla-Oued Ed-Dahab, Morocco", n_grid=130),
    "fesmeknes": dict(label="Fés-Meknés", query="Fès-Meknès, Morocco", n_grid=140),
    "bmk": dict(label="Béni Mellal-Khénifra", query="Béni Mellal-Khénifra, Morocco", n_grid=130),
}
for key, meta in REGIONS.items():
    meta["gdf"] = fetch_boundary(key, meta["query"])
national_gdf = fetch_national_boundary()
print("    done.", flush=True)

# ============================================================ 3. Raster stack =====
print("[3] Loading local terrain raster stack ...", flush=True)
# Elevation/Slope/Ruggedness read from physical_task2_corrected/, not physical/ --
# see 03_report_generation/make_paper2_region_maps.py's raster-loading block for the full
# investigation. Confirmed concretely: physical/elevation_meters.tif missed with
# raw -9999 nodata on 33% of real catalog site queries and was off by 500-1100m
# even where valid; physical_task2_corrected/ (documented as "registration-
# corrected via coastline calibration", and what code/02_extract_terrain_road_
# features.py actually used to build the CATALOG features the models were trained
# and evaluated on) + nodata-fill brings that to a ~130-200m median bias, in line
# with ordinary coarse-DEM-vs-ground-truth variation at 1.2km resolution.
# Dist_to_Highway_m has no corrected raster (only a "_PRE_CALIBRATION" file exists
# there) because the catalog computes it differently -- true vector distance to
# roads, not raster sampling -- so it stays sourced from physical/, which checked
# out fine against the catalog (quantization-scale differences only).
PHYS = os.path.join(BASE, "archive/gis_data/physical_task2_corrected")
PHYS_OLD = os.path.join(BASE, "archive/gis_data/physical")
raster_arrays = {}
raster_transforms = {}  # per-key: the corrected rasters use a different affine
                         # origin than the old ones (~24.85km Y-shift -- exactly
                         # the coastline calibration), so they are NOT
                         # interchangeable under one shared transform.
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
print("    done.", flush=True)

to_26191 = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
inv_transforms = {k: ~t for k, t in raster_transforms.items()}

def sample_local_stack(lon, lat):
    x, y = to_26191.transform(lon, lat)
    out = {}
    for key in ["Elevation_m", "Slope_deg", "Ruggedness", "Dist_to_Highway_m"]:
        col, row = inv_transforms[key] * (x, y)
        row_i = np.clip(np.round(row).astype(int), 0, raster_arrays[key].shape[0] - 1)
        col_i = np.clip(np.round(col).astype(int), 0, raster_arrays[key].shape[1] - 1)
        out[key] = raster_arrays[key][row_i, col_i]
    return out

# ============================================================ 4. Settlements ======
cities = pd.read_csv(os.path.join(BASE, "data/archive/pipeline_intermediates/morocco_reference_cities_geocoded.csv"))
clat, clon = cities["Latitude"].values, cities["Longitude"].values

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def dist_to_settlement(lon, lat):
    d = np.stack([haversine_m(lat, lon, la, lo) for la, lo in zip(clat, clon)], axis=-1)
    return d.min(axis=-1)

# ============================================================ 5. WorldCover =======
def tile_key(lat, lon):
    return int(np.floor(lat / 3) * 3), int(np.floor(lon / 3) * 3)

def tile_url(lat_tile, lon_tile):
    ns = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{-lat_tile:02d}"
    ew = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{-lon_tile:03d}"
    name = f"ESA_WorldCover_10m_2021_v200_{ns}{ew}_Map"
    return f"/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif"

def worldcover_friction_grid(lon_min, lat_min, lon_max, lat_max, ny, nx):
    """Fetch WorldCover class over the bbox at (ny, nx) via decimated windowed reads,
    tiling across 3-degree WorldCover tiles if the bbox spans more than one."""
    lat_tiles = sorted(set(int(np.floor(v / 3) * 3) for v in [lat_min, lat_max]))
    lon_tiles = sorted(set(int(np.floor(v / 3) * 3) for v in [lon_min, lon_max]))
    full = np.full((ny, nx), 60, dtype=np.uint8)  # default: bare/sparse
    lon_edges = np.linspace(lon_min, lon_max, nx + 1)
    lat_edges = np.linspace(lat_max, lat_min, ny + 1)  # north to south
    for lt in lat_tiles:
        for ln in lon_tiles:
            try:
                with rasterio.open(tile_url(lt, ln)) as src:
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
                    arr = src.read(1, window=win, out_shape=(sub_h, sub_w), resampling=Resampling.mode)
                    full[j0:j0 + sub_h, i0:i0 + sub_w] = arr
            except Exception as e:
                print(f"    WorldCover tile ({lt},{ln}) failed: {e}", flush=True)
    friction = np.vectorize(lambda c: WC_FRICTION.get(int(c), 0.3))(full)
    return friction.astype(float)

# ============================================================ 6. Hillshade ========
def hillshade(elev, dx=1183.4, dy=1183.4, azdeg=315, altdeg=45):
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    e = np.where(np.isfinite(elev), elev, np.nanmin(elev))
    return ls.hillshade(e, vert_exag=1.5, dx=dx, dy=dy)

# ============================================================ 7. Region pipeline ==
def build_region_grid(gdf, n_grid):
    minx, miny, maxx, maxy = gdf.total_bounds
    aspect = (maxx - minx) / max(1e-6, (maxy - miny))
    if aspect >= 1:
        nx, ny = n_grid, max(20, int(n_grid / aspect))
    else:
        nx, ny = max(20, int(n_grid * aspect)), n_grid
    lon_edges = np.linspace(minx, maxx, nx + 1)
    lat_edges = np.linspace(miny, maxy, ny + 1)
    lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
    lon2d, lat2d = np.meshgrid(lon_c, lat_c)  # lat2d[0] = south row
    return lon2d, lat2d, (minx, miny, maxx, maxy), nx, ny

def point_in_polygon_mask(lon2d, lat2d, gdf):
    union = gdf.union_all()
    path_list = []
    geoms = [union] if union.geom_type != "MultiPolygon" else list(union.geoms)
    mask = np.zeros(lon2d.shape, dtype=bool)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    for geom in geoms:
        xx, yy = geom.exterior.xy
        p = MplPath(np.column_stack([xx, yy]))
        inside = p.contains_points(pts)
        mask |= inside.reshape(lon2d.shape)
    return mask

if not _reuse_regional_cache:
    REGION_MODELS = {
        "eddakhla": dict(diff=fitted_eddakhla_diff, easy=fitted_eddakhla_easy, t_diff=0.5, t_easy=0.5),
        "fesmeknes": dict(diff=fitted_fesmeknes_diff, easy=fitted_fesmeknes_easy,
                           t_diff=thresh_fesmeknes_diff, t_easy=thresh_fesmeknes_easy),
        "bmk": dict(diff=fitted_bmk_diff, easy=fitted_bmk_easy, t_diff=thresh_bmk_diff, t_easy=thresh_bmk_easy),
    }
else:
    REGION_MODELS = {}  # unused: run_region() is never called when reusing the cache

def run_region(key, meta):
    print(f"[region] {meta['label']} ...", flush=True)
    gdf = meta["gdf"]
    lon2d, lat2d, bounds, nx, ny = build_region_grid(gdf, meta["n_grid"])
    minx, miny, maxx, maxy = bounds

    local = sample_local_stack(lon2d, lat2d)
    friction = worldcover_friction_grid(minx, miny, maxx, maxy, ny, nx)
    dsettle = dist_to_settlement(lon2d, lat2d)

    Xg = np.column_stack([
        local["Dist_to_Highway_m"].ravel(), local["Slope_deg"].ravel(), local["Ruggedness"].ravel(),
        local["Elevation_m"].ravel(), friction.ravel(), dsettle.ravel(),
    ])
    rm = REGION_MODELS[key]
    p_diff = predict_proba_ensemble(rm["diff"], Xg).reshape(lon2d.shape)
    p_easy = predict_proba_ensemble(rm["easy"], Xg).reshape(lon2d.shape)

    # Light display-only smoothing of the probability surfaces before thresholding into
    # classes. The own-region models (Fes-Meknes N=322, BMK N=157) produce a noticeably
    # noisier per-pixel classification than the pooled N=733 national model, since a
    # smaller training set gives less stable per-cell probabilities -- this does not
    # change the reported accuracy numbers (those come from held-out CV on the labeled
    # sites, computed above, not from this grid) or any per-site prediction; it only
    # reduces salt-and-pepper noise in the rendered map for readability.
    p_diff = gaussian_filter(p_diff, sigma=1.1)
    p_easy = gaussian_filter(p_easy, sigma=1.1)

    cls = np.full(lon2d.shape, 1, dtype=int)  # 0=Easy,1=Moderate,2=Difficult
    cls[p_diff >= rm["t_diff"]] = 2
    cls[(p_diff < rm["t_diff"]) & (p_easy >= rm["t_easy"])] = 0
    # No coarse-grid inside/outside masking here (previously `cls = np.where(inside,
    # cls, -1)`) -- that coarse cell-center-in-polygon test is imperfect right at the
    # boundary (grid too coarse relative to the true vector shape) and left visible
    # gaps of unmasked hillshade peeking through near edges even after the render-time
    # clip_path fix. The classification is now computed for the full grid and the
    # render script clips it to the EXACT polygon boundary at draw time instead --
    # correct regardless of grid resolution, with no gap and no bleed.

    return dict(lon2d=lon2d, lat2d=lat2d, bounds=bounds, cls=cls,
                elev=local["Elevation_m"], gdf=gdf)

# 2026-08-22: Paper 1 (national) update only -- the regional loop below
# (eddakhla/fesmeknes/bmk: WorldCover tile fetching + per-region grid search)
# is Paper 2's scope, deferred by the user's own sequencing. Reuse the
# existing cached regional entries unchanged rather than rerunning that
# network-dependent, slower pipeline for no reason right now.
import pickle as _pickle
_cache_path = os.path.join(FW, "results", "grids", "map_grids.pkl")
if os.path.exists(_cache_path):
    with open(_cache_path, "rb") as _f:
        _cached = _pickle.load(_f)
    results = _cached["results"]
    print(f"[region] Reusing cached regional entries: {list(results.keys())}", flush=True)
else:
    results = {}
    for key, meta in REGIONS.items():
        results[key] = run_region(key, meta)

print("[region] National ...", flush=True)
lon2d, lat2d, bounds, nx, ny = build_region_grid(national_gdf, 170)
minx, miny, maxx, maxy = bounds
local = sample_local_stack(lon2d, lat2d)
friction = worldcover_friction_grid(minx, miny, maxx, maxy, ny, nx)
dsettle = dist_to_settlement(lon2d, lat2d)
Xg = np.column_stack([
    local["Dist_to_Highway_m"].ravel(), local["Slope_deg"].ravel(), local["Ruggedness"].ravel(),
    local["Elevation_m"].ravel(), friction.ravel(), dsettle.ravel(),
])

# Difficult layer uses the deployed GP+Infra model (see 1b above), not the tree
# ensemble -- needs Infra features (tourism-POI density, settlement distance/type)
# gridded nationally via region_infra_grid.py. Rewritten 2026-09-01 to stream
# via pyosmium instead of pyrosm's bbox-filtered GeoDataFrame path (which OOM'd
# and even segfaulted on quarter-country-scale bboxes); the pyosmium version
# processes the whole country in ~98s at negligible memory, so this is a
# single direct call, no tiling needed.
print("  extracting national infra grid (tourism POI / settlement, via OSM PBF) ...", flush=True)
PBF_PATH = os.path.join(FW, "data/osm/morocco-latest.osm.pbf")
_infra_tmp = "/tmp/national_infra_grid"
os.makedirs(_infra_tmp, exist_ok=True)
np.save(os.path.join(_infra_tmp, "lon2d.npy"), lon2d)
np.save(os.path.join(_infra_tmp, "lat2d.npy"), lat2d)
_out_npz = os.path.join(_infra_tmp, "national.npz")
_r = subprocess.run([sys.executable, os.path.join(HERE, "region_infra_grid.py"),
                      str(minx - 0.2), str(miny - 0.2), str(maxx + 0.2), str(maxy + 0.2),
                      os.path.join(_infra_tmp, "lon2d.npy"), os.path.join(_infra_tmp, "lat2d.npy"),
                      _out_npz, PBF_PATH], capture_output=True, text=True, timeout=600)
print(f"    worker: {_r.stdout.strip()} {_r.stderr.strip()[-300:] if _r.returncode else ''}", flush=True)
if _r.returncode != 0 or not os.path.exists(_out_npz):
    raise RuntimeError("National infra grid extraction failed -- cannot render the GP+Infra Difficult layer.")
_infra_grid = np.load(_out_npz)
n_tourism_g = _infra_grid["n_tourism_poi_10km"]
dist_tourism_g = np.where(np.isfinite(_infra_grid["dist_nearest_tourism_poi_m"]), _infra_grid["dist_nearest_tourism_poi_m"], SENTINEL_DIST_M)
dist_settle_g = np.where(np.isfinite(_infra_grid["dist_nearest_settlement_town_m"]), _infra_grid["dist_nearest_settlement_town_m"], SENTINEL_DIST_M)
settle_code_g = _infra_grid["settlement_type_code"]
_rank_to_cat = {0: "None", 1: "hamlet", 2: "village", 3: "town", 4: "city"}
settle_onehot_g = {cat: np.zeros(lon2d.shape, dtype=float) for cat in SETTLEMENT_CATS}
for rank, cat in _rank_to_cat.items():
    settle_onehot_g[cat][settle_code_g == rank] = 1.0
Xg_infra = np.column_stack([
    Xg,
    n_tourism_g.ravel(), dist_tourism_g.ravel(), dist_settle_g.ravel(),
] + [settle_onehot_g[cat].ravel() for cat in SETTLEMENT_CATS])
Xg_infra_scaled = infra_scaler.transform(Xg_infra)
p_diff = gp_difficult.predict_proba(Xg_infra_scaled)[:, 1].reshape(lon2d.shape)
p_easy = predict_proba_ensemble(fitted_easy, Xg).reshape(lon2d.shape)
cls = np.full(lon2d.shape, 1, dtype=int)
cls[p_diff >= 0.5] = 2
cls[(p_diff < 0.5) & (p_easy >= 0.5)] = 0
# No inside/outside masking -- see run_region's comment above; the render
# script's clip_path handles the exact boundary shaping instead.
results["national"] = dict(lon2d=lon2d, lat2d=lat2d, bounds=(minx, miny, maxx, maxy),
                            cls=cls, elev=local["Elevation_m"], gdf=national_gdf)
print("    all regions done.", flush=True)

os.makedirs(os.path.join(FW, "results", "json", "other"), exist_ok=True)
json.dump({"generated": True}, open(os.path.join(FW, "results", "json", "other", "maps_generated.json"), "w"))
print("Pipeline stage complete. Rendering happens in make_maps_render.py", flush=True)

import pickle
os.makedirs(os.path.join(FW, "results", "grids"), exist_ok=True)
with open(os.path.join(FW, "results", "grids", "map_grids.pkl"), "wb") as f:
    pickle.dump({"results": results, "merged": merged}, f)
print("Saved grid results to results/grids/map_grids.pkl", flush=True)
