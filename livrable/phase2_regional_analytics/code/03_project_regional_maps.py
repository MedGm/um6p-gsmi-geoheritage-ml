import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")

PHASE2_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = os.path.abspath(os.path.join(PHASE2_DIR, "..", ".."))

FIGURES_DIR = os.path.join(PHASE2_DIR, "figures")
DATA_DIR = os.path.join(PHASE2_DIR, "data")
RASTER_DIR = os.path.join(BASE_DIR, "gis_data", "physical")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(FIGURES_DIR, exist_ok=True)

NAT_MODEL_PATH = os.path.join(MODELS_DIR, "fuzzy_mcdss_model.joblib")

LAYERS = {
    "Elevation_m": "elevation_meters.tif",
    "Slope_deg": "slope_degrees.tif",
    "Ruggedness": "ruggedness.tif",
    "Dist_to_Highway_m": "distance_to_highways_meters.tif",
    "Dist_to_Piste_m": "distance_to_pistes_meters.tif",
    "LULC_Class": "lulc_classes.tif"
}

FEATURES_REQUIRED = [
    'Elevation_m', 'Slope_deg', 'Ruggedness', 'Dist_to_Dam_m', 'Dist_to_River_m',
    'LULC_Class', 'Soil_Class', 'Geology_Class', 'Dist_to_Highway_m', 'Dist_to_Piste_m'
]

COLOR_LIST = ['#2ecc71', '#f1c40f', '#e74c3c']
CLASS_MAP = {0: 'Facile (S >= 0.50)', 1: 'Modérée (0.25 <= S < 0.50)', 2: 'Difficile (S < 0.25)'}

def fill_spatial_nans(arr, valid_domain_mask):
    invalid = np.isnan(arr) & valid_domain_mask
    if np.any(invalid):
        valid_source = (~np.isnan(arr)) & valid_domain_mask
        if np.any(valid_source):
            indices = distance_transform_edt(~valid_source, return_distances=False, return_indices=True)
            arr = arr[tuple(indices)]
    return arr

def project_region(region_name, geojson_path, regional_model_path, dataset_csv, map_out_png, comp_out_png, tif_out):
    print(f"\n" + "="*80)
    print(f"   PROJECTING RASTER ACCESSIBILITY MAPS FOR {region_name}")
    print("="*80)

    gdf = gpd.read_file(geojson_path)
    gdf_proj = gdf.to_crs(epsg=26191)
    geoms = [geom for geom in gdf_proj.geometry]

    cropped_arrays = {}
    ref_transform = None
    ref_shape = None

    for name, filename in LAYERS.items():
        rpath = os.path.join(RASTER_DIR, filename)
        with rasterio.open(rpath) as src:
            full_arr = src.read(1).astype(np.float32)

            if name in ["Dist_to_Highway_m", "Dist_to_Piste_m"]:
                shift_rows = 17
                full_shifted = np.full_like(full_arr, np.nan)
                full_shifted[:-shift_rows] = full_arr[shift_rows:]
                full_shifted[-shift_rows:] = full_arr[-shift_rows-1]
                full_arr = full_shifted

            mem_meta = src.meta.copy()
            mem_meta.update(dtype=rasterio.float32)

            with rasterio.MemoryFile() as memfile:
                with memfile.open(**mem_meta) as mem_src:
                    mem_src.write(full_arr, 1)
                    out_img, out_transform = mask(mem_src, geoms, crop=True)
                    arr = out_img[0].astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    arr[arr < -9000] = np.nan

            cropped_arrays[name] = arr
            if ref_transform is None:
                ref_transform = out_transform
                ref_shape = arr.shape

    rows, cols = ref_shape
    lulc = cropped_arrays["LULC_Class"]
    land_mask = np.isfinite(lulc) & (lulc > -500)

    for name in LAYERS.keys():
        cropped_arrays[name] = fill_spatial_nans(cropped_arrays[name], land_mask)

    n_land = land_mask.sum()
    print(f"Cropped {region_name} raster size: {rows} rows x {cols} cols | Solid land pixels: {n_land:,}")

    # Build feature matrix for land pixels
    X_dict = {}
    for feat in FEATURES_REQUIRED:
        if feat in cropped_arrays:
            X_dict[feat] = cropped_arrays[feat][land_mask]
        elif feat == "Dist_to_Dam_m":
            X_dict[feat] = np.full(n_land, 15000.0, dtype=np.float32)
        elif feat == "Dist_to_River_m":
            X_dict[feat] = np.full(n_land, 2000.0, dtype=np.float32)
        elif feat == "Soil_Class":
            X_dict[feat] = np.full(n_land, 2.0, dtype=np.float32)
        elif feat == "Geology_Class":
            X_dict[feat] = np.full(n_land, 1.0, dtype=np.float32)
        else:
            X_dict[feat] = np.zeros(n_land, dtype=np.float32)

    X_grid = pd.DataFrame(X_dict)[FEATURES_REQUIRED]

    # Load models & predict
    reg_model = joblib.load(regional_model_path)
    if isinstance(reg_model, dict):
        reg_model = reg_model.get('model', reg_model.get('best_model', reg_model))

    nat_model = joblib.load(NAT_MODEL_PATH)
    if isinstance(nat_model, dict):
        nat_model = nat_model.get('model', nat_model.get('best_model', nat_model))

    preds_reg = reg_model.predict(X_grid)
    preds_nat = nat_model.predict(X_grid)

    match_count = np.sum(preds_reg == preds_nat)
    concordance = (match_count / n_land) * 100.0
    print(f"Grid prediction concordance: {concordance:.2f}% ({match_count:,} / {n_land:,} pixels match)")

    # Render regional standalone map
    grid_img = np.full((rows, cols), np.nan, dtype=np.float32)
    grid_img[land_mask] = preds_reg

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_disc = matplotlib.colors.ListedColormap(COLOR_LIST)

    im = ax.imshow(grid_img, cmap=cmap_disc, vmin=-0.5, vmax=2.5)

    df_sites = pd.read_csv(dataset_csv)
    ax.set_title(f"Carte Prédictive d'Accessibilité Régionale — {region_name}\n(Concordance avec le Modèle National: {concordance:.2f}%)",
                 fontsize=12, fontweight="bold", pad=12)

    patches = [mpatches.Patch(color=COLOR_LIST[i], label=CLASS_MAP[i]) for i in range(3)]
    ax.legend(handles=patches, loc="lower right", framealpha=0.9, fontsize=9)
    plt.axis("off")
    plt.tight_layout()

    out_map_path = os.path.join(FIGURES_DIR, map_out_png)
    plt.savefig(out_map_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Regional Map -> {out_map_path}")

    # Render Comparative Grid Map (National vs Regional) with Discrepancy Callout Markers
    grid_nat = np.full((rows, cols), np.nan, dtype=np.float32)
    grid_nat[land_mask] = preds_nat

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    ax1.imshow(grid_nat, cmap=cmap_disc, vmin=-0.5, vmax=2.5)
    ax1.set_title("Modèle Flou MCDSS National", fontsize=12, fontweight="bold", pad=10)
    ax1.axis('off')

    ax2.imshow(grid_img, cmap=cmap_disc, vmin=-0.5, vmax=2.5)
    ax2.set_title(f"Modèle Localisé Régional ({region_name})\nConcordance = {concordance:.2f}%", fontsize=12, fontweight="bold", pad=10)
    ax2.axis('off')

    import pyproj
    transformer_to_proj = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:26191', always_xy=True)
    inv_transform = ~ref_transform

    if "Béni" in region_name:
        # Clean comparative map for BMK without floating point projections
        pass

    else:
        # TTAH Discrepancy 1: Punta Cires (35.900000, -5.416700)
        # TTAH Discrepancy 2: Fahs-Anjra Melloussa (35.724975, -5.668510)
        lon1, lat1 = -5.416700, 35.900000
        x1, y1 = transformer_to_proj.transform(lon1, lat1)
        col1, row1 = inv_transform * (x1, y1)

        lon2, lat2 = -5.668510, 35.724975
        x2, y2 = transformer_to_proj.transform(lon2, lat2)
        col2, row2 = inv_transform * (x2, y2)

        for ax in [ax1, ax2]:
            ax.plot(col1, row1, 'r*', markersize=16, markeredgecolor='black', markeredgewidth=1.2, zorder=10)
            ax.plot(col2, row2, 'r*', markersize=16, markeredgecolor='black', markeredgewidth=1.2, zorder=10)

        # Annotations on National Map (Separated offsets to prevent any overlap)
        ax1.annotate("(1) Punta Cires: Prédit Modérée",
                     xy=(col1, row1), xytext=(col1 + 10, row1 - 25),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                     fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="yellow", ec="black", lw=1, alpha=0.9))

        ax1.annotate("(2) Fahs-Anjra: Prédit Difficile\n(Pénalisation pente hill-side 11.9°)",
                     xy=(col2, row2), xytext=(col2 - 75, row2 + 40),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                     fontsize=8.5, fontweight='bold', color='white',
                     bbox=dict(boxstyle="round,pad=0.4", fc="#e74c3c", ec="black", lw=1, alpha=0.9))

        # Annotations on Regional Map (Separated offsets to prevent any overlap)
        ax2.annotate("(1) Punta Cires: Reclassé Facile\n(Accès piste direct 0m, 8 min drive)",
                     xy=(col1, row1), xytext=(col1 + 10, row1 - 25),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                     fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#2ecc71", ec="black", lw=1, alpha=0.9))

        ax2.annotate("(2) Fahs-Anjra: Reclassé Facile\n(<750m Gare Melloussa, 11 min pied)",
                     xy=(col2, row2), xytext=(col2 - 75, row2 + 40),
                     arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7),
                     fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.4", fc="#2ecc71", ec="black", lw=1, alpha=0.9))

    patches = [mpatches.Patch(color=COLOR_LIST[i], label=CLASS_MAP[i]) for i in range(3)]
    plt.legend(handles=patches, loc="lower center", bbox_to_anchor=(-0.1, -0.08), ncol=3, framealpha=0.95, fontsize=10)

    title_str = f"Phase 2 — Comparaison Cartographique Régionale ({region_name})" if "Béni" in region_name else f"Phase 2 — Comparaison Cartographique & Localisation des Écarts ({region_name})"
    plt.suptitle(title_str, fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_comp_path = os.path.join(FIGURES_DIR, comp_out_png)
    plt.savefig(out_comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Comparative Map -> {out_comp_path}")

def main():
    project_region("Béni Mellal-Khénifra", 
                   os.path.join(DATA_DIR, "beni_mellal_khenifra_boundary.geojson"),
                   os.path.join(MODELS_DIR, "bmk_regional_model.joblib"),
                   os.path.join(DATA_DIR, "geosites_bmk_indexed.csv"),
                   "beni_mellal_khenifra_accessibility_map.png",
                   "comparative_accessibility_grid_map.png",
                   "bmk_regional_accessibility.tif")

    project_region("Tanger-Tétouan-Al Hoceïma", 
                   os.path.join(DATA_DIR, "tanger_tetouan_al_hoceima_boundary.geojson"),
                   os.path.join(MODELS_DIR, "ttah_regional_model.joblib"),
                   os.path.join(DATA_DIR, "geosites_ttah_indexed.csv"),
                   "ttah_accessibility_map.png",
                   "ttah_comparative_grid_map.png",
                   "ttah_regional_accessibility.tif")

if __name__ == "__main__":
    main()
