import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from pyproj import Transformer
from scipy.ndimage import label, distance_transform_edt, binary_fill_holes

warnings.filterwarnings("ignore")

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "fuzzy_mcdss_model.joblib"))
GIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "gis_data", "physical"))
MASTER_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "geosites_fuzzy_mcdss_master.csv"))
OUT_GEOTIFF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "morocco_fuzzy_mcdss_accessibility.tif"))
OUT_PNG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures", "morocco_fuzzy_mcdss_accessibility_map.png"))
os.makedirs(os.path.dirname(OUT_GEOTIFF), exist_ok=True)

RASTER_FILES = {
    'Elevation_m': 'elevation_meters.tif',
    'Slope_deg': 'slope_degrees.tif',
    'Ruggedness': 'ruggedness.tif',
    'Dist_to_Dam_m': 'distance_to_dams_meters.tif',
    'Dist_to_River_m': 'distance_to_rivers_meters.tif',
    'LULC_Class': 'lulc_classes.tif',
    'Soil_Class': 'soil_classes.tif',
    'Geology_Class': 'geology_classes.tif',
    'Dist_to_Highway_m': 'distance_to_highways_meters.tif',
    'Dist_to_Piste_m': 'distance_to_pistes_meters.tif'
}

FEATURES = ['Elevation_m', 'Slope_deg', 'Ruggedness', 'Dist_to_Dam_m', 'Dist_to_River_m', 'LULC_Class', 'Soil_Class', 'Geology_Class', 'Dist_to_Highway_m', 'Dist_to_Piste_m']

def main():
    print(f"Loading trained spatial MCDSS model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    dem_path = os.path.join(GIS_DIR, RASTER_FILES['Elevation_m'])
    with rasterio.open(dem_path) as template_src:
        ref_transform = template_src.transform
        rows, cols = template_src.shape
        ref_crs = template_src.crs
        dem_data = template_src.read(1).astype(np.float32)
        nodata = template_src.nodata if template_src.nodata is not None else -9999
        
    print(f"Master DEM raster grid shape: {rows} rows × {cols} cols")
    
    raster_arrays = {}
    for feat in FEATURES:
        rpath = os.path.join(GIS_DIR, RASTER_FILES[feat])
        with rasterio.open(rpath) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr == src.nodata] = np.nan
            arr[arr < -9000] = np.nan
            
            if feat in ['Dist_to_Highway_m', 'Dist_to_Piste_m']:
                shift_rows = 17
                arr_shifted = np.full_like(arr, np.nan)
                arr_shifted[:-shift_rows] = arr[shift_rows:]
                arr_shifted[-shift_rows:] = 500000.0
                arr = arr_shifted
                
            raster_arrays[feat] = arr
            
    # Clean land mask using connected components to eliminate floating map text and border artifacts
    raw_land_mask = (np.isfinite(dem_data) & (dem_data > -500) & (dem_data != nodata)) | (np.isfinite(raster_arrays['LULC_Class']) & (raster_arrays['LULC_Class'] > -500))
    labeled_array, num_features = label(raw_land_mask)
    if num_features > 0:
        component_sizes = np.bincount(labeled_array.ravel())
        component_sizes[0] = 0  # Ignore background (label 0)
        largest_component = component_sizes.argmax()
        land_mask = (labeled_array == largest_component)
    else:
        land_mask = raw_land_mask
    land_mask = binary_fill_holes(land_mask)
    
    n_land = land_mask.sum()
    print(f"Clean contiguous Morocco land pixels for prediction: {n_land:,}")
    
    flat_features = {}
    for feat in FEATURES:
        arr = raster_arrays[feat]
        # Fill any minor internal NaNs using spatial nearest neighbor within land mask
        invalid = np.isnan(arr) & land_mask
        if np.any(invalid):
            valid_mask = (~np.isnan(arr)) & land_mask
            if np.any(valid_mask):
                indices = distance_transform_edt(~valid_mask, return_distances=False, return_indices=True)
                arr = arr[tuple(indices)]
        flat_features[feat] = arr[land_mask]
        
    df_grid = pd.DataFrame(flat_features, columns=FEATURES)
    for col in FEATURES:
        df_grid[col] = df_grid[col].fillna(df_grid[col].median()).fillna(0)
        
    print("Running spatial surrogate prediction across clean Morocco land grid...")
    pred_int = np.zeros(n_land, dtype=np.uint8)
    CHUNK = 50000
    for start in range(0, n_land, CHUNK):
        end = min(start + CHUNK, n_land)
        pred_int[start:end] = model.predict(df_grid.iloc[start:end])
        
    output_grid = np.full((rows, cols), 255, dtype=np.uint8)
    output_grid[land_mask] = pred_int
    
    # Save GeoTIFF
    os.makedirs(os.path.dirname(OUT_GEOTIFF), exist_ok=True)
    with rasterio.open(
        OUT_GEOTIFF, 'w', driver='GTiff',
        height=rows, width=cols, count=1,
        dtype=np.uint8, crs=ref_crs, transform=ref_transform, nodata=255
    ) as dst:
        dst.write(output_grid, 1)
    print(f"Exported continuous Fuzzy MCDSS GeoTIFF → {OUT_GEOTIFF}")
    
    # Render PNG Map with Refined Cartographic Palette
    print("Rendering high-aesthetic scientific publication map...")
    COLORS = {0: "#1E8A54", 1: "#F39C12", 2: "#B8332A"}  # Forest Emerald, Warm Amber, Terracotta Crimson
    
    rgb = np.ones((rows, cols, 3), dtype=np.float32)  # White background
    for cls_idx, hex_col in COLORS.items():
        r = int(hex_col[1:3], 16) / 255.0
        g = int(hex_col[3:5], 16) / 255.0
        b = int(hex_col[5:7], 16) / 255.0
        px = land_mask & (output_grid == cls_idx)
        rgb[px] = [r, g, b]
        
    # Load and project geosite points
    df_sites = pd.read_csv(MASTER_CSV)
    df_sites = df_sites.dropna(subset=["Latitude_WGS84", "Longitude_WGS84", "Fuzzy_Accessibility_Category"])
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:26191", always_xy=True)
    x_sites, y_sites = transformer.transform(df_sites["Longitude_WGS84"].values, df_sites["Latitude_WGS84"].values)
    df_sites["X_projected"] = x_sites
    df_sites["Y_projected"] = y_sites + 20000  # Shift points UP by 20,000m to align with raster
    
    fig, ax = plt.subplots(figsize=(15, 11), facecolor="white")
    ax.set_facecolor("#F8F9FA")
    
    xmin = ref_transform.c
    xmax = ref_transform.c + ref_transform.a * cols
    ymin = ref_transform.f + ref_transform.e * rows
    ymax = ref_transform.f
    
    ax.imshow(rgb, aspect="equal", extent=[xmin, xmax, ymin, ymax])
    
    site_colors = {"Easy": "#1E8A54", "Moderate": "#F39C12", "Difficult": "#B8332A"}
    site_markers = {"Easy": "o", "Moderate": "^", "Difficult": "s"}
    
    for cls in ["Easy", "Moderate", "Difficult"]:
        sub = df_sites[df_sites["Fuzzy_Accessibility_Category"] == cls]
        ax.scatter(sub["X_projected"], sub["Y_projected"],
                   c=site_colors[cls], marker=site_markers[cls],
                   edgecolors="white", linewidths=0.9,
                   s=36, zorder=6, label=f"Géosite ({cls})")
                   
    # Compute exact bounds of the cleaned land mass to avoid excess oceanic whitespace
    valid_rows, valid_cols = np.where(land_mask)
    x_min_land = ref_transform.c + ref_transform.a * valid_cols.min()
    x_max_land = ref_transform.c + ref_transform.a * valid_cols.max()
    y_max_land = ref_transform.f + ref_transform.e * valid_rows.min()
    y_min_land = ref_transform.f + ref_transform.e * valid_rows.max()
    
    margin_x = (x_max_land - x_min_land) * 0.03
    margin_y = (y_max_land - y_min_land) * 0.03
    ax.set_xlim(x_min_land - margin_x, x_max_land + margin_x)
    ax.set_ylim(y_min_land - margin_y, y_max_land + margin_y)
    
    ax.grid(True, linestyle="--", color="#BDC3C7", alpha=0.4, zorder=0)
    
    legend_patches = [
        mpatches.Patch(color="#1E8A54", label="Accès Facile (Routes Nationales / Régionales)"),
        mpatches.Patch(color="#F39C12", label="Accès Modéré (Pistes & Pentes Intermédiaires)"),
        mpatches.Patch(color="#B8332A", label="Accès Difficile (Terrain Enclavé & Reliefs Accidentés)"),
    ]
    ax.legend(handles=legend_patches, loc="lower left", fontsize=11,
              frameon=True, facecolor="white", edgecolor="#D5D8DC", framealpha=0.95,
              title="Classes d'Accessibilité (Modèle Flou AI)", title_fontsize=12)
              
    ax.set_title("Carte de Projection de l'Accessibilité des Géosites du Maroc\n"
                 "(Système d'Aide à la Décision Spatiale par IA — Fuzzy MCDSS & Ensemble Tree Distillation)",
                 fontsize=15, fontweight="bold", pad=15, color="#1A1A1A")
    ax.set_xlabel("Coordonnées Est (Lambert Nord — EPSG:26191, en mètres)", fontsize=12, labelpad=8)
    ax.set_ylabel("Coordonnées Nord (Lambert Nord — EPSG:26191, en mètres)", fontsize=12, labelpad=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved nationwide Fuzzy MCDSS map PNG → {OUT_PNG}")

if __name__ == "__main__":
    main()
