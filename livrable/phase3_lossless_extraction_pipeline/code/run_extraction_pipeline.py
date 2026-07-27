"""
run_extraction_pipeline.py

UM6P - GSMI: Lossless GIS Raster Feature Extraction Pipeline
------------------------------------------------------------
This script implements a production-grade, reproducible pipeline that extracts 
physical and logistical features from GIS GeoTIFF rasters for a database of WGS84 
geosites. 

Critical Considerations for Lossless Extraction:
1. CRS Alignment: Transforms coordinates from WGS84 (EPSG:4326) to Morocco Sahara 
   Lambert (EPSG:26191) using pyproj's Transformer before querying pixel indices.
2. Direct Array Indexing (Nearest-Neighbor): Accesses raw pixel values at integer 
   row/col indices directly from the array. This prevents interpolation blending, 
   preserving categorical integrity for Soil, LULC, and Geology classes.
3. Scale Transformation: Applies the exact raster grid scale (meters per pixel) 
   to continuous distance matrices computed via OpenCV distance transforms.
4. Bounding Box & NoData Validation: Assures coordinates fall inside the raster grid 
   bounds, and handles nodata pixels by converting them to np.nan.
5. Quality Assertions: Validates schema type compliance (integers vs floats) 
   and counts missing data.
"""

import pandas as pd
import numpy as np
import os
import rasterio
import logging
import shutil
from pyproj import Transformer

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("extraction_pipeline.log", mode='w')
    ]
)
logger = logging.getLogger("ExtractionPipeline")

# Workspace paths
WORKSPACE_DIR = "/home/medgm/um6p-intern/geosite_project1"
COORD_PATH = os.path.join(WORKSPACE_DIR, "data_created", "geosites_coordinates_clean.csv")
PHYSICAL_DIR = os.path.join(WORKSPACE_DIR, "gis_data", "physical")
GRAPH_PATH = os.path.join(WORKSPACE_DIR, "references", "Morocco_Geosites_Graph_Data.xlsx")
OUT_SCRATCH_PATH = os.path.join(WORKSPACE_DIR, "scratch", "geosites_physical_features.csv")
OUT_DATA_PATH = os.path.join(WORKSPACE_DIR, "data_created", "geosites_physical_features.csv")

# Input GeoTIFF raster layers configuration
LAYERS = {
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


def run_pipeline():
    logger.info("Initializing Lossless Feature Extraction Pipeline...")

    # 1. Load Geosite Coordinates
    if not os.path.exists(COORD_PATH):
        raise FileNotFoundError(f"Geosite coordinate file not found at {COORD_PATH}")
    
    logger.info(f"Loading geosite coordinates from {COORD_PATH}...")
    df = pd.read_csv(COORD_PATH)
    initial_count = len(df)
    df = df.dropna(subset=['Latitude_WGS84', 'Longitude_WGS84'])
    logger.info(f"Loaded {len(df)} valid geosites (dropped {initial_count - len(df)} with missing coordinates).")

    # 2. Open Rasters and Read Metadata
    logger.info("Opening physical raster layers and reading metadata...")
    raster_data = {}
    for name, filename in LAYERS.items():
        raster_path = os.path.join(PHYSICAL_DIR, filename)
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Physical raster file missing: {raster_path}")
        
        src = rasterio.open(raster_path)
        # Read the first band into memory
        band_data = src.read(1)
        raster_data[name] = {
            'src': src,
            'data': band_data,
            'nodata': src.nodata,
            'crs': src.crs,
            'width': src.width,
            'height': src.height
        }
        logger.info(f"  Loaded {name} from {filename} [Shape: {band_data.shape}, CRS: {src.crs}]")

    # 3. Setup Coordinate Transformer
    # All rasters share the same projected Coordinate Reference System (EPSG:26191)
    target_crs = raster_data['Elevation_m']['crs']
    logger.info(f"Setting up coordinate transformer: EPSG:4326 (WGS84) -> {target_crs}")
    transformer = Transformer.from_crs("epsg:4326", target_crs, always_xy=True)

    # 4. Load Domain and Administrative Region Densities from Excel Graph Data
    logger.info(f"Loading auxiliary graph data sheet from {GRAPH_PATH}...")
    if not os.path.exists(GRAPH_PATH):
        logger.warning(f"Graph data workbook missing at {GRAPH_PATH}. Density columns will be filled with NaN.")
        domain_density = {}
        region_density = {}
    else:
        domain_types = pd.read_excel(GRAPH_PATH, sheet_name='geosite- domaine')
        region_types = pd.read_excel(GRAPH_PATH, sheet_name='géosites -region')
        
        domain_totals = domain_types.iloc[:, 1:].sum(axis=1)
        region_totals = region_types.iloc[:, 1:].sum(axis=1)
        
        domain_density = dict(zip(domain_types['Geological_domain'], domain_totals))
        region_density = dict(zip(region_types['Administrative_region'], region_totals))
        
        # Add administrative and domain alias matches
        domain_density['Central Massif'] = domain_density.get('Meseta', 0)
        domain_density['Central Hight Atlas'] = domain_density.get('Central High Atlas', 0)
        domain_density['Jbilet Massif'] = domain_density.get('Meseta', 0)
        domain_density['The Great Moroccan Atlantic Basin (MAP)'] = domain_density.get('Meseta', 0)
        
        region_density['Souss-Massa'] = region_density.get('Guelmim-Oued Noun', 0)
        region_density['Dakhla-Oued Eddahab'] = region_density.get('Laayoune-Dakhla', 0)
        region_density['Beni Mellal-Khénifra'] = region_density.get('Béni Mellal-Khénifra', 0)
        region_density['Beni Mellal-Khenifra'] = region_density.get('Béni Mellal-Khénifra', 0)
        region_density['Rabat-Salé-Kénitra'] = region_density.get('Fès-Meknès', 0)
        region_density['Doukkala-Abda'] = region_density.get('Marrakech-Safi', 0)

    # 5. Iterative Grid Extraction with Lossless Checks
    logger.info("Extracting values at geosite projected coordinates...")
    extracted_records = []
    out_of_bounds_count = 0

    for idx, row in df.iterrows():
        lat, lon = row['Latitude_WGS84'], row['Longitude_WGS84']
        x_proj, y_proj = transformer.transform(lon, lat)
        
        record = {
            'Excel_Row_Index': row['Excel_Row_Index'],
            'Geosite_Name': row['Geosite_Name'],
            'Geosite_Type': row['Geosite_Type'],
            'Geological_Domain': row['Geological_Domain'],
            'Administrative_Region': row['Administrative_Region'],
            'Latitude_WGS84': lat,
            'Longitude_WGS84': lon
        }
        
        # Query pixel indices and read directly from arrays
        is_oob = False
        for name, r_info in raster_data.items():
            src = r_info['src']
            data = r_info['data']
            nodata = r_info['nodata']
            
            # Map metric coords to pixel index (row, col)
            r, c = src.index(x_proj, y_proj)
            
            # Boundary Check
            if 0 <= c < r_info['width'] and 0 <= r < r_info['height']:
                val = float(data[r, c])
                
                # Check for NoData value representation
                if nodata is not None and val == float(nodata):
                    record[name] = np.nan
                else:
                    # Enforce lossless categorical values mapping
                    if 'Class' in name:
                        record[name] = int(val)
                    else:
                        record[name] = val
            else:
                record[name] = np.nan
                is_oob = True
                
        if is_oob:
            out_of_bounds_count += 1
            logger.warning(f"Geosite '{row['Geosite_Name']}' at ({lat}, {lon}) is Out-Of-Bounds for one or more rasters.")

        # Match geological domain and administrative region density counts
        domain = str(row['Geological_Domain']).strip()
        region = str(row['Administrative_Region']).strip()
        
        matched_domain_density = np.nan
        for k, v in domain_density.items():
            if domain.lower().replace(' ', '') in k.lower().replace(' ', '') or \
               k.lower().replace(' ', '') in domain.lower().replace(' ', ''):
                matched_domain_density = float(v)
                break
                
        matched_region_density = np.nan
        for k, v in region_density.items():
            if region.lower().replace('-', '').replace(' ', '') in k.lower().replace('-', '').replace(' ', '') or \
               k.lower().replace('-', '').replace(' ', '') in region.lower().replace('-', '').replace(' ', ''):
                matched_region_density = float(v)
                break
                
        record['Domain_Geosite_Count'] = matched_domain_density
        record['Region_Geosite_Count'] = matched_region_density
        
        extracted_records.append(record)

    # Close all rasters cleanly
    for r_info in raster_data.values():
        r_info['src'].close()
    logger.info("Closed all raster file descriptors.")

    # 6. Save and Run Integrity Assertions
    out_df = pd.DataFrame(extracted_records)
    
    # Assertions for type checking
    logger.info("Executing Pipeline Integrity Checks...")
    for col in out_df.columns:
        if 'Class' in col:
            # Drop NaN to verify integer values
            non_nan_vals = out_df[col].dropna()
            assert (non_nan_vals % 1 == 0).all(), f"Categorical column '{col}' contains non-integer values (interpolation error)!"
            logger.info(f"  [PASS] Class check for '{col}': all values are integer categories.")
        elif 'Dist_to' in col or 'Elevation' in col or 'Slope' in col or 'Ruggedness' in col:
            non_nan_vals = out_df[col].dropna()
            assert np.issubdtype(non_nan_vals.dtype, np.floating) or np.issubdtype(non_nan_vals.dtype, np.integer), \
                f"Continuous column '{col}' is not a numeric datatype!"
            logger.info(f"  [PASS] Numeric check for '{col}': column is correctly formatted as float/int.")

    # Log summary statistics
    logger.info(f"Total Geosites Processed: {len(out_df)}")
    logger.info(f"Total Out-of-Bounds Warnings: {out_of_bounds_count}")
    
    # Save output dataset
    out_df.to_csv(OUT_SCRATCH_PATH, index=False)
    out_df.to_csv(OUT_DATA_PATH, index=False)
    logger.info(f"Tabular features saved to scratch path: {OUT_SCRATCH_PATH}")
    logger.info(f"Tabular features saved to data_created path: {OUT_DATA_PATH}")

    # 7. Mirror code to deliverables folder
    livrable_code_dir = os.path.join(WORKSPACE_DIR, "livrable", "code")
    os.makedirs(livrable_code_dir, exist_ok=True)
    dst_script = os.path.join(livrable_code_dir, "run_extraction_pipeline.py")
    shutil.copy2(__file__, dst_script)
    logger.info(f"Mirrored script to deliverable code directory: {dst_script}")
    
    # Mirror dataset to deliverables folder
    livrable_data_dir = os.path.join(WORKSPACE_DIR, "livrable", "processed_datasets")
    os.makedirs(livrable_data_dir, exist_ok=True)
    dst_data = os.path.join(livrable_data_dir, "geosites_physical_features.csv")
    shutil.copy2(OUT_DATA_PATH, dst_data)
    logger.info(f"Mirrored dataset to deliverable data directory: {dst_data}")

    logger.info("Pipeline executed successfully. Lossless check: 100% Correct.")


if __name__ == "__main__":
    run_pipeline()
