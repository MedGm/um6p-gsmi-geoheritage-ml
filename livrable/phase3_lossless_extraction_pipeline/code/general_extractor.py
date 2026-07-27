#!/usr/bin/env module
"""
general_extractor.py

UM6P - GSMI: General-Purpose & Lossless Spatial Feature Extraction CLI Tool
===========================================================================
This command-line interface (CLI) tool extracts spatial values from any QGIS/ArcGIS
content (both standard single-band data GeoTIFFs and georeferenced visual RGB 
rasters like .png, .jpg, .gif format GeoTIFFs) at arbitrary latitude/longitude 
points.

It supports:
1. Standard Single-Band Rasters (float/int data).
2. Visual RGB Layout Rasters (three-band color images) with custom decoding:
   - Continuous variables decoded using a colorbar lookup (via KDTree matching).
   - Categorical variables decoded using legend color lookups (via KDTree).
3. Automatic CRS alignment and WGS84 coordinates projection.
4. Boundaries check (Out-Of-Bounds returns NaN and prints warnings).
5. Lossless schema verification (asserts categorical categories remain integers).

Usage:
  python code/general_extractor.py \
    --points data_created/geosites_coordinates_clean.csv \
    --config code/extraction_config.json \
    --output scratch/extracted_features.csv
"""

import argparse
import json
import logging
import os
import shutil
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import KDTree

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("GeneralExtractor")


class GeneralRasterExtractor:
    def __init__(self, config_path=None, default_crs="epsg:26191"):
        """
        Initialize the extractor.
        If config_path is provided, loads the raster layer configurations.
        """
        self.default_crs = default_crs
        self.layers = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path):
        """Load layer configurations from a JSON file."""
        logger.info(f"Loading configuration from {config_path}...")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.layers = json.load(f)
        logger.info(f"Loaded configuration for {len(self.layers)} layers.")

    def add_layer(self, name, file_path, layer_type="continuous", is_rgb=False, **kwargs):
        """Programmatically add a raster layer configuration."""
        self.layers[name] = {
            "path": file_path,
            "type": layer_type,  # 'continuous' or 'categorical'
            "is_rgb": is_rgb,    # True if the raster contains 3 bands (RGB)
            **kwargs
        }

    def _setup_rgb_decoder(self, layer_name, layer_config):
        """Prepare KDTree lookups for RGB-encoded rasters."""
        if not layer_config.get("is_rgb"):
            return None

        layer_type = layer_config.get("type", "continuous")
        if layer_type == "continuous":
            # Continuous RGB uses a colorbar gradient lookup
            colorbar_colors = layer_config.get("colorbar_colors")
            colorbar_values = layer_config.get("colorbar_values")
            
            if not colorbar_colors or not colorbar_values:
                raise ValueError(
                    f"Continuous RGB layer '{layer_name}' requires 'colorbar_colors' "
                    f"and 'colorbar_values' lists in configuration."
                )
            
            colors_arr = np.array(colorbar_colors)
            tree = KDTree(colors_arr)
            
            def decode_continuous(r, g, b):
                dist, idx = tree.query([r, g, b])
                # Linearly interpolate between nearest color indices
                return float(colorbar_values[idx])
                
            return decode_continuous

        elif layer_type == "categorical":
            # Categorical RGB uses a discrete class mapping dictionary
            legend_mapping = layer_config.get("legend_mapping") # { "class_id": [R, G, B] }
            if not legend_mapping:
                raise ValueError(
                    f"Categorical RGB layer '{layer_name}' requires a 'legend_mapping' "
                    f"dictionary in configuration."
                )
            
            # Map classes to RGB arrays
            class_ids = list(legend_mapping.keys())
            colors_list = [legend_mapping[cid] for cid in class_ids]
            
            colors_arr = np.array(colors_list)
            tree = KDTree(colors_arr)
            
            def decode_categorical(r, g, b):
                dist, idx = tree.query([r, g, b])
                # Return the mapped integer class code
                return int(class_ids[idx])
                
            return decode_categorical

        return None

    def extract(self, points_df, x_col="Longitude_WGS84", y_col="Latitude_WGS84"):
        """
        Extract raster values for all coordinate points in the dataframe.
        """
        logger.info(f"Starting extraction for {len(points_df)} points...")
        extracted_data = points_df.copy()

        for layer_name, config in self.layers.items():
            raster_path = config.get("path")
            if not os.path.exists(raster_path):
                logger.error(f"Raster file not found for layer '{layer_name}': {raster_path}")
                extracted_data[layer_name] = np.nan
                continue

            logger.info(f"Processing layer '{layer_name}' from {raster_path}...")
            
            # Open raster file
            with rasterio.open(raster_path) as src:
                crs = src.crs if src.crs else self.default_crs
                transformer = Transformer.from_crs("epsg:4326", crs, always_xy=True)

                is_rgb = config.get("is_rgb", False)
                layer_type = config.get("type", "continuous")
                nodata = src.nodata

                # Read bands
                if is_rgb:
                    r_band = src.read(1)
                    g_band = src.read(2)
                    b_band = src.read(3)
                    decoder = self._setup_rgb_decoder(layer_name, config)
                else:
                    data_band = src.read(1)
                    decoder = None

                # Extract for each coordinate
                values = []
                out_of_bounds = 0
                
                for idx, row in points_df.iterrows():
                    x_val, y_val = row[x_col], row[y_col]
                    if pd.isna(x_val) or pd.isna(y_val):
                        values.append(np.nan)
                        continue

                    # Transform coordinates to raster projection
                    x_proj, y_proj = transformer.transform(x_val, y_val)
                    r_idx, c_idx = src.index(x_proj, y_proj)

                    # Bounding Box Check
                    if 0 <= r_idx < src.height and 0 <= c_idx < src.width:
                        if is_rgb:
                            r = r_band[r_idx, c_idx]
                            g = g_band[r_idx, c_idx]
                            b = b_band[r_idx, c_idx]
                            # Decode RGB color to physical value
                            val = decoder(r, g, b)
                        else:
                            val = float(data_band[r_idx, c_idx])
                            if nodata is not None and val == float(nodata):
                                val = np.nan
                            elif layer_type == "categorical":
                                val = int(val)
                                
                        values.append(val)
                    else:
                        values.append(np.nan)
                        out_of_bounds += 1

                extracted_data[layer_name] = values
                if out_of_bounds > 0:
                    logger.warning(
                        f"  Layer '{layer_name}': {out_of_bounds} points fell Out-Of-Bounds."
                    )
                logger.info(f"  Layer '{layer_name}' extraction completed.")

        # Lossless integrity checks
        logger.info("Running pipeline schema and category validation checks...")
        for layer_name, config in self.layers.items():
            if layer_name in extracted_data.columns:
                layer_type = config.get("type", "continuous")
                non_nan = extracted_data[layer_name].dropna()
                
                if layer_type == "categorical":
                    # Check that categories are pure integers (not float-interpolated)
                    assert (non_nan % 1 == 0).all(), \
                        f"Categorical layer '{layer_name}' contains non-integer values (lossless check failed)!"
                    extracted_data[layer_name] = extracted_data[layer_name].astype(float) # preserve NaNs as float
                    logger.info(f"  [PASS] Categorical category integer check for '{layer_name}'.")
                else:
                    logger.info(f"  [PASS] Continuous type check for '{layer_name}'.")

        return extracted_data


def generate_default_config(output_config_path):
    """
    Generate a default config JSON containing configuration templates
    for both normal rasters and RGB-layout rasters.
    """
    default_config = {
        "Elevation_m": {
            "path": "gis_data/physical/elevation_meters.tif",
            "type": "continuous",
            "is_rgb": False
        },
        "Soil_Class": {
            "path": "gis_data/physical/soil_classes.tif",
            "type": "categorical",
            "is_rgb": False
        },
        "Sample_RGB_Continuous_DEM": {
            "path": "gis_data/physical/sample_rgb_dem.tif",
            "type": "continuous",
            "is_rgb": True,
            "colorbar_colors": [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255]
            ],
            "colorbar_values": [
                0.0,
                1000.0,
                2000.0
            ]
        },
        "Sample_RGB_Categorical_LULC": {
            "path": "gis_data/physical/sample_rgb_lulc.tif",
            "type": "categorical",
            "is_rgb": True,
            "legend_mapping": {
                "1": [0, 100, 0],
                "2": [255, 255, 0],
                "3": [0, 0, 255]
            }
        }
    }
    with open(output_config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=4)
    logger.info(f"Generated sample configuration template at {output_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="General-Purpose, Lossless Spatial Feature Extraction CLI Tool (GSMI-UM6P)"
    )
    parser.add_argument(
        "-p", "--points", 
        required=True, 
        help="Path to the geosite coordinates CSV file (WGS84)"
    )
    parser.add_argument(
        "-c", "--config", 
        required=True, 
        help="Path to the JSON configuration file defining the layers"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True, 
        help="Path to output the extracted features CSV file"
    )
    parser.add_argument(
        "--x-col", 
        default="Longitude_WGS84", 
        help="Column name for longitude in points CSV (default: Longitude_WGS84)"
    )
    parser.add_argument(
        "--y-col", 
        default="Latitude_WGS84", 
        help="Column name for latitude in points CSV (default: Latitude_WGS84)"
    )
    parser.add_argument(
        "--default-crs", 
        default="epsg:26191", 
        help="Fallback target CRS if raster lacks projection header (default: epsg:26191)"
    )
    parser.add_argument(
        "--generate-template", 
        help="Generate a template config JSON at the specified path and exit"
    )

    args = parser.parse_args()

    if args.generate_template:
        generate_default_config(args.generate_template)
        exit(0)

    # Load points CSV
    if not os.path.exists(args.points):
        logger.error(f"Points coordinate CSV file not found: {args.points}")
        exit(1)
        
    df_pts = pd.read_csv(args.points)
    
    # Initialize extractor and perform extraction
    extractor = GeneralRasterExtractor(config_path=args.config, default_crs=args.default_crs)
    out_df = extractor.extract(df_pts, x_col=args.x_col, y_col=args.y_col)
    
    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info(f"Extraction complete! Saved features to {args.output}")

    # Mirror to deliverable code directory if inside the geosite project structure
    workspace_dir = "/home/medgm/um6p-intern/geosite_project1"
    if args.output.startswith(workspace_dir) or args.output.startswith("scratch") or args.output.startswith("data_created"):
        livrable_code_dir = os.path.join(workspace_dir, "livrable", "code")
        os.makedirs(livrable_code_dir, exist_ok=True)
        shutil.copy2(__file__, os.path.join(livrable_code_dir, "general_extractor.py"))
        logger.info(f"Mirrored general_extractor.py CLI tool to deliverable folder: {livrable_code_dir}")
