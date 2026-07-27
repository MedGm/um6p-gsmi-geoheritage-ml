# General-Purpose Lossless Spatial Feature Extraction CLI Tool

This directory contains a production-grade, highly generic Python command-line utility `code/general_extractor.py` that automates spatial feature extraction from **any QGIS/ArcGIS raster layers** (standard single-band GeoTIFFs, or visual RGB layout maps) at arbitrary coordinates.

It is designed to be fully decoupled from the specific geosite project and can be used on any set of WGS84 points and raster datasets.

---

## 1. Key Features
1. **Universal Raster Support**:
   - **Standard GeoTIFFs**: Reads raw floating-point or integer pixel grids directly.
   - **RGB Visual Layouts**: Automatically decodes three-band RGB rasters (e.g., exported map layouts) into continuous scales (via colorbar gradient lookups) or discrete labels (via legend color lookups) using a fast SciPy `KDTree`.
2. **Decoupled Configuration**: Uses a clean JSON configuration file defining the list of rasters, their data types, and any color decoding requirements.
3. **Lossless Thematic Integrity**: Guarantees that categorical layers (LULC, Soils, Geology) are queried with nearest-neighbor precision (direct array indexing), preventing interpolation artifacts.
4. **Boundary Validation**: Detects and logs coordinates falling outside the raster grid, mapping them safely to `NaN` without breaking execution.
5. **Auto-CRS Alignment**: Reads projection systems directly from raster metadata and projects coordinate points using pyproj's `Transformer`.

---

## 2. Command Line Interface (CLI) Usage

### A. Run Feature Extraction
To run extraction for a points CSV, execute:
```bash
python code/general_extractor.py \
  --points data_created/geosites_coordinates_clean.csv \
  --config code/extraction_config.json \
  --output scratch/geosites_physical_features.csv \
  --x-col Longitude_WGS84 \
  --y-col Latitude_WGS84
```

### B. Generate a Configuration Template
To generate a default JSON configuration template explaining all parameters (including RGB decoders):
```bash
python code/general_extractor.py --generate-template my_config_template.json
```

---

## 3. Configuration JSON Format

Below is an example of the JSON configuration (`code/extraction_config.json`):

```json
{
    "Elevation_m": {
        "path": "gis_data/physical/elevation_meters.tif",
        "type": "continuous",
        "is_rgb": false
    },
    "Soil_Class": {
        "path": "gis_data/physical/soil_classes.tif",
        "type": "categorical",
        "is_rgb": false
    },
    "Continuous_RGB_DEM": {
        "path": "gis_data/physical/dem_rgb_layout.tif",
        "type": "continuous",
        "is_rgb": true,
        "colorbar_colors": [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ],
        "colorbar_values": [
            -63.0,
            1500.0,
            3970.0
        ]
    },
    "Categorical_RGB_LULC": {
        "path": "gis_data/physical/lulc_rgb_layout.tif",
        "type": "categorical",
        "is_rgb": true,
        "legend_mapping": {
            "1": [0, 100, 0],
            "2": [255, 255, 0],
            "3": [0, 0, 255]
        }
    }
}
```

---

## 4. Best Practices for Lossless GIS Export

To ensure 100% data integrity when exporting rasters from QGIS or ArcGIS:
1. **Export Raw Float GeoTIFFs**: Prefer exporting raw single-band float grids directly instead of layout screens to avoid the need for visual RGB colorbar decoders.
2. **Nearest-Neighbor Resampling**: Always use **Nearest Neighbor** resampling for categorical data transformation to avoid decimal interpolation codes (e.g. creating a class code $1.5$ from codes $1$ and $2$).
3. **NoData Synchronization**: Retain a standard NoData flag (e.g., `-9999` or `np.nan`) to represent pixels outside the study domain.
