import rasterio
from rasterio.transform import Affine
import os

workspace_dir = "/home/medgm/um6p-intern/geosite_project1"
gis_dir = os.path.join(workspace_dir, "gis_data")

# Calibrated transform parameters (from Morocco Lambert Zone I metric grid ticks)
# Pixel size in X and Y is exactly 1183.432 meters
pixel_size_x = 1183.432
pixel_size_y = 1183.432

# Upper left corner projected coordinates corresponding to pixel (0, 0)
x_min = -1671597.6
y_max = 682840.2

new_transform = Affine(pixel_size_x, 0.0, x_min, 0.0, -pixel_size_y, y_max)
new_crs = "EPSG:26191"

# All target tif files in gis_data
tif_files = ["DEM.tif", "DAMS.tif", "geology.tif", "giosites.tif", "hydrography.tif", "LULC2009.tif", "SOIL.tif"]

print("Starting programmatic georeferencing...")

for filename in tif_files:
    input_path = os.path.join(gis_dir, filename)
    output_filename = filename.replace(".tif", "_modified.tif")
    output_path = os.path.join(gis_dir, output_filename)
    
    if os.path.exists(input_path):
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile.copy()
            
            # Update the profile with georeferencing metadata
            profile.update({
                'crs': new_crs,
                'transform': new_transform,
                'nodata': None
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)
                
        print(f"  Georeferenced {filename} -> {output_filename}")
    else:
        print(f"  Warning: File {filename} not found at {input_path}")

print("All layers georeferenced successfully.")
