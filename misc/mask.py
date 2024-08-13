import matplotlib.pyplot as plt
import os
import json
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import shape
from PIL import Image  # Import Image from PIL
import numpy as np

# Define paths
ResimPATH = '/data/malaria_proj/building_data/AOI_4_Shanghai_Train/RGB-PanSharpen'
GeoJSONPATH = '/data/malaria_proj/building_data/AOI_4_Shanghai_Train/geojson/buildings'
OutputPATH = '/data/malaria_proj/building_data/unification/masks_unified'  # Define your output path here

# Ensure directories exist
assert os.path.isdir(ResimPATH), f"Resim path not found: {ResimPATH}"
assert os.path.isdir(GeoJSONPATH), f"GeoJSON path not found: {GeoJSONPATH}"
os.makedirs(OutputPATH, exist_ok=True)  # Create the output directory if it doesn't exist

# List files in directories
ResimAdlari = sorted(os.listdir(ResimPATH))
GeoAdlari = sorted(os.listdir(GeoJSONPATH))

# Process each GeoJSON file
for DosyaNumarasi in range(len(GeoAdlari)):

    try:
        # Load GeoJSON data
        with open(os.path.join(GeoJSONPATH, GeoAdlari[DosyaNumarasi])) as f:
            data = json.load(f)

        # Load corresponding TIF file
        image_file = ResimAdlari[DosyaNumarasi]
        image_path = os.path.join(ResimPATH, image_file)

        with rasterio.open(image_path) as src:
            transform = src.transform
            width = src.width
            height = src.height

        # Create an empty mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Process each feature in the GeoJSON
        for feature in data['features']:
            geom = shape(feature['geometry'])
            if geom.is_valid:
                # Create a mask for this geometry
                geom_mask = geometry_mask([geom], transform=transform, invert=True, out_shape=(height, width))
                mask |= geom_mask.astype(np.uint8)  # Combine masks with bitwise OR

        # Convert mask to a PIL Image and save as PNG
        # Use the image file name for the mask
        base_name = os.path.splitext(image_file)[0]  # Remove file extension
        output_file = os.path.join(OutputPATH, f'{base_name}_mask.png')
        mask_image = Image.fromarray(mask * 255)  # Convert mask to 0-255 scale for PNG
        mask_image.save(output_file)

        print(f"Saved mask to {output_file}")

    except Exception as e:
        print(f"Error processing file {GeoAdlari[DosyaNumarasi]}: {e}")

