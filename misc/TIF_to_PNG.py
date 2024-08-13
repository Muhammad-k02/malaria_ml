import rasterio
import numpy as np
import pathlib
import logging
from rasterio.plot import reshape_as_image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_16bit_to_8bit(input_raster: pathlib.Path, output_raster: pathlib.Path, percentiles=[2, 98]):
    '''
    Convert 16-bit image to 8-bit PNG using Rasterio.
    '''
    try:
        with rasterio.open(input_raster) as src:
            img_array = src.read()

            if len(img_array.shape) == 3:  # Multiband image
                bands, height, width = img_array.shape
                img_array_8bit = np.zeros_like(img_array, dtype=np.uint8)

                for i in range(bands):
                    band = img_array[i]
                    bmin = np.percentile(band, percentiles[0])
                    bmax = np.percentile(band, percentiles[1])
                    band = np.clip(band, bmin, bmax)
                    band = 255 * (band - bmin) / (bmax - bmin)
                    img_array_8bit[i] = band.astype(np.uint8)

                img_8bit = reshape_as_image(img_array_8bit)

                with rasterio.open(
                    output_raster, 'w',
                    driver='PNG',
                    height=img_8bit.shape[0],
                    width=img_8bit.shape[1],
                    count=img_8bit.shape[2],
                    dtype=img_8bit.dtype
                ) as dst:
                    dst.write(img_8bit.transpose(2, 0, 1))

                logging.info(f"Successfully processed {input_raster} to {output_raster}")

            else:
                raise ValueError("Unsupported raster format or dimensions")

    except Exception as e:
        logging.error(f"Error processing {input_raster}: {e}")

def main():
    input_dir = pathlib.Path("/data/malaria_proj/building_data/unification/dino/test_originals")
    output_dir = pathlib.Path("/data/malaria_proj/building_data/unification/dino/test_images")

    output_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.glob('*.tif')]

    for file in files:
        output_file = output_dir / file.name.replace('.tif', '.png')
        
        if not output_file.exists():
            convert_16bit_to_8bit(file, output_file)
        else:
            logging.info(f"Output file {output_file} already exists. Skipping.")

if __name__ == "__main__":
    main()

