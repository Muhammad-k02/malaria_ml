-- mask.py
Description:
This script generates binary masks from GeoJSON files and TIF images. It reads geometrical features from GeoJSON files and creates masks corresponding to these features using the TIF images. The resulting masks are saved as PNG files.

Dependencies:

matplotlib
os
json
rasterio
shapely
PIL
numpy
Usage:

Ensure the following paths are correctly set:
ResimPATH: Directory containing TIF images.
GeoJSONPATH: Directory containing GeoJSON files.
OutputPATH: Directory where the masks will be saved.
Run the script to process each GeoJSON file and generate corresponding masks.
Features:

Converts geometrical features into binary masks.
Saves masks as PNG files.
Handles errors gracefully and provides informative error messages.


-- patch.py
Description:
This script extracts patches from a satellite image, upscales them, and saves them as separate PNG files. The patches are extracted using a sliding window approach with a specified stride and patch size. The patches are then upscaled by a defined factor.

Dependencies:

os
PIL
numpy
Usage:

Set the following parameters:
image_path: Path to the input satellite image.
output_dir: Directory to save the patches.
patch_size: Size of each patch (height, width).
stride: Step size for moving the window across the image.
upscale_factor: Factor by which to upscale the patches.
Run the script to generate and save the patches.
Features:

Extracts image patches using a sliding window.
Upscales patches before saving.
Provides feedback on the number of patches saved.


-- rename.py
Description:
This script renames files in a specified directory by removing a given substring from the filenames.

Dependencies:

os
Usage:

Set the following parameters:
directory: Path to the directory containing files to be renamed.
substring_to_remove: Substring to be removed from filenames.
Run the script to rename files in the specified directory.
Features:

Renames files by removing a specified substring.
Provides feedback on renamed files.


-- TIF_to_PNG.py
Description:
This script converts 16-bit TIF images to 8-bit PNG images. It reads TIF files, scales pixel values to 0-255 range, and saves the images as PNG files. The script processes all TIF files in the input directory and saves the results to the output directory.

Dependencies:

rasterio
numpy
pathlib
logging
Usage:

Set the following parameters:
input_dir: Directory containing the TIF files.
output_dir: Directory to save the PNG files.
Run the script to convert all TIF files in the input directory to PNG.
Features:

Converts 16-bit TIF images to 8-bit PNG.
Scales pixel values using percentile-based normalization.
Logs processing information and handles errors.

