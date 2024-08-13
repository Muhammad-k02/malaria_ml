import os
from PIL import Image
import numpy as np


def save_patches(image, patch_size, stride, output_dir, upscale_factor=2, base_name="patch"):
    """Splits an image into patches, upscales them, and saves them.

    Args:
        image (PIL.Image): The input satellite image.
        patch_size (tuple): The size of each patch (height, width).
        stride (int): The stride or step size for moving the window across the image.
        output_dir (str): Directory to save the patches.
        upscale_factor (int, optional): Factor by which to upscale the patches. Defaults to 2.
        base_name (str): Base name for the patch files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_np = np.array(image)
    image_height, image_width = image_np.shape[:2]
    patch_height, patch_width = patch_size

    patch_count = 0

    for y in range(0, image_height - patch_height + 1, stride):
        for x in range(0, image_width - patch_width + 1, stride):
            patch = image_np[y:y + patch_height, x:x + patch_width]
            patch_image = Image.fromarray(patch)
            
            # Upscale the patch
            upscale_patch = patch_image.resize(
                (patch_width * upscale_factor, patch_height * upscale_factor), 
                resample=Image.BICUBIC  # You can change to Image.BILINEAR for a different method
            )
            
            patch_filename = f"{base_name}_{patch_count}.png"
            upscale_patch.save(os.path.join(output_dir, patch_filename))
            patch_count += 1
            print(f"Saved: {patch_filename}")

    print(f"Total patches saved: {patch_count}")


if __name__ == "__main__":
    # Path to the satellite image
    image_path = "/data/malaria_proj/building_data/unification/dino/True_color_composite.png"

    # Output directory to save patches
    output_dir = "/data/malaria_proj/building_data/unification/dino/ibadan_patches"


    # Define the patch size and stride
    patch_size = (256, 256)  # Size of each patch (height, width)
    stride = 128  # Stride for sliding window

    # Factor by which to upscale the patches
    upscale_factor = 2  # 2x zoom, can increase for more zoom

    # Load the image
    image = Image.open(image_path).convert('RGB')

    # Generate and save patches with zoom
    save_patches(image, patch_size, stride, output_dir, upscale_factor)

