from ultralytics import YOLO
import os
from pathlib import Path

# Load the model
model = YOLO("best.pt")  # pretrained YOLOv8n model

# Directory containing images
image_dir = "/data/malaria_proj/building_data/unification/models/ibadan_patches"

# Directory to save the results
save_dir = "/data/malaria_proj/building_data/unification/models/yolo_test_ibadan/"
os.makedirs(save_dir, exist_ok=True)

# Process each image one by one
for img_path in Path(image_dir).glob("*.*"):
    # Perform inference on the current image
    result = model(str(img_path))
    for result in result:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs

    original_filename = os.path.basename(img_path)
    result.save(filename=save_dir+original_filename)



