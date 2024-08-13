import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def mask_to_boxes_and_segmentation(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    segmentations = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x, y, x + w, y + h])

        segmentation = contour.flatten().tolist()
        if len(segmentation) > 4:  # Ensure that the segmentation is valid
            segmentations.append(segmentation)

    return np.array(boxes, dtype=np.float32).tolist(), segmentations

def find_mask_for_image(image_name, masks_path):
    mask_name = image_name.replace('.jpg', '.png')
    mask_path = os.path.join(masks_path, mask_name)
    return mask_path

def create_coco_format(images_path, masks_path, categories, output_file):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": categories
    }

    annotation_id = 0
    image_id = 0

    for image_name in tqdm(os.listdir(images_path)):
        image_path = os.path.join(images_path, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Unable to read image {image_path}")
            continue

        height, width, _ = img.shape

        coco_format['images'].append({
            "file_name": image_name,
            "height": height,
            "width": width,
            "id": image_id
        })

        mask_path = find_mask_for_image(image_name, masks_path)
        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for image {image_name}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Unable to read mask {mask_path}")
            continue

        boxes, segmentations = mask_to_boxes_and_segmentation(mask)

        for box, segmentation in zip(boxes, segmentations):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            coco_format['annotations'].append({
                "segmentation": [segmentation], 
                "area": width * height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, width, height],
                "category_id": 1,
                "id": annotation_id
            })

            annotation_id += 1

        image_id += 1

    with open(output_file, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Example usage:
images_path = '/data/malaria_proj/building_data/unification/models/images'
masks_path = '/data/malaria_proj/building_data/unification/models/masks'
categories = [{"id": 1, "name": "building"}]
output_file = 'coco_annotations.json'

create_coco_format(images_path, masks_path, categories, output_file)

