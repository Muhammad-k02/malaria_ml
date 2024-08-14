import cv2
import numpy as np
import os
from tqdm import tqdm
import shutil


def mask_to_boxes_and_classes(mask):
    """
    Converts a mask into bounding boxes and class IDs.

    Args:
        mask (numpy.ndarray): Mask where different pixel values represent different classes.

    Returns:
        tuple: A tuple containing:
            - List of bounding boxes, where each bounding box is represented as [x_min, y_min, x_max, y_max].
            - List of class IDs corresponding to the bounding boxes.
    """
    unique_values = np.unique(mask)
    boxes = []
    class_ids = []

    for value in unique_values:
        if value == 0:
            continue  # Skip background if value is 0

        mask_class = (mask == value).astype(np.uint8)
        contours, _ = cv2.findContours(mask_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
            class_ids.append(1)  # Use 1 as class ID for buildings

    return np.array(boxes, dtype=np.float32).tolist(), class_ids


def convert_to_yolo_format(image_path, mask_path, label_file_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Unable to read image {image_path}")
        return

    height, width, _ = img.shape
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Warning: Unable to read mask {mask_path}")
        return

    boxes, class_ids = mask_to_boxes_and_classes(mask)

    with open(label_file_path, 'w') as f:
        for box, class_id in zip(boxes, class_ids):
            xmin, ymin, xmax, ymax = box
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            norm_width = (xmax - xmin) / width
            norm_height = (ymax - ymin) / height

            f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")


def create_yolo_dataset(images_path, masks_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')

    if not os.path.exists(images_output_dir):
        os.makedirs(images_output_dir)
    if not os.path.exists(labels_output_dir):
        os.makedirs(labels_output_dir)

    for image_name in tqdm(os.listdir(images_path)):
        if not image_name.lower().endswith('.png'):
            continue  # Skip non-PNG files

        image_path = os.path.join(images_path, image_name)
        mask_path = os.path.join(masks_path, image_name)

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for image {image_name}")
            continue

        # Copy image to output directory
        shutil.copy(image_path, images_output_dir)

        # Generate YOLO-format label file
        label_file_name = image_name.replace('.png', '.txt')
        label_file_path = os.path.join(labels_output_dir, label_file_name)

        convert_to_yolo_format(image_path, mask_path, label_file_path)

    print("Dataset conversion to YOLO format completed successfully.")


# Example usage
images_path = '../../images'
masks_path = '../../masks'
output_dir = 'yolo_dataset'

create_yolo_dataset(images_path, masks_path, output_dir)

