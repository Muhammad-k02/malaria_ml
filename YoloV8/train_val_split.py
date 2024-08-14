import os
import random
import shutil
from pathlib import Path

def split_dataset(base_dir, split_ratio=0.8):
    image_dir = os.path.join(base_dir, 'images')
    label_dir = os.path.join(base_dir, 'labels')

    train_image_dir = os.path.join(image_dir, 'train')
    val_image_dir = os.path.join(image_dir, 'val')
    train_label_dir = os.path.join(label_dir, 'train')
    val_label_dir = os.path.join(label_dir, 'val')

    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_files = list(Path(image_dir).glob("*.png"))
    random.shuffle(image_files)

    split_index = int(len(image_files) * split_ratio)

    train_images = image_files[:split_index]
    val_images = image_files[split_index:]

    for image_path in train_images:
        image_name = image_path.name
        label_name = image_name.replace('.png', '.txt')

        shutil.move(str(image_path), os.path.join(train_image_dir, image_name))
        shutil.move(os.path.join(label_dir, label_name), os.path.join(train_label_dir, label_name))

    for image_path in val_images:
        image_name = image_path.name
        label_name = image_name.replace('.png', '.txt')

        shutil.move(str(image_path), os.path.join(val_image_dir, image_name))
        shutil.move(os.path.join(label_dir, label_name), os.path.join(val_label_dir, label_name))

    print("Dataset split completed.")

# Example usage
base_dir = 'yolo_dataset'
split_dataset(base_dir, split_ratio=0.8)

