-- coco.py

This script converts image and mask data into COCO format annotations, which is widely used for training object detection and segmentation models. The script processes images and their corresponding masks, extracts bounding boxes and segmentation data, and outputs the annotations in a JSON file.

Features
Image and Mask Handling: The script reads images and their corresponding masks, ensuring that each mask matches the image.
Bounding Box and Segmentation Extraction: It uses contour detection to extract bounding boxes and segmentation polygons from the masks.
COCO Format Conversion: The extracted data is formatted into COCO-compliant JSON, including image metadata, bounding boxes, and segmentation annotations.

How to Use
Specify Paths: Update the paths for images_path and masks_path to point to your image and mask directories.
Define Categories: Customize the categories variable with the relevant class names and IDs.
Run the Script: Execute the script to generate a coco_annotations.json file with the COCO format annotations.

Example Usage

images_path = '/data/malaria_proj/building_data/unification/models/images'
masks_path = '/data/malaria_proj/building_data/unification/models/masks'
categories = [{"id": 1, "name": "building"}]
output_file = 'coco_annotations.json'


create_coco_format(images_path, masks_path, categories, output_file)

Requirements
Python 3.x
OpenCV
NumPy
tqdm

Notes
Ensure that the image and mask files are named consistently.
The script assumes that the masks are in grayscale format.
Warnings are provided if images or masks are not found or cannot be read.


-- train.py (Not Tested)

This repository contains code for training and evaluating an object detection model using the DETR (DEtection TRansformer) architecture on a custom COCO dataset. The primary objective is to detect and classify buildings from satellite imagery.

Key Components
CustomCocoDataset: A custom dataset class that extends the CocoDetection class to load images and corresponding annotations. The dataset can be transformed using various image augmentations.
Training Loop: The training loop fine-tunes the DETR model using an AdamW optimizer and a learning rate scheduler. The model is trained over multiple epochs, with logging of loss values and performance metrics.
Evaluation: The evaluation function uses COCO metrics to assess model performance on the validation set, including mean precision, recall, and F1 score.
Metrics Calculation: Precision, recall, and F1 score are calculated for the predictions and targets.


Dataset Preparation:

Ensure that the images and corresponding COCO annotations are available.
Set the paths for images_path and annotations_path.

Model Training:

The model is trained using a fine-tuning approach on the custom dataset. The training process includes data augmentation and logging of training progress.
The model is saved as detr_finetuned.pth after training.

Evaluation:

The evaluation function runs after each epoch to assess the model's performance using COCO evaluation metrics.

Dependencies
Python 3.x
PyTorch
torchvision
MONAI
Hugging Face Transformers
pycocotools
tqdm


Install Dependencies:
bash
Copy code
pip install torch torchvision monai transformers pycocotools tqdm

Train the Model:
Adjust the paths in the script and run it to start training:
bash
Copy code
python train_detr.py

Evaluate the Model:
The evaluation is integrated into the training loop and will run automatically after each epoch.
Output
The trained model is saved as detr_finetuned.pth.
Evaluation metrics such as loss, precision, recall, and F1 score are logged.

Notes
The model uses a pre-trained DETR backbone, which is fine-tuned on the custom dataset.
Logging is set up to monitor the training process, providing detailed feedback on model performance.



-- train_lighning.py


This repository contains a PyTorch Lightning-based implementation for training a DEtection TRansformer (DETR) model on a custom COCO-style dataset. The code is designed to train an object detection model using the DETR architecture, leveraging a custom dataset and data augmentation techniques.

Requirements
Python 3.8+
PyTorch 1.8+
PyTorch Lightning 1.4+
Transformers 4.5+
PIL (Pillow)
torchvision
Setup


Dataset Structure:

Images: Place your images in a directory, e.g., images_path.
Annotations: Create a COCO-style annotations file, e.g., coco_annotations.json.

Modify Paths:
Update the images_path and annotations_path variables with the correct paths to your images and annotations in the main script.

Running the Model
Training:
Run the script to start training:

bash
Copy code
python train.py
The training process uses data augmentation and splits the dataset into training and validation sets (80/20 split). The model is trained for 300 steps with checkpoints saved based on the lowest validation loss.

Model Saving:
After training, the model is saved to the saved_model/ directory.

Key Components
CustomCocoDataset: A custom dataset class that loads images and corresponding annotations from the COCO-style JSON file, applies transformations, and prepares data for the model.

Detr Module: A PyTorch Lightning module that wraps the DETR model, handles forward passes, and manages training/validation steps.

Training and Validation: The model is trained with an AdamW optimizer, and key metrics like loss are logged during the process.

Output
The model checkpoint with the lowest validation loss is saved in the checkpoints/ directory.
The final trained model is saved in the saved_model/ directory as final_model.pth.

Customization
Number of Classes: Adjust the num_labels parameter in the Detr class initialization based on your dataset.
Transforms: Modify the transform variable to apply different augmentations to your training images.

Additional Information
Model Checkpoints: Checkpoints are saved during training for easy resumption or analysis.
Validation: Validation is performed every epoch to monitor performance on unseen data.
