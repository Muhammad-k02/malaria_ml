This repository contains code for training and evaluating a Faster R-CNN model to detect buildings in satellite images. The model is implemented using PyTorch and torchvision, and it includes data preprocessing, model training, and evaluation functionalities.

Overview
The code performs the following tasks:

Data Preparation: Loads and preprocesses images and corresponding masks from specified directories.
Model Setup: Configures a Faster R-CNN model with a ResNet50 backbone.
Training: Trains the model using a dataset of images and masks.
Evaluation: Evaluates model performance on a validation set and computes precision, recall, and F1-score.
Logging and Checkpointing: Logs training progress and saves model checkpoints.
Installation
Ensure you have the following dependencies installed:

bash
Copy code
pip install torch torchvision opencv-python pillow tqdm
Data Preparation
Dataset
The BuildingDataset class loads images and corresponding masks from directories specified by image_dir and mask_dir. Masks are converted to bounding boxes to train the Faster R-CNN model.

Images should be in .jpg or .png format.
Masks should be in .png format.
The masks are converted to bounding boxes using the mask_to_boxes function, which extracts contours and computes bounding rectangles.

Training
Parameters
Epochs: 200
Batch Size: 12
Learning Rate: 5e-5
The model is trained using an AdamW optimizer with a learning rate scheduler that reduces the learning rate upon plateau in validation loss.

Logging
Training progress and evaluation metrics are logged to training.log.

Model Checkpoint
The model checkpoint is saved as model_checkpoint.pth.

Evaluation
Model performance is evaluated using precision, recall, and F1-score metrics computed by the compute_metrics function. The evaluation results are logged for each epoch.

Usage
Prepare your dataset and place images and masks in the specified directories.

Run the training script:

bash
Copy code
python train.py
Check training.log for detailed training and evaluation logs.
