test_batch.py:

Purpose: This script evaluates the Faster R-CNN model on a batch of images from a directory and saves the predictions to output files.
Key Functions:
load_model(): Loads the pre-trained Faster R-CNN model and prepares it for inference.
transform_image(): Transforms images for model input.
visualize_and_save_predictions(): Visualizes and saves predictions on images.
test_model_on_directory(): Runs the model on all images in a specified directory and saves the results.
Execution: It sets up the device, paths for input images, output directory, and model checkpoint, then calls test_model_on_directory() to perform the evaluation.


test.py:

Purpose: This script evaluates the Faster R-CNN model on a single image and saves the predictions.
Key Functions:
load_model(): Loads the pre-trained Faster R-CNN model and prepares it for inference (same as in test_batch.py).
transform_image(): Transforms images for model input (same as in test_batch.py).
visualize_and_save_predictions(): Visualizes and saves predictions on images (same as in test_batch.py).
test_model_on_image(): Runs the model on a single image and saves the result.
Execution: It sets up the device, paths for the input image, output path, and model checkpoint, then calls test_model_on_image() to perform the evaluation.



train.py:

Purpose: This script trains the Faster R-CNN model using a dataset of images and masks and logs training progress.
Key Components:
BuildingDataset: Custom dataset class for loading images and masks and converting masks to bounding boxes.
mask_to_boxes(): Converts binary masks to bounding boxes.
Data Loading and Transformation: Defines transformations, splits the dataset into training and validation sets, and creates DataLoaders.
Model Setup: Configures the Faster R-CNN model, optimizer, learning rate scheduler, and mixed precision training.
Training and Evaluation: Runs the training loop for a specified number of epochs, evaluates the model on the validation set, and logs metrics.
Execution: It trains the model, evaluates it on the validation set, and saves the model checkpoint at the end of training.
