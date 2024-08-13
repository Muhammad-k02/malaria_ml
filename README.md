# malaria_ml


Project Overview
This repository contains several machine learning and image processing scripts organized into different directories based on their functionality. The project includes implementations for object detection using DETR and Faster-RCNN models, along with various image processing utilities.

Directory Structure


-- detr
Contains scripts and resources related to the DETR (DEtection TRansformer) model for object detection.

--coco.py: Script for working with COCO dataset format, likely involving data loading and processing.
--readMe.txt: A readme file providing an overview and usage instructions specific to the DETR directory.
--train_lightning: Directory for training scripts or configurations, potentially using PyTorch Lightning.
--train.py: Main script for training the DETR model.

--Faster-RCNN
Contains scripts and resources for training and testing Faster-RCNN models, organized into two versions.

--faster_rcnn_v1
--readMe.txt: A readme file providing details about the Faster-RCNN v1 implementation and usage.
--test_batch.py: Script for testing the Faster-RCNN model on batches of data.
--test.py: Script for evaluating the model on individual images or data.
--training.log: Log file containing training progress and metrics for Faster-RCNN v1.
--train.py: Main script for training the Faster-RCNN v1 model.


--faster_rcnn_v2
--readMe.txt: A readme file providing details about the Faster-RCNN v2 implementation and usage.
--train.py: Main script for training the Faster-RCNN v2 model.

--misc
--Contains utility scripts for various image processing tasks.

--mask.py: Generates binary masks from GeoJSON files and TIF images.
--patch.py: Extracts patches from images, upscales them, and saves them as PNG files.
--readMe.txt: A readme file providing an overview and usage instructions specific to the misc directory.
--rename.py: Renames files in a directory by removing a specified substring from the filenames.
--TIF_to_PNG.py: Converts 16-bit TIF images to 8-bit PNG images.

--README.md

--README.md: The main README file providing an overview of the entire project, including setup instructions, dependencies, and general usage guidelines.


Getting Started
Setup Environment: Ensure you have the required dependencies installed for each script or model. Refer to the individual readMe.txt files for specific requirements and installation instructions.
Training Models: Use the train.py scripts in the detr and Faster-RCNN directories to train the respective models. Follow the instructions in the readMe.txt files for configuration and usage.
Image Processing: Utilize the utility scripts in the misc directory for tasks such as creating masks, extracting patches, renaming files, and converting image formats.
For more detailed instructions, refer to the readMe.txt files within each directory.
