Earth Observation Land Cover Pipeline: README
Overview
This project provides a complete, reproducible workflow for automated land cover classification using Sentinel-2 image patches and ESA WorldCover data. The pipeline enables extraction, labeling, and supervised modeling of large geospatial chip collections, supporting quantitative environmental audit and analysis.

Directory Structure
text
/Earth_Obervation_Pipeline/
    |-- data/
    |     |-- delhi_ncr_region.geojson
    |     |-- delhi_ncr_grid.geojson
    |     |-- worldcover_bbox_delhi_ncr_2021.tif
    |     |-- rgb/                          # Folder with Sentinel-2 chips
    |     |-- image_coords.csv
    |     |-- imgs_within_grid.csv
    |     |-- labelled_images_clean.csv
    |-- 01_grid_visualization.py
    |-- 02_label_extract_assignment.py
    |-- 03_train_test_split.py
    |-- 04_cnn_train_eval.py
    |-- requirements.txt
    |-- README.md
Prerequisites
Python 3.8 or newer
DATASET
https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_ncr_region.geojson
AI-Based Geospatial Audit of the Delhi Airshed
This repository contains the pipeline for an AI-based geospatial audit of the Delhi Airshed, focusing on identifying land use patterns and pollution sources through the classification of satellite imagery. The project utilizes Sentinel-2 RGB image patches and ESA WorldCover 2021 data to train a CNN classifier (ResNet18).

Project Overview
The Ministry of Environment has commissioned this audit to leverage Earth Observation data for environmental monitoring. The pipeline integrates spatial analysis, land cover raster processing, and deep learning to classify land use within the Delhi-NCR region.

Pipeline Components and Goals
The pipeline is structured into three main phases:

Spatial Reasoning & Data Filtering: Define the area of interest using the Delhi-NCR shapefile, create a uniform 60x60 km grid, and filter Sentinel-2 imagery whose center coordinates fall within this grid.

Label Construction & Dataset Preparation: Extract ground truth labels from the ESA WorldCover 2021 raster (land_cover.tif) for each Sentinel-2 image using a mode-based labeling approach. Standardize labels and prepare a train/test dataset.

Model Training & Evaluation: Train a ResNet18 classifier on the Sentinel-2 images using the generated labels, and evaluate performance using F1 score and a confusion matrix.

Datasets and Inputs
The following datasets are required for the pipeline:

Delhi-NCR shapefile (EPSG:4326): Defines the boundary for the gridding and analysis area.

Delhi-Airshed shapefile (EPSG:4326): (Provided, but primary analysis focuses on the Delhi-NCR extent for Q1).

Sentinel-2 RGB image patches (128x128 pixels, 10m/pixel): PNG files with associated metadata (center coordinates) for classification.

land_cover.tif (ESA WorldCover 2021, 10m resolution): The raster used for generating ground truth labels.

Technical Requirements and Setup
The pipeline requires a Python environment with the following dependencies:

Dependencies
geopandas: For handling shapefiles and spatial operations.

rasterio: For reading and manipulating the land_cover.tif raster.

numpy, pandas: For data manipulation.

matplotlib, seaborn: For plotting and visualization.

geemap or leafmap: For interactive geospatial visualization and basemaps.

torch, torchvision: For CNN model training (ResNet18).

torchmetrics: For standardized evaluation metrics (F1 Score).
Required packages (install with the following):

bash
pip install geopandas matplotlib numpy pandas rasterio scikit-learn seaborn torch torchvision torchmetrics geemap shapely scipy
Workflow Summary
1. Grid Generation & Image Placement
Loads NCR region boundary and creates a regular 60×60 km grid.

Assigns each Sentinel-2 chip center to a grid cell, filtering to the Area of Interest (AOI).

Scripts:

01_grid_visualization.py

2. Patch Extraction & Land Cover Label Assignment
For each image, extracts a 128×128 patch from the WorldCover raster.

Computes the mode (most frequent land cover class) as the label.

Handles missing data and edges to ensure label reliability.

Scripts:

02_label_extract_assignment.py

3. Dataset Cleaning and Split
Filters out images with poor/invalid labels based on patch validity.

Splits labeled dataset into train/test sets (stratified).

Visualizes class balance to ensure robustness.

Scripts:

03_train_test_split.py

4. CNN Model Training & Evaluation
Loads images and labels, applying standard normalization.

Trains a ResNet-18 CNN; computes metrics (accuracy, F1), and confusion matrices.

Visualizes correct and incorrect predictions.

Scripts:

04_cnn_train_eval.py

Usage
Step-wise Instructions
Prepare inputs: Place all geospatial files and Sentinel-2 images in the indicated folders.

Run grid generation:

bash
python 01_grid_visualization.py
Produces the regular grid and imgs_within_grid.csv.

Land cover patch labeling:

bash
python 02_label_extract_assignment.py
Outputs labelled_images_clean.csv with valid image-label pairs.

Train-test split and class analysis:

bash
python 03_train_test_split.py
Generates split CSVs and visualizations.

CNN training and evaluation:

bash
python 04_cnn_train_eval.py
Model metrics and confusion matrix are displayed after training.

Data Format Details
labelled_images_clean.csv:

filename	lat	lon	label	class_str
Train/test splits:
CSV files listing images, coordinates, and class labels.

Images:
RGB PNG chips, sized to match patch/raster dimensions.

Troubleshooting
FileNotFoundError:

Ensure all referenced files exist in the paths described above.

Update script paths if directory layout differs.

Missing images in splits:

Remove or restore files as needed; verify all images in splits are present in the rgb/ folder.

Label/key errors:

Always run the labeling and cleaning steps before model training.

Best Practices
Back up all data and intermediate outputs after each step.

Validate dataset balance and quality before training models.

Document and save model configurations and metrics for reproducibility.

Citation
If using ESA WorldCover or Sentinel-2 data, credit the relevant data providers as specified in their documentation.
