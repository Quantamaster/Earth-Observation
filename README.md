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
https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_ncr_region.geojson
https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=rgb
https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=delhi_airshed.geojson
https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed?select=worldcover_bbox_delhi_ncr_2021.tif
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
