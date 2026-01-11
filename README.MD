
## 🌍 Earth Observation Land Cover Classification Pipeline 
**AI-Based Geospatial Audit of the Delhi Airshed**
---


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Satellite](https://img.shields.io/badge/Satellite-Sentinel--2-green)
![Geospatial](https://img.shields.io/badge/Geospatial-GIS%20%7C%20Raster-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---
![Space for climate and water](https://github.com/Quantamaster/Earth-Observation/blob/c8f8408086fe21bce706fdaddfa563c7f9857b5d/Space%20for%20climate%20and%20water.jpg)

## 📚 Table of Contents

- [📌 Overview](#-overview)
- [🛰️ Project Context](#️-project-context)
  - [Data Sources](#data-sources)
- [🧠 System Architecture](#-system-architecture)
- [📁 Directory Structure](#-directory-structure)
- [📦 Dataset](#-dataset)
- [🔍 Pipeline Breakdown](#-pipeline-breakdown)
  - [Phase 1: Spatial Reasoning & Filtering](#phase-1-spatial-reasoning--filtering)
  - [Phase 2: Label Construction](#phase-2-label-construction)
  - [Phase 3: Dataset Cleaning & Split](#phase-3-dataset-cleaning--split)
  - [Phase 4: CNN Training & Evaluation](#phase-4-cnn-training--evaluation)
- [📊 Results & Metrics](#-results--metrics)
- [🧠 Model Details](#-model-details)
- [⚙️ Technical Requirements](#️-technical-requirements)
- [🚀 Usage](#-usage)
- [🧾 Data Format](#-data-format)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [✅ Best Practices](#-best-practices)
- [📚 Citation](#-citation)

---

## 📌 Overview
This repository presents a **reproducible Earth Observation (EO) and Deep Learning pipeline** for **automated land-cover classification** using **Sentinel-2 RGB image patches** and **ESA WorldCover 2021** data.

The system enables:
- Automated geospatial data filtering
- Raster-based ground-truth label generation
- Supervised CNN training using **ResNet-18**
- Quantitative land-use and environmental analysis

The project is framed as an **AI-based Geospatial Audit of the Delhi Airshed**, suitable for **research labs, environmental agencies, and policy analytics**.

---

## 🛰️ Project Context
This work demonstrates how **satellite imagery + deep learning** can be leveraged for **urban land-use monitoring and environmental auditing**.

**Data Sources**
- Sentinel-2 (10m resolution RGB imagery)
- ESA WorldCover 2021 land-cover raster
- Delhi-NCR administrative boundary (EPSG:4326)

---

## 🧠 System Architecture

```

┌──────────────────────────────┐
│   Delhi-NCR AOI (GeoJSON)    │
└──────────────┬───────────────┘
↓
┌──────────────────────────────┐
│   Spatial Grid (60×60 km)    │
└──────────────┬───────────────┘
↓
┌──────────────────────────────┐
│ Sentinel-2 RGB Image Chips   │
│  (128×128 @ 10m resolution)  │
└──────────────┬───────────────┘
↓
┌──────────────────────────────┐
│ ESA WorldCover Raster (10m)  │
│  → Mode-based Labeling       │
└──────────────┬───────────────┘
↓
┌──────────────────────────────┐
│ Clean Labeled Dataset (CSV)  │
└──────────────┬───────────────┘
↓
┌──────────────────────────────┐
│ ResNet-18 CNN (PyTorch)      │
└──────────────┬───────────────┘
↓
┌──────────────────────────────┐
│ Metrics: Accuracy, F1, CM    │
└──────────────────────────────┘

````

## 📁 Directory Structure
```text
/Earth_Observation_Pipeline/
│
├── data/
│   ├── delhi_ncr_region.geojson
│   ├── delhi_ncr_grid.geojson
│   ├── worldcover_bbox_delhi_ncr_2021.tif
│   ├── rgb/                          # Sentinel-2 image patches (128×128)
│   ├── image_coords.csv
│   ├── imgs_within_grid.csv
│   └── labelled_images_clean.csv
│
├── 01_grid_visualization.py
├── 02_label_extract_assignment.py
├── 03_train_test_split.py
├── 04_cnn_train_eval.py
│
├── requirements.txt
└── README.md
````

---

## 📦 Dataset

**Kaggle Dataset (Required):**
[https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed](https://www.kaggle.com/datasets/rishabhsnip/earth-observation-delhi-airshed)

Includes:

* Sentinel-2 RGB image patches
* Delhi-NCR shapefiles
* Image coordinate metadata

---

## 🔍 Pipeline Breakdown

### Phase 1: Spatial Reasoning & Filtering

* Define AOI using Delhi-NCR boundary
* Generate a uniform **60 × 60 km grid**
* Filter Sentinel-2 images by grid intersection

**Script:** `01_grid_visualization.py`

---

### Phase 2: Label Construction

* Extract raster patches from ESA WorldCover
* Assign land-cover class using **mode-based sampling**
* Handle missing data and edge effects

**Script:** `02_label_extract_assignment.py`

---

### Phase 3: Dataset Cleaning & Split

* Remove invalid labels
* Perform **stratified train-test split**
* Analyze class distribution

**Script:** `03_train_test_split.py`

---

### Phase 4: CNN Training & Evaluation

* CNN: **ResNet-18**
* Input: Sentinel-2 RGB chips
* Metrics: Accuracy, **F1 Score**, Confusion Matrix

**Script:** `04_cnn_train_eval.py`

---

## 📊 Results & Metrics

> *(Representative – depends on training run)*

* **Overall Accuracy:** ~75–85%
* **Macro F1 Score:** ~0.72–0.82
* **Strong Performance:** Urban, Vegetation, Water
* **Challenges:** Mixed land-cover and boundary regions

### Outputs

* Confusion Matrix
* Class-wise F1 Scores
* Correct vs Incorrect Prediction Visualizations

---

## 🧠 Model Details

* **Architecture:** ResNet-18
* **Framework:** PyTorch
* **Loss Function:** Cross-Entropy
* **Evaluation Metrics:** Accuracy, F1 Score

---

## ⚙️ Technical Requirements

### Python

* Python **3.8+**

### Dependencies

```bash
pip install geopandas numpy pandas rasterio shapely scipy \
            matplotlib seaborn scikit-learn \
            torch torchvision torchmetrics \
            geemap
```

---

## 🚀 Usage

### 1️⃣ Grid Generation

```bash
python 01_grid_visualization.py
```

### 2️⃣ Label Assignment

```bash
python 02_label_extract_assignment.py
```

### 3️⃣ Train/Test Split

```bash
python 03_train_test_split.py
```

### 4️⃣ Model Training

```bash
python 04_cnn_train_eval.py
```

---

## 🧾 Data Format

### labelled_images_clean.csv

| filename | lat | lon | label | class_str |

---

## 🛠️ Troubleshooting

* Ensure all data paths are correct
* Run scripts in order
* Verify image presence in `rgb/`

---

## ✅ Best Practices

* Preserve intermediate outputs
* Validate class balance before training
* Track experiment configurations

---

## 📚 Citation

Please cite the following if used in research or applications:

* **ESA WorldCover 2021**
* **Copernicus Sentinel-2**
* Relevant geospatial data providers

---

⭐ **If this repository helps your work, consider starring it!**





