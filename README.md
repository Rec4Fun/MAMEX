# MAMEX: Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MAMEX is a recommendation system designed for the item cold-start scenario. It uniquely integrates a Mixture of Experts (MoE) architecture to process multi-modal (visual and textual) features. This approach allows the model to learn flexibly from different data types and, crucially, to dynamically weigh the importance of each modality on an item-by-item basis, leading to more accurate recommendations.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)

---

## 📖 Overview

This project implements a cold-start recommendation system using advanced multi-modal learning techniques. It is built upon a solid foundation of well-established research:

- **Base Architecture**: Based on the **MILK** framework.
- **Data Structure**: Follows the organizational principles of **LATTICE**.
- **Feature Preprocessing**: Utilizes **CLIP-based** feature extraction, inspired by **MMSRec**.

## ✨ Key Features

- **Multi-modal Learning**: Effectively combines visual and textual features for a deeper understanding of products.
- **CLIP Integration**: Uses the powerful CLIP model to extract rich and robust semantic features.
- **Scalable Architecture**: Designed for easy training and scaling on large-scale datasets.
- **Specialized for Amazon Categories**: Optimized for Amazon product categories, including:
  - Baby Products
  - Clothing
  - Sports

## 🚀 Setup

To run this project, you need to set up the environment and install the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/MAMEX.git
    cd MAMEX
    ```

2.  **Install dependencies:**
    (Assuming you have a `requirements.txt` file)
    ```bash
    pip install -r requirements.txt
    ```

## 📦 Data Preparation

1.  **Download the Datasets**:

    | Dataset Name | Download Link | Target Directory |
    | :--- | :--- | :--- |
    | Amazon Baby | [Google Drive](https://drive.google.com/file/d/1C-18Y84lMS5xsRGwKQYa9P0Tot-gxgYY/view) | `Baby` |

2.  **Directory Structure**:
    After downloading and extracting, place the datasets in the `datasets/` directory following the structure below:

    ```
    MAMEX/
    ├── datasets/
    │   ├── Baby/
    │   ├── Clothing/
    │   └── Sport/
    ├── main.py
    ├── Environment.py
    └── ...
    ```

## ⚙️ Training

The training process is straightforward and can be customized via the `Environment.py` file.

1.  **Configure the Environment**:
    Open `Environment.py` and adjust the parameters for your experiment.

    ```python
    # Environment.py

    # Select the dataset to train on ('Baby', 'Clothing', or 'Sport')
    DATASET_PATH = "datasets/Baby"

    # Training hyperparameters
    BATCH_SIZE = 128
    NUM_EXPERTS = 8
    LEARNING_RATE = 1e-4
    ```

    For example, to train on the `Clothing` dataset, simply change the path:
    `DATASET_PATH = "datasets/Clothing"`

2.  **Run the Training Script**:
    After configuration, execute the following command from the project's root directory:
    ```bash
    python main.py
    ```
    The training process will start, and the results will be saved according to your configuration.
