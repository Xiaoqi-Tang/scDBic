# scDBic

**scDBic: A novel deep learning-based biclustering algorithm for analyzing scRNA-seq data**

scDBic is a R+Python package for single-cell RNA sequencing (scRNA-seq) biclustering analysis. It integrates dimensionality reduction and graph-based biclustering in a unified deep learning framework. It outperforms many state-of-the-art methods in accuracy and biological interpretability.

## 🔧 Features

- Cell clustering with a deep autoencoder
- Gene clustering
- The identification of key gene clusters by using the reverse strategy

##  💻 System Requirements
Hardware
- GPU: NVIDIA GPU with CUDA support is recommended for acceleration (tested on CUDA 11.x/12.x), -but the code supports CPU fallback.
- RAM: >16GB recommended for datasets >10k cells

Software
- OS: Linux (Ubuntu 20.04+ recommended) / Windows / macOS
- R: >= 4.0
- Python: >= 3.8

## 📦 Requirements

- python==3.10.19
- torch==2.9.1
- numpy==2.1.2
- pandas==2.3.3
- scipy==1.15.3
- scikit-learn==1.7.2
- scanpy==1.11.5
- igraph==1.0.0

## ⚙️ Installation
scDBic requires both R and Python environments. We use the R package reticulate to bridge them.

### Step 1: Set up Python Environment
We recommend using Conda to manage the Python environment:

conda create -n your_env python=3.10
conda activate your_env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 # Example for CUDA 11.8 (Please visit pytorch.org for the command specific to your CUDA version)
pip install numpy pandas

### Step 2: Install R Dependencies
Open R and run the following commands:

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install(c("SingleCellExperiment", "scater", "scran", "BiocParallel", "Matrix", "rhdf5"))
install.packages(c("dplyr", "reticulate", "igraph"))

### Step 3: Run scDBic 
#### 1. Configuration 
Before running the script, please open `scDBic.R` and update the following paths in the **User Configuration** section to match your local environment:

* **Conda Path**: Update the `conda` argument in `use_condaenv()` to point to your local conda executable (e.g., inside your anaconda/miniconda folder).
* **Input Data**: Set `input_file` to the path of your scRNA-seq expression matrix (CSV format).
* **Output Directory**: Set `output_dir` to where you want to save the results.

#### 2. Run the Script
You can run the analysis from the command line:
Rscript scDBic.R

#### 3. (Optional) Run with Demo Data
We provide a small demo dataset (demo_dataset.csv) in this repository to verify the installation and demonstrate the workflow.

