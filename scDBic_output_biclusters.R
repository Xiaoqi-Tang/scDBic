# ==============================================================================
# scDBic: Single-Cell Deep Biclustering using PyTorch and scran
# ==============================================================================
# Description: This script performs recursive biclustering on scRNA-seq data.
# It uses a PyTorch Autoencoder for feature extraction and SNN-graph 
# clustering for cell partitioning.
# ==============================================================================

start_time <- Sys.time()
set.seed(1L)
options(stringsAsFactors = FALSE)

# -------------------------
# 1. Directory Setup
# -------------------------
# Define relative paths for portability
input_file    <- "data/input_expression_matrix.csv" 
output_base   <- "results/output_run"
log_dir       <- file.path(output_base, "logs")
bicluster_dir <- file.path(output_base, "biclusters")

# Create directories if they don't exist
if (!dir.exists(log_dir)) dir.create(log_dir, recursive = TRUE)
if (!dir.exists(bicluster_dir)) dir.create(bicluster_dir, recursive = TRUE)

log_file <- file.path(log_dir, paste0("gpu_memory_", format(Sys.time(), "%Y%m%d_%H%M"), ".log"))

# -------------------------
# 2. Dependencies
# -------------------------
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(scater)
  library(scran)
  library(dplyr)
  library(reticulate)
  library(Matrix)
  library(BiocParallel)
})

# Python Environment Setup
# Note: Ensure you have a conda env with: torch, numpy
try({
  # It is recommended to use 'use_virtualenv' or 'use_condaenv' 
  # based on your local machine configuration.
  use_condaenv("r-pytorch", required = FALSE) 
}, silent = TRUE)

# -------------------------
# 3. Python Backend (PyTorch)
# -------------------------
# Embedded Python code for the Deep Learning Autoencoder
py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # SELU activation is used for self-normalizing properties
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.SELU(),
            nn.Linear(1024, 512), nn.SELU(),
            nn.Linear(512, 256), nn.SELU(),
            nn.Linear(256, 128), nn.SELU(),
            nn.AlphaDropout(0.05)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.SELU(),
            nn.Linear(256, 512), nn.SELU(),
            nn.Linear(512, 1024), nn.SELU(),
            nn.Linear(1024, input_dim) 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def cosine_loss(y_pred, y_true):
    target = torch.ones(y_pred.size(0)).to(y_pred.device)
    return 1 - F.cosine_embedding_loss(y_pred, y_true, target, reduction='mean')

def train_ae(data_mat, epochs=100, batch_size=32):
    input_dim = data_mat.shape[1]
    tensor_x = torch.FloatTensor(data_mat)
    dataset = TensorDataset(tensor_x, tensor_x)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    
    model = Autoencoder(input_dim).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    # Early stopping logic
    best_loss, patience, counter = float('inf'), 15, 0
    best_model_state = None
    
    model.train()
    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad()
            _, decoded = model(batch_x)
            loss = cosine_loss(decoded, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        if avg_loss < best_loss:
            best_loss, counter, best_model_state = avg_loss, 0, model.state_dict()
        else:
            counter += 1
        if counter >= patience: break
            
    if best_model_state: model.load_state_dict(best_model_state)
    
    model.eval()
    with torch.no_grad():
        encoded_final, _ = model(tensor_x.to(DEVICE))
    
    # Clean up GPU memory
    del model, tensor_x
    torch.cuda.empty_cache()
    return encoded_final.cpu().numpy()

def get_gpu_memory():
    if torch.cuda.is_available():
        return f'{torch.cuda.memory_allocated()/1024**3:.2f}GB (Alloc)'
    return 'CPU Mode'
")

# -------------------------
# 4. Helper Functions
# -------------------------

# Log GPU status to file and console
monitor_gpu_memory <- function(step_name = "") {
  tryCatch({
    mem_info <- py$get_gpu_memory()
    msg <- sprintf("[%s] GPU Memory: %s\n", step_name, mem_info)
    cat(msg)
    cat(msg, file = log_file, append = TRUE)
  }, error = function(e) { })
}

# Sub-matrix extraction for cells
rm1 <- function(x1, x2, a, m) { 
  idx <- which(!is.na(x2) & x2 == (m + 1 - a))
  if (length(idx) == 0) return(x1[, 0, drop = FALSE])
  return(x1[, idx, drop = FALSE])
} 

# -------------------------
# 5. Core Biclustering Logic
# -------------------------

# Autoencoder wrapper for R
auto_encode_features <- function(x){ 
  if (inherits(x, "sparseMatrix")) x <- as.matrix(x) 
  x_scaled <- scale(log1p(t(x)))
  x_scaled[is.na(x_scaled) | !is.finite(x_scaled)] <- 0
  
  encoded_data <- py$train_ae(x_scaled, 100L, 32L)
  return(t(encoded_data)) 
} 

# Cell Clustering using SNN-graph on AE features
cell_cluster <- function(x){ 
  if (ncol(x) < 6 || nrow(x) < 10) return(data.frame(clust = rep(1L, ncol(x)))) 
  
  reduced_dims <- auto_encode_features(x)
  sce <- SingleCellExperiment(assays = list(logcounts = x))
  reducedDim(sce, "AE") <- t(reduced_dims) 
  
  k_val <- min(ncol(sce) - 1L, 20L, max(3L, floor(ncol(sce)/3))) 
  
  tryCatch({ 
    g <- scran::buildSNNGraph(sce, use.dimred = "AE", k = k_val)
    clust <- igraph::cluster_walktrap(g)$membership 
    return(data.frame(clust = clust)) 
  }, error = function(e) return(data.frame(clust = rep(1L, ncol(x)))))
} 

# Recursive Biclustering Function
recursive_biclust <- function(x1, a, biclust_path, data = list(), depth = 0, max_depth = 10){ 
  
  save_bicluster <- function(mat, root_id, path_vec) {
    p_str <- paste(path_vec, collapse = "-")
    f_name <- sprintf("bicluster_r%s_p%s.csv", root_id, p_str)
    write.csv(as.matrix(mat), file = file.path(biclust_path, f_name), quote = FALSE)
  }
  
  # Termination conditions
  if (depth >= max_depth || nrow(x1) < 20 || ncol(x1) < 6) { 
    save_bicluster(x1, a, depth)
    return(data) 
  } 
  
  # Gene selection (kmeans) and Cell sub-clustering
  # (Simplification of the original genecell_c for stability)
  clust_res <- cell_cluster(x1)
  k <- max(clust_res$clust)
  
  if (k == 1) {
    save_bicluster(x1, a, depth)
  } else {
    for (b in 1:k) {
      sub_m <- rm1(x1, clust_res$clust, b, k)
      if (ncol(sub_m) >= 6) {
        data <- recursive_biclust(sub_m, a, biclust_path, data, depth + 1, max_depth)
      }
    }
  }
  return(data)
}

# -------------------------
# 6. Main Execution
# -------------------------
main <- function(matrix_input, save_path) {
  monitor_gpu_memory("Process Started")
  
  # Step 1: Initial Global Clustering
  initial_clusters <- cell_cluster(matrix_input)
  u_clusters <- sort(unique(initial_clusters$clust))
  
  cat(sprintf("Initial clustering found %d groups.\n", length(u_clusters)))
  
  # Step 2: Recursive deep-dive for each cluster
  for (cluster_id in u_clusters) {
    sub_mat <- rm1(matrix_input, initial_clusters$clust, cluster_id, max(initial_clusters$clust))
    if (ncol(sub_mat) >= 6 && nrow(sub_mat) >= 20) {
      recursive_biclust(sub_mat, cluster_id, save_path)
    }
  }
  
  monitor_gpu_memory("Process Finished")
}

# Execute if input exists
if (file.exists(input_file)) {
  mat <- read.csv(input_file, row.names = 1, check.names = FALSE)
  matrix_sparse <- Matrix(as.matrix(mat), sparse = TRUE)
  main(matrix_sparse, bicluster_dir)
  cat("Success: Biclusters saved to", bicluster_dir, "\n")
} else {
  cat("Error: Please place your data in", input_file, "\n")
}