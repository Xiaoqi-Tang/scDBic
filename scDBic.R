# ==============================================================================
# scDBic: Deep Learning-based Biclustering for scRNA-seq Data
# Author: [Your Name/Lab Name]
# Date: [Current Date]
# Description: Main script for running scDBic with PyTorch Autoencoder
# ==============================================================================

# ========================= 
# 1. USER CONFIGURATION 
# (Please modify these paths before running)
# ========================= 

# --- Environment Settings ---
# Name of your Conda environment containing PyTorch
CONDA_ENV_NAME <- "r-pytorch-txq" 
# Path to Conda executable (Optional: set to NULL to let reticulate find it)
# Example: "/home/username/anaconda3/bin/conda"
CONDA_PATH <- NULL 

# --- Input/Output Paths ---
# Path to your expression matrix (Rows=Genes, Cols=Cells)
INPUT_FILE <- "./data/your_dataset.csv" 
# Path to cell labels (Optional: set to NULL if no labels)
LABEL_FILE <- "./data/your_labels.csv" 
# Directory to save results
OUTPUT_DIR <- "./results/your_dataset_output" 

# --- Logging ---
# Directory to save GPU memory logs
LOG_DIR <- "./logs"

# ========================= 
# 2. Initialization 
# ========================= 
start_time <- Sys.time() 
set.seed(1L) 
options(stringsAsFactors = FALSE)

# Setup Logging
if (!dir.exists(LOG_DIR)) dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(LOG_DIR, paste0("gpu_memory_torch_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".log"))

cat("==============================================\n", file = log_file)
cat("GPU Memory Monitoring Log (PyTorch Version)\n", file = log_file, append = TRUE)
cat(paste("Start Time:", Sys.time(), "\n"), file = log_file, append = TRUE)
cat("==============================================\n\n", file = log_file, append = TRUE) 

# Load Libraries
suppressPackageStartupMessages({ 
  library(SingleCellExperiment) 
  library(scater) 
  library(scran) 
  library(dplyr) 
  library(reticulate) 
  library(Matrix) 
  library(BiocParallel) 
}) 

# Activate Conda Environment
tryCatch({
  if (!is.null(CONDA_PATH)) {
    use_condaenv(CONDA_ENV_NAME, conda = CONDA_PATH, required = TRUE)
  } else {
    use_condaenv(CONDA_ENV_NAME, required = TRUE)
  }
  message(paste("Activated Conda Environment:", CONDA_ENV_NAME))
}, error = function(e) {
  message("Warning: Failed to activate conda environment. Please check CONDA_ENV_NAME.")
})

# ========================= 
# 3. Python Embed: PyTorch Model 
# ========================= 
py_run_string("
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import sys

# Set Random Seed
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
        # Encoder: 1024 -> 512 -> 256 -> 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.SELU(),
            nn.Linear(512, 256),
            nn.SELU(),
            nn.Linear(256, 128),
            nn.SELU(),
            nn.AlphaDropout(0.05)
        )
        
        # Decoder: 128 -> 256 -> 512 -> 1024 -> Output
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.SELU(),
            nn.Linear(256, 512),
            nn.SELU(),
            nn.Linear(512, 1024),
            nn.SELU(),
            nn.Linear(1024, input_dim) 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def cosine_loss(y_pred, y_true):
    target = torch.ones(y_pred.size(0)).to(y_pred.device)
    loss = 1 - F.cosine_embedding_loss(y_pred, y_true, target, reduction='mean')
    return loss

def train_ae(data_mat, epochs=100, batch_size=32):
    input_dim = data_mat.shape[1]
    tensor_x = torch.FloatTensor(data_mat)
    dataset = TensorDataset(tensor_x, tensor_x)
    dataloader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True, drop_last=False)
    
    model = Autoencoder(input_dim).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(int(epochs)):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            _, decoded = model(batch_x)
            loss = cosine_loss(decoded, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    encoded_features = []
    pred_loader = DataLoader(TensorDataset(tensor_x), batch_size=int(batch_size), shuffle=False)
    
    with torch.no_grad():
        for batch in pred_loader:
            batch_x = batch[0].to(DEVICE)
            enc, _ = model(batch_x)
            encoded_features.append(enc.cpu().numpy())
            
    encoded_final = np.vstack(encoded_features)
    
    del model, tensor_x, dataset, dataloader, optimizer
    torch.cuda.empty_cache()
    
    return encoded_final

def get_gpu_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        res = torch.cuda.memory_reserved() / 1024**3
        return f'{alloc:.2f} GB (Alloc), {res:.2f} GB (Res)'
    return 'No GPU'
")

# ========================= 
# 4. Helper Functions
# ========================= 
monitor_gpu_memory <- function(step_name = "") {
  tryCatch({
    mem_info <- py$get_gpu_memory()
    cat(sprintf("[%s] GPU Memory: %s\n", step_name, mem_info))
    cat(sprintf("[%s] GPU Memory: %s\n", step_name, mem_info), file = log_file, append = TRUE)
  }, error = function(e) { })
}

mean_pairwise_sq_euclid <- function(X) { 
  M <- ncol(X) 
  if (M < 2) return(0) 
  sum_norm2 <- sum(colSums(X * X)) 
  sum_vec <- rowSums(X) 
  total_sq <- M * sum_norm2 - sum(sum_vec * sum_vec) 
  total_sq / (M * (M - 1) / 2) 
} 

rm1_g <- function(x1, x2, a, m) { 
  x2v <- if (is.data.frame(x2)) as.vector(unlist(x2)) else as.vector(x2) 
  idx <- which(!is.na(x2v) & x2v == (m + 1 - a)) 
  if (length(idx) == 0) { 
    ma <- matrix(nrow = 0, ncol = ncol(x1)) 
    colnames(ma) <- colnames(x1) 
    return(ma) 
  } 
  ma <- x1[idx, , drop = FALSE] 
  rownames(ma) <- rownames(x1)[idx] 
  colnames(ma) <- colnames(x1) 
  ma 
} 

rm1 <- function(x1, x2, a, m) { 
  x2v <- if (is.data.frame(x2)) as.vector(unlist(x2)) else as.vector(x2) 
  idx <- which(!is.na(x2v) & x2v == (m + 1 - a)) 
  if (length(idx) == 0) { 
    ma <- matrix(nrow = nrow(x1), ncol = 0) 
    rownames(ma) <- rownames(x1) 
    return(ma) 
  } 
  ma <- x1[, idx, drop = FALSE] 
  colnames(ma) <- colnames(x1)[idx] 
  ma 
} 

# ========================= 
# 5. Core Algorithm Functions
# ========================= 
auto <- function(x){ 
  monitor_gpu_memory("Autoencoder-Start")
  if (inherits(x, "sparseMatrix")) x <- as.matrix(x) 
  x[!is.finite(x)] <- 0 
  x <- x[rowSums(x != 0) > 0, , drop = FALSE] 
  
  x_trans <- log1p(t(x)) 
  x_scaled <- scale(x_trans)
  x_scaled[is.na(x_scaled)] <- 0
  finite_cols <- apply(is.finite(x_scaled), 2, all) 
  x_scaled <- x_scaled[, finite_cols, drop = FALSE] 
  
  row_names <- rownames(x_scaled)
  encoded_data <- py$train_ae(x_scaled, 100L, 32L)
  rownames(encoded_data) <- row_names
  
  monitor_gpu_memory("Autoencoder-End")
  return(t(encoded_data)) 
} 

cell_c <- function(x){ 
  if (ncol(x) < 6) return(data.frame(clust = rep(1L, ncol(x)))) 
  if (nrow(x) < 10) return(data.frame(clust = rep(1L, ncol(x)))) 
  
  x_dense_for_check <- as.matrix(x) 
  bad_rows <- apply(x_dense_for_check, 1, function(row) any(!is.finite(row))) 
  if (any(bad_rows)) x <- x[!bad_rows, , drop = FALSE] 
  if (nrow(x) < 10) return(data.frame(clust = rep(1L, ncol(x)))) 
  
  x_dense <- as.matrix(x) 
  gene_var <- apply(x_dense, 1, stats::var) 
  gene_var[!is.finite(gene_var)] <- 0 
  keep_idx <- which(gene_var > 1e-6) 
  x_filtered <- x[keep_idx, , drop = FALSE] 
  
  if (nrow(x_filtered) < 5) return(data.frame(clust = rep(1L, ncol(x)))) 
  
  reduced_dims <- auto(x_filtered)
  sce <- SingleCellExperiment(assays = list(logcounts = x_filtered))
  reducedDim(sce, "AE") <- t(reduced_dims) 
  
  k_val <- min(ncol(sce) - 1L, 20L, max(3L, floor(ncol(sce)/3))) 
  
  tryCatch({ 
    g <- scran::buildSNNGraph(sce, use.dimred = "AE", k = k_val, BPPARAM = BiocParallel::SerialParam())
    clust <- igraph::cluster_walktrap(g)$membership 
    data.frame(clust = clust) 
  }, error = function(e) { 
    cat("Clustering failed:", e$message, ", returning single cluster\n") 
    data.frame(clust = rep(1L, ncol(x))) 
  }) 
} 

genecell_c <- function(x1){ 
  if (ncol(x1) < 6 || nrow(x1) < 10) { 
    cell_names <- colnames(x1) 
    if (is.null(cell_names)) cell_names <- paste0("cell_", seq_len(ncol(x1))) 
    return(data.frame(v1 = cell_names, v2 = rep(1L, ncol(x1)))) 
  } 
  
  has_bad <- function(m) inherits(m, "dgCMatrix") && any(!is.finite(m@x)) 
  if (inherits(x1, "dgCMatrix") && !has_bad(x1)) { 
    x1_clean <- x1 
  } else { 
    x1_dense_for_check <- as.matrix(x1) 
    x1_clean <- x1[!apply(x1_dense_for_check, 1, function(row) any(!is.finite(row))), , drop = FALSE] 
  } 
  
  if (nrow(x1_clean) < 10) { 
    cell_names <- colnames(x1) 
    if (is.null(cell_names)) cell_names <- paste0("cell_", seq_len(ncol(x1))) 
    return(data.frame(v1 = cell_names, v2 = rep(1L, ncol(x1)))) 
  } 
  
  x1_clean_dense <- as.matrix(x1_clean) 
  centers <- min(5L, nrow(x1_clean_dense)) 
  set.seed(1L) 
  result <- kmeans(x1_clean_dense, centers = centers, nstart = 10, iter.max = 100) 
  clust1 <- data.frame(result$cluster) 
  kmax <- max(clust1[[1]]) 
  unique_clust <- sort(unique(clust1[[1]])) 
  
  min_mean_dist <- Inf 
  best_new <- NULL 
  
  for (a in unique_clust) { 
    rm11_g <- rm1_g(x1, clust1[[1]], a, kmax) 
    if (is.null(dim(rm11_g)) || nrow(rm11_g) < 2 || ncol(rm11_g) < 6) next 
    
    all_zeros <- which(colSums(rm11_g) == 0) 
    if (length(all_zeros) > 0) next 
    
    col_vars <- apply(as.matrix(rm11_g), 2, var) 
    if (any(col_vars < 1e-8)) next 
    
    new <- cell_c(rm11_g) 
    
    if (is.numeric(new)) { 
      df <- data.frame(v1 = colnames(rm11_g), v2 = rep(as.integer(new), ncol(rm11_g))) 
      mean_dist <- mean_pairwise_sq_euclid(rm11_g) 
      if (mean_dist < min_mean_dist) { 
        min_mean_dist <- mean_dist 
        best_new <- df 
      } 
    } else { 
      df <- data.frame(v1 = colnames(rm11_g), v2 = as.integer(new$clust)) 
      clust2 <- data.frame(clust2 = df$v2) 
      unique_clust2 <- sort(unique(clust2$clust2)) 
      mean_ddd <- numeric(length(unique_clust2)) 
      for (ii in seq_along(unique_clust2)) { 
        b <- unique_clust2[ii] 
        rm11_cc <- rm1(rm11_g, clust2$clust2, b, max(clust2$clust2)) 
        if (ncol(rm11_cc) < 2) { 
          mean_ddd[ii] <- NA 
        } else { 
          mean_ddd[ii] <- mean_pairwise_sq_euclid(rm11_cc) 
        } 
      } 
      mean_dist <- mean(stats::na.omit(mean_ddd)) 
      if (!is.na(mean_dist) && mean_dist < min_mean_dist) { 
        min_mean_dist <- mean_dist 
        best_new <- df 
      } 
    } 
  } 
  
  if (is.null(best_new)) { 
    cell_names <- colnames(x1) 
    if (is.null(cell_names)) cell_names <- paste0("cell_", seq_len(ncol(x1))) 
    C <- ncol(x1) 
    best_new <- data.frame(v1 = cell_names, v2 = rep(1L, C)) 
  } 
  return(best_new) 
} 

list_frame <- function(x) { 
  data_frame <- data.frame(v1 = character(), v2 = integer()) 
  path_counter <- 0 
  for (path in names(x)) { 
    path_counter <- path_counter + 1 
    temp_data <- x[[path]] 
    temp_data$v2 <- temp_data$v2 + path_counter - 1 
    data_frame <- rbind(data_frame, temp_data) 
  } 
  row.names(data_frame) <- NULL 
  return(data_frame) 
} 

re <- function(x1, a, data = list(), path = c(), max_depth = 10){ 
  if (length(path) >= max_depth || nrow(x1) < 20 || ncol(x1) < 6) { 
    clust1 <- data.frame(v1 = colnames(x1), v2 = rep(1L, ncol(x1))) 
    data[[paste0(a, ":", paste0("(", paste(path, collapse = "->"), ")"))]] <- clust1 
    return(data) 
  } 
  
  clust1 <- genecell_c(x1) 
  clust1 <- data.frame(clust1) 
  k <- max(clust1$v2) 
  
  if (k == 1 || nrow(clust1) <= 6) { 
    data[[paste0(a, ":", paste0("(", paste(path, collapse = "->"), ")"))]] <- clust1 
  } else { 
    for (b in 1:k) { 
      new_path <- c(path, b) 
      sub_matrix <- rm1(x1, clust1$v2, b, k) 
      if (ncol(sub_matrix) >= 6 && nrow(sub_matrix) >= 20) { 
        data <- re(sub_matrix, a, data, new_path, max_depth) 
        cat(paste("Processing Cluster", b, "- Cells:", ncol(sub_matrix), "\n")) 
      } else { 
        sub_clust <- data.frame(v1 = colnames(sub_matrix), v2 = rep(1L, ncol(sub_matrix))) 
        data[[paste0(a, ":", paste0("(", paste(new_path, collapse = "->"), ")"))]] <- sub_clust 
      } 
    } 
  } 
  return(data) 
} 

main <- function(x){ 
  monitor_gpu_memory("Main-Start")
  clust <- cell_c(x) 
  unique_clust <- sort(unique(clust$clust)) 
  data_list <- list() 
  for (a in unique_clust) { 
    rm11_c <- rm1(x, clust$clust, a, max(clust$clust)) 
    if (ncol(rm11_c) >= 6 && nrow(rm11_c) >= 20) { 
      data_list <- c(data_list, re(rm11_c, a)) 
    } else { 
      clust_df <- data.frame(v1 = colnames(rm11_c), v2 = rep(1L, ncol(rm11_c))) 
      data_list[[paste0(a, ":()")]] <- clust_df 
    } 
  } 
  data <- list_frame(data_list) 
  monitor_gpu_memory("Main-End")
  return(data) 
} 

# ========================= 
# 6. Execution Block
# ========================= 
if(file.exists(INPUT_FILE)) {
  message("Loading data: ", INPUT_FILE)
  mat <- read.csv(INPUT_FILE, row.names = 1, check.names = FALSE) 
  mat <- as.matrix(mat) 
  mat[!is.finite(mat)] <- 0 
  matrix1 <- Matrix(mat, sparse = TRUE) 
  
  if (is.null(rownames(matrix1))) rownames(matrix1) <- paste0("g", seq_len(nrow(matrix1))) 
  if (is.null(colnames(matrix1))) colnames(matrix1) <- paste0("c", seq_len(ncol(matrix1))) 
  
  print(dim(matrix1))
  monitor_gpu_memory("Data Loaded")
  
  # Run Algorithm
  result <- main(matrix1) 
  print("Clustering complete") 
  
  # Label Mapping (Optional)
  if(!is.null(LABEL_FILE) && file.exists(LABEL_FILE)) {
    message("Mapping labels from: ", LABEL_FILE)
    label <- read.csv(LABEL_FILE, stringsAsFactors = FALSE) 
    unique_values <- unique(result$v2) 
    data <- data.frame() 
    
    for (i in unique_values) { 
      sub_result <- result[result$v2 == i, ] 
      sub_result$v2 <- NA 
      
      # Vectorized matching (Faster)
      match_idx <- match(sub_result$v1, label$V1)
      valid_labels <- label$V2[match_idx]
      
      label_count <- table(valid_labels, useNA = "no") 
      
      if(length(label_count) > 0) {
        max_label <- names(label_count)[which.max(label_count)] 
        sub_result$v2 <- rep(max_label, nrow(sub_result)) 
      } else {
        sub_result$v2 <- "Unknown"
      }
      data <- rbind(data, sub_result) 
    } 
    
    data_sorted <- data[order(match(data$v1, label$V1)), ] 
    
    if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE) 
    
    out_file <- file.path(OUTPUT_DIR, "scDBic_results_labeled.csv")
    write.csv(data_sorted, file = out_file, row.names = FALSE, quote = FALSE) 
    message(paste("Results saved to:", out_file))
    
  } else {
    message("Label file not found or not provided, saving raw cluster IDs.")
    if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE) 
    
    out_file <- file.path(OUTPUT_DIR, "scDBic_results_raw.csv")
    write.csv(result, file = out_file, row.names = FALSE, quote = FALSE) 
    message(paste("Results saved to:", out_file))
  }
} else {
  stop("Error: Input file not found. Please check INPUT_FILE path in the configuration section.")
}

monitor_gpu_memory("Program Finished")
end_time <- Sys.time() 
print(paste("Total Runtime (min):", as.numeric(end_time - start_time, units = "mins")))