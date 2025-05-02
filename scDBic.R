start_time <- Sys.time()

library(SingleCellExperiment)
library(scater)
library(scran)
library(dplyr)
library(reticulate)
library(tensorflow)

use_condaenv("r-tensorflow", conda = "/home/anaconda3/condabin/conda")

gpu_options <- tf$compat$v1$GPUOptions(allow_growth = TRUE)
config <- tf$compat$v1$ConfigProto(gpu_options = gpu_options)
sess <- tf$compat$v1$Session(config = config)
gpus <- tf$config$list_physical_devices('GPU')
if (length(gpus) > 0) {
  for (gpu in gpus) {
    tf$config$experimental$set_memory_growth(gpu, TRUE)
  }
}


rm1_g <- function(x1, x2, a, m) {
  class_indices <- which(x2 == (m+1-a), arr.ind = TRUE)
  class_indices <- matrix(class_indices[,1], ncol = 1)
  n <- ncol(x1)
  c <- nrow(class_indices)
  ma <- matrix(nrow = c, ncol = n)
  rownames(ma) <- rownames(x1)[class_indices]
  colnames(ma) <- colnames(x1)

  for (i in 1:c) {
    ma[i ,] <- x1[class_indices[i] ,]
  }

  return(ma)
}

rm1 <- function(x1, x2, a, m) {          
  class_indices <- which(x2 == (m+1-a), arr.ind = TRUE)
  class_indices <- matrix(class_indices[,1], ncol = 1)
  n <- nrow(x1)
  c <- nrow(class_indices)
  ma <- matrix(nrow = n, ncol = c)
  rownames(ma) <- rownames(x1)
  colnames(ma) <- colnames(x1)[class_indices]

  for (i in 1:c) {
    ma[, i] <- x1[, class_indices[i]]
  }

  return(ma)
}


# gene clustering + cell clustering
genecell_c <- function(x1){
  x1_clean <- x1[!apply(x1, 1, function(row) any(is.na(row) | is.nan(row) | is.infinite(row))), ]
  result <- kmeans(x1_clean , centers = 5) 
  #result <- specc(x1, centers = 5)
  #clust1 <- result@.Data
  #print(clust1)
  #clust1 <- data.frame(result@.Data)
  clust1 <- data.frame(result$cluster)
  unique_clust <- unique(clust1)
  unique_clust <- sort(unique_clust$result.cluster)
  min_mean_dist <- Inf  
  best_new <- NULL 
  for (a in unique_clust) {
    rm11_g <- rm1_g(x1, clust1, a, max(clust1))
    #rm11_g <- data.frame(rm11_g)
    all_zeros <- which(colSums(rm11_g) == 0)
    if (nrow(rm11_g) < 2 || length(all_zeros) > 0 ) {
      next
    }else{
      new <- cell_c(rm11_g)
      if (is.numeric(new)){
        new <- rep(new, ncol(rm11_g))
        new <- data.frame(new)
        new$v1 <- colnames(rm11_g)
        new$v2 <- new$new
        new <- new[, -1]
        dist_matrix <- dist(t(rm11_g), method = "euclidean")
        mean_dist <- mean(dist_matrix)
        if (mean_dist < min_mean_dist) {
          min_mean_dist <- mean_dist
          best_new <- new
        }
      }else{
        new$v1 <- colnames(rm11_g)
        new$v2 <- new$clust
        new <- new[, -1]
        clust2 <- new[, -1]####
        clust2 <- data.frame(clust2)
        unique_clust2 <- unique(clust2)
        unique_clust2 <- sort(unique_clust2$clust2)

        mean_ddd <- vector("numeric", length(unique_clust2))
        for (b in unique_clust2) {
          rm11_cc <- rm1(rm11_g, clust2, b, max(clust2))
          dist_matrix <- dist(t(rm11_cc), method = "euclidean")
          mean_d <- mean(dist_matrix)
          mean_ddd[b] <- mean_d


        }
        mean_dist <- mean(na.omit(mean_ddd))
        if (mean_dist < min_mean_dist) {
          min_mean_dist <- mean_dist
          best_new <- new
        }
      }
    }
  }
  return(best_new)
}

cell_c <- function(x){
  sce <- SingleCellExperiment(assays = list(counts = auto(x)))
  assays(sce)$logcounts <- assays(sce)$counts
  #sce <- scater::runPCA(sce)
  #g <- buildSNNGraph(sce, k = min(nrow(sce) - 1, 20), use.dimred = 'PCA')
  g <- buildSNNGraph(sce, k=min(nrow(sce) - 1, 20))
  #g <- buildSNNGraph(sce, use.dimred = NULL, k = min(nrow(sce) - 1, 20), d = NULL)
  clust <- igraph::cluster_walktrap(g)$membership
  clust1<-data.frame(clust)
  return(clust1)
}

list_frame <- function(x) {
  data_frame <- data.frame(v1 = character(), v2 = integer())
  path_counter <- 0 
  max_length <- 0  
  longest_path <- ""  

  for (path in names(x)) {
    path_counter <- path_counter + 1  
    temp_data <- x[[path]]
    temp_data$v2 <- temp_data$v2 + path_counter - 1
    data_frame <- rbind(data_frame, temp_data)

    path_numbers <- unlist(strsplit(path, "->"))
    num_count <- length(path_numbers)

    if (num_count > max_length) {
      max_length <- num_count
      longest_path <- path
    }
  }

  row.names(data_frame) <- NULL 

  return(data_frame)
}

re <- function(x1, a, data = list(), path = c()){
  clust1 <- genecell_c(x1) 
  clust1 <- data.frame(clust1)
  #print(clust1)
  k <- max(clust1$v2) 
  if (k == 1 || nrow(clust1) <= 5) {
    data[[paste0(a, ":", paste0("(", paste(path, collapse = "->"), ")"))]] <- clust1
  } else {
    for (b in 1:k) {
      new_path <- c(path, b)
      data <- re(rm1(x1, clust1, b, k), a, data, new_path)
      print(b)
    }
  }

  return(data)
}




main<-function(x){
  clust <- cell_c(x)
  unique_clust <- unique(clust)
  unique_clust <- sort(unique_clust$clust)
  data_list <- list()
  for (a in unique_clust) {
    rm11_c <- rm1(x, clust, a, max(clust))
    data_list <- c(data_list, re(rm11_c, a)) 
  }
  data <- list_frame(data_list)
  return(data)
}


tf <- import("tensorflow")
keras <- import("keras")

auto <- function(x){
  x <- x[rowSums(x != 0) > 0, ]
  x <- log1p(t(x))
  x <- scale(x)
  finite_cols <- apply(is.finite(x), 2, all)
  x <- x[, finite_cols]
  x <- as.array(x, dtype = 'float32')


  input_dim <- ncol(x)
  inputs <- keras$layers$Input(shape = list(input_dim), dtype = 'float32')
  #inputs <- keras$layers$Input(shape = input_dim)

  encoded <- keras$layers$Dense(units = as.integer(1024), activation = 'selu')(inputs)
  encoded <- keras$layers$Dense(units = as.integer(512), activation = 'selu')(encoded)
  encoded <- keras$layers$Dense(units = as.integer(256), activation = 'selu')(encoded)
  encoded <- keras$layers$Dense(units = as.integer(128), activation = 'selu')(encoded)

  decoded <- keras$layers$Dense(units = as.integer(256), activation = 'selu')(encoded)
  decoded <- keras$layers$Dense(units = as.integer(512), activation = 'selu')(decoded)
  decoded <- keras$layers$Dense(units = as.integer(1024), activation = 'selu')(decoded)
  decoded <- keras$layers$Dense(units = input_dim, activation = 'sigmoid')(decoded)

  autoencoder <- keras$Model(inputs = inputs, outputs = decoded)

  autoencoder$compile(
    optimizer = keras$optimizers$RMSprop(),
    #optimizer = keras$optimizers$Adam(),
    loss = function(x, decoded) {
      x <- tf$cast(x, dtype = 'float32')
      decoded <- tf$cast(decoded, dtype = 'float32')
      dot_prod <- tf$reduce_sum(x * decoded)
      cos_sim <- dot_prod / (tf$sqrt(tf$reduce_sum(x^2)) * tf$sqrt(tf$reduce_sum(decoded^2)))
      scale_cos_err <- 1 - cos_sim
      return(scale_cos_err)
    }
  )


  early_stopping <- keras$callbacks$EarlyStopping(
    monitor = "loss",
    patience = 15,
    verbose = 0,
    restore_best_weights = TRUE
  )


  autoencoder$fit(
    x, x,
    epochs = as.integer(70),
    batch_size = as.integer(16),
    shuffle = TRUE,
    verbose = 0, 
    callbacks = list(early_stopping)
  )

  encoder_model <- keras$Model(inputs = autoencoder$input, outputs = encoded)

  encoded_data <- encoder_model$predict(x)

  rownames(encoded_data) <- rownames(x)


  tf$keras$backend$clear_session()
  tf$compat$v1$reset_default_graph()


  return(t(encoded_data))
}




matrix <- read.csv("/home/biclust/data/E-MTAB-3321/E-MTAB-3321.csv")
#matrix <- matrix[, -1] 
matrix1 <- apply(matrix,2,as.numeric)

print(matrix1)
result  <- main(matrix1)
print("Clustering complete")

label <- read.csv("/home/biclust/data/E-MTAB-3321/label.csv", stringsAsFactors = FALSE)
unique_values <- unique(result$v2) 

data <- data.frame()
for (i in unique_values) {
  sub_result <- result[result$v2 == i, ]  
  sub_result$v2 <- NA
  for (j in 1:length(sub_result$v1)) {
    element <- sub_result$v1[j]
    match_idx <- match(element, label$V1)
    if (!is.na(match_idx)) {
      label_value <- label$V2[match_idx]
      sub_result$v2[j] <- label_value
    }
  }

  label_count <- table(sub_result$v2)
  max_label <- names(label_count)[which.max(label_count)]
  sub_result$v2 <- rep(max_label, nrow(sub_result))
  data <- rbind(data, sub_result)
}

data_sorted <- data[order(match(data$v1, label$V1)), ]
write.csv(data_sorted , file = "/home/biclust/DD/E-MTAB-3321/scDBic.csv", row.names = FALSE, quote = FALSE)


end_time <- Sys.time()


run_time <- end_time - start_time
run_time_minutes <- as.numeric(run_time, units = "mins")
print(paste("time:", run_time_minutes))
