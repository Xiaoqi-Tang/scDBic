# ==============================================================================
# Title:        KEGG Pathway Enrichment Analysis for scDBic Biclusters
# Description:  Performs gene ID mapping and KEGG enrichment analysis for 
#               gene lists derived from biclustering results.
# Dependencies: clusterProfiler, org.Hs.eg.db, org.Mm.eg.db, ggplot2
# ==============================================================================

# Load necessary libraries
suppressPackageStartupMessages({
  library(clusterProfiler)
  library(org.Hs.eg.db)
  library(org.Mm.eg.db)
  library(ggplot2)
})

#' Perform KEGG Enrichment Analysis
#'
#' @param file_path String. Path to the .csv file containing the gene matrix.
#'                  Expects column 1 to be gene names/IDs.
#' @param species String. "human" or "mouse".
#' @param id_type String. Input gene ID type: "SYMBOL" or "ENSEMBL".
#' @param pvalue_cutoff Numeric. Cutoff for enrichment significance.
#' @param output_dir String. Directory to save results. Default is current dir.
#'
#' @return Generates dotplots and barplots in the output directory.
run_kegg_analysis <- function(file_path, 
                              species = "human", 
                              id_type = "SYMBOL", 
                              pvalue_cutoff = 0.05,
                              output_dir = "./results") {
  
  # 1. Setup Species Database and Code
  if (species == "human") {
    org_db <- org.Hs.eg.db
    kegg_code <- "hsa"
  } else if (species == "mouse") {
    org_db <- org.Mm.eg.db
    kegg_code <- "mmu"
  } else {
    stop("Species must be 'human' or 'mouse'")
  }
  
  # 2. Read Data
  if (!file.exists(file_path)) stop("File not found: ", file_path)
  
  message(paste0(">>> Processing file: ", file_path))
  data <- read.csv(file_path, header = TRUE)
  
  # Assuming the first column contains Gene IDs
  gene_list <- as.character(data[, 1])
  
  # Clean ENSEMBL IDs if necessary (remove version numbers like ENSG000.1)
  if (id_type == "ENSEMBL") {
    gene_list <- sub("\\..*", "", gene_list)
  }
  
  # 3. ID Mapping (Convert to ENTREZID for KEGG)
  message(">>> Mapping Gene IDs to Entrez IDs...")
  tryCatch({
    gene.df <- bitr(gene_list, 
                    fromType = id_type,
                    toType = "ENTREZID",
                    OrgDb = org_db)
  }, error = function(e) {
    stop("ID mapping failed. Please check if your gene IDs match the 'id_type' parameter.")
  })
  
  entrez_ids <- gene.df$ENTREZID
  
  # 4. Run KEGG Enrichment
  message(">>> Running KEGG enrichment...")
  kk <- enrichKEGG(gene         = entrez_ids,
                   organism     = kegg_code,
                   pvalueCutoff = pvalue_cutoff)
  
  # 5. Visualization and Saving
  if (is.null(kk) || nrow(kk) == 0) {
    warning("No significant KEGG pathways found.")
    return(NULL)
  } else {
    # Create output directory
    if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
    
    # Get base filename for saving
    base_name <- tools::file_path_sans_ext(basename(file_path))
    
    # Plot: Dotplot
    p1 <- dotplot(kk, showCategory = 20) + ggtitle(paste("KEGG Dotplot:", base_name))
    ggsave(filename = file.path(output_dir, paste0(base_name, "_dotplot.png")), plot = p1, width = 8, height = 6)
    
    # Plot: Barplot
    p2 <- barplot(kk, showCategory = 20) + ggtitle(paste("KEGG Barplot:", base_name))
    ggsave(filename = file.path(output_dir, paste0(base_name, "_barplot.png")), plot = p2, width = 8, height = 6)
    
    # Save Results Table
    write.csv(as.data.frame(kk), file.path(output_dir, paste0(base_name, "_kegg_results.csv")))
    
    message(paste(">>> Results saved to:", output_dir))
  }
}
# ==============================================================================
# Example Usage (User should modify this part)
# ==============================================================================

# Example 1: Analyze a single file for Mouse
# run_kegg_analysis(file_path = "data/bicluster_b9.csv", 
#                   species = "mouse", 
#                   id_type = "ENSEMBL")

# Example 2: Batch processing all CSVs in a folder
# files <- list.files("data/biclusters", pattern = "*.csv", full.names = TRUE)
# for (f in files) {
#   run_kegg_analysis(f, species = "human", id_type = "SYMBOL")
# }
