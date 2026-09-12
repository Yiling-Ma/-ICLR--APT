library(Matrix)
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; meta_path <- args[2]; out <- args[3]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
meta <- read.csv(meta_path, stringsAsFactors=FALSE)
raw <- readRDS(file.path(root, 'data/ALL_PBMC_Gene_Cell_matrix.rds'))
stopifnot(inherits(raw, 'sparseMatrix'), !anyDuplicated(rownames(raw)))
ix <- match(meta$cell_id, rownames(raw)); stopifnot(!anyNA(ix))
raw <- raw[ix,,drop=FALSE]
lib <- rowSums(raw); stopifnot(all(lib > 0), all(raw@x >= 0))
# Broad PBMC contextual markers, not validated signatures of the 27 subtypes.
genes <- c('CD3D','CD3E','IL7R','CCR7','S100A4','CD8A','MS4A1','CD79A',
           'TCL1A','CD14','LYZ','FCGR3A','MS4A7','GNLY','NKG7','FCER1A',
           'CST3','PPBP','PF4','GZMB','PRF1','S100A8')
writeLines(setdiff(genes,colnames(raw)), file.path(out,'missing_markers.txt'))
genes <- intersect(genes,colnames(raw))
stopifnot(length(genes)>0)
x <- as.matrix(raw[,genes,drop=FALSE])
x <- log1p(x / lib * 1e4)
colnames(x) <- paste0('RNA:',genes)
mt <- grep('^MT-',colnames(raw))
qc <- data.frame(cell_id=meta$cell_id, RNA_total=lib,
                 RNA_detected=rowSums(raw > 0))
if(length(mt)>0) qc$RNA_mito_percent <- rowSums(raw[,mt,drop=FALSE])/lib*100
write.csv(cbind(qc,x),file.path(out,'markers_qc.csv'),row.names=FALSE)
writeLines(paste('mitochondrial_genes',length(mt)),file.path(out,'rna_qc_audit.txt'))
