suppressMessages(library(Matrix))
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]
out <- args[2]
raw <- readRDS(file.path(root, "data/ALL_PBMC_Gene_Cell_matrix.rds"))
meta <- read.csv(file.path(out, "cells.csv"), stringsAsFactors=FALSE)
stopifnot(inherits(raw, "sparseMatrix"), !anyDuplicated(rownames(raw)))
idx <- match(meta$cell_id, rownames(raw))
stopifnot(!anyNA(idx))
raw <- raw[idx, , drop=FALSE]
lib <- rowSums(raw)
stopifnot(all(lib > 0), all(raw@x >= 0))
norm <- as(Diagonal(x=1e4/lib) %*% raw, "CsparseMatrix")
norm@x <- log1p(norm@x)
rm(raw); gc()
for (manifest in sort(list.files(out, pattern="_g[0-4]_train.txt$", full.names=TRUE))) {
  prefix <- sub("_train.txt$", "", manifest)
  if (file.exists(paste0(prefix, "_done.txt"))) next
  tr <- scan(manifest, quiet=TRUE)+1L
  te <- scan(paste0(prefix, "_test.txt"), quiet=TRUE)+1L
  stopifnot(!any(tr %in% te))
  stopifnot(!any(meta$sample_id[tr] %in% meta$sample_id[te]))
  x <- norm[tr, , drop=FALSE]
  mu <- colMeans(x)
  variance <- pmax(colMeans(x^2)-mu^2, 0)
  dispersion <- variance/(mu+1e-12)
  bin <- pmin(20L, ceiling(rank(mu, ties.method="first")/length(mu)*20L))
  score <- ave(dispersion, bin, FUN=function(v) {
    s <- sd(v)
    if (!is.finite(s) || s==0) rep(0,length(v)) else (v-mean(v))/s
  })
  score[mu==0] <- -Inf
  genes <- head(order(score, decreasing=TRUE), 2000)
  writeMM(norm[tr,genes,drop=FALSE], paste0(prefix, "_train.mtx"))
  writeMM(norm[te,genes,drop=FALSE], paste0(prefix, "_test.mtx"))
  writeLines(colnames(norm)[genes], paste0(prefix, "_genes.txt"))
  writeLines("PASS", paste0(prefix, "_done.txt"))
  cat("PREPARED", basename(prefix), "\n"); flush.console()
}
