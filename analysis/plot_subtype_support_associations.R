#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[[1]] else "analysis/generated"
output_path <- if (length(args) >= 2) args[[2]] else "figures/subtype_support_associations.png"

subtypes <- read.csv(file.path(input_dir, "subtype_per_class.csv"), check.names = FALSE)
summary <- read.csv(file.path(input_dir, "subtype_model_summary.csv"), check.names = FALSE)
models <- c("DropCascade", "XGBoost")
columns <- c("dropcascade_f1", "xgboost_f1")
colors <- c("#A01822", "#2369A1")

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
png(output_path, width = 1800, height = 720, res = 180)
par(mfrow = c(1, 2), mar = c(4.2, 4.4, 2.4, 1.0), oma = c(0, 0, 0, 0))

draw_panel <- function(x, xlab, title, rho_column) {
  plot(x, subtypes[[columns[[1]]]], type = "n", xlab = xlab,
       ylab = "Pooled out-of-fold subtype F1", ylim = c(0, 1), main = title)
  grid(col = "#E4E4E4", lty = 1)
  for (i in seq_along(models)) {
    points(x, subtypes[[columns[[i]]]], pch = if (i == 1) 16 else 17,
           col = adjustcolor(colors[[i]], alpha.f = 0.72), cex = 0.85)
  }
  rho <- summary[match(models, summary$model), rho_column]
  legend("topleft", legend = sprintf("%s (rho = %.2f)", models, rho),
         col = colors, pch = c(16, 17), bty = "n", cex = 0.78)
}

draw_panel(log10(subtypes$cell_count), expression(log[10]("subtype cell count")),
           "Cell abundance", "rho_cell")
draw_panel(subtypes$patient_count, "Patients containing subtype",
           "Patient coverage (16/27 at ceiling)", "rho_patient")
dev.off()
