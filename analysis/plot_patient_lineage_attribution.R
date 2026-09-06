suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

input_path <- "analysis/results/patient_lineage_attribution.csv"
output_dir <- "figures"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

d <- read_csv(input_path, show_col_types = FALSE)
d$disease <- factor(d$disease, levels = c("CRC", "GC", "HCC", "PC", "BTC", "Normal"))
d$lineage <- factor(d$lineage, levels = c("T/NK", "B", "Myeloid"))
d$aptamer <- factor(d$aptamer, levels = unique(d$aptamer))

group_n <- c(CRC = 8, GC = 7, HCC = 7, PC = 6, BTC = 6, Normal = 6)
x_labels <- setNames(
  paste0(names(group_n), "\n", "n=", group_n),
  names(group_n)
)
palette <- c(
  CRC = "#315D6B",
  GC = "#4D7C8A",
  HCC = "#C56A3D",
  PC = "#D59B45",
  BTC = "#739B72",
  Normal = "#8A8175"
)
shared_limits <- range(d$patient_lineage_mean)
shared_limits <- shared_limits + c(-1, 1) * diff(shared_limits) * 0.05

make_plot <- function(signal_name, title) {
  panel <- d[d$signal == signal_name, ]
  ggplot(panel, aes(x = disease, y = patient_lineage_mean, color = disease)) +
    geom_hline(yintercept = 0, color = "#AEB6B8", linewidth = 0.25) +
    geom_boxplot(
      aes(fill = disease),
      width = 0.62,
      outlier.shape = NA,
      alpha = 0.18,
      linewidth = 0.35,
      color = "#596164"
    ) +
    geom_point(
      position = position_jitter(width = 0.13, height = 0, seed = 42),
      size = 1.35,
      alpha = 0.9
    ) +
    facet_grid(rows = vars(lineage), cols = vars(aptamer)) +
    scale_color_manual(values = palette, guide = "none") +
    scale_fill_manual(values = palette, guide = "none") +
    scale_x_discrete(labels = x_labels) +
    scale_y_continuous(limits = shared_limits) +
    labs(
      title = title,
      x = NULL,
      y = "Patient-lineage mean APT value"
    ) +
    theme_bw(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 12),
      strip.background = element_rect(fill = "#EEF1EF", color = "#CBD2D0", linewidth = 0.35),
      strip.text = element_text(face = "bold", size = 9),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "#E1E5E4", linewidth = 0.3),
      axis.text.x = element_text(size = 6.4, lineheight = 0.9),
      axis.text.y = element_text(size = 7),
      axis.title.y = element_text(size = 8.5),
      plot.margin = margin(6, 7, 5, 5)
    )
}

raw_plot <- make_plot("Raw", "Raw APT signal")
centered_plot <- make_plot("Patient-centered", "Patient-centered APT signal")

ggsave(
  file.path(output_dir, "patient_lineage_attribution_raw.png"),
  raw_plot,
  width = 14.2,
  height = 6.8,
  units = "in",
  dpi = 300,
  bg = "white"
)
ggsave(
  file.path(output_dir, "patient_lineage_attribution_centered.png"),
  centered_plot,
  width = 14.2,
  height = 6.8,
  units = "in",
  dpi = 300,
  bg = "white"
)
