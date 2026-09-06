suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
})

result_dir <- "analysis/results/psas_controls_v2"
summary <- read_csv(file.path(result_dir, "random_panel_summary_n100.csv"), show_col_types = FALSE)
summary$model <- factor(summary$model, levels = c("LR", "XGBoost"))

colors <- c(LR = "#2474A6", XGBoost = "#D94738")
plot <- ggplot(summary, aes(x = B, color = model, fill = model)) +
  geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.45, color = "#62696B") +
  geom_ribbon(
    aes(ymin = `random_empirical_2.5pct`, ymax = `random_empirical_97.5pct`),
    alpha = 0.13,
    color = NA
  ) +
  geom_line(aes(y = random_mean), linetype = "dotted", linewidth = 0.75) +
  geom_point(aes(y = random_mean), shape = 21, size = 2.3, stroke = 0.6) +
  geom_line(aes(y = top_B_patient_macro_f1), linewidth = 0.85) +
  geom_point(aes(y = top_B_patient_macro_f1), size = 2.4) +
  scale_color_manual(values = colors, name = NULL) +
  scale_fill_manual(values = colors, name = NULL) +
  scale_x_continuous(breaks = c(5, 10, 20)) +
  coord_cartesian(ylim = c(0.35, 1.015)) +
  labs(
    title = "Compact-panel performance against 100 random panels",
    subtitle = "Solid: fold-specific Top-B; dotted and ribbon: random mean and empirical 95% interval",
    x = "Aptamers retained (B)",
    y = "Patient macro-F1"
  ) +
  theme_bw(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 8.5, color = "#4A5355"),
    legend.position = "top",
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#E1E5E4", linewidth = 0.35)
  )

ggsave(
  "figures/panel_size_curve.png",
  plot,
  width = 7.2,
  height = 3.7,
  units = "in",
  dpi = 300,
  bg = "white"
)
