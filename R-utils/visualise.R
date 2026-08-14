#!/usr/bin/env Rscript

library(argparse)
library(tidyverse)
library(ggrepel)

parser <- ArgumentParser(description = "plot 2D semantic map coordinates (columns: discourseme, item, x, y)")
parser$add_argument("path_in", help = "path to .tsv/.tsv.gz file with columns discourseme, item, x, y")
parser$add_argument("--path_out", default = NULL, help = "path to save the plot to (.pdf, .png, ...); default: <path_in> with plot extension")
parser$add_argument("--width", type = "double", default = 20, help = "plot width in inches")
parser$add_argument("--height", type = "double", default = 10, help = "plot height in inches")
parser$add_argument("--facet_by", default = NULL, help = "optional column to facet by, one panel per value (e.g. 'am')")
parser$add_argument("--size_by", default = NULL, help = "optional column to size labels by")
parser$add_argument("--size_direction", default = "desc", choices = c("asc", "desc"),
                     help = "for --size_by: 'desc' = larger value gets a larger label (e.g. an AM score like conservative_log_ratio); 'asc' = smaller value gets a larger label (e.g. 'rank', where 1 is most salient)")
args <- parser$parse_args()

path_out <- if (is.null(args$path_out)) paste0(tools::file_path_sans_ext(args$path_in), ".pdf") else args$path_out

d <- read_tsv(args$path_in, show_col_types = FALSE) |> mutate(discourseme = as.factor(discourseme))

if (!is.null(args$size_by)) {
  d$.size <- if (args$size_direction == "asc") -d[[args$size_by]] else d[[args$size_by]]
}

p <- d |>
  ggplot(aes(x = x, y = y, label = item, colour = discourseme))

if (is.null(args$size_by)) {
  p <- p + geom_label_repel(max.overlaps = Inf, point.size = NA, min.segment.length = Inf)
} else {
  p <- p + geom_label_repel(aes(size = .size), max.overlaps = Inf, point.size = NA, min.segment.length = Inf)
}

# qualitative (non-gradient) palette with clearly distinct hues, extended if there are more than 12 discoursemes
n_discoursemes <- nlevels(d$discourseme)
discourseme_colours <- if (n_discoursemes <= 12) {
  RColorBrewer::brewer.pal(max(3, n_discoursemes), "Paired")[seq_len(n_discoursemes)]
} else {
  colorRampPalette(RColorBrewer::brewer.pal(12, "Paired"))(n_discoursemes)
}

p <- p +
  xlab("") + ylab("") +
  scale_colour_manual(values = discourseme_colours) +
  guides(colour = guide_legend(title = "discourseme", override.aes = list(size = 5)), size = "none") +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), axis.text.y = element_blank(), axis.ticks.y = element_blank(),
        legend.position = "right", panel.grid.major = element_blank(), panel.grid.minor = element_blank(), panel.background = element_blank())

if (!is.null(args$facet_by)) {
  p <- p + facet_wrap(vars(.data[[args$facet_by]]), ncol = 1)
}

ggsave(path_out, plot = p, width = args$width, height = args$height)
cat(paste0("wrote plot to '", path_out, "'\n"))
