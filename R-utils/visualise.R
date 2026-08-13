library(tidyverse)

d  <- read_tsv(path_in)

d |> ggplot(aes(x = x, y = y, label = text)
