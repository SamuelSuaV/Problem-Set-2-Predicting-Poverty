# Prueba: CART con las 6 variables y las 3 interacciones más importantes del
# CART con todas las variables (importancia por reducción de impureza, rpart).
# Base aumentada; mismos folds por hogar que 04 y 05.
suppressMessages({
  library(here)
  library(tidyverse)
  library(rpart)
  library(caret)
})

train <- read_rds(here("data", "train_clean_augmented.rds"))
y <- factor(train$pobre, levels = c(1, 0), labels = c("pobre", "no_pobre"))
id <- train$id

x <- train |>
  select(
    prop_ocupados_hogar, arriendo_pc_sobre_lp, educ_max_adultos, linea_pobreza,
    cuartos_dormir, regimen_salud
  ) |>
  mutate(
    regimen_salud = factor(regimen_salud),
    educ_max_adultos_x_prop_ocupados_hogar = educ_max_adultos *
      prop_ocupados_hogar,
    prop_ocupados_hogar_x_linea_pobreza = prop_ocupados_hogar * linea_pobreza,
    arriendo_pc_sobre_lp_x_linea_pobreza = arriendo_pc_sobre_lp * linea_pobreza
  ) |>
  as.data.frame()
rm(train)
invisible(gc())

set.seed(123)
k <- 10
hogares <- unique(id)
fold <- sample(rep(1:k, length.out = length(hogares)))[match(id, hogares)]
index <- map(1:k, ~ which(fold != .x)) |> setNames(sprintf("Fold%02d", 1:k))
cat("Filas:", nrow(x), "- columnas:", ncol(x), "\n")
