###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   03_naive_bayes.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar un modelo naive bayes para predecir pobreza y
#                      evaluarlo con F1 por validación cruzada.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Validación cruzada

################################################################################

# Input:  Base limpia creada en 01_data_cleaning.r; train_clean.rds .
# Output: F1 por fold y promedio (en consola).

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  e1071       # Modelos naive bayes.
)

################################################################################

## 2. Cargar datos

train <- read_rds(here("data", "train_clean.rds"))

# Predictores: se quitan las variables que solo existen en train (ingreso
# define pobre), los identificadores y los factores de expansión. max_educ,
# regimen_salud y tenencia_vivienda son códigos del DANE; se usan sus dummies.
y <- factor(train$pobre)
x <- train |>
  select(-c(id, pobre, ingreso, Indigente, Npobres, Nindigentes, Estrato1,
            fex_c, fex_dpto, max_educ, regimen_salud, tenencia_vivienda)) |>
  mutate(across(where(is.character), factor))

# Dummies 0/1 como factor: naiveBayes trata las numéricas como normales, lo
# que no tiene sentido para una variable binaria.
x <- x |> mutate(across(where(~ is.numeric(.x) && all(.x %in% c(0, 1, NA))), factor))

set.seed(123) # Para reproducibilidad

# Folds por hogar: todas las personas de un hogar quedan en el mismo fold,
# porque comparten el valor de pobre.
k       <- 10
hogares <- unique(train$id)
fold    <- sample(rep(1:k, length.out = length(hogares)))
fold    <- fold[match(train$id, hogares)]

# Ya no se necesita la base completa; se libera la memoria.
rm(train, hogares); gc()

################################################################################

## 3. Validación cruzada

f1 <- numeric(k)
for (i in 1:k) {
  prueba <- fold == i
  modelo <- naiveBayes(x[!prueba, ], y[!prueba])
  pred   <- predict(modelo, x[prueba, ])
  real   <- y[prueba]

  vp <- sum(pred == "1" & real == "1")
  precision <- vp / sum(pred == "1")
  recall    <- vp / sum(real == "1")
  f1[i]     <- 2 * precision * recall / (precision + recall)
  cat("Fold", i, "- F1:", round(f1[i], 4), "\n")
}

cat("F1 promedio:", round(mean(f1), 4), "\n")




### NOTAS:

# El modelo de naive bayes requiere mucho poder de computo cuando hay muchas variables.