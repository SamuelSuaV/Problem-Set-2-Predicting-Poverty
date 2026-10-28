###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   08_logit_interacciones.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar el logit lasso de 06 con más flexibilidad
#                      (más interacciones, splines e interacciones por
#                      dominio) y escoger lambda y umbral por F1 en
#                      validación cruzada.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos y armar la matriz de diseño
# 3. Validación cruzada de lambda y umbral
# 4. Predicción en test

# Input:  Bases a nivel hogar creadas en 02_bases_hogar.R; train_hogar.rds ,
#         test_hogar.rds .
# Output: F1 de validación cruzada (en consola);
#         02_outputs/predictions/logit_lasso_inter2_lambda_<l>_threshold_<c>.csv
#         ;
#         02_outputs/logs/logit_interacciones/logit_oof.rds (probabilidades
#         fuera
#         de muestra y de test, para 11_xgboost_hogar y 15_comparacion_modelos).
#
# Con el argumento --prueba (Rscript 01_code/08_logit_interacciones.R --prueba)
# corre con el 10% de los hogares de train y de test y no guarda la
# submission; sirve para verificar que el script corre.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  glmnet, # Logit con penalización (lasso / elastic net).
  splines # Splines naturales (ns).
)
# Funciones comunes (calcular_f1, marca, ...).
source(here("01_code", "functions", "funciones_comunes.R"), encoding = "UTF-8")

prueba <- "--prueba" %in% commandArgs(trailingOnly = TRUE)

# 2. Cargar datos y armar la matriz de diseño ---------------------------------

train <- read_rds(here("data", "train_hogar.rds"))
test <- read_rds(here("data", "test_hogar.rds"))

if (prueba) {
  set.seed(1)
  train <- slice_sample(train, prop = 0.1)
  test <- slice_sample(test, prop = 0.1)
  cat(
    "MODO PRUEBA:", nrow(train),
    "hogares de train; no se guarda la submission\n"
  )
}
y <- train$pobre

# Categóricas con códigos DANE: entran como factor (una dummy por nivel).
categoricas <- c(
  "dominio", "depto", "tenencia", "jefe_posicion",
  "jefe_regimen", "jefe_tamano", "jefe_oficio"
)

# Todas las variables de la base a nivel hogar entran al modelo.
x_base <- setdiff(names(train), c("id", "pobre"))

# Interacciones de a pares: las 18 variables de 06 más 18 que resumen trabajo,
# formalidad, educación, demografía, vivienda e ingresos no laborales.
x_clave <- c(
  "log_arriendo_pc_lp", "horas_pc", "ocup_pc", "prop_contributivo",
  "prop_subsidiado", "prop_menores", "log_n_ug", "educ_media_adultos",
  "jefe_educ", "cuartos_pc", "formal_pc", "beneficios_pc",
  "dependencia", "n_fuentes_no_lab", "cabecera", "any_ayudas",
  "educ_ocup_media", "horas_formal_pc",
  # Nuevas frente a 06.
  "n_ocup", "n_formal", "prop_formal_ocup", "ocup_por_adulto",
  "educ_max", "prop_superior", "jefe_edad", "n_menor6",
  "pers_por_dormitorio", "log_cuota_pc_lp", "arriendo_es_pagado",
  "any_pension", "any_arriendos", "n_con_beneficios",
  "max_log_meses", "n_empresa_grande", "jefe_formal", "log_lp"
)

# Splines naturales (4 grados de libertad) en lugar de los cuadrados de 06.
x_spline <- c(
  "log_arriendo_pc_lp", "horas_pc", "educ_media_adultos", "jefe_edad",
  "prop_contributivo", "cuartos_pc", "ocup_pc", "educ_ocup_media"
)

# Variables cuyo efecto cambia por dominio (ciudad).
x_dominio <- c(
  "log_arriendo_pc_lp", "horas_pc", "prop_contributivo",
  "educ_media_adultos", "ocup_pc", "formal_pc", "cuartos_pc",
  "prop_menores", "jefe_educ", "log_n_ug"
)

formula_x <- paste(
  "~", paste(x_base, collapse = " + "),
  "+ (", paste(x_clave, collapse = " + "), ")^2",
  "+", paste0("ns(", x_spline, ", df = 4)", collapse = " + "),
  "+ dominio:(", paste(x_dominio, collapse = " + "), ")",
  # El arriendo pesa distinto según la tenencia y las horas según la posición
  # ocupacional del jefe.
  "+ tenencia:log_arriendo_pc_lp + jefe_posicion:horas_pc"
) |> as.formula()

# La matriz se arma con train y test juntos para que los factores tengan los
# mismos niveles en ambas (como en 06). Los nodos de los splines salen de los
# cuantiles de train y test juntos.
todo <- bind_rows(
  train |> select(all_of(x_base)),
  test |> select(all_of(x_base))
) |>
  mutate(across(all_of(categoricas), factor))
# [, -1] (quitar el intercepto) al partir y no sobre la matriz completa, y
# `todo` se borra antes: es la misma matriz con una copia menos en memoria.
mat <- model.matrix(formula_x, todo)
rm(todo)
invisible(gc())
mat_train <- mat[seq_len(nrow(train)), -1]
mat_test <- mat[nrow(train) + seq_len(nrow(test)), -1]
rm(mat)
gc()
cat("Matriz de diseño:", ncol(mat_train), "columnas\n")

# 3. Validación cruzada de lambda y umbral ------------------------------------

# Mismos folds que 06 (una fila = un hogar).
set.seed(123)
k <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

# Lasso (alpha = 1). La grilla de lambda baja hasta 1e-6 x lambda máximo
# porque en 06 el óptimo quedó en el borde inferior (1e-4). Sin parallel:
# en Windows copia la matriz a cada proceso.
inicio <- Sys.time()
cv <- cv.glmnet(mat_train, y,
  family = "binomial", alpha = 1, foldid = fold,
  keep = TRUE, nlambda = 30, lambda.min.ratio = 1e-6
)
cat(
  "cv.glmnet tardó", round(difftime(Sys.time(), inicio, units = "mins"), 1),
  "minutos\n"
)

# fit.preval viene en escala de probabilidad en versiones recientes de glmnet
# y en log odds en otras; se lleva a probabilidad en ambos casos.
prob_cv <- cv$fit.preval
if (any(prob_cv < 0 | prob_cv > 1, na.rm = TRUE)) prob_cv <- plogis(prob_cv)

# F1 promedio de los folds para cada lambda y umbral.
grid_umbral <- seq(0.2, 0.5, by = 0.01)
grid <- map_dfr(seq_along(cv$lambda), function(j) {
  map_dfr(1:k, function(i) {
    p <- prob_cv[fold == i, j]
    real <- y[fold == i]
    tibble(
      lambda = cv$lambda[j], umbral = grid_umbral,
      f1 = map_dbl(grid_umbral, ~ calcular_f1(as.integer(p > .x), real))
    )
  })
}) |>
  summarize(f1 = mean(f1), .by = c(lambda, umbral))

# Mejor umbral para cada lambda, y la mejor combinación.
grid |>
  slice_max(f1, n = 1, by = lambda, with_ties = FALSE) |>
  print(n = Inf)
mejor <- grid |> slice_max(f1, n = 1, with_ties = FALSE)
cat(
  "Mejor: lambda =", signif(mejor$lambda, 3), "- umbral =", mejor$umbral,
  "- F1 =", round(mejor$f1, 4), "\n"
)
cat(
  "Coeficientes distintos de 0:", cv$nzero[cv$lambda == mejor$lambda],
  "de", ncol(mat_train), "\n"
)
cat(
  "Lambda en el borde inferior de la grilla:", mejor$lambda == min(cv$lambda),
  "- en el borde superior:", mejor$lambda == max(cv$lambda), "\n"
)

# 4. Predicción en test -------------------------------------------------------

# cv.glmnet ya estimó el modelo con todo train para cada lambda (glmnet.fit).
prob_test <- predict(
  cv$glmnet.fit, mat_test,
  s = mejor$lambda, type = "response"
)[, 1]
predicciones <- tibble(
  id = test$id, pobre = as.integer(prob_test > mejor$umbral)
)
nrow(predicciones) # una fila por hogar de test
mean(predicciones$pobre) # proporción predicha de pobres (en train es 0,20)

# Nombre con formato Kaggle: modelo_hiperparámetro_valor, sin punto decimal ni
# signo (p. ej. lambda 1.8e-05 -> 18e05, umbral 0.36 -> 036).
sin_punto <- function(v) gsub("[.-]", "", as.character(v))
archivo <- paste0(
  "logit_lasso_inter2_lambda_",
  sin_punto(format(signif(mejor$lambda, 2), scientific = TRUE)),
  "_threshold_", sin_punto(round(mejor$umbral, 2)), ".csv"
)
if (!prueba) {
  dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
  predicciones |> write_csv(here("02_outputs", "predictions", archivo))
}
cat("Submission:", archivo, "\n")

# Probabilidades del mejor lambda, fuera de muestra (en el orden de
# train_hogar.rds) y en test, para combinar con otros modelos (11, sección 13).
dir_out <- here("02_outputs", "logs", "logit_interacciones")
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
saveRDS(
  list(
    id_train = train$id, prob_cv = prob_cv[, which(cv$lambda == mejor$lambda)],
    id_test = test$id, prob_test = prob_test, lambda = mejor$lambda,
    umbral = mejor$umbral, f1 = mejor$f1
  ),
  file.path(dir_out, if (prueba) "logit_oof_prueba.rds" else "logit_oof.rds")
)

# Notas -----------------------------------------------------------------------

# Matriz de diseño: 1159 columnas (497 en 06). Lo que más agrega son las 630
# interacciones de a pares entre las 36 variables de x_clave y las
# interacciones con dominio (23 dummies x 10 variables).

# Resultado: F1 de CV 0.7451 (06: 0.7417) con lambda = 5.4e-05, umbral 0.37 y
# 794 de 1159 coeficientes distintos de 0. Al bajar la grilla hasta 1e-6 x
# lambda máximo, el óptimo ya no queda en el borde: el F1 es plano (0.744-
# 0.745) desde lambda 1.4e-04 hacia abajo, así que más flexibilidad ya no
# mejora y la penalización empieza a ayudar un poco. glmnet paró en 25 de los
# 30 lambdas porque la devianza dejó de cambiar.

# Los cuadrados de 06 se reemplazan por splines naturales (ns, df = 4). ns()
# avisa que mueve nodos interiores en prop_contributivo (muchos 0 y 1); no
# afecta la estimación.

# Proporción predicha de pobres en test: 0.231 (06: 0.232). Tarda ~55 minutos
# (cv.glmnet) y ocupa ~9 GB de RAM con la matriz densa.
