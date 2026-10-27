# COPIA de script/06_logit_interacciones.r de main (commit 63fde6d, María José
# Pérez), para combinar su logit con el XGBoost por hogar de 05b (sección 13)
# sin tocar el original. Dos cambios, marcados con "CAMBIO 05b":
#   1. Menos memoria al armar la matriz de diseño (en el original ocupa
#      ~9 GB y este Mac tiene 8 GB): sin la copia de [, -1] sobre la matriz
#      completa, borrando `todo` antes de partirla. Es la misma matriz. (La
#      matriz dispersa ocupa menos pero glmnet tarda horas con ella.)
#   2. Al final guarda las probabilidades fuera de muestra (mismos 5 folds
#      que 05b) y de test en output/pruebas/logit/logit_oof.rds.
# Correr desde la raíz: Rscript output/pruebas/logit/06_logit_interacciones_main.r

###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   06_logit_interacciones.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar el logit lasso de 04 con más flexibilidad
#                      (más interacciones, splines e interacciones por
#                      dominio) y escoger lambda y umbral por F1 en
#                      validación cruzada.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos y armar la matriz de diseño
# 3. Validación cruzada de lambda y umbral
# 4. Predicción en test

################################################################################

# Input:  Bases a nivel hogar creadas en 01b_bases_hogar.r; train_hogar.rds ,
#         test_hogar.rds .
# Output: F1 de validación cruzada (en consola);
#         output/submissions/logit_lasso_inter2_lambda_<l>_threshold_<c>.csv .
#
# Con el argumento --prueba (Rscript script/06_logit_interacciones.r --prueba)
# corre con el 10% de los hogares de train y de test y no guarda la
# submission; sirve para verificar que el script corre.

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  glmnet,     # Logit con penalización (lasso / elastic net).
  splines     # Splines naturales (ns).
)

prueba <- "--prueba" %in% commandArgs(trailingOnly = TRUE)

################################################################################

## 2. Cargar datos y armar la matriz de diseño

train <- read_rds(here("data", "train_hogar.rds"))
test  <- read_rds(here("data", "test_hogar.rds"))

if (prueba) {
  set.seed(1)
  train <- slice_sample(train, prop = 0.1)
  test  <- slice_sample(test, prop = 0.1)
  cat("MODO PRUEBA:", nrow(train), "hogares de train; no se guarda la submission\n")
}
y <- train$pobre

# Categóricas con códigos DANE: entran como factor (una dummy por nivel).
categoricas <- c("dominio", "depto", "tenencia", "jefe_posicion",
                 "jefe_regimen", "jefe_tamano", "jefe_oficio")

# Todas las variables de la base a nivel hogar entran al modelo.
x_base <- setdiff(names(train), c("id", "pobre"))

# Interacciones de a pares: las 18 variables de 04 más 18 que resumen trabajo,
# formalidad, educación, demografía, vivienda e ingresos no laborales.
x_clave <- c("log_arriendo_pc_lp", "horas_pc", "ocup_pc", "prop_contributivo",
             "prop_subsidiado", "prop_menores", "log_n_ug", "educ_media_adultos",
             "jefe_educ", "cuartos_pc", "formal_pc", "beneficios_pc",
             "dependencia", "n_fuentes_no_lab", "cabecera", "any_ayudas",
             "educ_ocup_media", "horas_formal_pc",
             # Nuevas frente a 04.
             "n_ocup", "n_formal", "prop_formal_ocup", "ocup_por_adulto",
             "educ_max", "prop_superior", "jefe_edad", "n_menor6",
             "pers_por_dormitorio", "log_cuota_pc_lp", "arriendo_es_pagado",
             "any_pension", "any_arriendos", "n_con_beneficios",
             "max_log_meses", "n_empresa_grande", "jefe_formal", "log_lp")

# Splines naturales (4 grados de libertad) en lugar de los cuadrados de 04.
x_spline <- c("log_arriendo_pc_lp", "horas_pc", "educ_media_adultos", "jefe_edad",
              "prop_contributivo", "cuartos_pc", "ocup_pc", "educ_ocup_media")

# Variables cuyo efecto cambia por dominio (ciudad).
x_dominio <- c("log_arriendo_pc_lp", "horas_pc", "prop_contributivo",
               "educ_media_adultos", "ocup_pc", "formal_pc", "cuartos_pc",
               "prop_menores", "jefe_educ", "log_n_ug")

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
# mismos niveles en ambas (como en 04). Los nodos de los splines salen de los
# cuantiles de train y test juntos.
todo <- bind_rows(train |> select(all_of(x_base)), test |> select(all_of(x_base))) |>
  mutate(across(all_of(categoricas), factor))
# CAMBIO 05b: [, -1] (quitar el intercepto) al partir, no sobre la matriz
# completa, y `todo` se borra antes.
X <- model.matrix(formula_x, todo)
rm(todo); invisible(gc())
X_train <- X[seq_len(nrow(train)), -1]
X_test  <- X[nrow(train) + seq_len(nrow(test)), -1]
rm(X)
gc()
cat("Matriz de diseño:", ncol(X_train), "columnas\n")

################################################################################

## 3. Validación cruzada de lambda y umbral

# F1 a partir de predicciones y valores reales (0/1).
calcular_f1 <- function(pred, real) {
  vp <- sum(pred == 1 & real == 1)
  2 * vp / (sum(pred == 1) + sum(real == 1))
}

# Mismos folds que 04 (una fila = un hogar).
set.seed(123)
k    <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

# Lasso (alpha = 1). La grilla de lambda baja hasta 1e-6 x lambda máximo
# porque en 04 el óptimo quedó en el borde inferior (1e-4). Sin parallel:
# en Windows copia la matriz a cada proceso.
inicio <- Sys.time()
cv <- cv.glmnet(X_train, y, family = "binomial", alpha = 1, foldid = fold,
                keep = TRUE, nlambda = 30, lambda.min.ratio = 1e-6)
cat("cv.glmnet tardó", round(difftime(Sys.time(), inicio, units = "mins"), 1), "minutos\n")

# fit.preval viene en escala de probabilidad en versiones recientes de glmnet
# y en log odds en otras; se lleva a probabilidad en ambos casos.
prob_cv <- cv$fit.preval
if (any(prob_cv < 0 | prob_cv > 1, na.rm = TRUE)) prob_cv <- plogis(prob_cv)

# F1 promedio de los folds para cada lambda y umbral.
grid_umbral <- seq(0.2, 0.5, by = 0.01)
grid <- map_dfr(seq_along(cv$lambda), function(j) {
  map_dfr(1:k, function(i) {
    p    <- prob_cv[fold == i, j]
    real <- y[fold == i]
    tibble(lambda = cv$lambda[j], umbral = grid_umbral,
           f1 = map_dbl(grid_umbral, ~ calcular_f1(as.integer(p > .x), real)))
  })
}) |>
  summarize(f1 = mean(f1), .by = c(lambda, umbral))

# Mejor umbral para cada lambda, y la mejor combinación.
grid |> slice_max(f1, n = 1, by = lambda, with_ties = FALSE) |> print(n = Inf)
mejor <- grid |> slice_max(f1, n = 1, with_ties = FALSE)
cat("Mejor: lambda =", signif(mejor$lambda, 3), "- umbral =", mejor$umbral,
    "- F1 =", round(mejor$f1, 4), "\n")
cat("Coeficientes distintos de 0:", cv$nzero[cv$lambda == mejor$lambda],
    "de", ncol(X_train), "\n")
cat("Lambda en el borde inferior de la grilla:", mejor$lambda == min(cv$lambda),
    "- en el borde superior:", mejor$lambda == max(cv$lambda), "\n")

################################################################################

## 4. Predicción en test

# cv.glmnet ya estimó el modelo con todo train para cada lambda (glmnet.fit).
prob_test <- predict(cv$glmnet.fit, X_test, s = mejor$lambda, type = "response")[, 1]
predicciones <- tibble(id = test$id, pobre = as.integer(prob_test > mejor$umbral))
nrow(predicciones)          # una fila por hogar de test
mean(predicciones$pobre)    # proporción predicha de pobres (en train es 0,20)

# Nombre con formato Kaggle: modelo_hiperparámetro_valor, sin punto decimal ni
# signo (p. ej. lambda 1.8e-05 -> 18e05, umbral 0.36 -> 036).
sin_punto <- function(v) gsub("[.-]", "", as.character(v))
archivo <- paste0("logit_lasso_inter2_lambda_", sin_punto(format(signif(mejor$lambda, 2), scientific = TRUE)),
                  "_threshold_", sin_punto(round(mejor$umbral, 2)), ".csv")
if (!prueba) {
  dir.create(here("output", "submissions"), showWarnings = FALSE)
  predicciones |> write_csv(here("output", "submissions", archivo))
}
cat("Submission:", archivo, "\n")

# CAMBIO 05b: probabilidades del mejor lambda, fuera de muestra (train, en el
# orden de train_hogar.rds) y en test, para la sección 13 de 05b.
saveRDS(list(id_train = train$id, prob_cv = prob_cv[, which(cv$lambda == mejor$lambda)],
             id_test = test$id, prob_test = prob_test, lambda = mejor$lambda,
             umbral = mejor$umbral, f1 = mejor$f1),
        here("output", "pruebas", "logit", if (prueba) "logit_oof_prueba.rds" else "logit_oof.rds"))

### NOTAS:

# Matriz de diseño: 1159 columnas (497 en 04). Lo que más agrega son las 630
# interacciones de a pares entre las 36 variables de x_clave y las
# interacciones con dominio (23 dummies x 10 variables).

# Resultado: F1 de CV 0.7451 (04: 0.7417) con lambda = 5.4e-05, umbral 0.37 y
# 794 de 1159 coeficientes distintos de 0. Al bajar la grilla hasta 1e-6 x
# lambda máximo, el óptimo ya no queda en el borde: el F1 es plano (0.744-
# 0.745) desde lambda 1.4e-04 hacia abajo, así que más flexibilidad ya no
# mejora y la penalización empieza a ayudar un poco. glmnet paró en 25 de los
# 30 lambdas porque la devianza dejó de cambiar.

# Los cuadrados de 04 se reemplazan por splines naturales (ns, df = 4). ns()
# avisa que mueve nodos interiores en prop_contributivo (muchos 0 y 1); no
# afecta la estimación.

# Proporción predicha de pobres en test: 0.231 (04: 0.232). Tarda ~55 minutos
# (cv.glmnet) y ocupa ~9 GB de RAM con la matriz densa.
