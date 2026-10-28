###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   06_logit.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar un logit con penalización lasso a nivel hogar,
#                      con interacciones, y escoger lambda y umbral por F1 en
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
#         02_outputs/predictions/logit_lasso_lambda_<l>_threshold_<c>.csv .
#
# Con el argumento --prueba (Rscript 01_code/06_logit.R --prueba) corre con el
# 10% de los hogares de train y no guarda la submission; sirve para verificar
# que el script corre.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  glmnet # Logit con penalización (lasso / elastic net).
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

# Interacciones de a pares entre las variables que más predicen solas (AUC) y
# las que resumen cada bloque: arriendo, trabajo, formalidad, salud,
# demografía, educación y vivienda. El lasso decide cuáles se quedan.
x_clave <- c(
  "log_arriendo_pc_lp", "horas_pc", "ocup_pc", "prop_contributivo",
  "prop_subsidiado", "prop_menores", "log_n_ug", "educ_media_adultos",
  "jefe_educ", "cuartos_pc", "formal_pc", "beneficios_pc",
  "dependencia", "n_fuentes_no_lab", "cabecera", "any_ayudas",
  "educ_ocup_media", "horas_formal_pc"
)

formula_x <- paste(
  "~", paste(x_base, collapse = " + "),
  "+ (", paste(x_clave, collapse = " + "), ")^2",
  # El retorno del arriendo, el trabajo y la educación cambia por ciudad.
  paste0(
    "+ dominio:(log_arriendo_pc_lp + horas_pc + prop_contributivo + ",
    "educ_media_adultos)"
  ),
  # Términos cuadráticos de las continuas más importantes.
  "+ I(log_arriendo_pc_lp^2) + I(horas_pc^2) + I(ocup_pc^2)",
  "+ I(educ_media_adultos^2) + I(prop_menores^2)"
) |> as.formula()

# La matriz se arma con train y test juntos para que los factores tengan los
# mismos niveles en ambas (test no tiene Bogotá; algunos oficios solo
# aparecen en una base). Un nivel que no está en train queda como columna de
# ceros en train y el lasso le da coeficiente 0.
todo <- bind_rows(
  train |> select(all_of(x_base)),
  test |> select(all_of(x_base))
) |>
  mutate(across(all_of(categoricas), factor))
mat <- model.matrix(formula_x, todo)[, -1]
mat_train <- mat[seq_len(nrow(train)), ]
mat_test <- mat[nrow(train) + seq_len(nrow(test)), ]
rm(mat, todo)
cat("Matriz de diseño:", ncol(mat_train), "columnas\n")

# 3. Validación cruzada de lambda y umbral ------------------------------------

# 5 folds (una fila = un hogar). Con 10 folds tarda el doble y el F1 cambia
# en la cuarta decimal.
set.seed(123)
k <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

# Lasso (alpha = 1). keep = TRUE guarda las predicciones fuera de muestra de
# cada lambda, para evaluar el F1 sin volver a estimar.
cv <- cv.glmnet(mat_train, y,
  family = "binomial", alpha = 1, foldid = fold,
  keep = TRUE, nlambda = 25, lambda.min.ratio = 1e-4
)

# fit.preval viene en escala de probabilidad en versiones recientes de glmnet
# y en log odds en otras; se lleva a probabilidad en ambos casos.
prob_cv <- cv$fit.preval
if (any(prob_cv < 0 | prob_cv > 1, na.rm = TRUE)) prob_cv <- plogis(prob_cv)

# F1 promedio de los folds para cada lambda y umbral. El umbral óptimo queda
# por debajo de 0,5 porque solo 20% de los hogares es pobre.
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
cat("Coeficientes distintos de 0:", cv$nzero[cv$lambda == mejor$lambda], "\n")

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
  "logit_lasso_lambda_",
  sin_punto(format(signif(mejor$lambda, 2), scientific = TRUE)),
  "_threshold_", sin_punto(round(mejor$umbral, 2)), ".csv"
)
if (!prueba) {
  dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
  predicciones |> write_csv(here("02_outputs", "predictions", archivo))
}
cat("Submission:", archivo, "\n")

# Notas -----------------------------------------------------------------------

# Estimar a nivel hogar (una fila por hogar con agregados de sus miembros) es
# lo que más importa: pobre y la submission son del hogar, los hogares grandes
# no pesan más y los NA de las preguntas que no aplican quedan como 0 en los
# conteos. glm() elimina las filas con NA, así que no puede quedar ninguno.

# F1 de validación cruzada (5 folds, mejor umbral de cada modelo) al agregar
# bloques a un logit sin penalización:
#   demografía 0.468; + trabajo 0.547; + calidad del empleo 0.617;
#   + salud y educación 0.665; + vivienda e ingresos no laborales 0.726;
#   + ubicación 0.730; + splines e interacciones a mano 0.733;
#   + grupos de oficio, educación de ocupados, oficio del jefe, depto y
#     arriendo pagado vs estimado 0.738.
# El lasso con interacciones de a pares (496 columnas) llega a 0.7415.

# El lasso escoge un lambda casi en el extremo inferior de la grilla y deja
# casi todos los coeficientes: la ganancia viene de las interacciones, no de
# la penalización. Más interacciones pueden mejorar el F1.

# Variables más predictivas por sí solas (AUC): log_arriendo_pc_lp 0.79,
# prop_contributivo 0.76, horas_pc 0.74, cuartos_pc 0.73, ocup_pc 0.72,
# prop_menores 0.72. A diferencia de naive bayes, en logit la ubicación y las
# medidas repetidas (varias de educación) no empeoran el modelo.

# Umbral: el óptimo está entre 0.34 y 0.37, no en 0.5, porque solo 20% de los
# hogares es pobre.
