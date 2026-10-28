###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   04_naive_bayes.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar un modelo naive bayes para predecir pobreza y
#                      evaluarlo con F1 por validación cruzada.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Validación cruzada
# 4. Grid search de laplace y umbral
# 5. Predicción en test

# Input:  Bases limpias creadas en 01_data_cleaning.R; train_clean.rds (o
#         train_clean_augmented.rds), test_clean.rds .
# Output: F1 por fold y promedio y resultados del grid search (en consola);
#         02_outputs/predictions/NB_<base>_laplace_<l>_threshold_<c>.csv .
#
# Con PRUEBA=1 usa el 10% de los hogares y no guarda la submission.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  e1071 # Modelos naive bayes.
)
# Variables y funciones comunes (categoricas, x1, preparar, prob_hogar,
# calcular_f1).
source(here("01_code", "functions", "funciones_comunes.R"))

# 2. Cargar datos -------------------------------------------------------------

# Base de entrenamiento:
#   "train_clean": base original.
#   "train_clean_augmented": hogares pobres replicados hasta balancear las
#     clases (ver 01). Las copias conservan el id y quedan en el mismo fold; al
#     validar, prob_hogar() las junta en un solo hogar.
#   "train_clean_sin_ricos": se entrena sin el 10% de hogares más ricos, pero
#     se valida con todos los hogares (si los ricos salieran también de la
#     validación, el F1 se mediría sobre una población distinta a la de test).
#     Por eso se carga train_clean y se usa hogar_rico.
base <- "train_clean"
train <- read_rds(here(
  "data",
  paste0(if (base == "train_clean_sin_ricos") "train_clean" else base, ".rds")
))
test <- read_rds(here("data", "test_clean.rds"))

# Modo de prueba (PRUEBA=1, como lo corre 00_rundirectory.R): el 10% de los
# hogares de train y de test, y sin submission. Sirve para verificar que corre.
modo_prueba <- Sys.getenv("PRUEBA") == "1"
if (modo_prueba) {
  set.seed(1)
  train <- filter(
    train, id %in% sample(unique(id), round(0.1 * n_distinct(id)))
  )
  test <- filter(test, id %in% sample(unique(id), round(0.1 * n_distinct(id))))
  cat(
    "MODO PRUEBA:", n_distinct(train$id),
    "hogares de train; no se guarda la submission\n"
  )
}

# Filas con las que se entrenan los modelos (en la validación cruzada y en el
# modelo final). La validación usa siempre todas las filas del fold.
entrenar <- if (base == "train_clean_sin_ricos") {
  train$hogar_rico == 0
} else {
  rep(
    TRUE, nrow(train)
  )
}

y <- factor(train$pobre)

# Conjuntos de variables. Se parte de edad y educación y se agrega una
# variable a la vez; si mejora el F1, se conserva. Los resultados están en
# las NOTAS del final.

# Dummies de educación (se usan dentro de select() con all_of()).
x_educ <- c(
  "educ_preescolar", "educ_primaria", "educ_secundaria",
  "educ_media", "educ_superior", "educ_no_sabe"
)

# Edad y educación.
x1 <- train |> select(anos, educ_ninguno)
x2 <- train |> select(anos, anos2, educ_ninguno) # + edad al cuadrado
x3 <- train |> select(anos, anos2, educ_ninguno, educ_preescolar) # + preescolar
# preescolar en vez de ninguno
x4 <- train |> select(anos, anos2, educ_preescolar)
# todas las dummies de educación
x5 <- train |> select(anos, anos2, all_of(x_educ))
# años de educación en vez de dummies
x6 <- train |> select(anos, anos2, anos_educ_imp)

# A partir de x5 (edad y dummies de educación), una variable más de la persona.
x7 <- train |> select(anos, anos2, all_of(x_educ), mujer)
# sexo con menos educación
x8 <- train |> select(anos, anos2, educ_ninguno, mujer)
x9 <- train |> select(anos, anos2, all_of(x_educ), ocupado)
x10 <- train |> select(anos, anos2, all_of(x_educ), afiliado_salud)
x11 <- train |>
  select(anos, anos2, all_of(x_educ), afiliado_salud, regimen_salud)
x12 <- train |> select(anos, anos2, all_of(x_educ), actividad_sem_pasada)
x13 <- train |> select(anos, anos2, all_of(x_educ), cotiza_pension)
# Empleo.
x14 <- train |> select(anos, anos2, all_of(x_educ), meses_empresa)
x15 <- train |> select(anos, anos2, all_of(x_educ), tamano_empresa)
x16 <- train |> select(anos, anos2, all_of(x_educ), oficio)
x17 <- train |> select(anos, anos2, all_of(x_educ), posicion_ocupacional)
x18 <- train |> select(anos, anos2, all_of(x_educ), horas_trabajo_semana)
# Pagos del trabajo y segundo trabajo.
x19 <- train |> select(anos, anos2, all_of(x_educ), recibio_horas_extra)
x20 <- train |> select(anos, anos2, all_of(x_educ), recibio_primas)
x21 <- train |> select(anos, anos2, all_of(x_educ), recibio_bonificaciones)
x22 <- train |> select(anos, anos2, all_of(x_educ), segundo_trabajo)

# Conjunto final: las variables que mejoraron el F1, con años de educación en
# vez de dummies (ver NOTAS). Es el x1 de funciones_comunes.R que usan CART (09)
# y XGBoost por persona (10); aquí x1 es solo la primera prueba.
xtodas <- train |> select(
  anos, anos_educ_imp,
  mujer, ocupado, afiliado_salud, regimen_salud,
  actividad_sem_pasada, log_meses_empresa, tamano_empresa,
  posicion_ocupacional, horas_trabajo_semana, arriendo_pc_sobre_lp,
  prop_ocupados_hogar, tenencia_vivienda, recibio_arriendos_pensiones,
  ayudas_instituciones
)

set.seed(123)

# Folds por hogar: todas las personas de un hogar quedan en el mismo fold,
# porque comparten el valor de pobre.
k <- 10
hogares <- unique(train$id)
fold <- sample(rep(1:k, length.out = length(hogares)))
fold <- fold[match(train$id, hogares)]

# 3. Validación cruzada -------------------------------------------------------

# Probabilidad de que cada hogar sea pobre: promedio de los log odds de sus
# miembros (prob_hogar() de funciones_comunes.R). El hogar se predice pobre si
# pasa del umbral.
p_pobre <- function(modelo, x) predict(modelo, x, type = "raw")[, "1"]

# F1 por validación cruzada de naive bayes con covariables x. Se predice
# igual que en test: se promedian los log odds de los miembros de cada
# hogar y el F1 se calcula con una predicción por hogar. preparar() (de
# funciones_comunes.R) pasa a factor las categóricas y las 0/1, para que naive
# bayes no las modele como normales.
cv_naive_bayes <- function(x, y, fold, id = train$id) {
  x <- preparar(x, binarias = TRUE)

  k <- max(fold)
  f1 <- numeric(k)
  for (i in 1:k) {
    prueba <- fold == i
    modelo <- naiveBayes(x[!prueba & entrenar, ], y[!prueba & entrenar])

    # Todos los miembros de un hogar comparten pobre: se toma uno por hogar.
    hogar <- prob_hogar(p_pobre(modelo, x[prueba, ]), id[prueba]) |>
      left_join(distinct(tibble(id = id[prueba], real = y[prueba])), by = "id")
    f1[i] <- calcular_f1(as.integer(hogar$prob > 0.5), hogar$real)
    cat("Fold", i, "- F1:", round(f1[i], 4), "\n")
  }
  # Promedio entre folds (cada F1 ya usa las probabilidades promediadas).
  cat("F1 promedio de los", k, "folds:", round(mean(f1), 4), "\n")
  invisible(f1)
}

# Pruebas de los conjuntos de variables (comentadas porque tardan; el
# resultado de cada una está en las NOTAS).
# f1_x1 <- cv_naive_bayes(x1, y, fold)
# f1_x2 <- cv_naive_bayes(x2, y, fold)
# f1_x3 <- cv_naive_bayes(x3, y, fold)
# f1_x4 <- cv_naive_bayes(x4, y, fold)
# f1_x5 <- cv_naive_bayes(x5, y, fold)
# f1_x6 <- cv_naive_bayes(x6, y, fold)
# f1_x7 <- cv_naive_bayes(x7, y, fold)
# f1_x8 <- cv_naive_bayes(x8, y, fold)
# f1_x9 <- cv_naive_bayes(x9, y, fold)
# f1_x10 <- cv_naive_bayes(x10, y, fold)
# f1_x11 <- cv_naive_bayes(x11, y, fold)
# f1_x12 <- cv_naive_bayes(x12, y, fold)
# f1_x13 <- cv_naive_bayes(x13, y, fold)
# f1_x14 <- cv_naive_bayes(x14, y, fold)
# f1_x15 <- cv_naive_bayes(x15, y, fold)
# f1_x16 <- cv_naive_bayes(x16, y, fold)
# f1_x17 <- cv_naive_bayes(x17, y, fold)
# f1_x18 <- cv_naive_bayes(x18, y, fold)
# f1_x19 <- cv_naive_bayes(x19, y, fold)
# f1_x20 <- cv_naive_bayes(x20, y, fold)
# f1_x21 <- cv_naive_bayes(x21, y, fold)
# f1_x22 <- cv_naive_bayes(x22, y, fold)

f1_xtodas <- cv_naive_bayes(xtodas, y, fold)

# 4. Grid search de laplace y umbral ------------------------------------------

# laplace: suavizado de las categóricas en naiveBayes() (0 = sin suavizado).
# umbral (c): probabilidad promedio del hogar desde la que se predice pobre.
# Por cada laplace se estima el modelo una vez por fold; todos los umbrales se
# evalúan con esas mismas probabilidades, así que probar muchos casi no cuesta.
grid_laplace <- c(0, 0.5, 1, 2, 5, 10, 50)
grid_umbral <- seq(0.1, 0.6, by = 0.01)

# F1 promedio de los folds para cada combinación de laplace y umbral.
grid_search <- function(x, y, fold, id = train$id) {
  x <- preparar(x, binarias = TRUE)
  # Un proceso por valor de laplace (en paralelo).
  resultados <- parallel::mclapply(grid_laplace, function(l) {
    map_dfr(1:max(fold), function(i) {
      prueba <- fold == i
      modelo <- naiveBayes(
        x[!prueba & entrenar, ], y[!prueba & entrenar],
        laplace = l
      )
      hogar <- prob_hogar(p_pobre(modelo, x[prueba, ]), id[prueba]) |>
        left_join(
          distinct(tibble(id = id[prueba], real = y[prueba])),
          by = "id"
        )
      tibble(
        laplace = l, fold = i, umbral = grid_umbral,
        f1 = map_dbl(
          grid_umbral,
          ~ calcular_f1(as.integer(hogar$prob > .x), hogar$real)
        )
      )
    })
  }, mc.cores = min(length(grid_laplace), parallel::detectCores() - 2))
  bind_rows(resultados) |> summarize(f1 = mean(f1), .by = c(laplace, umbral))
}

grid_xtodas <- grid_search(xtodas, y, fold)

# Mejor umbral para cada laplace, y la mejor combinación.
grid_xtodas |>
  slice_max(f1, n = 1, by = laplace, with_ties = FALSE) |>
  print()
# Desempate: entre las combinaciones a menos de 0.0005 del mejor F1 (ruido),
# se toma el menor laplace (el modelo con menos suavizado).
mejor <- grid_xtodas |>
  filter(f1 >= max(f1) - 0.0005) |>
  arrange(laplace, desc(f1)) |>
  slice(1)
cat(
  "Mejor: laplace =", mejor$laplace, "- umbral =", mejor$umbral,
  "- F1 =", round(mejor$f1, 4), "\n"
)

# 5. Predicción en test -------------------------------------------------------

# Mejor modelo (xtodas) con los hiperparámetros del grid search y el mismo
# tratamiento de variables que en la validación cruzada.
modelo_xtodas <- naiveBayes(
  preparar(xtodas, binarias = TRUE)[entrenar, ], y[entrenar],
  laplace = mejor$laplace
)
# Una predicción por hogar: promedio de los log odds de sus miembros.
predicciones <- prob_hogar(
  p_pobre(
    modelo_xtodas,
    preparar(test |> select(all_of(names(xtodas))), binarias = TRUE)
  ),
  test$id
)
nrow(predicciones) # una fila por hogar de test

predicciones <- predicciones |>
  transmute(id, pobre = as.integer(prob > mejor$umbral))

# Nombre con formato Kaggle: modelo_hiperparámetro_valor, sin punto decimal
# (p. ej. NB_laplace_0_threshold_045.csv; con otra base, NB_aug_... o
# NB_sinricos_...). Sale de los valores del grid search.
sin_punto <- function(v) gsub(".", "", as.character(round(v, 2)), fixed = TRUE)
etiqueta_base <- c(
  train_clean = "", train_clean_augmented = "aug_",
  train_clean_sin_ricos = "sinricos_"
)[[base]]
archivo <- paste0(
  "NB_", etiqueta_base, "laplace_", sin_punto(mejor$laplace),
  "_threshold_", sin_punto(mejor$umbral), ".csv"
)
if (!modo_prueba) {
  dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
  predicciones |> write_csv(here("02_outputs", "predictions", archivo))
}

# Notas -----------------------------------------------------------------------

# Tipos de variables: naiveBayes() modela toda columna numérica como normal,
# aunque sea el código de una categórica. Las categóricas y las 0/1 deben ir
# como factor (preparar()), con el mismo tipo en train y en test, o predict()
# falla.

# NA: naive bayes ignora las variables en NA. Si el NA significa algo hay que
# recodificarlo: pasar ocupado de NA a 0 (no ocupado) subió el F1 de 0.29 a
# 0.41.

# Predicción por hogar: pobre es del hogar, así que se combinan las
# probabilidades de sus miembros. Promediar los log odds da mejor F1 que
# promediar probabilidades (0.671 vs 0.660, mejor en los 10 folds). Usar solo
# los adultos o solo el jefe empeora: los niños también informan (los hogares
# pobres tienen más dependientes).

# Umbral: las probabilidades de naive bayes están descalibradas (sobrestiman la
# pobreza porque cuentan varias veces la misma evidencia), así que el umbral se
# escoge por validación cruzada. Para comparar conjuntos de variables hay que
# usar el mejor umbral de cada uno: con 0.5, cotiza_pension como factor parecía
# mucho peor que como numérica y no lo era.

# Variables repetidas: naive bayes supone independencia, así que dos variables
# que miden lo mismo cuentan dos veces la misma información. Tener las dummies
# educ_*, max_educ y anos_educ_imp a la vez cuenta tres veces la educación:
# dejar solo anos_educ_imp subió el F1 de 0.590 a 0.612, y quitar anos2 (repite
# anos) lo subió a 0.623. Por lo mismo empeoran la educación del jefe y la
# máxima del hogar, y los cruces de categóricas parecidas (además tienen muchos
# niveles con pocos datos). Residualizar las variables tampoco sirve: quita la
# correlación lineal pero no la dependencia, y vuelve continuas las dummies.

# Qué aporta: ocupado y régimen de salud (más que afiliación) aportan mucho;
# sexo y formalidad, poco. cotiza_pension solo se pregunta a los ocupados y
# repite lo que dicen salud y actividad de la semana pasada. Los pagos del
# trabajo, otros ingresos y el segundo trabajo aportan poco; lo más útil es
# recibio_arriendos_pensiones. Del hogar, arriendo_pc_sobre_lp y
# prop_ocupados_hogar aportan mucho; personas, cuartos y línea de pobreza
# empeoran.

# Ubicación (dominio, depto, cabecera) empeora en los 10 folds, aunque la
# pobreza varía mucho entre regiones: esa información ya está en las variables
# de la persona (régimen de salud, educación, arriendo) y naive bayes la cuenta
# otra vez. En árboles puede ser distinto.

# Variables asimétricas: la normal les queda mal. Los arriendos crudos empeoran
# el modelo, pero el arriendo per cápita sobre la línea de pobreza en log
# (arriendo_pc_sobre_lp) es la variable que más aporta. log_meses_empresa (NA
# para no ocupados) es mejor que meses_empresa con 0 para los no ocupados.

# Base aumentada: no ayuda en naive bayes, probablemente porque cambia las
# distribuciones que estima. Laplace casi no cambia el F1 (quinta decimal): con
# 500 mil observaciones los conteos por nivel son grandes.

# Tiempo: predict() de e1071 calcula fila por fila, así que con muchas
# variables tarda varios minutos por validación cruzada.
