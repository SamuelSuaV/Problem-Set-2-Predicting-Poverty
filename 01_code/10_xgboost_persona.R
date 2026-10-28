###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   10_xgboost_persona.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar modelos XGBoost para predecir pobreza: elegir
#                      hiperparámetros y umbral por validación cruzada con
#                      caret (F1 por hogar), con distintas bases, y
#                      predecir en test.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Configuración: bases, conjuntos de variables y grillas
# 3. Funciones
#    3.1 Datos: cargar una base y pasar a matriz
#    3.2 Evaluación por hogar
#    3.3 XGBoost para caret
#    3.4 Validación cruzada, importancia y predicción
# 4. Pruebas
# 5. Modelo final: importancia de las variables y predicción en test

# Input:  Bases limpias creadas en 01_data_cleaning.R; train_clean.rds ,
#         train_clean_augmented.rds , test_clean.rds .
# Output: F1 de validación por combinación de hiperparámetros, importancia de
#         las variables (en consola);
#         02_outputs/predictions/XGB_<conjunto>_<base_>nrounds_<n>_depth_<d>_
#         eta_<e>_threshold_<c>.csv .
#
# Con PRUEBA=1 usa el 10% de los hogares y la grilla rápida, y no guarda la
# submission.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  xgboost, # Gradient boosting.
  caret, # Validación cruzada y fine tuning.
  doParallel # Folds de la validación cruzada en paralelo.
)
# Variables y funciones comunes (categoricas, x1, prob_hogar, calcular_f1).
source(here("01_code", "functions", "funciones_comunes.R"))

# 2. Configuración: bases, conjuntos de variables y grillas -------------------

# Bases de entrenamiento disponibles (archivos de data/). Para usar otra, se
# agrega aquí y se pasa su nombre a cv_xgb(base = ...).
#   train_clean: base original.
#   train_clean_augmented: hogares pobres replicados hasta balancear las
#     clases (ver 01). Las copias conservan el id y quedan en el mismo fold; al
#     validar, prob_hogar() las junta en un solo hogar, así que el F1 se mide
#     sobre los hogares originales.
#   train_clean_sin_ricos: se entrena sin el 10% de hogares más ricos, pero se
#     valida con todos los hogares (si los ricos salieran también de la
#     validación, el F1 se mediría sobre una población distinta a la de test).
#     Por eso se carga train_clean y se usa hogar_rico.
bases <- c("train_clean", "train_clean_augmented", "train_clean_sin_ricos")

# Categóricas con códigos DANE: pasan a dummies (una por nivel). Si quedan
# numéricas, los árboles las cortan como si los códigos estuvieran ordenados.
# El texto (dominio, depto) también pasa a dummies. Las de funciones_comunes.R
# más el parentesco con el jefe.
categoricas <- c(categoricas, "parentesco_jefe_hogar")

# Conjuntos de variables (vectores de nombres). Para probar uno nuevo, se
# define aquí y se agrega una línea en la sección 4. x1, el mejor conjunto de
# naive bayes y CART (04 y 09), está en funciones_comunes.R.

# x2: x1 + variables más importantes del CART con todas las variables
# (02_outputs/pruebas/todas/) que no están en x1.
x2 <- c(
  x1, "educ_max_adultos", "linea_pobreza", "cuartos_dormir",
  "prop_dependientes", "personas_hogar"
)

# x3: x2 + ubicación. El boosting combina muchos árboles, así que puede
# aprovechar
# variables que en un solo árbol solo agregaban ruido.
x3 <- c(x2, "dominio", "depto", "cabecera")

# Grillas de hiperparámetros (columnas que pide caret para xgbTree).
#   nrounds: número de árboles. Para cada combinación de los demás, se estima
#     un solo modelo con el mayor nrounds y se evalúa con menos árboles.
#   max_depth: profundidad de cada árbol.
#   eta: tasa de aprendizaje (peso de cada árbol nuevo).
#   gamma: reducción mínima de la pérdida para hacer un corte.
#   min_child_weight: peso mínimo en cada hoja (~ observaciones mínimas).
#   colsample_bytree: fracción de variables que ve cada árbol.
#   subsample: fracción de observaciones que ve cada árbol.
grid_xgb <- expand.grid(
  nrounds = c(250, 500),
  max_depth = c(4, 6),
  eta = c(0.05, 0.1),
  gamma = 0,
  min_child_weight = c(10, 50),
  colsample_bytree = 0.8,
  subsample = 0.63
)

# Grilla ampliada: con grid_xgb el mejor modelo quedó en el borde (500
# rondas, profundidad 6), así que se prueban más rondas y árboles más
# profundos. eta y min_child_weight casi no cambiaban el F1.
grid_ampliada <- expand.grid(
  nrounds = c(500, 1000, 1500),
  max_depth = c(6, 8, 10),
  eta = 0.05,
  gamma = 0,
  min_child_weight = c(10, 50),
  colsample_bytree = 0.8,
  subsample = 0.63
)

# Grilla pequeña para probar rápido que todo corre.
grid_rapida <- expand.grid(
  nrounds = 100, max_depth = 4, eta = 0.1, gamma = 0,
  min_child_weight = 10, colsample_bytree = 0.8,
  subsample = 0.63
)

# Umbral: probabilidad del hogar desde la que se predice pobre.
grid_umbral <- seq(0.10, 0.90, by = 0.01)
nombres_f1 <- sprintf("F1_%03d", round(100 * grid_umbral)) # F1_045 = 0.45

# Paralelismo. El xgboost de CRAN para macOS viene sin OpenMP: el parámetro
# nthread no tiene efecto y cada modelo usa un solo núcleo. Por eso se estiman
# varios folds a la vez (caret con allowParallel = TRUE), cada uno en un
# núcleo. Cada proceso copia los datos del fold: con la base aumentada, más de
# 4 puede quedarse sin memoria (el Mac tiene 8 GB).
nucleos <- 4
registerDoParallel(cores = nucleos)

# Modo de prueba (PRUEBA=1, como lo corre 00_rundirectory.R): el 10% de los
# hogares, la grilla rápida y sin submission. Sirve para verificar que corre.
prueba <- Sys.getenv("PRUEBA") == "1"
if (prueba) grid_xgb <- grid_rapida
submuestra <- function(d) {
  if (!prueba) {
    return(d)
  }
  set.seed(1)
  filter(d, id %in% sample(unique(id), round(0.1 * n_distinct(id))))
}

# 3. Funciones ----------------------------------------------------------------

## 3.1 Datos: cargar una base y pasar a matriz --------------------------------

# Carga una base de entrenamiento y arma lo que no depende de las variables:
# y, id del hogar y folds. Se guarda la última base cargada para no leerla
# otra vez en cada prueba (y no tener dos bases grandes en memoria).
base_actual <- new.env()
cargar_base <- function(base) {
  stopifnot(base %in% bases)
  if (!identical(base_actual$nombre, base)) {
    rm(list = ls(base_actual), envir = base_actual)
    invisible(gc())
    archivo <- if (base == "train_clean_sin_ricos") "train_clean" else base
    datos <- submuestra(read_rds(here("data", paste0(archivo, ".rds"))))
    # caret necesita niveles con nombres válidos para devolver probabilidades
    # (classProbs). La clase positiva va primero.
    base_actual$y <- factor(
      datos$pobre,
      levels = c(1, 0), labels = c("pobre", "no_pobre")
    )
    base_actual$id <- datos$id

    # Folds por hogar (los mismos de 04 y 09: la base aumentada conserva el
    # orden de los hogares de train_clean).
    set.seed(123)
    k <- 10
    hogares <- unique(datos$id)
    fold <- sample(rep(
      1:k,
      length.out = length(hogares)
    ))[match(datos$id, hogares)]
    # Filas con las que se entrenan los modelos (en la validación cruzada y en
    # el modelo final). La validación (index_out) usa todas las filas del fold.
    entrenar <- if (base == "train_clean_sin_ricos") {
      datos$hogar_rico == 0
    } else {
      rep(
        TRUE, nrow(datos)
      )
    }
    nombres <- sprintf("Fold%02d", 1:k)
    base_actual$entrenar <- entrenar
    base_actual$index <- map(1:k, ~ which(fold != .x & entrenar)) |>
      setNames(nombres)
    base_actual$index_out <- map(1:k, ~ which(fold == .x)) |> setNames(nombres)
    base_actual$datos <- datos
    base_actual$nombre <- base
  }
  base_actual
}

# Predictores como matriz numérica, que es lo que recibe xgboost: las
# categóricas pasan a una dummy por nivel. na.pass conserva las filas con NA;
# xgboost trata los NA como faltantes y aprende hacia dónde mandarlos.
# niveles: los de train, para que test tenga exactamente las mismas columnas.
a_matriz <- function(df, variables, niveles = NULL) {
  x <- df |>
    select(all_of(variables)) |>
    mutate(
      across(any_of(categoricas), ~ as.character(.x)),
      across(where(is.character), factor)
    )
  if (!is.null(niveles)) {
    x <- mutate(
      x, across(names(niveles), ~ factor(.x, levels = niveles[[cur_column()]]))
    )
  }
  factores <- names(x)[map_lgl(x, is.factor)]
  mf <- model.frame(~., x, na.action = na.pass)
  m <- model.matrix(~., mf,
    contrasts.arg = map(x[factores], ~ contrasts(.x, contrasts = FALSE))
  )
  # Variable original de cada columna (para sumar la importancia de las
  # dummies).
  origen <- c("(Intercept)", names(x))[attr(m, "assign") + 1]
  m <- m[, -1, drop = FALSE]
  colnames(m) <- make.names(colnames(m), unique = TRUE)
  attr(m, "origen") <- setNames(origen[-1], colnames(m))
  attr(m, "niveles") <- map(x[factores], levels)
  m
}

## 3.2 Evaluación por hogar ---------------------------------------------------

# summaryFunction de caret: F1 por hogar de un fold de validación, uno por
# umbral. Como en 04 y 09, la probabilidad del hogar es el promedio de los log
# odds de sus miembros (prob_hogar()). rowIndex es la fila en la base, con la
# que se recupera el id.
f1_hogar <- function(data, lev = NULL, model = NULL) {
  id <- base_actual$id[data$rowIndex]
  hogar <- prob_hogar(data$pobre, id) |>
    left_join(distinct(tibble(id = id, real = data$obs == "pobre")), by = "id")
  f1 <- map_dbl(
    grid_umbral, ~ calcular_f1(as.integer(hogar$prob > .x), hogar$real)
  )
  setNames(f1, nombres_f1)
}

# selectionFunction de caret: la combinación cuyo mejor umbral da el mayor F1
# promedio. Así el modelo final (finalModel) se estima con esa combinación.
elegir <- function(x, metric, maximize) {
  which.max(apply(x[nombres_f1], 1, max, na.rm = TRUE))
}

## 3.3 XGBoost para caret -----------------------------------------------------

# Modelo propio para caret: method = "xgbTree" no funciona con xgboost 3.x.
# caret agrega campos al modelo estimado (xNames, obsLevels...) y en xgboost 3
# el modelo no se puede modificar, así que se devuelve dentro de una lista
# (modelFit$booster). Mismos hiperparámetros que xgbTree.
hiperparametros <- c(
  "nrounds", "max_depth", "eta", "gamma", "colsample_bytree",
  "min_child_weight", "subsample"
)
xgb_caret <- list(
  label = "XGBoost (xgboost 3)", library = "xgboost", type = "Classification",
  parameters = data.frame(
    parameter = hiperparametros, class = "numeric",
    label = hiperparametros
  ),
  grid = function(x, y, len = NULL, search = "grid") grid_rapida,

  # Para cada combinación de los demás hiperparámetros se estima solo el mayor
  # nrounds; los nrounds menores se evalúan con los primeros árboles.
  loop = function(grid) {
    otros <- setdiff(names(grid), "nrounds")
    loop <- grid |>
      group_by(across(all_of(otros))) |>
      summarize(nrounds = max(nrounds), .groups = "drop") |>
      as.data.frame()
    submodels <- map(seq_len(nrow(loop)), function(i) {
      mismos <- grid |> semi_join(loop[i, otros, drop = FALSE], by = otros)
      data.frame(nrounds = setdiff(mismos$nrounds, loop$nrounds[i]))
    })
    list(loop = loop[names(grid)], submodels = submodels)
  },
  # classProbs y modelFit (nolint): caret pasa estos argumentos por nombre.
  fit = function(x, y, wts, param, lev, last, classProbs, ...) { # nolint
    d <- xgb.DMatrix(
      as.matrix(x),
      label = as.integer(y == lev[1]), missing = NA
    )
    if (!is.null(wts)) setinfo(d, "weight", wts)
    # nthread = 1: el paralelismo está en los folds (ver sección 2). seed fija
    # el submuestreo de cada modelo, sin depender de en qué proceso corre.
    booster <- xgb.train(
      params = list(
        objective = "binary:logistic", nthread = 1, seed = 123,
        eta = param$eta, max_depth = param$max_depth,
        gamma = param$gamma,
        colsample_bytree = param$colsample_bytree,
        min_child_weight = param$min_child_weight,
        subsample = param$subsample
      ),
      data = d, nrounds = param$nrounds, verbose = 0
    )
    list(booster = booster)
  },

  # Probabilidad de cada clase. Con submodels, una tabla por cada nrounds
  # (iterationrange = c(1, n) usa los primeros n árboles).
  prob = function(modelFit, newdata, submodels = NULL) { # nolint
    d <- xgb.DMatrix(as.matrix(newdata), missing = NA)
    a_tabla <- function(p) data.frame(pobre = p, no_pobre = 1 - p)
    out <- a_tabla(predict(modelFit$booster, d))
    if (is.null(submodels)) {
      return(out)
    }
    c(list(out), map(
      submodels$nrounds,
      ~ a_tabla(predict(modelFit$booster, d, iterationrange = c(1, .x)))
    ))
  },
  predict = function(modelFit, newdata, submodels = NULL) { # nolint
    clase <- function(t) {
      factor(ifelse(t$pobre > 0.5, "pobre", "no_pobre"),
        levels = c("pobre", "no_pobre")
      )
    }
    p <- xgb_caret$prob(modelFit, newdata, submodels)
    if (is.data.frame(p)) clase(p) else map(p, clase)
  },
  sort = function(x) x[order(x$nrounds, x$max_depth, x$eta), ],
  levels = function(x) c("pobre", "no_pobre")
)

## 3.4 Validación cruzada, importancia y predicción ---------------------------

# Fine tuning de hiperparámetros y umbral por validación cruzada.
#   variables: conjunto de variables (sección 2).
#   base: una de `bases` (para balancear las clases, la base aumentada).
#   grid: grilla de hiperparámetros.
#   nombre: nombre del conjunto de variables, para la submission (por defecto,
#     el del objeto que se pasa, p. ej. "x1").
# Imprime el mejor resultado y devuelve el modelo, el F1 por combinación y la
# mejor combinación.
cv_xgb <- function(variables, base = "train_clean", grid = grid_xgb,
                   nombre = deparse(substitute(variables))) {
  b <- cargar_base(base)
  x <- a_matriz(b$datos, variables)

  # Semilla en cada llamada: el submuestreo de xgboost (subsample,
  # colsample_bytree) es aleatorio, y así cada prueba da el mismo resultado sin
  # importar qué se corrió antes.
  set.seed(123)
  modelo <- train(
    x = x, y = b$y, method = xgb_caret,
    metric = "F1_050", # caret exige una métrica; elegir() decide.
    tuneGrid = grid,
    trControl = trainControl(
      method = "cv", index = b$index, indexOut = b$index_out,
      classProbs = TRUE, summaryFunction = f1_hogar,
      selectionFunction = elegir,
      returnData = FALSE, allowParallel = TRUE
    )
  )

  # Mejor umbral y F1 para cada combinación de hiperparámetros.
  por_combinacion <- modelo$results |>
    select(all_of(hiperparametros), all_of(nombres_f1)) |>
    pivot_longer(all_of(nombres_f1), names_to = "umbral", values_to = "f1") |>
    mutate(umbral = as.numeric(sub("F1_", "", umbral)) / 100) |>
    slice_max(f1, n = 1, by = all_of(hiperparametros), with_ties = FALSE) |>
    arrange(desc(f1))
  mejor <- por_combinacion[1, ]

  # caret estima el modelo final con todas las filas. Si se entrena sin una
  # parte de la base (train_clean_sin_ricos), se vuelve a estimar solo con las
  # filas de entrenamiento y los hiperparámetros elegidos.
  if (!all(b$entrenar)) {
    set.seed(123)
    modelo <- train(
      x = x[b$entrenar, , drop = FALSE], y = b$y[b$entrenar],
      method = xgb_caret,
      tuneGrid = as.data.frame(mejor[hiperparametros]),
      trControl = trainControl(method = "none", classProbs = TRUE)
    )
  }

  cat(
    nombre, "(", length(variables), "variables ) -", base,
    "- nrounds =", mejor$nrounds, "- max_depth =", mejor$max_depth,
    "- eta =", mejor$eta, "- min_child_weight =", mejor$min_child_weight,
    "- umbral =", mejor$umbral, "- F1 =", round(mejor$f1, 4), "\n"
  )
  invisible(list(
    modelo = modelo, por_combinacion = por_combinacion, mejor = mejor,
    variables = variables, nombre = nombre, base = base,
    origen = attr(x, "origen"), niveles = attr(x, "niveles")
  ))
}

# Importancia (ganancia: reducción de la pérdida en los cortes de cada
# variable) del modelo final, normalizada para que sume 100. Las dummies de una
# categórica se suman en su variable original.
importancia_xgb <- function(resultado) {
  xgb.importance(model = resultado$modelo$finalModel$booster) |>
    as_tibble() |>
    mutate(variable = resultado$origen[Feature]) |>
    summarize(importancia = sum(Gain), .by = variable) |>
    mutate(importancia = 100 * importancia / sum(importancia)) |>
    arrange(desc(importancia))
}

# Predicción en test de un resultado de cv_xgb(): una fila por hogar, con los
# hiperparámetros y el umbral elegidos. Guarda la submission con formato Kaggle
# (modelo_hiperparámetro_valor, sin punto decimal).
predecir_test <- function(resultado) {
  test <- submuestra(read_rds(here("data", "test_clean.rds")))
  # Mismos niveles que train: un nivel que no está en train queda NA y las
  # columnas de la matriz son las mismas.
  x_test <- a_matriz(test, resultado$variables, niveles = resultado$niveles)
  stopifnot(identical(colnames(x_test), names(resultado$origen)))

  p <- predict(resultado$modelo, newdata = x_test, type = "prob")[, "pobre"]
  predicciones <- prob_hogar(p, test$id) |>
    transmute(id, pobre = as.integer(prob > resultado$mejor$umbral))

  sin_punto <- function(v) {
    gsub(
      ".", "", format(signif(v, 2), scientific = FALSE),
      fixed = TRUE
    )
  }
  m <- resultado$mejor
  archivo <- paste0(
    "XGB_", resultado$nombre, "_",
    c(
      train_clean = "", train_clean_augmented = "aug_",
      train_clean_sin_ricos = "sinricos_"
    )[[resultado$base]],
    "nrounds_", m$nrounds, "_depth_", m$max_depth,
    "_eta_", sin_punto(m$eta), "_threshold_", sin_punto(m$umbral), ".csv"
  )
  if (!prueba) {
    dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
    write_csv(predicciones, here("02_outputs", "predictions", archivo))
  }
  cat("Submission:", archivo, "-", nrow(predicciones), "hogares\n")
  invisible(predicciones)
}

# 4. Pruebas ------------------------------------------------------------------

# Prueba rápida de que todo corre (una sola combinación de hiperparámetros):
# cv_rapida <- cv_xgb(x1, grid = grid_rapida)

# Con grid_xgb (16 combinaciones, 8 modelos por fold) y 4 núcleos, cada prueba
# tarda ~11 minutos con train_clean y ~17 con la base aumentada. Las pruebas
# comentadas se corrieron con los scripts de 01_code/pruebas/xgb/ (bases.R,
# ampliada.R y conjuntos.R), que guardan cada resultado. Arriba de cada una,
# su resultado: F1 (nrounds, max_depth, eta, min_child_weight, umbral).

# F1 = 0.693 (500, 6, 0.1, 50, 0.29)
cv_x1 <- cv_xgb(x1)
# F1 = 0.693 (mismos hiperparámetros, umbral 0.64)
# cv_x1_aug <- cv_xgb(x1, base = "train_clean_augmented")
# F1 = 0.694 (mismos hiperparámetros, umbral 0.29)
# cv_x1_sinricos <- cv_xgb(x1, base = "train_clean_sin_ricos")
# Grilla ampliada (18 combinaciones, 6 modelos de 1.500 árboles por fold).
# F1 = 0.693 (500, 8, 0.05, 10, 0.31)
# cv_x1_amp <- cv_xgb(x1, grid = grid_ampliada)
# F1 = 0.702 (500, 6, 0.05, 50, 0.32)
# cv_x2 <- cv_xgb(x2)
# F1 = 0.706 (500, 6, 0.1, 50, 0.32)
# cv_x3 <- cv_xgb(x3)

# F1 de cada combinación de hiperparámetros de una prueba:
# print(cv_x1$por_combinacion, n = Inf)

# 5. Modelo final: importancia de las variables y predicción en test ----------

# El mejor conjunto es x3 (F1 0.706); su submission salió de
# 01_code/pruebas/xgb/conjuntos.R. Para repetirla aquí: final <- cv_xgb(x3).
final <- cv_x1

print(importancia_xgb(final), n = Inf)

predecir_test(final)

# Notas -----------------------------------------------------------------------

# Base: la aumentada y la que entrena sin los ricos dan el mismo F1 que la
# original (0.693-0.694). Balancear las clases solo mueve el umbral (0.29 ->
# 0.64); el orden de los hogares por probabilidad casi no cambia.

# Hiperparámetros: la grilla ampliada (más rondas, árboles más profundos) no
# mejora (0.693). eta y min_child_weight casi no cambian el F1.

# Variables: lo que sí ayuda son más variables. x2 (+ educación de los adultos,
# línea de pobreza, cuartos, dependientes y personas) sube a 0.702 y x3
# (+ ubicación) a 0.706. A diferencia de naive bayes y de un solo árbol, el
# boosting aprovecha la ubicación.

# Límite del modelo por persona: predice a cada persona y promedia en el hogar.
# El XGBoost a nivel hogar (11), con agregados de todos los miembros, llega
# mucho más alto.
