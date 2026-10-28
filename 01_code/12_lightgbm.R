###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   12_lightgbm.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar gradient boosting (LightGBM) a nivel hogar con
#                      las mismas variables y folds que el logit, escoger
#                      hiperparámetros y umbral por F1 en validación cruzada
#                      y medir la importancia de las variables.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos y armar la matriz
# 3. Funciones: F1 y validación cruzada de una configuración
# 4. Configuraciones: grilla de clase y random search
# 5. Validación cruzada de todas las configuraciones (en paralelo)
# 6. Tabla de las mejores configuraciones
# 7. Modelo final y predicción en test
# 8. Importancia por ganancia

# Input:  Bases a nivel hogar creadas en 02_bases_hogar.R; train_hogar.rds ,
#         test_hogar.rds .
# Output: 02_outputs/tables/boosting_grid.tex ;
#         02_outputs/figures/boosting_importancia_ganancia.png ;
#         02_outputs/predictions/LGBM_lr_<lr>_leaves_<l>_depth_<d>_rounds_<n>_
#         threshold_<c>.csv ;
#         02_outputs/logs/boosting/: grid.rds (todas las configuraciones),
#         oof.rds (probabilidades fuera de muestra de la mejor),
#         importancia.rds y modelo_final.txt (modelo final).
#
# Correr con source(..., encoding = "UTF-8") para que las tildes salgan bien
# en Windows. Con prueba <- TRUE antes del source() corre 2 configuraciones
# con pocas iteraciones y no guarda nada.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos y gráficas.
  lightgbm, # Gradient boosting.
  knitr, # Exportar la tabla a LaTeX.
  parallel # Procesos de R en paralelo.
)
# Funciones comunes (calcular_f1, marca, ...).
source(here("01_code", "functions", "funciones_comunes.R"), encoding = "UTF-8")

prueba <- exists("prueba") && isTRUE(prueba)
dir_boost <- here("02_outputs", "logs", "boosting")
dir.create(dir_boost, showWarnings = FALSE, recursive = TRUE)

# 2. Cargar datos y armar la matriz -------------------------------------------

train <- read_rds(here("data", "train_hogar.rds"))
test <- read_rds(here("data", "test_hogar.rds"))
y <- train$pobre

# Categóricas con códigos DANE (las mismas de 06).
categoricas <- c(
  "dominio", "depto", "tenencia", "jefe_posicion",
  "jefe_regimen", "jefe_tamano", "jefe_oficio"
)

# Level playing field: exactamente las variables del logit, sin interacciones
# ni splines (el boosting las encuentra solo).
x_base <- setdiff(names(train), c("id", "pobre"))

# Las categóricas pasan a códigos enteros con los mismos niveles en train y
# test: los niveles se arman con las dos bases juntas, como en 06.
todo <- bind_rows(
  train |> select(all_of(x_base)),
  test |> select(all_of(x_base))
) |>
  mutate(across(all_of(categoricas), ~ as.integer(factor(.x))))
mat <- data.matrix(todo)
mat_train <- mat[seq_len(nrow(train)), ]
mat_test <- mat[nrow(train) + seq_len(nrow(test)), ]
rm(mat, todo)
cat(
  "Matriz:", nrow(mat_train), "hogares de train,", ncol(mat_train),
  "variables\n"
)

# Mismos folds que 06 (una fila = un hogar). lgb.cv recibe la lista de
# índices de validación de cada fold.
set.seed(123)
k <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))
folds <- map(1:k, ~ which(fold == .x))

# 3. Funciones: F1 y validación cruzada de una configuración ------------------

grid_umbral <- seq(0.1, 0.6, by = 0.01)

# Validación cruzada de una configuración: lgb.cv con early stopping en
# logloss y, con los boosters de cada fold en la mejor iteración, las
# probabilidades fuera de muestra y el F1 promedio de los folds por umbral.
# Sin ajuste de desbalance: se maneja con el umbral, como en el logit.
cv_config <- function(config, nrounds = 5000, num_threads = 4) {
  inicio <- Sys.time()
  params <- list(
    objective = "binary", metric = "binary_logloss",
    num_threads = num_threads, seed = 123, verbose = -1
  ) |>
    c(config$params)
  # El Dataset se arma de nuevo en cada configuración: algunos parámetros
  # (min_data_in_leaf) no se pueden cambiar en un Dataset ya construido.
  dtrain <- lgb.Dataset(
    data = mat_train, label = y,
    categorical_feature = categoricas
  )
  cv <- lgb.cv(
    params = params, data = dtrain, nrounds = nrounds,
    folds = folds, early_stopping_rounds = 100, verbose = -1
  )

  oof <- numeric(length(y))
  for (i in seq_along(folds)) {
    oof[folds[[i]]] <- predict(
      cv$boosters[[i]]$booster, mat_train[folds[[i]], ],
      num_iteration = cv$best_iter
    )
  }
  f1_umbral <- map_dfr(1:k, function(i) {
    p <- oof[fold == i]
    real <- y[fold == i]
    tibble(
      umbral = grid_umbral,
      f1 = map_dbl(grid_umbral, ~ calcular_f1(as.integer(p > .x), real))
    )
  }) |>
    summarize(f1 = mean(f1), .by = umbral)
  mejor <- slice_max(f1_umbral, f1, n = 1, with_ties = FALSE)

  resultado <- list(
    id = config$id, origen = config$origen, params = config$params,
    num_threads = num_threads, mejor_iter = cv$best_iter,
    logloss = cv$best_score, umbral = mejor$umbral, f1 = mejor$f1,
    minutos = as.numeric(difftime(Sys.time(), inicio, units = "mins")),
    oof = oof
  )
  if (!prueba) {
    write_rds(
      resultado, file.path(dir_boost, sprintf("config_%02d.rds", config$id))
    )
  }
  resultado
}

# 4. Configuraciones: grilla de clase y random search -------------------------

# (a) Grilla de clase (complementaria 10): 18 configuraciones, el resto de
# parámetros por defecto.
grilla_clase <- expand.grid(
  learning_rate = c(0.01, 0.05),
  num_leaves = c(4L, 16L, 32L),
  max_depth = c(-1L, 2L, 6L)
)

# (b) Random search: 30 configuraciones al azar sobre una grilla más amplia.
set.seed(123)
n_random <- 30
grilla_random <- tibble(
  learning_rate = sample(c(0.02, 0.05, 0.1), n_random, replace = TRUE),
  num_leaves = sample(
    c(15L, 31L, 63L, 127L, 255L), n_random,
    replace = TRUE
  ),
  max_depth = sample(c(-1L, 4L, 6L, 8L), n_random, replace = TRUE),
  min_data_in_leaf = sample(c(20L, 50L, 100L, 300L), n_random, replace = TRUE),
  feature_fraction = sample(c(0.5, 0.7, 0.9), n_random, replace = TRUE),
  bagging_fraction = sample(c(0.7, 0.9, 1), n_random, replace = TRUE),
  lambda_l2 = sample(c(0, 1, 10), n_random, replace = TRUE)
)

# Lista de configuraciones; bagging_freq = 1 solo si hay bagging.
configs <- c(
  map(seq_len(nrow(grilla_clase)), ~ list(
    origen = "Grilla de clase",
    params = as.list(grilla_clase[.x, ])
  )),
  map(seq_len(n_random), function(i) {
    p <- as.list(grilla_random[i, ])
    if (p$bagging_fraction < 1) p$bagging_freq <- 1L
    list(origen = "Random search", params = p)
  })
)
configs <- imap(configs, ~ c(list(id = .y), .x))
cat("Configuraciones:", length(configs), "\n")

# 5. Validación cruzada de todas las configuraciones (en paralelo) ------------

# 8 procesos de R con 4 hilos cada uno (32 núcleos). Cada proceso recibe la
# configuración como argumento de la función y guarda su resultado en
# 02_outputs/logs/boosting/config_XX.rds a medida que termina.
n_nucleos <- detectCores()
n_procesos <- 8
n_hilos <- max(1, n_nucleos %/% n_procesos)
cat(
  "Núcleos:", n_nucleos, "- procesos:", n_procesos, "- hilos por proceso:",
  n_hilos, "\n"
)

if (prueba) {
  configs <- configs[c(1, 19)]
  nrounds <- 50
} else {
  nrounds <- 5000
}

inicio <- Sys.time()
cl <- makeCluster(n_procesos, outfile = file.path(dir_boost, "procesos.log"))
clusterEvalQ(cl, suppressMessages({
  library(tidyverse)
  library(lightgbm)
}))
clusterExport(cl, c(
  "mat_train", "y", "fold", "folds", "k", "categoricas",
  "calcular_f1", "grid_umbral", "cv_config", "prueba", "dir_boost"
))
resultados <- clusterApplyLB(cl, configs, function(config, nrounds, n_hilos) {
  cv_config(config, nrounds = nrounds, num_threads = n_hilos)
}, nrounds = nrounds, n_hilos = n_hilos)
stopCluster(cl)
cat(
  "Búsqueda de hiperparámetros:",
  round(difftime(Sys.time(), inicio, units = "mins"), 1),
  "minutos\n"
)

# Tabla con todas las configuraciones.
grid <- map_dfr(resultados, function(r) {
  tibble(
    id = r$id, origen = r$origen,
    learning_rate = r$params$learning_rate,
    num_leaves = r$params$num_leaves,
    max_depth = r$params$max_depth,
    min_data_in_leaf = r$params$min_data_in_leaf %||% 20L,
    feature_fraction = r$params$feature_fraction %||% 1,
    bagging_fraction = r$params$bagging_fraction %||% 1,
    lambda_l2 = r$params$lambda_l2 %||% 0,
    num_threads = r$num_threads, mejor_iter = r$mejor_iter,
    logloss = r$logloss, umbral = r$umbral, f1 = r$f1, minutos = r$minutos
  )
}) |>
  arrange(desc(f1))
print(grid, n = Inf, width = Inf)

# Mejor de cada estrategia de búsqueda.
grid |>
  slice_max(f1, n = 1, by = origen) |>
  select(origen, id, f1, umbral) |>
  print()

mejor <- grid |> slice(1)
mejor_res <- keep(resultados, ~ .x$id == mejor$id)[[1]]
cat(
  "Mejor configuración: id", mejor$id, "(", mejor$origen, ") - F1 =",
  round(mejor$f1, 4), "- umbral =", mejor$umbral, "- iteraciones =",
  mejor$mejor_iter, "\n"
)

if (!prueba) {
  write_rds(grid, file.path(dir_boost, "grid.rds"))
  write_rds(
    tibble(id = train$id, fold = fold, pobre = y, prob = mejor_res$oof),
    file.path(dir_boost, "oof.rds")
  )
}

# 6. Tabla de las mejores configuraciones -------------------------------------

f1_logit <- 0.7451 # Logit lasso con interacciones de 08 (F1 de CV).

tabla <- grid |>
  slice_head(n = 10) |>
  transmute(origen, learning_rate, num_leaves, max_depth, min_data_in_leaf,
    feature_fraction, bagging_fraction, lambda_l2,
    mejor_iter = as.character(mejor_iter), umbral, f1
  ) |>
  bind_rows(tibble(
    origen = "Logit lasso con interacciones (06)",
    mejor_iter = "", umbral = 0.37, f1 = f1_logit
  ))

if (!prueba) {
  options(knitr.kable.NA = "")
  kable(tabla,
    format = "latex", booktabs = TRUE,
    digits = c(0, 2, 0, 0, 0, 1, 1, 0, 0, 2, 4),
    col.names = c(
      "Búsqueda", "Learning rate", "Hojas", "Profundidad",
      "Mín. datos hoja", "Fracción variables", "Fracción bagging",
      "L2", "Iteraciones", "Umbral", "F1 (CV)"
    ),
    caption = paste0(
      "Las 10 mejores configuraciones de LightGBM por F1 de validación ",
      "cruzada (5 folds)"
    ),
    label = "boosting_grid"
  ) |>
    # enc2utf8 + useBytes: en Windows writeLines pasaría las tildes a CP1252.
    enc2utf8() |>
    writeLines(
      here("02_outputs", "tables", "boosting_grid.tex"),
      useBytes = TRUE
    )
}

# 7. Modelo final y predicción en test ----------------------------------------

# Se estima con todo train y la mejor configuración. Como hay un 25% más de
# datos que en cada fold, se usan 10% más iteraciones que la mejor de la CV.
n_final <- round(mejor$mejor_iter * 1.1)
params_final <- list(
  objective = "binary", metric = "binary_logloss",
  num_threads = n_nucleos, seed = 123, verbose = -1
) |>
  c(mejor_res$params)
dtrain <- lgb.Dataset(
  data = mat_train, label = y, categorical_feature = categoricas
)
modelo <- lgb.train(
  params = params_final, data = dtrain, nrounds = n_final,
  verbose = -1
)

prob_test <- predict(modelo, mat_test)
predicciones <- tibble(
  id = test$id, pobre = as.integer(prob_test > mejor$umbral)
)
nrow(predicciones) # una fila por hogar de test
mean(predicciones$pobre) # proporción predicha de pobres (en train es 0,20)

# Nombre con formato Kaggle, sin punto decimal (0.05 -> 005, umbral 0.37 ->
# 037). Profundidad -1 (sin límite) se escribe m1.
sin_punto <- function(v) gsub("[.]", "", sub("^-", "m", as.character(v)))
archivo <- paste0(
  "LGBM_lr_", sin_punto(mejor$learning_rate),
  "_leaves_", mejor$num_leaves,
  "_depth_", sin_punto(mejor$max_depth),
  "_rounds_", n_final,
  "_threshold_", sin_punto(round(mejor$umbral, 2)), ".csv"
)
if (!prueba) {
  dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
  predicciones |> write_csv(here("02_outputs", "predictions", archivo))
  lgb.save(modelo, file.path(dir_boost, "modelo_final.txt"))
}
cat("Submission:", archivo, "\n")

# 8. Importancia por ganancia -------------------------------------------------

# lgb.importance da la ganancia como fracción del total (suma 1). Las
# variables que el modelo nunca usa no aparecen: se agregan con ganancia 0.
importancia <- lgb.importance(modelo, percentage = TRUE) |>
  as_tibble() |>
  select(
    variable = Feature, ganancia = Gain, cobertura = Cover,
    frecuencia = Frequency
  )
importancia <- tibble(variable = x_base) |>
  left_join(importancia, by = "variable") |>
  mutate(across(c(ganancia, cobertura, frecuencia), ~ replace_na(.x, 0))) |>
  arrange(desc(ganancia))
print(importancia, n = Inf)
cat(
  "Variables con ganancia < 0,1% del total:", sum(importancia$ganancia < 0.001),
  "de", nrow(importancia), "\n"
)

grafica <- importancia |>
  slice_head(n = 20) |>
  ggplot(aes(x = ganancia, y = fct_reorder(variable, ganancia))) +
  geom_col(fill = "steelblue") +
  scale_x_continuous(labels = scales::percent) +
  labs(
    title = "LightGBM: las 20 variables más importantes",
    x = "Ganancia (% del total)", y = "Variable"
  ) +
  theme_minimal()

if (!prueba) {
  write_rds(importancia, file.path(dir_boost, "importancia.rds"))
  ggsave(
    here(
      "02_outputs", "figures", "boosting_importancia_ganancia.png"
    ), grafica,
    width = 8, height = 6
  )
}

# Notas -----------------------------------------------------------------------

# Resultado: el mejor LightGBM llega a F1 de CV 0.7537 (umbral 0.37), contra
# 0.7451 del logit lasso con interacciones (08) y 0.7417 de 06, con las
# mismas 94 variables y los mismos folds. El boosting encuentra solo las
# interacciones y no linealidades que en el logit hay que escribir a mano.

# Mejor configuración (random search): learning_rate 0.02, num_leaves 127,
# max_depth 6, min_data_in_leaf 20, feature_fraction 0.7, sin bagging,
# lambda_l2 0; 1313 iteraciones en CV y 1444 en el modelo final. Con
# max_depth 6 un árbol tiene como máximo 64 hojas, así que num_leaves 127 y
# 255 dan el mismo modelo (ids 37 y 38, mismo F1).

# Mejor de la grilla de clase: 0.7529 (learning_rate 0.01, num_leaves 16,
# sin límite de profundidad); mejor del random search: 0.7537. La superficie
# es plana: 42 de las 48 configuraciones quedan entre 0.749 y 0.754. Lo que
# sí empeora el F1 son los árboles muy pequeños (4 hojas o profundidad 2,
# 0.743-0.750): con learning_rate 0.01 llegan al tope de 5000 iteraciones
# sin activar el early stopping. En la grilla de clase, profundidad 2 limita
# a 4 hojas y por eso num_leaves 4, 16 y 32 dan el mismo resultado.

# Importancia por ganancia (modelo final): log_arriendo_pc_lp 13.2%,
# horas_pc 12.3%, prop_contributivo 11.4%, educ_ocup_suma_pc 10.6%,
# formal_pc 5.0%, n_fuentes_no_lab 4.1%, tenencia 3.6%, any_arriendos 3.2%,
# beneficios_pc 2.9%, jefe_oficio 2.7%, dominio 2.4%, depto 2.1%,
# jefe_regimen 2.0%, ocup_pc 1.9%, cuartos_pc 1.4%. Las 4 primeras suman casi
# la mitad de la ganancia. 35 de las 94 variables tienen ganancia menor a
# 0.1% del total. Las mismas variables encabezan el ranking del lasso de 07.

# Proporción predicha de pobres en test: 0.233. Coincide con la submission
# de 08 en el 96% de los hogares.

# Tiempo: ~26 minutos la búsqueda (48 configuraciones en 8 procesos de 4
# hilos, 32 núcleos) y ~1 minuto el modelo final; ~11 GB de RAM en total.
