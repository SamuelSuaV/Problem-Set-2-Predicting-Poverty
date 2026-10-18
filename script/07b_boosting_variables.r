###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   07b_boosting_variables.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Probar si variables nuevas (factor de expansión, mejor
#                      ocupado del hogar, target encoding del oficio e
#                      ingreso predicho) y un ensamble con el logit suben el
#                      F1 del boosting de 07, con sus hiperparámetros fijos.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Funciones comunes (F1, AUC, hiperparámetros)
# 4. Variables que no dependen de los folds (V1 y V2)
# 5. Variables que dependen de los folds (V3 y V4) y logit 06, en paralelo
# 6. Experimentos E0-E4 (en paralelo)
# 7. Experimentos E5 y E6
# 8. Ensamble con el logit 06
# 9. Modelos finales y submissions

################################################################################

# Input:  data/train_hogar.rds , data/test_hogar.rds (01b);
#         data/train_clean.rds , data/test_clean.rds (01);
#         data/train_personas.csv (solo Ingtot, para el ingreso predicho).
# Output: output/tables/boosting_variables.tex ;
#         output/logs/boosting_variables/: probabilidades fuera de muestra
#         (oof_E0.rds ... oof_E6.rds, oof_logit06.rds), variables por fold,
#         importancia por ganancia de E6 y tablas del ensamble;
#         output/submissions/ (S1-S4, solo si mejoran el F1 de 07 en más de
#         0,001).
#
# Correr con source(..., encoding = "UTF-8"). Con prueba <- TRUE antes del
# source() usa el 10% de los hogares y pocas iteraciones y no guarda nada.

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  lightgbm,   # Gradient boosting (hogares) y regresión del ingreso (personas).
  glmnet,     # Logit lasso de 06 para el ensamble.
  splines,    # Splines naturales de 06.
  knitr,      # Exportar la tabla a LaTeX.
  parallel,   # Procesos de R en paralelo.
  callr       # El logit 06 corre en un proceso aparte, en segundo plano.
)

prueba  <- exists("prueba") && isTRUE(prueba)
dir_out <- here("output", "logs", "boosting_variables")
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
inicio_total <- Sys.time()
marca <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "-", ..., "\n")

################################################################################

## 2. Cargar datos

train <- read_rds(here("data", "train_hogar.rds"))
test  <- read_rds(here("data", "test_hogar.rds"))
pers_tr <- read_rds(here("data", "train_clean.rds"))
pers_te <- read_rds(here("data", "test_clean.rds"))

# Ingreso total de la persona (Ingtot: suma de todas sus fuentes de ingreso,
# observadas e imputadas). Solo existe en train y solo se usa como variable
# objetivo del modelo auxiliar de personas (V4).
ingtot <- read_csv(here("data", "train_personas.csv"), guess_max = Inf,
                   col_select = c(id, Orden, Ingtot), show_col_types = FALSE)
pers_tr <- left_join(pers_tr, ingtot, by = c("id", orden = "Orden"),
                     relationship = "one-to-one")
rm(ingtot)

if (prueba) {
  set.seed(1)
  train   <- slice_sample(train, prop = 0.1)
  test    <- slice_sample(test, prop = 0.1)
  pers_tr <- filter(pers_tr, id %in% train$id)
  pers_te <- filter(pers_te, id %in% test$id)
  cat("MODO PRUEBA:", nrow(train), "hogares de train; no se guarda nada\n")
}
y <- train$pobre

# Mismos folds que 04 y 07 (una fila = un hogar). Todas las personas de un
# hogar quedan en el fold de su hogar.
set.seed(123)
k    <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

# Categóricas con códigos DANE (las mismas de 04 y 07).
categoricas <- c("dominio", "depto", "tenencia", "jefe_posicion",
                 "jefe_regimen", "jefe_tamano", "jefe_oficio")
x_base <- setdiff(names(train), c("id", "pobre"))

################################################################################

## 3. Funciones comunes (F1, AUC, hiperparámetros)

# F1 a partir de predicciones y valores reales (0/1).
calcular_f1 <- function(pred, real) {
  vp <- sum(pred == 1 & real == 1)
  2 * vp / (sum(pred == 1) + sum(real == 1))
}
grid_umbral <- seq(0.1, 0.6, by = 0.01)

# F1 promedio de los folds para cada umbral y el mejor umbral (como en 04).
f1_cv <- function(prob) {
  map_dfr(1:k, function(i) {
    p    <- prob[fold == i]
    real <- y[fold == i]
    tibble(umbral = grid_umbral,
           f1 = map_dbl(grid_umbral, ~ calcular_f1(as.integer(p > .x), real)))
  }) |>
    summarize(f1 = mean(f1), .by = umbral) |>
    slice_max(f1, n = 1, with_ties = FALSE)
}

# AUC (probabilidad de que un pobre tenga mayor puntaje que un no pobre).
calcular_auc <- function(puntaje, real) {
  r  <- rank(puntaje)
  n1 <- sum(real == 1)
  n0 <- sum(real == 0)
  (sum(r[real == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# Hiperparámetros fijos: la mejor configuración de 07.
params_hogar <- list(objective = "binary", metric = "binary_logloss",
                     learning_rate = 0.02, num_leaves = 127L, max_depth = 6L,
                     min_data_in_leaf = 20L, feature_fraction = 0.7,
                     bagging_fraction = 1, lambda_l2 = 0, seed = 123, verbose = -1)
nrounds_hogar <- if (prueba) 30 else 5000

# Modelo auxiliar de personas (V4): regresión del log del ingreso.
params_pers <- list(objective = "regression", metric = "l2", learning_rate = 0.05,
                    num_leaves = 63L, seed = 123, verbose = -1)
nrounds_pers <- if (prueba) 30 else 5000

# Escribir un .tex en UTF-8 (en Windows writeLines pasaría a CP1252).
escribir_tex <- function(x, archivo) writeLines(enc2utf8(x), archivo, useBytes = TRUE)

################################################################################

## 4. Variables que no dependen de los folds (V1 y V2)

# Marcas por persona, como en agregar_hogar() de 01b: unidad de gasto
# (parentesco 6-8 fuera), jefe y ocupado de la unidad de gasto.
marcar <- function(df) {
  df |>
    mutate(en_ug = !parentesco_jefe_hogar %in% c(6, 7, 8),
           jefe  = parentesco_jefe_hogar == 1,
           ocup  = ocupado == 1 & en_ug,
           horas = coalesce(horas_trabajo_semana, 0) + coalesce(horas_segundo_trabajo, 0),
           pet   = !is.na(edad_trabajar))
}
pers_tr <- marcar(pers_tr)
pers_te <- marcar(pers_te)

# V1. Factor de expansión del hogar (está en test).
v1 <- function(p) p |> summarize(.by = id, fex_c = first(fex_c), fex_dpto = first(fex_dpto))

# V2. El mejor ocupado del hogar (0 si nadie trabaja). El máximo
# log_meses_empresa entre ocupados no se agrega: es igual a max_log_meses,
# que ya está en la base.
v2 <- function(p) {
  p |>
    filter(ocup) |>
    summarize(.by = id,
              ocup_max_educ   = max(coalesce(anos_educ_imp, 0)),
              ocup_max_tamano = max(tamano_empresa),
              # Posición ocupacional del ocupado con más educación (el primero
              # si hay empate).
              ocup_posicion_educ = coalesce(posicion_ocupacional[which.max(coalesce(anos_educ_imp, -1))], 0),
              ocup_max_horas  = max(horas))
}

cols_bloque <- list(
  V1 = c("fex_c", "fex_dpto"),
  V2 = c("ocup_max_educ", "ocup_max_tamano", "ocup_posicion_educ", "ocup_max_horas"),
  V3 = c("te_oficio_media", "te_oficio_min", "te_oficio_max"),
  V4 = c("ing_pred_total", "log_ing_pred_pc_lp", "max_ing_pred_lp", "prop_ing_jefe")
)
categoricas_ext <- c(categoricas, "ocup_posicion_educ")

# Base de hogares (train y test juntos) con las variables estáticas. Las
# categóricas pasan a códigos enteros con los mismos niveles en train y test.
hog <- bind_rows(train |> select(id, all_of(x_base)) |> mutate(base = "train"),
                 test  |> select(id, all_of(x_base)) |> mutate(base = "test")) |>
  left_join(bind_rows(v1(pers_tr), v1(pers_te)), by = "id") |>
  left_join(bind_rows(v2(pers_tr), v2(pers_te)), by = "id") |>
  mutate(across(all_of(cols_bloque$V2), ~ coalesce(.x, 0)),
         across(all_of(categoricas_ext), ~ as.integer(factor(.x))))
hog_tr <- hog |> filter(base == "train")
hog_te <- hog |> filter(base == "test")
stopifnot(identical(hog_tr$id, train$id), identical(hog_te$id, test$id))
rm(hog)

################################################################################

## 5. Variables que dependen de los folds (V3 y V4) y logit 06, en paralelo

# Para evitar fuga de información, V3 (usa pobre) y V4 (usa Ingtot) se
# estiman solo con hogares de entrenamiento, con validación cruzada anidada:
#   - validación del fold f: modelo con los otros 4 folds;
#   - entrenamiento del fold f (cada fold g distinto de f): modelo con los 3
#     folds que no son ni f ni g. Así la etiqueta de f nunca entra en las
#     variables del modelo que se evalúa en f;
#   - modelo final: en train, la variable fuera de muestra de 5 folds (la
#     misma de la validación de cada fold); en test, modelo con todo train.

# Personas de train y test juntas, con las variables del modelo de ingreso.
# NUNCA variables que no estén en test_clean.
x_pers <- c("anos", "mujer", "parentesco_jefe_hogar", "max_educ", "anos_educ_imp",
            "ocupado", "desocupado", "inactivo", "actividad_sem_pasada",
            "posicion_ocupacional", "tamano_empresa", "oficio",
            "horas_trabajo_semana", "horas_segundo_trabajo", "segundo_trabajo",
            "cotiza_pension", "regimen_salud", "afiliado_salud", "log_meses_empresa",
            grep("^(recibio_|prima_|subsidio_|dinero_)", names(pers_te), value = TRUE),
            "ayudas_instituciones", "dominio", "depto", "cabecera")
stopifnot(all(x_pers %in% names(pers_te)))
cat_pers <- c("parentesco_jefe_hogar", "max_educ", "actividad_sem_pasada",
              "posicion_ocupacional", "oficio", "cotiza_pension", "regimen_salud",
              "dominio", "depto")

pers <- bind_rows(
  pers_tr |> left_join(tibble(id = train$id, fold = fold), by = "id") |> mutate(base = "train"),
  pers_te |> mutate(base = "test", fold = NA_integer_, pobre = NA_integer_, Ingtot = NA_real_)
)
X_pers <- pers |>
  select(all_of(x_pers)) |>
  mutate(across(all_of(cat_pers), ~ as.integer(factor(.x)))) |>
  data.matrix()
pers <- pers |>
  select(id, base, fold, en_ug, jefe, ocup, pet, oficio, pobre, Ingtot,
         lp = linea_pobreza, n_ug = personas_unidad_gasto)
rm(pers_tr, pers_te)
cat("Personas:", nrow(pers), "- variables del modelo de ingreso:", ncol(X_pers), "\n")

# V3 y V4 para las personas `i_pred`, estimadas con las personas `i_ent`.
# Devuelve una fila por hogar.
tarea_v34 <- function(tarea, n_hilos) {
  i_ent  <- which(pers$base == "train" & pers$fold %in% tarea$entren)
  i_pred <- if (is.null(tarea$pred)) which(pers$base == "test") else
    which(pers$base == "train" & pers$fold %in% tarea$pred)

  # V3. Target encoding del oficio con los ocupados de entrenamiento:
  # tasa de pobreza por oficio suavizada hacia la media con peso 20.
  ocup_ent <- i_ent[pers$ocup[i_ent]]
  media <- mean(pers$pobre[ocup_ent])
  tasas <- tibble(oficio = pers$oficio[ocup_ent], pobre = pers$pobre[ocup_ent]) |>
    summarize(.by = oficio, n = n(), tasa = mean(pobre)) |>
    mutate(te = (n * tasa + 20 * media) / (n + 20))

  # V4. Regresión de log1p(Ingtot) con las personas en edad de trabajar
  # (las demás no tienen Ingtot y se les asigna ingreso 0). El 10% de los
  # hogares de entrenamiento se separa para el early stopping.
  i_lab <- i_ent[pers$pet[i_ent]]
  set.seed(tarea$semilla)
  hog_val <- sample(unique(pers$id[i_lab]), round(0.1 * n_distinct(pers$id[i_lab])))
  es_val  <- pers$id[i_lab] %in% hog_val
  dtr <- lgb.Dataset(X_pers[i_lab[!es_val], ], label = log1p(pers$Ingtot[i_lab[!es_val]]),
                     categorical_feature = cat_pers)
  dva <- lgb.Dataset.create.valid(dtr, X_pers[i_lab[es_val], ],
                                  label = log1p(pers$Ingtot[i_lab[es_val]]))
  modelo <- lgb.train(params = c(params_pers, num_threads = n_hilos), data = dtr,
                      nrounds = nrounds_pers, valids = list(valid = dva),
                      early_stopping_rounds = 100, verbose = -1)
  pred_log <- predict(modelo, X_pers[i_pred, ], num_iteration = modelo$best_iter)

  pp <- pers[i_pred, ] |>
    mutate(te = coalesce(tasas$te[match(oficio, tasas$oficio)], media),
           ing = if_else(pet, pmax(expm1(pred_log), 0), 0))
  te_hogar <- pp |>
    filter(ocup) |>
    summarize(.by = id, te_oficio_media = mean(te), te_oficio_min = min(te),
              te_oficio_max = max(te))
  hogar <- pp |>
    summarize(.by = id,
              ing_pred_total = sum(ing[en_ug]),
              log_ing_pred_pc_lp = log((ing_pred_total + 1) / first(n_ug) / first(lp)),
              max_ing_pred_lp = max(c(ing[en_ug], 0)) / first(lp),
              prop_ing_jefe = if (ing_pred_total > 0) sum(ing[jefe & en_ug]) / ing_pred_total else 0) |>
    left_join(te_hogar, by = "id") |>
    mutate(across(starts_with("te_oficio"), ~ coalesce(.x, media)))

  # Para el R² fuera de muestra (solo tareas que predicen un fold de train).
  r2 <- NULL
  if (!is.null(tarea$pred)) {
    obs <- pers$pet[i_pred]
    r2 <- tibble(real = log1p(pers$Ingtot[i_pred][obs]), pred = pred_log[obs])
  }
  list(tarea = tarea, hogar = hogar, iter = modelo$best_iter, r2 = r2)
}

# Logit 06 (misma matriz de diseño y folds) con una grilla corta de lambda
# que termina en el óptimo de 06 (5.4e-05). Corre en un proceso aparte.
tarea_logit <- function(train, test, fold, lambdas) {
  suppressMessages({ library(tidyverse); library(glmnet); library(splines) })
  categoricas <- c("dominio", "depto", "tenencia", "jefe_posicion",
                   "jefe_regimen", "jefe_tamano", "jefe_oficio")
  x_base <- setdiff(names(train), c("id", "pobre"))
  x_clave <- c("log_arriendo_pc_lp", "horas_pc", "ocup_pc", "prop_contributivo",
               "prop_subsidiado", "prop_menores", "log_n_ug", "educ_media_adultos",
               "jefe_educ", "cuartos_pc", "formal_pc", "beneficios_pc",
               "dependencia", "n_fuentes_no_lab", "cabecera", "any_ayudas",
               "educ_ocup_media", "horas_formal_pc",
               "n_ocup", "n_formal", "prop_formal_ocup", "ocup_por_adulto",
               "educ_max", "prop_superior", "jefe_edad", "n_menor6",
               "pers_por_dormitorio", "log_cuota_pc_lp", "arriendo_es_pagado",
               "any_pension", "any_arriendos", "n_con_beneficios",
               "max_log_meses", "n_empresa_grande", "jefe_formal", "log_lp")
  x_spline <- c("log_arriendo_pc_lp", "horas_pc", "educ_media_adultos", "jefe_edad",
                "prop_contributivo", "cuartos_pc", "ocup_pc", "educ_ocup_media")
  x_dominio <- c("log_arriendo_pc_lp", "horas_pc", "prop_contributivo",
                 "educ_media_adultos", "ocup_pc", "formal_pc", "cuartos_pc",
                 "prop_menores", "jefe_educ", "log_n_ug")
  formula_x <- paste(
    "~", paste(x_base, collapse = " + "),
    "+ (", paste(x_clave, collapse = " + "), ")^2",
    "+", paste0("ns(", x_spline, ", df = 4)", collapse = " + "),
    "+ dominio:(", paste(x_dominio, collapse = " + "), ")",
    "+ tenencia:log_arriendo_pc_lp + jefe_posicion:horas_pc"
  ) |> as.formula()
  todo <- bind_rows(train |> select(all_of(x_base)), test |> select(all_of(x_base))) |>
    mutate(across(all_of(categoricas), factor))
  X <- model.matrix(formula_x, todo)[, -1]
  X_train <- X[seq_len(nrow(train)), ]
  X_test  <- X[nrow(train) + seq_len(nrow(test)), ]
  rm(X, todo)
  cv <- cv.glmnet(X_train, train$pobre, family = "binomial", alpha = 1,
                  foldid = fold, keep = TRUE, lambda = lambdas)
  j <- length(cv$lambda)
  prob_cv <- cv$fit.preval[, j]
  if (any(prob_cv < 0 | prob_cv > 1, na.rm = TRUE)) prob_cv <- plogis(prob_cv)
  prob_test <- predict(cv$glmnet.fit, X_test, s = cv$lambda[j], type = "response")[, 1]
  list(oof = prob_cv, test = prob_test, lambda = cv$lambda[j], columnas = ncol(X_train))
}

lambdas_logit <- exp(seq(log(1e-3), log(5.4e-5), length.out = 6))
proceso_logit <- r_bg(tarea_logit,
                      args = list(train = train, test = test, fold = fold,
                                  lambdas = lambdas_logit),
                      package = FALSE)
marca("Logit 06 lanzado en segundo plano (PID", proceso_logit$get_pid(), ")")

# Tareas de V3/V4: 5 de validación, 20 internas y 1 para test.
tareas <- c(
  map(1:k, ~ list(tipo = "validacion", f = .x, entren = setdiff(1:k, .x), pred = .x)),
  unlist(map(1:k, function(f) map(setdiff(1:k, f), function(g)
    list(tipo = "interna", f = f, g = g, entren = setdiff(1:k, c(f, g)), pred = g))),
    recursive = FALSE),
  list(list(tipo = "test", entren = 1:k, pred = NULL))
)
tareas <- imap(tareas, ~ c(.x, semilla = 100 + .y))
cat("Tareas de V3/V4:", length(tareas), "\n")

n_nucleos <- detectCores()
n_proc_1  <- 8
hilos_1   <- max(1, (n_nucleos - 1) %/% n_proc_1)   # un núcleo queda para el logit
marca("Núcleos:", n_nucleos, "- procesos:", n_proc_1, "- hilos por proceso:", hilos_1)

cl <- makeCluster(n_proc_1, outfile = file.path(dir_out, "procesos.log"))
clusterEvalQ(cl, suppressMessages({ library(tidyverse); library(lightgbm) }))
clusterExport(cl, c("pers", "X_pers", "cat_pers", "params_pers", "nrounds_pers", "tarea_v34"))
res_v34 <- clusterApplyLB(cl, tareas, tarea_v34, n_hilos = hilos_1)
stopCluster(cl)
rm(X_pers)
marca("V3/V4 listas")

# Ensamblar las variables por fold.
hogar_de <- function(filtro) bind_rows(map(keep(res_v34, filtro), "hogar"))
v34_val    <- map(1:k, function(f) hogar_de(~ .x$tarea$tipo == "validacion" && .x$tarea$f == f))
v34_entren <- map(1:k, function(f) hogar_de(~ .x$tarea$tipo == "interna" && .x$tarea$f == f))
v34_oof    <- bind_rows(v34_val)
v34_test   <- hogar_de(~ .x$tarea$tipo == "test")
stopifnot(setequal(v34_oof$id, train$id), setequal(v34_test$id, test$id),
          all(map_lgl(1:k, ~ setequal(v34_entren[[.x]]$id, train$id[fold != .x]))))

# Diagnóstico del ingreso predicho: R² fuera de muestra del modelo de
# personas (log1p del ingreso) y AUC del ingreso predicho per cápita solo.
r2_datos <- bind_rows(map(keep(res_v34, ~ .x$tarea$tipo == "validacion"), "r2"))
r2_pers <- 1 - sum((r2_datos$real - r2_datos$pred)^2) /
  sum((r2_datos$real - mean(r2_datos$real))^2)
ing_oof <- v34_oof$log_ing_pred_pc_lp[match(train$id, v34_oof$id)]
auc_ing <- calcular_auc(-ing_oof, y)
cat("Modelo de personas: iteraciones",
    paste(map_int(res_v34, "iter"), collapse = " "), "\n")
cat("R² fuera de muestra (log1p Ingtot):", round(r2_pers, 4), "\n")
cat("AUC de log(ingreso predicho pc / lp) solo:", round(auc_ing, 4), "\n")
if (!prueba) {
  write_rds(list(v34_val = v34_val, v34_entren = v34_entren, v34_test = v34_test,
                 r2 = r2_pers, auc = auc_ing),
            file.path(dir_out, "variables_por_fold.rds"))
}

################################################################################

## 6. Experimentos E0-E4 (en paralelo)

# Matriz de un experimento para las filas `filas` de hogares (train o test),
# con las variables estáticas y, si aplica, las de V3/V4 en `v34`.
armar_X <- function(base, filas, bloques, v34 = NULL) {
  cols <- c(x_base, unlist(cols_bloque[intersect(bloques, c("V1", "V2"))]))
  X <- base[filas, cols]
  dinam <- unlist(cols_bloque[intersect(bloques, c("V3", "V4"))])
  if (length(dinam) > 0) {
    X <- bind_cols(X, v34[match(base$id[filas], v34$id), dinam])
  }
  data.matrix(X)
}

# Validación cruzada de un experimento con los hiperparámetros fijos. Cada
# fold se estima por separado (las variables V3/V4 cambian por fold) con
# early stopping sobre el fold de validación; como en lgb.cv, la mejor
# iteración es la que minimiza el logloss promedio de los 5 folds.
cv_experimento <- function(exp, n_hilos) {
  inicio <- Sys.time()
  modelos <- list()
  perdida <- list()
  for (f in 1:k) {
    i_tr <- which(fold != f)
    i_va <- which(fold == f)
    X_tr <- armar_X(hog_tr, i_tr, exp$bloques, v34_entren[[f]])
    X_va <- armar_X(hog_tr, i_va, exp$bloques, v34_val[[f]])
    cats <- intersect(categoricas_ext, colnames(X_tr))
    dtr <- lgb.Dataset(X_tr, label = y[i_tr], categorical_feature = cats)
    dva <- lgb.Dataset.create.valid(dtr, X_va, label = y[i_va])
    modelos[[f]] <- lgb.train(params = c(params_hogar, num_threads = n_hilos), data = dtr,
                              nrounds = nrounds_hogar, valids = list(valid = dva),
                              early_stopping_rounds = 100, verbose = -1)
    perdida[[f]] <- unlist(modelos[[f]]$record_evals$valid$binary_logloss$eval)
  }
  largo <- min(lengths(perdida))
  mejor_iter <- which.min(rowMeans(sapply(perdida, `[`, 1:largo)))
  oof <- numeric(length(y))
  for (f in 1:k) {
    i_va <- which(fold == f)
    oof[i_va] <- predict(modelos[[f]], armar_X(hog_tr, i_va, exp$bloques, v34_val[[f]]),
                         num_iteration = mejor_iter)
  }
  mejor <- f1_cv(oof)
  list(nombre = exp$nombre, bloques = exp$bloques, mejor_iter = mejor_iter,
       umbral = mejor$umbral, f1 = mejor$f1, oof = oof,
       minutos = as.numeric(difftime(Sys.time(), inicio, units = "mins")))
}

experimentos <- list(
  list(nombre = "E0", bloques = character(0)),
  list(nombre = "E1", bloques = "V1"),
  list(nombre = "E2", bloques = "V2"),
  list(nombre = "E3", bloques = "V3"),
  list(nombre = "E4", bloques = "V4")
)
correr_experimentos <- function(exps, n_hilos) {
  cl <- makeCluster(length(exps), outfile = file.path(dir_out, "procesos.log"))
  clusterEvalQ(cl, suppressMessages({ library(tidyverse); library(lightgbm) }))
  clusterExport(cl, c("hog_tr", "y", "fold", "k", "x_base", "cols_bloque",
                      "categoricas_ext", "v34_val", "v34_entren", "params_hogar",
                      "nrounds_hogar", "calcular_f1", "grid_umbral", "f1_cv",
                      "armar_X", "cv_experimento"))
  res <- clusterApplyLB(cl, exps, cv_experimento, n_hilos = n_hilos)
  stopCluster(cl)
  set_names(res, map_chr(res, "nombre"))
}
res_exp <- correr_experimentos(experimentos, n_hilos = (n_nucleos - 1) %/% length(experimentos))
marca("E0-E4 listos")

################################################################################

## 7. Experimentos E5 y E6

# E5: E0 + los bloques que mejoraron el F1 en E1-E3 (sin V4). E6: E5 + V4.
# Si E5 o E6 coinciden con un experimento ya corrido, se reutiliza.
f1_E0 <- res_exp$E0$f1
mejoran <- c(V1 = "E1", V2 = "E2", V3 = "E3") |>
  keep(~ res_exp[[.x]]$f1 > f1_E0) |>
  names()
cat("Bloques que mejoran el F1 frente a E0:", if (length(mejoran)) mejoran else "ninguno", "\n")

nuevos <- list(list(nombre = "E5", bloques = mejoran),
               list(nombre = "E6", bloques = c(mejoran, "V4")))
reutilizar <- function(exp) {
  igual <- keep(res_exp, ~ setequal(.x$bloques, exp$bloques))
  if (length(igual) > 0) modifyList(igual[[1]], list(nombre = exp$nombre)) else NULL
}
por_correr <- discard(nuevos, ~ !is.null(reutilizar(.x)))
res_nuevos <- if (length(por_correr) > 0)
  correr_experimentos(por_correr, n_hilos = (n_nucleos - 1) %/% length(por_correr)) else list()
for (exp in nuevos) {
  res_exp[[exp$nombre]] <- res_nuevos[[exp$nombre]] %||% reutilizar(exp)
}
marca("E5 y E6 listos")

# Tabla de experimentos.
describir <- function(bloques) {
  if (length(bloques) == 0) return("Variables de train_hogar (base)")
  paste(map_chr(bloques, ~ paste0(.x, ": ", paste(cols_bloque[[.x]], collapse = ", "))),
        collapse = "; ")
}
tabla_exp <- map_dfr(res_exp, function(r) {
  tibble(experimento = r$nombre, variables = describir(r$bloques),
         mejor_iter = r$mejor_iter, umbral = r$umbral, f1 = r$f1,
         dif_E0 = r$f1 - f1_E0)
}) |>
  arrange(experimento)
print(tabla_exp |> mutate(variables = str_trunc(variables, 60)), width = Inf)

if (!prueba) {
  kable(tabla_exp |> mutate(variables = str_replace_all(variables, "_", "\\\\_")),
        format = "latex", booktabs = TRUE, escape = FALSE,
        digits = c(0, 0, 0, 2, 4, 4),
        col.names = c("Experimento", "Variables agregadas", "Iteraciones",
                      "Umbral", "F1 (CV)", "Diferencia vs. E0"),
        align = c("l", "p{8cm}", "r", "r", "r", "r"),
        caption = "LightGBM con variables nuevas (hiperparámetros fijos de 07, mismos 5 folds)",
        label = "boosting_variables") |>
    escribir_tex(here("output", "tables", "boosting_variables.tex"))
  iwalk(res_exp, ~ write_rds(tibble(id = train$id, fold = fold, pobre = y, prob = .x$oof),
                             file.path(dir_out, paste0("oof_", .y, ".rds"))))
}

################################################################################

## 8. Ensamble con el logit 06

proceso_logit$wait()
res_logit <- proceso_logit$get_result()
f1_logit  <- f1_cv(res_logit$oof)
marca("Logit 06: lambda =", signif(res_logit$lambda, 3), "- columnas =", res_logit$columnas,
      "- F1 de CV =", round(f1_logit$f1, 4), "- umbral =", f1_logit$umbral)
if (!prueba) {
  write_rds(tibble(id = train$id, fold = fold, pobre = y, prob = res_logit$oof),
            file.path(dir_out, "oof_logit06.rds"))
}

# Ensamble = w x p_boosting + (1 - w) x p_logit, con el mejor umbral para cada w.
ensamble <- function(oof_boost) {
  map_dfr(seq(0, 1, by = 0.1), function(w) {
    mejor <- f1_cv(w * oof_boost + (1 - w) * res_logit$oof)
    tibble(w = w, umbral = mejor$umbral, f1 = mejor$f1)
  })
}
ens_E5 <- ensamble(res_exp$E5$oof)
ens_E6 <- ensamble(res_exp$E6$oof)
cat("Ensamble con E5 (sin ingreso predicho):\n"); print(ens_E5, n = Inf)
cat("Ensamble con E6 (con ingreso predicho):\n"); print(ens_E6, n = Inf)
if (!prueba) write_rds(list(E5 = ens_E5, E6 = ens_E6), file.path(dir_out, "ensamble.rds"))

################################################################################

## 9. Modelos finales y submissions

# Modelo con todo train: en train, V3/V4 fuera de muestra de 5 folds; en
# test, V3/V4 de los modelos con todo train. nrounds = mejor iteración x 1,1.
modelo_final <- function(r) {
  n_final <- round(r$mejor_iter * 1.1)
  X_tr <- armar_X(hog_tr, seq_len(nrow(hog_tr)), r$bloques, v34_oof)
  X_te <- armar_X(hog_te, seq_len(nrow(hog_te)), r$bloques, v34_test)
  dtr <- lgb.Dataset(X_tr, label = y,
                     categorical_feature = intersect(categoricas_ext, colnames(X_tr)))
  modelo <- lgb.train(params = c(params_hogar, num_threads = n_nucleos), data = dtr,
                      nrounds = n_final, verbose = -1)
  list(modelo = modelo, n_final = n_final, prob_test = predict(modelo, X_te))
}
final_E5 <- modelo_final(res_exp$E5)
final_E6 <- modelo_final(res_exp$E6)
marca("Modelos finales listos")

# Importancia por ganancia de E6.
importancia_E6 <- lgb.importance(final_E6$modelo, percentage = TRUE) |>
  as_tibble() |>
  select(variable = Feature, ganancia = Gain, cobertura = Cover, frecuencia = Frequency)
print(importancia_E6 |> slice_head(n = 20))
if (!prueba) write_rds(importancia_E6, file.path(dir_out, "importancia_E6.rds"))

# Una submission se genera solo si mejora el F1 de CV de 07 en más de 0,001.
f1_07  <- 0.7537
minimo <- f1_07 + 0.001
sin_punto <- function(v) gsub("[.]", "", as.character(v))
tags <- c(V1 = "fex", V2 = "ocup", V3 = "te", V4 = "ingpred")

verificar <- function(s) {
  stopifnot(nrow(s) == nrow(test), identical(names(s), c("id", "pobre")),
            all(s$pobre %in% c(0, 1)), !anyNA(s), !anyDuplicated(s$id),
            setequal(s$id, test$id))
  mean(s$pobre)
}
guardar <- function(nombre, archivo, prob, umbral, f1) {
  if (f1 <= minimo) {
    cat(nombre, ": F1 de CV", round(f1, 4), "no supera", minimo, "- no se genera\n")
    return(invisible(NULL))
  }
  s <- tibble(id = test$id, pobre = as.integer(prob > umbral))
  prop <- verificar(s)
  if (!prueba) write_csv(s, here("output", "submissions", archivo))
  cat(nombre, ":", archivo, "- F1 de CV", round(f1, 4), "- proporción de pobres",
      round(prop, 4), "\n")
}

# S1 y S2: mejor boosting sin y con ingreso predicho.
nombre_lgbm <- function(r, n_final) {
  tag <- if (length(r$bloques) > 0) paste(tags[r$bloques], collapse = "_") else "base"
  paste0("LGBM_lr_002_leaves_127_depth_6_rounds_", n_final, "_vars_", tag,
         "_threshold_", sin_punto(r$umbral), ".csv")
}
guardar("S1", nombre_lgbm(res_exp$E5, final_E5$n_final), final_E5$prob_test,
        res_exp$E5$umbral, res_exp$E5$f1)
guardar("S2", nombre_lgbm(res_exp$E6, final_E6$n_final), final_E6$prob_test,
        res_exp$E6$umbral, res_exp$E6$f1)

# S3 y S4: mejor ensamble sin y con ingreso predicho. Si el mejor w es 1, el
# ensamble es el boosting solo (ya está en S1/S2) y no se repite.
guardar_ens <- function(nombre, ens, final, sufijo) {
  mejor <- ens |> slice_max(f1, n = 1, with_ties = FALSE)
  if (mejor$w == 1) {
    cat(nombre, ": el mejor w es 1 (boosting solo); no se genera\n")
    return(invisible(NULL))
  }
  prob <- mejor$w * final$prob_test + (1 - mejor$w) * res_logit$test
  archivo <- paste0("ENS_lgbm_logit_w_", sin_punto(mejor$w), "_threshold_",
                    sin_punto(mejor$umbral), sufijo, ".csv")
  guardar(nombre, archivo, prob, mejor$umbral, mejor$f1)
}
guardar_ens("S3", ens_E5, final_E5, "")
guardar_ens("S4", ens_E6, final_E6, "_ingpred")

marca("Tiempo total:", round(difftime(Sys.time(), inicio_total, units = "mins"), 1), "minutos")

### NOTAS:

# (Se completan después de correr el script.)
