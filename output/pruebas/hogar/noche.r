# Pruebas nocturnas a nivel hogar (LightGBM), comparables con 07 y 07b de
# origin/main: mismos 5 folds, mismos hiperparámetros y mismo F1.
#   C0: base por hogar (01b)                              -> 07b E0 = 0.7530
#   C1: C0 + ingreso predicho (V4, CV anidada)            -> 07b E4 = 0.7594
#   C2: C1 + V1 (fex), V2 (mejor ocupado), V3 (oficio)    -> 07b E6 = 0.7596
#   R1: regresión de log(ingreso / línea) con las variables de C2
#   R2: igual con pérdida de Huber
#   B1: combinación de C2 y R1 (pesos por F1)
#   C3: C2 promediando 5 semillas
#   C4: C2 con 8 configuraciones de hiperparámetros al azar
# Correr desde la raíz del repo (requiere data/train_hogar.rds, de
# 01b_bases_hogar_main.r):
#   Rscript output/pruebas/hogar/noche.r            (completo)
#   PRUEBA=1 Rscript output/pruebas/hogar/noche.r   (10% de hogares, rápido)
# Guarda cada paso en output/pruebas/hogar/resultados/ y retoma si se corta.

suppressMessages({library(here); library(tidyverse); library(lightgbm); library(parallel)})

prueba  <- Sys.getenv("PRUEBA") == "1"
dir_out <- here("output", "pruebas", "hogar", if (prueba) "resultados_prueba" else "resultados")
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
marca <- function(...) {
  linea <- paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "-", ...)
  cat(linea, "\n"); cat(linea, "\n", file = file.path(dir_out, "progreso.log"), append = TRUE)
}
guardar <- function(obj, nombre) saveRDS(obj, file.path(dir_out, paste0(nombre, ".rds")))
hecho   <- function(nombre) file.exists(file.path(dir_out, paste0(nombre, ".rds")))
leer    <- function(nombre) readRDS(file.path(dir_out, paste0(nombre, ".rds")))
procesos <- 4   # sin OpenMP en macOS: el paralelismo es entre procesos (fork)

################################################################################
## 1. Datos (como 07b)

train <- read_rds(here("data", "train_hogar.rds"))
test  <- read_rds(here("data", "test_hogar.rds"))
pers_tr <- read_rds(here("data", "train_clean.rds"))
pers_te <- read_rds(here("data", "test_clean.rds"))
ingtot <- read_csv(here("data", "train_personas.csv"), guess_max = Inf,
                   col_select = c(id, Orden, Ingtot), show_col_types = FALSE)
pers_tr <- left_join(pers_tr, ingtot, by = c("id", orden = "Orden"), relationship = "one-to-one")
rm(ingtot)

# Ingreso per cápita del hogar (solo train) para la regresión R1/R2.
ing_hogar <- distinct(pers_tr, id, ingreso, linea_pobreza)
train <- left_join(train, ing_hogar, by = "id", relationship = "one-to-one")
rm(ing_hogar)

if (prueba) {
  set.seed(1)
  train   <- slice_sample(train, prop = 0.1)
  test    <- slice_sample(test, prop = 0.1)
  pers_tr <- filter(pers_tr, id %in% train$id)
  pers_te <- filter(pers_te, id %in% test$id)
}
y <- train$pobre
# Objetivo de la regresión: log del ingreso relativo a la línea (pobre si < 0).
z <- log1p(train$ingreso) - log(train$linea_pobreza)

set.seed(123)
k    <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))   # mismos folds que 04/07/07b

categoricas <- c("dominio", "depto", "tenencia", "jefe_posicion", "jefe_regimen",
                 "jefe_tamano", "jefe_oficio")
x_base <- setdiff(names(train), c("id", "pobre", "ingreso", "linea_pobreza"))

calcular_f1 <- function(pred, real) { vp <- sum(pred == 1 & real == 1); 2 * vp / (sum(pred == 1) + sum(real == 1)) }
grid_umbral <- seq(0.1, 0.6, by = 0.01)
# F1 promedio de los folds para cada corte y el mejor. `menor = TRUE`: pobre si
# el puntaje es menor que el corte (regresión).
f1_cv <- function(puntaje, cortes = grid_umbral, menor = FALSE) {
  map_dfr(1:k, function(i) {
    p <- puntaje[fold == i]; real <- y[fold == i]
    tibble(corte = cortes, f1 = map_dbl(cortes, ~ calcular_f1(as.integer(if (menor) p < .x else p > .x), real)))
  }) |> summarize(f1 = mean(f1), .by = corte) |> slice_max(f1, n = 1, with_ties = FALSE)
}

params_hogar <- list(objective = "binary", metric = "binary_logloss",
                     learning_rate = 0.02, num_leaves = 127L, max_depth = 6L,
                     min_data_in_leaf = 20L, feature_fraction = 0.7,
                     bagging_fraction = 1, lambda_l2 = 0, seed = 123, verbose = -1,
                     num_threads = 1)
params_pers <- list(objective = "regression", metric = "l2", learning_rate = 0.05,
                    num_leaves = 63L, seed = 123, verbose = -1, num_threads = 1)
nrounds <- if (prueba) 60 else 5000

################################################################################
## 2. Variables V1-V4 (copiado de 07b)

marcar <- function(df) {
  df |> mutate(en_ug = !parentesco_jefe_hogar %in% c(6, 7, 8), jefe = parentesco_jefe_hogar == 1,
               ocup = ocupado == 1 & en_ug,
               horas = coalesce(horas_trabajo_semana, 0) + coalesce(horas_segundo_trabajo, 0),
               pet = !is.na(edad_trabajar))
}
pers_tr <- marcar(pers_tr); pers_te <- marcar(pers_te)
v1 <- function(p) p |> summarize(.by = id, fex_c = first(fex_c), fex_dpto = first(fex_dpto))
v2 <- function(p) {
  p |> filter(ocup) |>
    summarize(.by = id, ocup_max_educ = max(coalesce(anos_educ_imp, 0)),
              ocup_max_tamano = max(tamano_empresa),
              ocup_posicion_educ = coalesce(posicion_ocupacional[which.max(coalesce(anos_educ_imp, -1))], 0),
              ocup_max_horas = max(horas))
}
cols <- list(V1 = c("fex_c", "fex_dpto"),
             V2 = c("ocup_max_educ", "ocup_max_tamano", "ocup_posicion_educ", "ocup_max_horas"),
             V3 = c("te_oficio_media", "te_oficio_min", "te_oficio_max"),
             V4 = c("ing_pred_total", "log_ing_pred_pc_lp", "max_ing_pred_lp", "prop_ing_jefe"))
categoricas_ext <- c(categoricas, "ocup_posicion_educ")

hog <- bind_rows(train |> select(id, all_of(x_base)) |> mutate(base = "train"),
                 test  |> select(id, all_of(x_base)) |> mutate(base = "test")) |>
  left_join(bind_rows(v1(pers_tr), v1(pers_te)), by = "id") |>
  left_join(bind_rows(v2(pers_tr), v2(pers_te)), by = "id") |>
  mutate(across(all_of(cols$V2), ~ coalesce(.x, 0)),
         across(all_of(categoricas_ext), ~ as.integer(factor(.x))))
hog_tr <- hog |> filter(base == "train"); hog_te <- hog |> filter(base == "test")
stopifnot(identical(hog_tr$id, train$id))
rm(hog)

x_pers <- c("anos", "mujer", "parentesco_jefe_hogar", "max_educ", "anos_educ_imp",
            "ocupado", "desocupado", "inactivo", "actividad_sem_pasada",
            "posicion_ocupacional", "tamano_empresa", "oficio",
            "horas_trabajo_semana", "horas_segundo_trabajo", "segundo_trabajo",
            "cotiza_pension", "regimen_salud", "afiliado_salud", "log_meses_empresa",
            grep("^(recibio_|prima_|subsidio_|dinero_)", names(pers_te), value = TRUE),
            "ayudas_instituciones", "dominio", "depto", "cabecera")
cat_pers <- c("parentesco_jefe_hogar", "max_educ", "actividad_sem_pasada",
              "posicion_ocupacional", "oficio", "cotiza_pension", "regimen_salud",
              "dominio", "depto")
pers <- bind_rows(
  pers_tr |> left_join(tibble(id = train$id, fold = fold), by = "id") |> mutate(base = "train"),
  pers_te |> mutate(base = "test", fold = NA_integer_, pobre = NA_integer_, Ingtot = NA_real_))
X_pers <- pers |> select(all_of(x_pers)) |> mutate(across(all_of(cat_pers), ~ as.integer(factor(.x)))) |> data.matrix()
pers <- pers |> select(id, base, fold, en_ug, jefe, ocup, pet, oficio, pobre, Ingtot,
                       lp = linea_pobreza, n_ug = personas_unidad_gasto)
rm(pers_tr, pers_te); invisible(gc())

tarea_v34 <- function(tarea) {
  i_ent  <- which(pers$base == "train" & pers$fold %in% tarea$entren)
  i_pred <- if (is.null(tarea$pred)) which(pers$base == "test") else which(pers$base == "train" & pers$fold %in% tarea$pred)
  ocup_ent <- i_ent[pers$ocup[i_ent]]
  media <- mean(pers$pobre[ocup_ent])
  tasas <- tibble(oficio = pers$oficio[ocup_ent], pobre = pers$pobre[ocup_ent]) |>
    summarize(.by = oficio, n = n(), tasa = mean(pobre)) |> mutate(te = (n * tasa + 20 * media) / (n + 20))
  i_lab <- i_ent[pers$pet[i_ent]]
  set.seed(tarea$semilla)
  hog_val <- sample(unique(pers$id[i_lab]), round(0.1 * n_distinct(pers$id[i_lab])))
  es_val  <- pers$id[i_lab] %in% hog_val
  dtr <- lgb.Dataset(X_pers[i_lab[!es_val], ], label = log1p(pers$Ingtot[i_lab[!es_val]]), categorical_feature = cat_pers)
  dva <- lgb.Dataset.create.valid(dtr, X_pers[i_lab[es_val], ], label = log1p(pers$Ingtot[i_lab[es_val]]))
  modelo <- lgb.train(params = params_pers, data = dtr, nrounds = nrounds, valids = list(valid = dva),
                      early_stopping_rounds = 100, verbose = -1)
  pred_log <- predict(modelo, X_pers[i_pred, ], num_iteration = modelo$best_iter)
  pp <- pers[i_pred, ] |> mutate(te = coalesce(tasas$te[match(oficio, tasas$oficio)], media),
                                 ing = if_else(pet, pmax(expm1(pred_log), 0), 0))
  te_hogar <- pp |> filter(ocup) |>
    summarize(.by = id, te_oficio_media = mean(te), te_oficio_min = min(te), te_oficio_max = max(te))
  pp |> summarize(.by = id, ing_pred_total = sum(ing[en_ug]),
                  log_ing_pred_pc_lp = log((ing_pred_total + 1) / first(n_ug) / first(lp)),
                  max_ing_pred_lp = max(c(ing[en_ug], 0)) / first(lp),
                  prop_ing_jefe = if (ing_pred_total > 0) sum(ing[jefe & en_ug]) / ing_pred_total else 0) |>
    left_join(te_hogar, by = "id") |> mutate(across(starts_with("te_oficio"), ~ coalesce(.x, media)))
}

# Tareas: validación de f (entrena con los otros 4), entrenamiento de f para
# cada g != f (entrena con los 3 que no son ni f ni g) y test (todo train).
tareas <- list()
for (f in 1:k) {
  tareas[[paste0("val_", f)]] <- list(entren = setdiff(1:k, f), pred = f, semilla = 100 + f)
  for (g in setdiff(1:k, f)) tareas[[paste0("ent_", f, "_", g)]] <- list(entren = setdiff(1:k, c(f, g)), pred = g, semilla = 1000 + 10 * f + g)
}
tareas[["test"]] <- list(entren = 1:k, pred = NULL, semilla = 999)

if (!hecho("v34")) {
  marca("V3/V4:", length(tareas), "tareas en", procesos, "procesos")
  v34 <- mclapply(names(tareas), function(n) tarea_v34(tareas[[n]]), mc.cores = procesos, mc.preschedule = FALSE)
  names(v34) <- names(tareas)
  errores <- names(v34)[map_lgl(v34, ~ inherits(.x, "try-error"))]
  if (length(errores)) stop("Fallaron tareas V3/V4: ", paste(errores, collapse = ", "))
  guardar(v34, "v34"); marca("V3/V4 listo")
}
v34 <- leer("v34")
rm(pers, X_pers); invisible(gc())

# Matrices de un fold: entrenamiento (V3/V4 de las tareas internas) y
# validación (V3/V4 de la tarea val_f).
matrices <- function(f, vars) {
  agrega <- function(ids, tabla) {
    base <- hog_tr[match(ids, hog_tr$id), ]
    if (any(vars %in% c(cols$V3, cols$V4))) base <- left_join(select(base, -any_of(c(cols$V3, cols$V4))), tabla, by = "id")
    data.matrix(base[vars])
  }
  i_ent <- which(fold != f); i_val <- which(fold == f)
  tabla_ent <- bind_rows(map(setdiff(1:k, f), ~ v34[[paste0("ent_", f, "_", .x)]]))
  list(i_ent = i_ent, i_val = i_val,
       X_ent = agrega(hog_tr$id[i_ent], tabla_ent), X_val = agrega(hog_tr$id[i_val], v34[[paste0("val_", f)]]))
}

################################################################################
## 3. Experimentos

vars_c0 <- x_base
vars_c1 <- c(x_base, cols$V4)
vars_c2 <- c(x_base, cols$V1, cols$V2, cols$V3, cols$V4)

# Validación cruzada de un modelo LightGBM: predicción fuera de muestra (OOF)
# de cada hogar con early stopping en su fold. objetivo "binary" usa y;
# "regression"/"huber" usan z.
cv_lgb <- function(vars, params, objetivo = "binary") {
  etiqueta <- if (objetivo == "binary") y else z
  cats <- intersect(vars, categoricas_ext)
  oof <- rep(NA_real_, nrow(train)); iters <- integer(k)
  for (f in 1:k) {
    m <- matrices(f, vars)
    dtr <- lgb.Dataset(m$X_ent, label = etiqueta[m$i_ent], categorical_feature = cats)
    dva <- lgb.Dataset.create.valid(dtr, m$X_val, label = etiqueta[m$i_val])
    p <- params; p$objective <- objetivo
    p$metric <- if (objetivo == "binary") "binary_logloss" else "l2"
    b <- lgb.train(params = p, data = dtr, nrounds = nrounds, valids = list(valid = dva),
                   early_stopping_rounds = 100, verbose = -1)
    oof[m$i_val] <- predict(b, m$X_val, num_iteration = b$best_iter)
    iters[f] <- b$best_iter
  }
  list(oof = oof, iters = iters)
}

experimentos <- list(
  C0 = list(vars = vars_c0, params = params_hogar, objetivo = "binary"),
  C1 = list(vars = vars_c1, params = params_hogar, objetivo = "binary"),
  C2 = list(vars = vars_c2, params = params_hogar, objetivo = "binary"),
  R1 = list(vars = vars_c2, params = modifyList(params_hogar, list(learning_rate = 0.02)), objetivo = "regression"),
  R2 = list(vars = vars_c2, params = params_hogar, objetivo = "huber")
)
for (s in 1:4) experimentos[[paste0("C3_semilla", s)]] <- list(vars = vars_c2, params = modifyList(params_hogar, list(seed = 123 + s, feature_fraction_seed = 123 + s, bagging_seed = 123 + s)), objetivo = "binary")
set.seed(2024)
for (j in 1:8) {
  experimentos[[paste0("C4_config", j)]] <- list(vars = vars_c2, objetivo = "binary", params = modifyList(params_hogar, list(
    learning_rate = sample(c(0.01, 0.02, 0.03), 1), num_leaves = sample(c(31L, 63L, 127L, 255L), 1),
    max_depth = sample(c(-1L, 6L, 8L, 10L), 1), min_data_in_leaf = sample(c(10L, 20L, 50L, 100L), 1),
    feature_fraction = sample(c(0.5, 0.6, 0.7, 0.8, 0.9), 1), bagging_fraction = sample(c(0.7, 0.8, 1), 1),
    bagging_freq = 1L, lambda_l2 = sample(c(0, 1, 5, 10), 1))))
}

pendientes <- names(experimentos)[!map_lgl(names(experimentos), hecho)]
marca("Experimentos pendientes:", length(pendientes))
invisible(mclapply(pendientes, function(n) {
  e <- experimentos[[n]]
  t0 <- Sys.time()
  r <- tryCatch(cv_lgb(e$vars, e$params, e$objetivo), error = function(err) list(error = conditionMessage(err)))
  r$minutos <- as.numeric(difftime(Sys.time(), t0, units = "mins")); r$params <- e$params; r$objetivo <- e$objetivo
  guardar(r, n)
  marca(n, if (!is.null(r$error)) paste("ERROR:", r$error) else paste("listo en", round(r$minutos), "min"))
  NULL
}, mc.cores = procesos, mc.preschedule = FALSE))

################################################################################
## 4. Resumen (F1 de CV de cada experimento y combinaciones)

res <- map(set_names(names(experimentos)), function(n) if (hecho(n)) leer(n) else NULL)
fila <- function(nombre, puntaje, cortes = grid_umbral, menor = FALSE) {
  b <- f1_cv(puntaje, cortes, menor); tibble(experimento = nombre, corte = b$corte, f1 = b$f1)
}
cortes_reg <- seq(-1.5, 1.5, by = 0.01)
tabla <- bind_rows(imap(compact(res), function(r, n) {
  if (!is.null(r$error)) return(tibble(experimento = n, corte = NA, f1 = NA))
  if (r$objetivo == "binary") fila(n, r$oof) else fila(n, r$oof, cortes_reg, menor = TRUE)
}))

# C3: promedio de C2 y sus 4 semillas.
semillas <- c("C2", paste0("C3_semilla", 1:4))
if (all(map_lgl(semillas, ~ !is.null(res[[.x]]) && is.null(res[[.x]]$error)))) {
  tabla <- bind_rows(tabla, fila("C3 (promedio de 5 semillas)", rowMeans(sapply(semillas, \(n) res[[n]]$oof))))
}
# B1: combinación de C2 (probabilidad) y R1 (regresión) en rangos, con peso w
# para C2. Cortes: cuantiles del puntaje combinado.
if (!is.null(res$C2$oof) && !is.null(res$R1$oof)) {
  rc <- rank(res$C2$oof) / length(y); rr <- rank(-res$R1$oof) / length(y)
  cortes_q <- seq(0.6, 0.9, by = 0.0025)
  for (w in seq(0, 1, by = 0.1)) tabla <- bind_rows(tabla, fila(sprintf("B1 (w = %.1f C2 + %.1f R1)", w, 1 - w), w * rc + (1 - w) * rr, cortes_q))
}
tabla <- arrange(tabla, desc(f1))
guardar(tabla, "tabla_resumen")
marca("=== TERMINADO")
print(as.data.frame(tabla), digits = 4)
