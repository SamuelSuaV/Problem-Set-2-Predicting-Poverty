###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   14_lightgbm_ingreso.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Subir el F1 del boosting con variables auxiliares
#                      predichas (ingreso de las personas, ingreso con
#                      arriendo imputado, ingreso laboral y no laboral,
#                      ingreso del hogar y estrato), controlando los errores
#                      de anidar predicciones.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Paso 0: fórmula del ingreso del DANE y regla del arriendo imputado
# 4. Funciones comunes (F1, AUC, modelos auxiliares)
# 5. Variables auxiliares con validación cruzada anidada (en paralelo)
# 6. Experimentos F0-F5
# 7. Reajuste: random search sobre las variables de F5
# 8. Promedio de 5 semillas
# 9. Modelo final, diagnóstico de la variable predicha y submissions
# 10. Tabla de resultados

# Input:  data/train_hogar.rds , data/test_hogar.rds (02);
#         data/train_clean.rds , data/test_clean.rds (01);
#         data/train_personas.csv (Ingtot y sus componentes, Estrato1) y
#         data/train_hogares.csv (Ingpcug y agregados de ingreso), solo para
#         construir las variables objetivo de los modelos auxiliares.
# Output: 02_outputs/tables/boosting_ingreso.tex ;
#         02_outputs/logs/boosting_ingreso/: probabilidades fuera de muestra de
#         cada experimento, variables por fold, importancia del mejor modelo
#         y diagnóstico de la variable predicha;
#         02_outputs/predictions/ (F0 siempre; las demás si superan 0,7596 en
#         más de 0,001).
#
# Correr con source(..., encoding = "UTF-8"). Con prueba <- TRUE antes del
# source() usa el 10% de los hogares y pocas iteraciones y no guarda nada.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  lightgbm, # Gradient boosting (clasificador y modelos auxiliares).
  knitr, # Exportar la tabla a LaTeX.
  parallel # Procesos de R en paralelo.
)
# Funciones comunes (calcular_f1, marca, ...).
source(here("01_code", "functions", "funciones_comunes.R"), encoding = "UTF-8")

prueba <- exists("prueba") && isTRUE(prueba)
dir_out <- here("02_outputs", "logs", "boosting_ingreso")
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
inicio_total <- Sys.time()

# Núcleos para esta parte (la parte B corre al mismo tiempo con el resto).
n_nucleos <- 20
n_proc <- 5
n_hilos <- n_nucleos %/% n_proc

# 2. Cargar datos -------------------------------------------------------------

train <- read_rds(here("data", "train_hogar.rds"))
test <- read_rds(here("data", "test_hogar.rds"))
pers_tr <- read_rds(here("data", "train_clean.rds"))
pers_te <- read_rds(here("data", "test_clean.rds"))

# Ingresos crudos de train: solo sirven como variables objetivo de los
# modelos auxiliares (nunca como predictores).
crudo_pers <- read_csv(here("data", "train_personas.csv"),
  guess_max = Inf,
  show_col_types = FALSE,
  col_select = c(
    id, Orden, P6050, Estrato1, Ingtot,
    Impa, Isa, Ie, Imdi, Iof1, Iof2, Iof3h, Iof3i, Iof6,
    Impaes, Isaes, Iees, Imdies, Iof1es, Iof2es,
    Iof3hes, Iof3ies, Iof6es, Ingtotob, Ingtotes
  )
)
crudo_hog <- read_csv(here("data", "train_hogares.csv"),
  guess_max = Inf,
  show_col_types = FALSE,
  col_select = c(
    id, P5090, P5100, P5130, P5140, Npersug,
    Ingtotug, Ingtotugarr, Ingpcug, Lp
  )
)

# 3. Paso 0: fórmula del ingreso del DANE y regla del arriendo imputado -------

suma <- function(...) rowSums(cbind(...), na.rm = TRUE)

# a) Ingpcug = Ingtotugarr / Npersug, e Ingtotug = suma de Ingtot de la
# unidad de gasto (P6050 fuera de 6, 7 y 8). Tolerancia de 1 peso.
suma_ug <- crudo_pers |>
  filter(!P6050 %in% c(6, 7, 8)) |>
  summarize(.by = id, suma_ingtot = sum(Ingtot, na.rm = TRUE))
paso0 <- crudo_hog |>
  left_join(suma_ug, by = "id") |>
  mutate(
    suma_ingtot = coalesce(suma_ingtot, 0),
    arr_imputado = Ingtotugarr - Ingtotug
  )
cat(
  "Paso 0a. Ingpcug == Ingtotugarr / Npersug:",
  round(
    100 * mean(abs(paso0$Ingpcug - paso0$Ingtotugarr / paso0$Npersug) <= 1), 2
  ),
  "%\n"
)
cat(
  "Paso 0a. Ingtotug == suma de Ingtot de la unidad de gasto:",
  round(100 * mean(abs(paso0$Ingtotug - paso0$suma_ingtot) <= 1), 2), "%\n"
)

# Componentes de Ingtot (diccionario): Ingtot = Ingtotob + Ingtotes; lo
# observado es Impa + Isa + Ie + Imdi + Iof*, y lo imputado lo mismo con
# sufijo es. Laboral = primera y segunda actividad, especie y trabajo de
# desocupados e inactivos; no laboral = Iof* (intereses, pensiones, ayudas,
# arriendos y otros).
crudo_pers <- crudo_pers |>
  mutate(
    laboral = suma(Impa, Isa, Ie, Imdi, Impaes, Isaes, Iees, Imdies),
    no_laboral = suma(
      Iof1, Iof2, Iof3h, Iof3i, Iof6,
      Iof1es, Iof2es, Iof3hes, Iof3ies, Iof6es
    )
  )
con_ingtot <- !is.na(crudo_pers$Ingtot)
cat(
  "Paso 0a. Ingtot == laboral + no laboral:",
  round(
    100 *
      mean(abs(
        crudo_pers$Ingtot - crudo_pers$laboral - crudo_pers$no_laboral
      )[con_ingtot] <= 1),
    2
  ),
  "%\n"
)

# b) Arriendo imputado = Ingtotugarr - Ingtotug, frente a P5130 (estimado) y
# P5140 (pagado) por tenencia (P5090).
regla_arriendo <- function(tenencia, estimado, cuota) {
  # estimado y cuota con 98/99 (no sabe, no informa) ya como NA.
  case_when(
    tenencia %in% c(1, 4) ~ coalesce(estimado, 0), # propia pagada, usufructo
    # la están pagando
    tenencia == 2 ~ pmax(coalesce(estimado, 0) - coalesce(cuota, 0), 0),
    TRUE ~ 0 # arriendo, posesión, otra
  )
}
na_98 <- function(x) if_else(x %in% c(98, 99), NA, x)
paso0 <- paso0 |>
  mutate(regla = regla_arriendo(P5090, na_98(P5130), na_98(P5100)))
tabla_paso0 <- paso0 |>
  summarize(
    .by = P5090, hogares = n(),
    igual_P5130 = mean(abs(arr_imputado - P5130) <= 1, na.rm = TRUE),
    igual_P5140 = mean(abs(arr_imputado - P5140) <= 1, na.rm = TRUE),
    arriendo_cero = mean(abs(arr_imputado) <= 1),
    igual_regla = mean(abs(arr_imputado - regla) <= 1),
    # NA donde no hay pares completos o no hay variación.
    cor_P5130 = if (sum(!is.na(P5130)) > 2) {
      suppressWarnings(cor(arr_imputado, P5130, use = "complete.obs"))
    } else {
      NA
    },
    cor_regla = suppressWarnings(cor(arr_imputado, regla))
  ) |>
  arrange(P5090)
cat("Paso 0b. Arriendo imputado (Ingtotugarr - Ingtotug) por tenencia:\n")
print(as.data.frame(tabla_paso0), digits = 3)
cat(
  "Paso 0b. La regla coincide (1 peso) en el",
  round(100 * mean(abs(paso0$arr_imputado - paso0$regla) <= 1), 2),
  "% de los hogares\n"
)

# Variables objetivo de los modelos auxiliares.
obj_pers <- crudo_pers |> select(id, orden = Orden, Ingtot, laboral, no_laboral)
obj_hog <- crudo_pers |>
  summarize(.by = id, estrato = first(Estrato1)) |>
  left_join(select(crudo_hog, id, Ingpcug, Lp), by = "id") |>
  # log(Ingpcug / Lp) con piso de 1% de la línea (hay hogares con ingreso 0):
  # pobre sigue siendo exactamente objetivo < 0.
  mutate(log_ing_hog_lp = log(pmax(Ingpcug, 0.01 * Lp) / Lp))
cat("Estrato1 (hogares):\n")
print(table(obj_hog$estrato, useNA = "ifany"))
rm(crudo_pers, crudo_hog, suma_ug)

# 4. Funciones comunes (F1, AUC, modelos auxiliares) --------------------------

if (prueba) {
  set.seed(1)
  train <- slice_sample(train, prop = 0.1)
  test <- slice_sample(test, prop = 0.1)
  pers_tr <- filter(pers_tr, id %in% train$id)
  pers_te <- filter(pers_te, id %in% test$id)
  cat("MODO PRUEBA:", nrow(train), "hogares de train; no se guarda nada\n")
}
y <- train$pobre

# Mismos folds que 06, 12 y 13.
set.seed(123)
k <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

categoricas <- c(
  "dominio", "depto", "tenencia", "jefe_posicion",
  "jefe_regimen", "jefe_tamano", "jefe_oficio"
)
x_base <- setdiff(names(train), c("id", "pobre"))

grid_umbral <- seq(0.1, 0.6, by = 0.01)
f1_cv <- function(prob) {
  map_dfr(1:k, function(i) {
    p <- prob[fold == i]
    real <- y[fold == i]
    tibble(
      umbral = grid_umbral,
      f1 = map_dbl(grid_umbral, ~ calcular_f1(as.integer(p > .x), real))
    )
  }) |>
    summarize(f1 = mean(f1), .by = umbral) |>
    slice_max(f1, n = 1, with_ties = FALSE)
}
r2 <- function(real, pred) 1 - sum((real - pred)^2) / sum((real - mean(real))^2)

# Hiperparámetros del clasificador: la mejor configuración de 12.
params_07 <- list(
  learning_rate = 0.02, num_leaves = 127L, max_depth = 6L,
  min_data_in_leaf = 20L, feature_fraction = 0.7,
  bagging_fraction = 1, lambda_l2 = 0
)
params_fijos <- list(
  objective = "binary", metric = "binary_logloss",
  seed = 123, verbose = -1
)
nrounds_hogar <- if (prueba) 30 else 5000

# Modelos auxiliares (regresión).
params_aux <- list(
  objective = "regression", metric = "l2", learning_rate = 0.05,
  num_leaves = 63L, seed = 123, verbose = -1
)
nrounds_aux <- if (prueba) 30 else 5000

# 5. Variables auxiliares con validación cruzada anidada (en paralelo) --------

# Errores de anidar predicciones y su control:
# (1) Fuga de la etiqueta: las variables auxiliares usan Ingtot, Ingpcug y
#     Estrato1, así que se estiman con CV anidada (como en 13). Para el
#     fold f, los hogares de entrenamiento reciben la variable de modelos que
#     no vieron ni f ni su propio fold, y los de f la de un modelo con los
#     otros 4 folds.
# (2) Diferencia de error entre train y test: el clasificador se entrena con
#     variables fuera de muestra (modelos con 80% de train). Para que test
#     tenga el mismo error, su variable es el promedio de los 5 modelos de
#     los folds externos, no la de un modelo reestimado con todo train (que
#     se guarda solo para comparar).

# Variables del hogar como matriz (categóricas como códigos enteros con los
# mismos niveles en train y test).
hog_mat <- bind_rows(
  train |> select(id, all_of(x_base)), test |> select(id, all_of(x_base))
) |>
  mutate(across(all_of(categoricas), ~ as.integer(factor(.x))))
mat_hog_tr <- data.matrix(hog_mat |> filter(id %in% train$id) |> select(-id))
mat_hog_te <- data.matrix(hog_mat |> filter(id %in% test$id) |> select(-id))
stopifnot(identical(hog_mat$id[hog_mat$id %in% train$id], train$id))
obj_hog <- obj_hog[match(train$id, obj_hog$id), ]

# Personas de train y test con las variables del modelo de ingreso (las de
# 13, que están en test) y la información de su hogar.
x_pers <- c(
  "anos", "mujer", "parentesco_jefe_hogar", "max_educ", "anos_educ_imp",
  "ocupado", "desocupado", "inactivo", "actividad_sem_pasada",
  "posicion_ocupacional", "tamano_empresa", "oficio",
  "horas_trabajo_semana", "horas_segundo_trabajo", "segundo_trabajo",
  "cotiza_pension", "regimen_salud", "afiliado_salud", "log_meses_empresa",
  grep("^(recibio_|prima_|subsidio_|dinero_)", names(pers_te), value = TRUE),
  "ayudas_instituciones", "dominio", "depto", "cabecera"
)
stopifnot(all(x_pers %in% names(pers_te)))
cat_pers <- c(
  "parentesco_jefe_hogar", "max_educ", "actividad_sem_pasada",
  "posicion_ocupacional", "oficio", "cotiza_pension", "regimen_salud",
  "dominio", "depto"
)

pers <- bind_rows(
  pers_tr |>
    left_join(tibble(id = train$id, fold = fold), by = "id") |>
    left_join(obj_pers, by = c("id", "orden")) |>
    mutate(base = "train"),
  pers_te |> mutate(base = "test", fold = NA_integer_)
) |>
  mutate(
    en_ug = !parentesco_jefe_hogar %in% c(6, 7, 8),
    jefe = parentesco_jefe_hogar == 1,
    pet = !is.na(edad_trabajar),
    # Arriendo imputado con la regla del Paso 0 (variables que están en test).
    arr_regla = regla_arriendo(
      tenencia_vivienda, arriendo_estimado, cuota_amortizacion
    )
  )
mat_pers <- pers |>
  select(all_of(x_pers)) |>
  mutate(across(all_of(cat_pers), ~ as.integer(factor(.x)))) |>
  data.matrix()
# V6: el modelo de personas con las variables de su hogar.
mat_hog_pers <- rbind(mat_hog_tr, mat_hog_te)[
  match(pers$id, c(train$id, test$id)), ,
  drop = FALSE
]
colnames(mat_hog_pers) <- paste0("hog_", colnames(mat_hog_tr))
mat_pers6 <- cbind(mat_pers, mat_hog_pers)
cat_pers6 <- c(cat_pers, paste0("hog_", categoricas))
rm(mat_hog_pers)
pers <- pers |>
  transmute(id, base, fold, en_ug, jefe, pet, arr_regla,
    lp = linea_pobreza, n_ug = personas_unidad_gasto,
    y_ingtot = log1p(Ingtot), y_lab = log1p(laboral),
    y_nolab = log1p(no_laboral)
  )
rm(pers_tr, pers_te, obj_pers, hog_mat)
cat(
  "Personas:", nrow(pers), "- variables V4:", ncol(mat_pers), "- variables V6:",
  ncol(mat_pers6), "\n"
)

# Modelo auxiliar de regresión con early stopping en el 10% de los hogares
# de entrenamiento.
entrenar_aux <- function(mat, label, filas, ids, cats, n_hilos, semilla) {
  filas <- filas[!is.na(label[filas])]
  set.seed(semilla)
  hog_val <- sample(unique(ids[filas]), round(0.1 * n_distinct(ids[filas])))
  es_val <- ids[filas] %in% hog_val
  dtr <- lgb.Dataset(mat[filas[!es_val], , drop = FALSE],
    label = label[filas[!es_val]],
    categorical_feature = cats
  )
  dva <- lgb.Dataset.create.valid(dtr, mat[filas[es_val], , drop = FALSE],
    label = label[filas[es_val]]
  )
  lgb.train(
    params = c(params_aux, num_threads = n_hilos), data = dtr,
    nrounds = nrounds_aux, valids = list(valid = dva),
    early_stopping_rounds = 100, verbose = -1
  )
}

# Agregar las predicciones de las personas al hogar (unidad de gasto).
agregar <- function(pp) {
  pp |>
    summarize(
      .by = id,
      # V4: ingreso total predicho (como en 13).
      ing_pred_total = sum(ing4[en_ug]),
      log_ing_pred_pc_lp = log((ing_pred_total + 1) / first(n_ug) / first(lp)),
      max_ing_pred_lp = max(c(ing4[en_ug], 0)) / first(lp),
      prop_ing_jefe = if (ing_pred_total > 0) {
        sum(ing4[jefe & en_ug]) /
          ing_pred_total
      } else {
        0
      },
      # V5: fórmula del DANE con el arriendo imputado por la regla.
      log_ingarr_pc_lp = log(
        (ing_pred_total + first(arr_regla) + 1) / first(n_ug) / first(lp)
      ),
      # V6: ingreso laboral y no laboral con contexto del hogar.
      ing6_lab_total = sum(lab6[en_ug]),
      ing6_nolab_total = sum(nolab6[en_ug]),
      log_ing6_pc_lp = log(
        (ing6_lab_total + ing6_nolab_total + 1) / first(n_ug) / first(lp)
      ),
      log_ing6arr_pc_lp = log((
        ing6_lab_total + ing6_nolab_total + first(arr_regla) + 1
      ) /
        first(n_ug) / first(lp)),
      max_ing6_lp = max(c((lab6 + nolab6)[en_ug], 0)) / first(lp),
      prop_lab6 = if (ing6_lab_total + ing6_nolab_total > 0) {
        ing6_lab_total / (ing6_lab_total + ing6_nolab_total)
      } else {
        0
      }
    )
}

# Una tarea: modelos auxiliares con los hogares de los folds `entren`, que
# predicen los hogares de los folds `pred` (train) y, si `test`, los de test.
tarea_aux <- function(tarea, n_hilos) {
  ent_p <- which(pers$base == "train" & pers$fold %in% tarea$entren)
  lab_p <- ent_p[pers$pet[ent_p]] # personas con ingreso (edad de trabajar)
  ent_h <- which(fold %in% tarea$entren)

  m4 <- entrenar_aux(
    mat_pers, pers$y_ingtot, lab_p, pers$id, cat_pers, n_hilos, tarea$semilla
  )
  m6lab <- entrenar_aux(
    mat_pers6, pers$y_lab, lab_p, pers$id, cat_pers6, n_hilos, tarea$semilla + 1
  )
  m6nol <- entrenar_aux(
    mat_pers6, pers$y_nolab, lab_p, pers$id, cat_pers6, n_hilos,
    tarea$semilla + 2
  )
  m7 <- entrenar_aux(
    mat_hog_tr, obj_hog$log_ing_hog_lp, ent_h, train$id, categoricas, n_hilos,
    tarea$semilla + 3
  )
  m8 <- entrenar_aux(
    mat_hog_tr, obj_hog$estrato, ent_h, train$id, categoricas, n_hilos,
    tarea$semilla + 4
  )

  pred <- function(m, mat) predict(m, mat, num_iteration = m$best_iter)
  predecir <- function(filas_p, mat_h, ids_h) {
    pp <- pers[filas_p, ] |>
      mutate(
        p4 = pred(m4, mat_pers[filas_p, , drop = FALSE]),
        p6lab = pred(m6lab, mat_pers6[filas_p, , drop = FALSE]),
        p6nol = pred(m6nol, mat_pers6[filas_p, , drop = FALSE]),
        # Fuera de la edad de trabajar no hay ingreso.
        ing4 = if_else(pet, pmax(expm1(p4), 0), 0),
        lab6 = if_else(pet, pmax(expm1(p6lab), 0), 0),
        nolab6 = if_else(pet, pmax(expm1(p6nol), 0), 0)
      )
    hogar <- agregar(pp) |>
      left_join(tibble(
        id = ids_h, inghog_pred = pred(m7, mat_h),
        estrato_pred = pred(m8, mat_h)
      ), by = "id")
    list(hogar = hogar, personas = pp)
  }

  salida <- list(tarea = tarea, iter = c(
    m4$best_iter, m6lab$best_iter, m6nol$best_iter,
    m7$best_iter, m8$best_iter
  ))
  if (!is.null(tarea$pred)) {
    filas_h <- which(fold %in% tarea$pred)
    pr <- predecir(
      which(pers$base == "train" & pers$fold %in% tarea$pred),
      mat_hog_tr[filas_h, , drop = FALSE], train$id[filas_h]
    )
    salida$val <- pr$hogar
    # Para el R² fuera de muestra de los modelos auxiliares.
    if (tarea$tipo == "validacion") {
      pp <- pr$personas |> filter(pet)
      salida$r2 <- list(
        ingtot = tibble(
          real = pp$y_ingtot, v4 = pp$p4,
          v6 = log1p(pmax(expm1(pp$p6lab), 0) + pmax(expm1(pp$p6nol), 0))
        ),
        hogar = tibble(
          real7 = obj_hog$log_ing_hog_lp[filas_h],
          v7 = pred(m7, mat_hog_tr[filas_h, , drop = FALSE]),
          real8 = obj_hog$estrato[filas_h],
          v8 = pred(m8, mat_hog_tr[filas_h, , drop = FALSE])
        )
      )
    }
  }
  if (isTRUE(tarea$test)) {
    salida$test <- predecir(
      which(pers$base == "test"), mat_hog_te, test$id
    )$hogar
  }
  salida
}

# Tareas: 5 de validación (también predicen test), 20 internas y 1 con todo
# train (solo para comparar la variable de test).
tareas <- c(
  map(
    1:k,
    ~ list(
      tipo = "validacion", f = .x, entren = setdiff(1:k, .x), pred = .x,
      test = TRUE
    )
  ),
  unlist(
    map(1:k, function(f) {
      map(setdiff(1:k, f), function(g) {
        list(
          tipo = "interna", f = f, g = g, entren = setdiff(1:k, c(f, g)),
          pred = g, test = FALSE
        )
      })
    }),
    recursive = FALSE
  ),
  list(list(tipo = "completo", entren = 1:k, pred = NULL, test = TRUE))
)
tareas <- imap(tareas, ~ c(.x, semilla = 100 + 10 * .y))
marca(
  "Tareas auxiliares:", length(tareas), "- procesos:", n_proc, "- hilos:",
  n_hilos
)

cl <- makeCluster(n_proc, outfile = file.path(dir_out, "procesos.log"))
clusterEvalQ(cl, suppressMessages({
  library(tidyverse)
  library(lightgbm)
}))
clusterExport(cl, c(
  "pers", "mat_pers", "mat_pers6", "mat_hog_tr", "mat_hog_te", "obj_hog",
  "train", "test", "fold", "cat_pers", "cat_pers6", "categoricas",
  "params_aux", "nrounds_aux", "entrenar_aux", "agregar", "tarea_aux"
))
res_aux <- clusterApplyLB(cl, tareas, tarea_aux, n_hilos = n_hilos)
stopCluster(cl)
rm(mat_pers, mat_pers6)
marca("Variables auxiliares listas")

# Variables por fold.
hogar_de <- function(filtro, parte = "val") {
  bind_rows(map(
    keep(res_aux, filtro), parte
  ))
}
v_val <- map(
  1:k, function(f) hogar_de(~ .x$tarea$tipo == "validacion" && .x$tarea$f == f)
)
v_entren <- map(
  1:k, function(f) hogar_de(~ .x$tarea$tipo == "interna" && .x$tarea$f == f)
)
v_oof <- bind_rows(v_val)
# Test (ii): promedio de los 5 modelos de los folds externos.
v_test_prom <- hogar_de(~ .x$tarea$tipo == "validacion", "test") |>
  summarize(.by = id, across(everything(), mean))
# Test (iii): modelo con todo train (solo para comparar).
v_test_todo <- hogar_de(~ .x$tarea$tipo == "completo", "test")
stopifnot(
  setequal(v_oof$id, train$id), setequal(v_test_prom$id, test$id),
  all(map_lgl(1:k, ~ setequal(v_entren[[.x]]$id, train$id[fold != .x])))
)

# Calidad de los modelos auxiliares (fuera de muestra) y AUC de cada
# variable sola para predecir pobre (menor ingreso o estrato = más pobre).
r2_pers <- bind_rows(map(
  keep(res_aux, ~ .x$tarea$tipo == "validacion"), ~ .x$r2$ingtot
))
r2_hog <- bind_rows(map(
  keep(res_aux, ~ .x$tarea$tipo == "validacion"), ~ .x$r2$hogar
))
cat(
  "R² fuera de muestra: V4 log1p(Ingtot)",
  round(r2(r2_pers$real, r2_pers$v4), 4),
  "- V6 (laboral + no laboral)", round(r2(r2_pers$real, r2_pers$v6), 4),
  "- V7 log(Ingpcug/Lp)", round(r2(r2_hog$real7, r2_hog$v7), 4),
  "- V8 estrato", round(r2(r2_hog$real8, r2_hog$v8), 4), "\n"
)
oof_var <- function(v) v_oof[[v]][match(train$id, v_oof$id)]
auc_vars <- tibble(variable = c(
  "log_ing_pred_pc_lp", "log_ingarr_pc_lp", "log_ing6_pc_lp",
  "log_ing6arr_pc_lp", "inghog_pred", "estrato_pred"
)) |>
  mutate(
    bloque = c("V4", "V5", "V6", "V6 (con arriendo)", "V7", "V8"),
    auc = map_dbl(variable, ~ calcular_auc(-oof_var(.x), y))
  )
print(auc_vars)
cat("Iteraciones de los modelos auxiliares (V4, V6 lab, V6 no lab, V7, V8):\n")
print(do.call(rbind, map(res_aux, "iter")) |> apply(2, range))
if (!prueba) {
  write_rds(
    list(
      v_val = v_val, v_entren = v_entren, v_test_prom = v_test_prom,
      v_test_todo = v_test_todo, auc = auc_vars
    ),
    file.path(dir_out, "variables_por_fold.rds")
  )
}

# 6. Experimentos F0-F5 -------------------------------------------------------

cols_bloque <- list(
  V4 = c(
    "ing_pred_total", "log_ing_pred_pc_lp", "max_ing_pred_lp", "prop_ing_jefe"
  ),
  V5 = "log_ingarr_pc_lp",
  V6 = c(
    "ing6_lab_total", "ing6_nolab_total", "log_ing6_pc_lp", "log_ing6arr_pc_lp",
    "max_ing6_lp", "prop_lab6"
  ),
  V7 = "inghog_pred",
  V8 = "estrato_pred"
)
tags <- c(
  V4 = "ingpred", V5 = "ingarr", V6 = "ingpers2", V7 = "inghog", V8 = "estrato"
)

hog_tr <- bind_cols(id = train$id, as_tibble(mat_hog_tr))
hog_te <- bind_cols(id = test$id, as_tibble(mat_hog_te))

armar_mat <- function(base, bloques, v) {
  dinam <- unlist(cols_bloque[bloques])
  bind_cols(select(base, all_of(x_base)), v[match(base$id, v$id), dinam]) |>
    data.matrix()
}

# Validación cruzada con los parámetros `params` (cada fold por separado,
# como en 13: la mejor iteración minimiza el logloss promedio de los folds).
cv_modelo <- function(exp, n_hilos) {
  inicio <- Sys.time()
  params <- c(params_fijos, exp$params, num_threads = n_hilos)
  if (!is.null(exp$semilla)) {
    params$seed <- exp$semilla
    params$bagging_seed <- exp$semilla
    params$feature_fraction_seed <- exp$semilla
  }
  if (isTRUE(params$bagging_fraction < 1)) params$bagging_freq <- 1L
  modelos <- list()
  perdida <- list()
  mat_va <- list()
  for (f in 1:k) {
    i_tr <- which(fold != f)
    i_va <- which(fold == f)
    mat_tr <- armar_mat(hog_tr[i_tr, ], exp$bloques, v_entren[[f]])
    mat_va[[f]] <- armar_mat(hog_tr[i_va, ], exp$bloques, v_val[[f]])
    dtr <- lgb.Dataset(
      mat_tr,
      label = y[i_tr], categorical_feature = categoricas
    )
    dva <- lgb.Dataset.create.valid(dtr, mat_va[[f]], label = y[i_va])
    modelos[[f]] <- lgb.train(
      params = params, data = dtr, nrounds = nrounds_hogar,
      valids = list(valid = dva), early_stopping_rounds = 100,
      verbose = -1
    )
    perdida[[f]] <- unlist(modelos[[f]]$record_evals$valid$binary_logloss$eval)
  }
  largo <- min(lengths(perdida))
  mejor_iter <- which.min(rowMeans(sapply(perdida, `[`, 1:largo)))
  oof <- numeric(length(y))
  for (f in 1:k) {
    oof[fold == f] <- predict(modelos[[f]], mat_va[[f]],
      num_iteration = mejor_iter
    )
  }
  mejor <- f1_cv(oof)
  list(
    nombre = exp$nombre, bloques = exp$bloques, params = exp$params,
    semilla = exp$semilla, mejor_iter = mejor_iter, umbral = mejor$umbral,
    f1 = mejor$f1, oof = oof,
    minutos = as.numeric(difftime(Sys.time(), inicio, units = "mins"))
  )
}

correr <- function(exps) {
  n <- min(length(exps), n_proc)
  cl <- makeCluster(n, outfile = file.path(dir_out, "procesos.log"))
  clusterEvalQ(cl, suppressMessages({
    library(tidyverse)
    library(lightgbm)
  }))
  clusterExport(cl, c(
    "hog_tr", "y", "fold", "k", "x_base", "cols_bloque", "categoricas",
    "v_val", "v_entren", "params_fijos", "nrounds_hogar", "calcular_f1",
    "grid_umbral", "f1_cv", "armar_mat", "cv_modelo"
  ))
  res <- clusterApplyLB(cl, exps, cv_modelo, n_hilos = n_nucleos %/% n)
  stopCluster(cl)
  set_names(res, map_chr(res, "nombre"))
}

experimentos <- list(
  list(nombre = "F0", bloques = "V4", params = params_07),
  list(nombre = "F1", bloques = c("V4", "V5"), params = params_07),
  list(nombre = "F2", bloques = c("V4", "V6"), params = params_07),
  list(nombre = "F3", bloques = c("V4", "V7"), params = params_07),
  list(nombre = "F4", bloques = c("V4", "V8"), params = params_07)
)
res <- correr(experimentos)
marca("F0-F4 listos")

# F5: F0 + los bloques que mejoran el F1 de F0 en más de 0,001.
mejoran <- c(V5 = "F1", V6 = "F2", V7 = "F3", V8 = "F4") |>
  keep(~ res[[.x]]$f1 - res$F0$f1 > 0.001) |>
  names()
cat(
  "Bloques que mejoran F0 en más de 0,001:",
  if (length(mejoran)) mejoran else "ninguno", "\n"
)
bloques_f5 <- c("V4", mejoran)
igual <- keep(res, ~ setequal(.x$bloques, bloques_f5))
res$F5 <- if (length(igual) > 0) {
  modifyList(igual[[1]], list(nombre = "F5"))
} else {
  correr(list(list(nombre = "F5", bloques = bloques_f5, params = params_07)))$F5
}
marca("F5 listo: F1 =", round(res$F5$f1, 4))

# 7. Reajuste: random search sobre las variables de F5 ------------------------

# 20 configuraciones al azar con los rangos de 12.
set.seed(123)
n_random <- if (prueba) 2 else 20
grilla <- tibble(
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
configs <- map(seq_len(n_random), ~ list(
  nombre = sprintf("R%02d", .x), bloques = bloques_f5,
  params = as.list(grilla[.x, ])
))
res_rs <- correr(configs)
grid_rs <- map_dfr(res_rs, ~ as_tibble(.x$params) |>
  mutate(
    nombre = .x$nombre, mejor_iter = .x$mejor_iter,
    umbral = .x$umbral, f1 = .x$f1, .before = 1
  )) |>
  arrange(desc(f1))
print(grid_rs, n = Inf, width = Inf)
# El mejor entre la configuración fija de 12 (F5) y las 20 nuevas.
mejor_rs <-
  if (grid_rs$f1[1] > res$F5$f1) res_rs[[grid_rs$nombre[1]]] else res$F5
marca("Reajuste listo: mejor", mejor_rs$nombre, "- F1 =", round(mejor_rs$f1, 4))

# 8. Promedio de 5 semillas ---------------------------------------------------

# La mejor configuración con 5 semillas distintas (seed, bagging_seed y
# feature_fraction_seed); se promedian las probabilidades fuera de muestra.
semillas <- map(1:5, ~ list(
  nombre = paste0("S", .x), bloques = bloques_f5,
  params = mejor_rs$params, semilla = 1000 + .x
))
res_sem <- correr(semillas)
oof_sem <- rowMeans(sapply(res_sem, `[[`, "oof"))
f1_sem <- f1_cv(oof_sem)
cat("F1 de cada semilla:", round(map_dbl(res_sem, "f1"), 4), "\n")
marca(
  "Promedio de 5 semillas: F1 =", round(f1_sem$f1, 4), "- umbral =",
  f1_sem$umbral
)

# 9. Modelo final, diagnóstico de la variable predicha y submissions ----------

# Modelo con todo train: en train, las variables auxiliares fuera de muestra
# de 5 folds; en test, el promedio de los 5 modelos de los folds (ii). Se
# predice también con la variable del modelo con todo train (iii), solo para
# comparar. nrounds = mejor iteración x 1,1.
modelo_final <- function(r) {
  params <- c(params_fijos, r$params, num_threads = n_nucleos)
  if (!is.null(r$semilla)) {
    params$seed <- r$semilla
    params$bagging_seed <- r$semilla
    params$feature_fraction_seed <- r$semilla
  }
  if (isTRUE(params$bagging_fraction < 1)) params$bagging_freq <- 1L
  n_final <- round(r$mejor_iter * 1.1)
  dtr <- lgb.Dataset(armar_mat(hog_tr, r$bloques, v_oof),
    label = y,
    categorical_feature = categoricas
  )
  modelo <- lgb.train(
    params = params, data = dtr, nrounds = n_final, verbose = -1
  )
  list(
    modelo = modelo, n_final = n_final,
    prob_prom = predict(modelo, armar_mat(hog_te, r$bloques, v_test_prom)),
    prob_todo = predict(modelo, armar_mat(hog_te, r$bloques, v_test_todo))
  )
}

f1_07b <- 0.7596
minimo <- f1_07b + 0.001
sin_punto <- function(v) gsub("[.]", "", sub("^-", "m", as.character(v)))
verificar <- function(s) {
  stopifnot(
    nrow(s) == nrow(test), identical(names(s), c("id", "pobre")),
    all(s$pobre %in% c(0, 1)), !anyNA(s), !anyDuplicated(s$id),
    setequal(s$id, test$id)
  )
  mean(s$pobre)
}
nombre_lgbm <- function(p, n_final, bloques, umbral, sufijo = "") {
  paste0(
    "LGBM_lr_", sin_punto(p$learning_rate), "_leaves_", p$num_leaves,
    "_depth_", sin_punto(p$max_depth), "_rounds_", n_final,
    "_vars_", paste(tags[bloques], collapse = "_"),
    "_threshold_", sin_punto(umbral), sufijo, ".csv"
  )
}
guardar <- function(nombre, archivo, prob, umbral, f1, siempre = FALSE) {
  if (!siempre && f1 <= minimo) {
    cat(
      nombre, ": F1 de CV", round(f1, 4), "no supera", minimo,
      "- no se genera\n"
    )
    return(invisible(NULL))
  }
  s <- tibble(id = test$id, pobre = as.integer(prob > umbral))
  prop <- verificar(s)
  if (!prueba) write_csv(s, here("02_outputs", "predictions", archivo))
  cat(
    nombre, ":", archivo, "- F1 de CV", round(f1, 4), "- proporción de pobres",
    round(prop, 4), "\n"
  )
}

# F0 (E4 de 13 con la corrección de test): siempre.
final_f0 <- modelo_final(res$F0)
guardar("F0", nombre_lgbm(
  params_07, final_f0$n_final, "V4", res$F0$umbral, ""
) |>
  sub(pattern = "_vars_ingpred_", replacement = "_vars_ingpred_fix_"),
final_f0$prob_prom, res$F0$umbral, res$F0$f1,
siempre = TRUE
)

# Mejor de F1-F5 con la configuración de 12.
candidatos <- res[c("F1", "F2", "F3", "F4", "F5")]
mejor_f <- candidatos[[which.max(map_dbl(candidatos, "f1"))]]
final_f <- modelo_final(mejor_f)
guardar(
  mejor_f$nombre,
  nombre_lgbm(params_07, final_f$n_final, mejor_f$bloques, mejor_f$umbral),
  final_f$prob_prom, mejor_f$umbral, mejor_f$f1
)

# Mejor del reajuste (si no es la configuración de 12).
if (!identical(mejor_rs$nombre, "F5")) {
  final_rs <- modelo_final(mejor_rs)
  guardar(
    paste("Reajuste", mejor_rs$nombre),
    nombre_lgbm(
      mejor_rs$params, final_rs$n_final, mejor_rs$bloques, mejor_rs$umbral
    ),
    final_rs$prob_prom, mejor_rs$umbral, mejor_rs$f1
  )
}

# Promedio de 5 semillas.
finales_sem <- map(res_sem, modelo_final)
prob_sem_prom <- rowMeans(sapply(finales_sem, `[[`, "prob_prom"))
prob_sem_todo <- rowMeans(sapply(finales_sem, `[[`, "prob_todo"))
guardar(
  "Semillas", nombre_lgbm(
    mejor_rs$params, round(mean(map_dbl(finales_sem, "n_final"))),
    bloques_f5, f1_sem$umbral, "_seeds5"
  ),
  prob_sem_prom, f1_sem$umbral, f1_sem$f1
)
marca("Modelos finales listos")

# Importancia por ganancia del mejor modelo (promedio de las 5 semillas).
importancia <- map_dfr(finales_sem, ~ lgb.importance(
  .x$modelo,
  percentage = TRUE
) |>
  as_tibble()) |>
  summarize(.by = Feature, ganancia = sum(Gain) / length(finales_sem)) |>
  rename(variable = Feature) |>
  arrange(desc(ganancia))
print(importancia |> slice_head(n = 20))

# Diagnóstico de la variable predicha principal (la de mayor ganancia entre
# las auxiliares): train fuera de muestra (i), test con el promedio de los
# folds (ii) y test con el modelo de todo train (iii).
principal <-
  importancia$variable[importancia$variable %in% unlist(cols_bloque)][1]
resumen <- function(x, fuente) {
  tibble(
    fuente = fuente, media = mean(x), de = sd(x),
    p10 = quantile(x, 0.1), p50 = quantile(x, 0.5), p90 = quantile(x, 0.9)
  )
}
diagnostico <- bind_rows(
  resumen(v_oof[[principal]], "(i) train fuera de muestra"),
  resumen(v_test_prom[[principal]], "(ii) test, promedio de los 5 folds"),
  resumen(v_test_todo[[principal]], "(iii) test, modelo con todo train")
) |>
  # Proporción predicha de pobres con el promedio de las 5 semillas.
  mutate(prop_pobres = c(
    mean(oof_sem > f1_sem$umbral), mean(prob_sem_prom > f1_sem$umbral),
    mean(prob_sem_todo > f1_sem$umbral)
  ))
cat("Variable predicha principal:", principal, "\n")
print(as.data.frame(diagnostico), digits = 4)

if (!prueba) {
  iwalk(c(res, res_sem), ~ write_rds(
    tibble(id = train$id, fold = fold, pobre = y, prob = .x$oof),
    file.path(dir_out, paste0("oof_", .y, ".rds"))
  ))
  write_rds(
    tibble(id = train$id, fold = fold, pobre = y, prob = oof_sem),
    file.path(dir_out, "oof_semillas.rds")
  )
  write_rds(grid_rs, file.path(dir_out, "random_search.rds"))
  write_rds(importancia, file.path(dir_out, "importancia.rds"))
  write_rds(diagnostico, file.path(dir_out, "diagnostico_variable.rds"))
}

# 10. Tabla de resultados -----------------------------------------------------

describir <- function(bloques) {
  extra <- setdiff(bloques, "V4")
  if (length(extra) == 0) {
    return("Base + V4 (ingreso predicho de las personas)")
  }
  paste(
    "F0 +",
    paste(map_chr(extra, ~ paste0(.x, " (", tags[[.x]], ")")), collapse = " + ")
  )
}
fila <- function(nombre, r, f1, umbral, desc) {
  tibble(
    experimento = nombre, variables = desc,
    mejor_iter = as.character(r$mejor_iter),
    umbral = umbral, f1 = f1, dif_F0 = f1 - res$F0$f1
  )
}
tabla <- bind_rows(
  map_dfr(
    c("F0", "F1", "F2", "F3", "F4", "F5"),
    ~ fila(
      .x, res[[.x]], res[[.x]]$f1, res[[.x]]$umbral,
      describir(res[[.x]]$bloques)
    )
  ),
  fila(
    "Reajuste", mejor_rs, mejor_rs$f1, mejor_rs$umbral,
    paste0(
      "Variables de F5; mejor de 20 configuraciones (lr ",
      mejor_rs$params$learning_rate,
      ", hojas ", mejor_rs$params$num_leaves, ", profundidad ",
      mejor_rs$params$max_depth, ")"
    )
  ),
  fila(
    "Semillas",
    list(mejor_iter = paste(map_int(res_sem, "mejor_iter"), collapse = "/")),
    f1_sem$f1, f1_sem$umbral, "Reajuste, promedio de 5 semillas"
  )
)
print(tabla, width = Inf)
if (!prueba) {
  kable(tabla,
    format = "latex", booktabs = TRUE, digits = c(0, 0, 0, 2, 4, 4),
    col.names = c(
      "Experimento", "Variables", "Iteraciones", "Umbral", "F1 (CV)",
      "Diferencia vs. F0"
    ),
    align = c("l", "p{7cm}", "r", "r", "r", "r"),
    caption = paste0(
      "LightGBM con variables auxiliares predichas (CV anidada; test con el ",
      "promedio de los modelos de los folds)"
    ),
    label = "boosting_ingreso"
  ) |>
    escribir_tex(here("02_outputs", "tables", "boosting_ingreso.tex"))
  write_rds(
    list(tabla = tabla, f1_mejor = max(tabla$f1)),
    file.path(dir_out, "resultados.rds")
  )
}
marca(
  "Tiempo total:", round(difftime(Sys.time(), inicio_total, units = "mins"), 1),
  "minutos"
)

# Notas -----------------------------------------------------------------------

# Errores de anidar predicciones (para las diapositivas):
# (1) Fuga de la etiqueta: si el modelo auxiliar viera el ingreso de los
#     hogares con los que se evalúa, la variable predicha copiaría la
#     respuesta y el F1 de CV estaría inflado. Control: CV anidada; ningún
#     modelo auxiliar usa la etiqueta del fold que se evalúa.
# (2) Error distinto en train y test: el clasificador aprende con variables
#     predichas fuera de muestra (con su error); si test recibe una variable
#     más precisa, el umbral y los cortes no le sirven. Control: en test, el
#     promedio de los 5 modelos de los folds. En la práctica la diferencia
#     es mínima: la variable principal tiene media 0.537 con (ii) y 0.537 con
#     (iii), y la proporción de pobres 24.08% y 24.03%.

# Paso 0: Ingpcug = Ingtotugarr / Npersug e Ingtotug = suma de Ingtot de la
# unidad de gasto se cumplen en el 100% de los hogares, e Ingtot = laboral +
# no laboral en el 100% de las personas. Arriendo imputado = Ingtotugarr -
# Ingtotug: P5130 en propia pagada (99.5% exacto) y usufructo (92%), P5130 -
# P5100 en propia pagando (92%), 0 en arriendo, posesión sin título y otra
# (100%), y 0 si P5130 es 98/99. La regla coincide en el 98.4% de los
# hogares; las diferencias son hogares con arriendo imputado 0 pese a
# reportar P5130 (sobre todo en usufructo y con valores extremos).

# Modelos auxiliares (fuera de muestra): R² 0.942 en log1p(Ingtot) (V4),
# 0.943 con laboral + no laboral y contexto del hogar (V6), 0.757 en
# log(Ingpcug / Lp) (V7) y 0.836 en el estrato (V8). AUC de cada variable
# sola para pobre: V4 0.922, V5 0.940, V6 0.922 (0.942 con arriendo), V7
# 0.943, V8 0.700.

# Experimentos (F1 de CV): F0 0.7584, F1 (+V5) 0.7584, F2 (+V6) 0.7599, F3
# (+V7) 0.7598, F4 (+V8) 0.7587, F5 (V6 + V7) 0.7623. Reajuste (mejor de 20:
# lr 0.02, 31 hojas, sin límite de profundidad, min_data_in_leaf 50,
# feature_fraction 0.9, bagging 0.9) 0.7629. Promedio de 5 semillas 0.7622:
# las semillas solas dan 0.7618-0.7626, así que el 0.7629 del reajuste tiene
# algo de suerte y la mejora real sobre F5 es nula. La superficie del random
# search es plana (0.758-0.763).

# V5 no aporta: el boosting ya combina el ingreso predicho con las
# variables de arriendo. Lo que suma es V6 (laboral y no laboral con el
# contexto del hogar, con arriendo imputado: 63% de la ganancia) y V7
# (ingreso del hogar predicho directamente: 20%).

# F0 da 0.7584 y no 0.7594 como E4 en 13, con el mismo código: cambian solo
# las semillas de los modelos auxiliares (la muestra para su early
# stopping). Es la variación que introduce anidar predicciones (~0.001), del
# mismo orden que varias de las mejoras; por eso solo se aceptan mejoras de
# más de 0.001.

# Para pasar a 02: nada. Todas las variables de este script dependen de
# Ingtot, Ingpcug o Estrato1 y se calculan dentro de cada fold. La regla del
# arriendo imputado sí se podría agregar en agregar_hogar().

# Tiempo: ~65 minutos con 20 núcleos (26 tareas de modelos auxiliares en 5
# procesos: 25 minutos; experimentos 10; random search 16; semillas 6).
# RAM máxima ~20 GB.
