###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   11_xgboost_hogar.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           XGBoost a nivel hogar (una fila por hogar), el modelo
#                      final. Además de la base por hogar de 02, usa variables
#                      que salen de modelos auxiliares por persona (ingreso
#                      predicho, probabilidad de ser pobre), estimadas con
#                      validación cruzada anidada para no filtrar pobre. Se
#                      combina con una regresión del ingreso. Mismos 5 folds y
#                      mismo F1 que los demás modelos por hogar, para comparar.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Configuración: hiperparámetros, bases y conjuntos de variables
# 3. Datos y variables
#    3.1 Bases por hogar y por persona
#    3.2 Variables que no dependen de los folds
#    3.3 Modelos auxiliares por persona (dependen de los folds)
#    3.4 Versiones afinadas de los modelos auxiliares
# 4. Funciones de validación cruzada, combinación y predicción
# 5. Pruebas, en el orden en que se hicieron: cada una parte del mejor modelo
#    hasta ese punto (5.1-5.14: variables, bases, hiperparámetros y
#    combinaciones; 5.15-5.25: modelos auxiliares mejorados)
# 6. Modelo final y submissions
# 7. Naive bayes: con la base por hogar y combinado con el mejor modelo
# 8. Comparación de codificaciones de las categóricas
# 9. Random forest por hogar y combinación con XGBoost
# 10. Análisis de errores del mejor modelo
# 11. Modelo especialista para los hogares dudosos
# 12. Ruido: el mejor modelo con otras particiones en folds
# 13. Combinación con el logit lasso (08)
# 14. Importancia por permutación del mejor modelo
# 15. Importancia de extremo a extremo por concepto
# 16. Importancia de extremo a extremo por variable de la persona

# Input:  data/train_hogar.rds , data/test_hogar.rds (02);
#         data/train_clean.rds , data/test_clean.rds (01);
#         data/train_personas.csv (solo Ingtot y sus componentes, objetivos del
#         ingreso predicho); 02_outputs/logs/logit_interacciones/logit_oof.rds
#         (sección 13; lo genera 01_code/08_logit_interacciones.R).
# Output: 02_outputs/pruebas/xgb_hogar_nativo/: resultado de cada prueba (.rds)
# y
#         progreso.log (en 14, 15 y 16, importancia_permutacion.rds,
#         importancia_conceptos.rds e importancia_variables_e2e.rds);
#         02_outputs/predictions/XGBH_<prueba>_..._threshold_<c>.csv .
#
# Correr desde la raíz. Cada prueba se guarda al terminar y, si ya existe, se
# lee: volver a correr el script solo estima lo que falta. Con PRUEBA=1 usa el
# 10% de los hogares y pocas iteraciones, y guarda en
# 02_outputs/pruebas/xgb_hogar_prueba_nativo/.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  xgboost, # Gradient boosting.
  e1071, # Naive bayes (sección 7).
  ranger, # Random forest (sección 9).
  parallel # Procesos en paralelo (fork).
)
# Funciones comunes (calcular_f1, calcular_auc, evaluar_umbral).
source(here("01_code", "functions", "funciones_comunes.R"))

prueba <- Sys.getenv("PRUEBA") == "1"
# Codificación de las categóricas (variable de entorno CODIFICACION):
#   "nativo" (por defecto): categóricas nativas de xgboost (factores) en el
#     modelo por hogar. Es la que mejor F1 dio (sección 8 y NOTAS).
#   "codigos": códigos enteros, como en 13 (pensado para LightGBM, que los
#     trata como categorías); xgboost los trata como números ordenados.
#   "onehot": una dummy por nivel en el modelo por hogar.
# Con onehot y nativo los modelos por persona (V4, P) usan categóricas nativas:
# el one-hot por persona (unas 170 columnas para 760 mil personas) no cabe en
# memoria. Las pruebas ya descartadas con códigos (P2, ruido, stacking de la
# regresión, más no lineales, naive bayes) solo corren con códigos.
codificacion <- Sys.getenv("CODIFICACION", "nativo")
stopifnot(codificacion %in% c("codigos", "onehot", "nativo"))
completo <- codificacion == "codigos"
# NORMALIZAR=1: variables numéricas de la base por hogar estandarizadas con la
# media y la desviación de train (se aplica en armar()). Para árboles no
# debería cambiar nada: una transformación afín da los mismos cortes.
normalizar <- Sys.getenv("NORMALIZAR") == "1"
dir_out <- here("02_outputs", "pruebas", paste0(
  if (prueba) "xgb_hogar_prueba" else "xgb_hogar",
  if (completo) "" else paste0("_", codificacion),
  if (normalizar) "_normalizada" else ""
))
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
marca <- function(...) {
  linea <- paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "-", ...)
  cat(linea, "\n")
  cat(linea, "\n", file = file.path(dir_out, "progreso.log"), append = TRUE)
}
# lapply en `nucleos` procesos (fork) que se detiene si falla alguno; `que`
# describe la tarea en el mensaje de error.
en_paralelo <- function(x, f, ..., nucleos, que) {
  res <- mclapply(x, f, ..., mc.cores = nucleos, mc.preschedule = FALSE)
  fallas <- keep(res, ~ inherits(.x, "try-error"))
  if (length(fallas)) {
    stop(
      "Falló ", que, ": ", conditionMessage(attr(fallas[[1]], "condition"))
    )
  }
  res
}

# 2. Configuración: hiperparámetros, bases y conjuntos de variables -----------

# Paralelismo: el xgboost de CRAN para macOS no tiene OpenMP (un núcleo por
# modelo), así que se reparten modelos entre procesos (fork, sin copiar datos).
nucleos <- 5

# Hiperparámetros base del clasificador (los mejores del XGBoost por persona,
# 10). Se usa parada temprana: nrounds máximo y 100 rondas sin mejorar en el
# fold de validación (como lgb.cv en 12).
params_base <- list(
  objective = "binary:logistic", eval_metric = "logloss",
  eta = 0.05, max_depth = 6, min_child_weight = 50,
  subsample = 0.8, colsample_bytree = 0.8, gamma = 0,
  nthread = 1, seed = 123
)
nrounds_max <- if (prueba) 60 else 5000
parada <- 100

# Grilla de afinamiento (sección 5) sobre el mejor conjunto de variables.
grid_afinar <- expand_grid(
  max_depth = c(4, 6, 8), min_child_weight = c(10, 50),
  colsample_bytree = c(0.6, 0.8)
)

# Umbrales: probabilidad del hogar desde la que se predice pobre.
grid_umbral <- seq(0.10, 0.85, by = 0.01)

# Bases de entrenamiento a nivel hogar:
#   "normal": todos los hogares.
#   "aumentada": hogares pobres con peso (no pobres / pobres) ~ 4, que en
#     xgboost equivale a replicarlos (como train_clean_augmented).
#   "sin_ricos": se entrena sin el 10% de hogares más ricos (hogar_rico, 01)
#     y se valida con todos.
bases <- c("normal", "aumentada", "sin_ricos")

# 3. Datos y variables --------------------------------------------------------

# Grupos de variables (lista `cols`). Sus nombres aparecen en los nombres de
# las pruebas (p. ej. "H1_V1aV4" o "..._mas_V4t"):
#   x_base: las 94 variables por hogar de 02.
#   V1 factor de expansión; V2 el ocupado más educado del hogar; nuestras:
#     educación de los adultos y coeficientes de variación (de 01).
#   No lineales: N1 productos, N2 razones por ocupado, N3 diferencia con la
#     mediana de la ciudad, N4 combinaciones de las más importantes y T
#     logaritmos y cuadrados. I: índices (promedio de variables débiles del
#     mismo tema).
#   De modelos auxiliares por persona (3.3 y 3.4; dependen de los folds):
#     V3 tasa de pobreza del oficio; V4 ingreso predicho; P probabilidad de
#     ser pobre; PI de ser indigente; C080 y C125 de tener un ingreso menor a
#     0,8 y 1,25 líneas; IS índices supervisados (logit por tema).
#     Versiones de V4: _t afinado, _h con variables del hogar, _c laboral y no
#     laboral por separado, _e y _f con las preguntas que no se usaban, _w y
#     _x con objetivo Tweedie. Versiones de P: Pt afinado, Pe con las
#     preguntas no usadas, Pa con el arriendo imputado.
#   A: ingreso predicho más arriendo imputado (como Ingpcug del DANE).
#   Z: regresión del ingreso del hogar; P2: P con las variables del hogar.

## 3.1 Bases por hogar y por persona ------------------------------------------

train <- read_rds(here("data", "train_hogar.rds"))
test <- read_rds(here("data", "test_hogar.rds"))
pers_tr <- read_rds(here("data", "train_clean.rds"))
pers_te <- read_rds(here("data", "test_clean.rds"))
# Ingreso total de la persona: solo en train, solo como objetivo del ingreso
# predicho (V4). Ingtot = laboral + no laboral (observado más imputado por el
# DANE, *es); se verificó que la suma coincide en todas las personas.
ingtot <- read_csv(here("data", "train_personas.csv"),
  guess_max = Inf,
  col_select = c(
    id, Orden, Ingtot, Impa, Isa, Ie, Imdi, Impaes, Isaes, Iees, Imdies
  ),
  show_col_types = FALSE
) |>
  mutate(
    ing_lab = rowSums(
      pick(Impa, Isa, Ie, Imdi, Impaes, Isaes, Iees, Imdies),
      na.rm = TRUE
    ),
    ing_nolab = if_else(is.na(Ingtot), NA_real_, Ingtot - ing_lab)
  ) |>
  select(id, Orden, Ingtot, ing_lab, ing_nolab)
pers_tr <- left_join(
  pers_tr, ingtot,
  by = c("id", orden = "Orden"), relationship = "one-to-one"
)
rm(ingtot)

# Variables del hogar que solo están en train: ingreso (para la regresión) y
# hogar_rico (para la base sin ricos).
solo_train <- distinct(pers_tr, id, ingreso, linea_pobreza, hogar_rico)
train <- left_join(train, solo_train, by = "id", relationship = "one-to-one")
rm(solo_train)

if (prueba) {
  set.seed(1)
  train <- slice_sample(train, prop = 0.1)
  test <- slice_sample(test, prop = 0.1)
  pers_tr <- filter(pers_tr, id %in% train$id)
  pers_te <- filter(pers_te, id %in% test$id)
  marca("MODO PRUEBA:", nrow(train), "hogares")
}
y <- train$pobre
# Objetivo de la regresión: log del ingreso relativo a la línea (pobre si < 0).
z <- log1p(train$ingreso) - log(train$linea_pobreza)

# Mismos 5 folds que los demás modelos por hogar (una fila = un hogar).
# particion() arma otras particiones para medir el ruido del F1 (5.10).
k <- 5
particion <- function(semilla) {
  set.seed(semilla)
  sample(rep(1:k, length.out = nrow(train)))
}
fold <- particion(123)

## 3.2 Variables que no dependen de los folds ---------------------------------

# Marcas por persona (como agregar_hogar() de 02 y 13).
marcar <- function(df) {
  df |> mutate(
    en_ug = !parentesco_jefe_hogar %in% c(6, 7, 8),
    jefe = parentesco_jefe_hogar == 1,
    ocup = ocupado == 1 & en_ug,
    horas = coalesce(horas_trabajo_semana, 0) +
      coalesce(horas_segundo_trabajo, 0),
    pet = !is.na(edad_trabajar)
  )
}
pers_tr <- marcar(pers_tr)
pers_te <- marcar(pers_te)

# V1 (de 13): factor de expansión. V2 (de 13): el mejor ocupado del hogar.
v1 <- function(p) {
  p |>
    summarize(.by = id, fex_c = first(fex_c), fex_dpto = first(fex_dpto))
}
v2 <- function(p) {
  p |>
    filter(ocup) |>
    summarize(
      .by = id, ocup_max_educ = max(coalesce(anos_educ_imp, 0)),
      ocup_max_tamano = max(tamano_empresa),
      ocup_posicion_educ = coalesce(
        posicion_ocupacional[which.max(coalesce(anos_educ_imp, -1))], 0
      ),
      ocup_max_horas = max(horas)
    )
}
# Nuestras (de 01), que no están en la base de 02: educación de los
# adultos (02 la calcula con todos los miembros) y coeficientes de variación.
nuestras <- function(p) {
  distinct(
    p, id, educ_max_adultos, cv_educ_adultos, cv_horas_ocupados
  )
}

cols <- list(
  V1 = c("fex_c", "fex_dpto"),
  V2 = c(
    "ocup_max_educ", "ocup_max_tamano", "ocup_posicion_educ", "ocup_max_horas"
  ),
  V3 = c("te_oficio_media", "te_oficio_min", "te_oficio_max"),
  V4 = c(
    "ing_pred_total", "log_ing_pred_pc_lp", "max_ing_pred_lp", "prop_ing_jefe"
  ),
  P = c("p_pers_media", "p_pers_max", "p_pers_min", "p_pers_jefe"),
  # P2: como P, pero el clasificador por persona también ve las variables del
  # hogar (stacking más rico).
  P2 = c("p2_pers_media", "p2_pers_max", "p2_pers_min", "p2_pers_jefe"),
  # Z: predicción de la regresión de log(ingreso / línea) del hogar (stacking
  # de la regresión como variable del clasificador).
  Z = "z_pred",
  # N4: combinaciones de las variables más importantes del mejor modelo
  # (ingreso predicho, stacking por persona, arriendo). Dependen del fold.
  N4 = c(
    "ing_x_pmin", "ing_menos_arriendo", "ing_x_ocup_pc", "ing_x_educ_max",
    "ing_x_dependencia", "p_rango", "p_jefe_menos_media", "pmin_x_ocup_pc",
    "arriendo_x_cuartos", "horas_x_educ_ocup"
  ),
  # T: logaritmos y cuadrados. Con los cortes por cuantiles de xgboost una
  # transformación monótona da los mismos cortes; solo puede cambiar qué tan
  # seguido entra la variable en los árboles (colsample_bytree).
  T = c(
    "log_ing_pred_total", "log_horas_pc", "log_cuartos_pc",
    "log_educ_ocup_suma_pc",
    "log_beneficios_pc", "ing_pred_pc_lp_2", "p_pers_min_2", "arriendo_2",
    "educ_max_2", "dependencia_2"
  ),
  nuestras = c("educ_max_adultos", "cv_educ_adultos", "cv_horas_ocupados"),
  # N1: productos (interacciones que un árbol no arma con un solo corte).
  N1 = c(
    "educ_adultos_x_ocup_pc", "ocup_pc_x_log_lp", "arriendo_x_log_lp",
    "horas_pc_x_formal", "educ_ocup_x_formal_pc", "jefe_educ_x_formal"
  ),
  # N2: razones por ocupado y brecha de educación (ing_pred_por_ocup depende
  # del fold: se calcula con V4 en matrices()).
  N2 = c(
    "dep_por_ocup", "beneficios_por_ocup", "horas_por_ocup", "brecha_educ_jefe",
    "ing_pred_por_ocup"
  ),
  # N3: relativas a la ciudad (diferencia con la mediana del dominio, con
  # train y test juntos; no usa pobre).
  N3 = c(
    "rel_dom_log_arriendo", "rel_dom_educ_max", "rel_dom_horas_pc",
    "rel_dom_ocup_pc", "rel_dom_educ_media"
  )
)
# Ingreso predicho mejorado (3.4; dependen del fold): mismas variables que V4
# (más ing_pred_por_ocup) con un sufijo. _t: hiperparámetros afinados; _h:
# además, variables del hogar; _c: ingreso laboral y no laboral por separado;
# _e: como _t, con las preguntas que no se usaban (x_pers_extra); _f: con
# esas preguntas, afinado.
sufijos_v4 <- c("t", "h", "c", "e", "f", "w", "x")
for (s in sufijos_v4) {
  cols[[paste0("V4", s)]] <- paste0(
    c(cols$V4, "ing_pred_por_ocup"),
    "_", s
  )
}
# PI: probabilidad de ser indigente por persona, resumida por hogar (como P).
cols$PI <- c("p_ind_media", "p_ind_max", "p_ind_min", "p_ind_jefe")
# Pt, Pe: clasificador por persona afinado (Pe, con x_pers_extra). C080, C125:
# probabilidad de que el ingreso del hogar sea menor que 0.8 y 1.25 veces la
# línea (3.4).
resumen_p <- c("media", "max", "min", "jefe")
cols$Pt <- paste0("pt_", resumen_p)
cols$Pe <- paste0("pe_", resumen_p)
cols$C080 <- paste0("pc080_", resumen_p)
cols$C125 <- paste0("pc125_", resumen_p)
# A: arriendo imputado (01) per cápita sobre la línea y, para cada ingreso
# predicho, log((ingreso predicho + arriendo imputado) / personas / línea): la
# fórmula del DANE (Ingpcug) con el ingreso de las personas predicho. Las
# ing_arr_pc_lp* dependen del fold (se calculan en armar()). V4w: ingreso
# predicho con objetivo Tweedie (3.4). Pa: clasificador por persona como Pe,
# más el arriendo imputado del hogar.
cols$A <- c("arr_imp_pc_lp", paste0("ing_arr_pc_lp", c("", "_t", "_e", "_f")))
cols$Pa <- paste0("pa_", resumen_p)

# Índices (I): resumen de varias variables débiles del mismo tema, que un
# árbol tendría que sumar con muchos cortes. Signo: +1 si la variable sube con
# la pobreza. I es el promedio de las variables estandarizadas con la media y
# la desviación de train (no usa pobre); IS (3.4) es el log odds de un logit de
# pobre sobre las mismas variables (usa pobre: CV anidada).
grupos_indice <- list(
  informalidad = c(
    prop_formal_ocup = -1, formal_pc = -1, prop_contributivo = -1,
    prop_subsidiado = 1, beneficios_pc = -1, horas_informal_pc = 1,
    prop_cuenta_propia = 1, prop_empresa_sola = 1, prop_sin_pago = 1,
    prop_asalariado = -1, prop_empresa_grande = -1, no_afiliado_pc = 1,
    jefe_formal = -1
  ),
  capital_humano = c(
    educ_max = -1, educ_media_adultos = -1, educ_ocup_media = -1,
    prop_superior = -1, sin_educ_pc = 1, jefe_educ = -1
  ),
  vivienda = c(
    pers_por_dormitorio = 1, cuartos_pc = -1, log_arriendo_pc_lp = -1
  ),
  no_laboral = c(
    any_pension = -1, any_arriendos = -1, any_intereses = -1,
    any_cesantias = -1,
    any_exterior = -1, any_ayudas = 1, any_hogares_pais = 1
  ),
  dependencia = c(
    dependencia = 1, prop_menores = 1, ocup_pc = -1, ocup_por_adulto = -1,
    desocup_pc = 1, inactivo_pc = 1
  )
)
cols$I <- paste0("idx_", names(grupos_indice))
cols$IS <- paste0("idx_", names(grupos_indice), "_sup")
categoricas <- c(
  "dominio", "depto", "tenencia", "jefe_posicion", "jefe_regimen",
  "jefe_tamano", "jefe_oficio", "ocup_posicion_educ"
)
x_base <- setdiff(names(test), "id") # las 94 variables de 02

hog <- bind_rows(
  train |> select(id, all_of(x_base)) |> mutate(base = "train"),
  test |> select(id, all_of(x_base)) |> mutate(base = "test")
) |>
  left_join(bind_rows(v1(pers_tr), v1(pers_te)), by = "id") |>
  left_join(bind_rows(v2(pers_tr), v2(pers_te)), by = "id") |>
  left_join(bind_rows(nuestras(pers_tr), nuestras(pers_te)), by = "id") |>
  left_join(
    bind_rows(
      distinct(pers_tr, id, arriendo_imputado),
      distinct(pers_te, id, arriendo_imputado)
    ),
    by = "id", relationship = "one-to-one"
  ) |>
  mutate(arr_imp_pc_lp = arriendo_imputado / n_ug / exp(log_lp)) |>
  mutate(across(all_of(cols$V2), ~ coalesce(.x, 0)),
    # N1
    educ_adultos_x_ocup_pc = educ_max_adultos * ocup_pc,
    ocup_pc_x_log_lp = ocup_pc * log_lp,
    arriendo_x_log_lp = log_arriendo_pc_lp * log_lp,
    horas_pc_x_formal = horas_pc * prop_formal_ocup,
    educ_ocup_x_formal_pc = educ_ocup_suma_pc * formal_pc,
    jefe_educ_x_formal = jefe_educ * jefe_formal,
    # N2 (sin ing_pred_por_ocup)
    dep_por_ocup = (n_menor18 + n_mayor65) / pmax(n_ocup, 1),
    beneficios_por_ocup = beneficios_pc * n_ug / pmax(n_ocup, 1),
    horas_por_ocup = horas_pc * n_ug / pmax(n_ocup, 1),
    brecha_educ_jefe = educ_max_adultos - jefe_educ,
    # Proporciones que solo entran en los índices (I e IS).
    across(
      c(
        n_cuenta_propia, n_empresa_sola, n_sin_pago, n_asalariado,
        n_empresa_grande
      ),
      ~ .x / pmax(n_ocup, 1),
      .names = "prop_{sub('^n_', '', .col)}"
    ),
    no_afiliado_pc = n_no_afiliado / n_ug, sin_educ_pc = n_sin_educ / n_ug,
    inactivo_pc = n_inactivo_adulto / n_ug
  ) |>
  # N3
  mutate(
    .by = dominio,
    rel_dom_log_arriendo = log_arriendo_pc_lp - median(
      log_arriendo_pc_lp,
      na.rm = TRUE
    ),
    rel_dom_educ_max = educ_max - median(educ_max, na.rm = TRUE),
    rel_dom_horas_pc = horas_pc - median(horas_pc, na.rm = TRUE),
    rel_dom_ocup_pc = ocup_pc - median(ocup_pc, na.rm = TRUE),
    rel_dom_educ_media = educ_media_adultos - median(
      educ_media_adultos,
      na.rm = TRUE
    )
  ) |>
  # Categóricas a códigos enteros con los mismos niveles en train y test.
  mutate(across(all_of(categoricas), ~ as.integer(factor(.x))))
# I: promedio de las variables del grupo estandarizadas (media y desviación de
# train), con su signo.
es_tr <- hog$base == "train"
for (g in names(grupos_indice)) {
  s <- grupos_indice[[g]]
  hog[[paste0("idx_", g)]] <- rowMeans(sapply(names(s), function(v) {
    x <- hog[[v]]
    s[[v]] * (x - mean(x[es_tr])) / sd(x[es_tr])
  }))
}
hog_tr <- hog |> filter(base == "train")
hog_te <- hog |> filter(base == "test")
stopifnot(identical(hog_tr$id, train$id), identical(hog_te$id, test$id))
rm(hog)
# Normalización: media y desviación de train de las numéricas que no son
# categóricas ni 0/1. Se aplica al final de armar() (sección 4), sobre los
# predictores del modelo por hogar: las variables que armar() calcula
# (ing_pred_por_ocup, ing_arr_pc_lp*, N4, T) usan los valores originales de
# n_ocup, n_ug, log_lp, arriendo_imputado, etc.
escala <- list()
if (normalizar) {
  numericas <- names(hog_tr)[map_lgl(hog_tr, is.numeric)] |>
    setdiff(c("id", categoricas)) |>
    keep(~ !all(hog_tr[[.x]] %in% c(0, 1, NA)))
  escala <- map(
    set_names(numericas),
    ~ c(m = mean(hog_tr[[.x]], na.rm = TRUE), d = sd(hog_tr[[.x]],
      na.rm = TRUE
    ))
  ) |>
    keep(~ is.finite(.x[["d"]]) && .x[["d"]] > 0)
  marca(
    "Base normalizada:", length(escala), "variables numéricas estandarizadas"
  )
}
# Niveles de cada categórica (train y test juntos), para one-hot y nativo.
niveles_cat <- map(
  set_names(categoricas), ~ sort(unique(c(hog_tr[[.x]], hog_te[[.x]])))
)

## 3.3 Modelos auxiliares por persona (dependen de los folds) ----------------

# Usan pobre (V3, P) o Ingtot (V4), así que se estiman solo con hogares de
# entrenamiento, con CV anidada como en 13:
#   - validación del fold f: modelos con los otros 4 folds;
#   - entrenamiento del fold f (para cada g != f): modelos con los 3 folds que
#     no son ni f ni g, que predicen g. Así pobre de f nunca entra en las
#     variables del modelo que se evalúa en f;
#   - test: modelos con todo train.
# V3: tasa de pobreza del oficio (suavizada). V4: regresión de log1p(Ingtot)
# por persona. P: probabilidad de ser pobre por persona (clasificador XGBoost
# por persona), resumida por hogar.
x_pers <- c(
  "anos", "mujer", "parentesco_jefe_hogar", "max_educ", "anos_educ_imp",
  "ocupado", "desocupado", "inactivo", "actividad_sem_pasada",
  "posicion_ocupacional", "tamano_empresa", "oficio",
  "horas_trabajo_semana", "horas_segundo_trabajo", "segundo_trabajo",
  "cotiza_pension", "regimen_salud", "afiliado_salud", "log_meses_empresa",
  grep("^(recibio_|prima_|subsidio_|dinero_)", names(pers_te), value = TRUE),
  "ayudas_instituciones", "dominio", "depto", "cabecera"
)
cat_pers <- c(
  "parentesco_jefe_hogar", "max_educ", "actividad_sem_pasada",
  "posicion_ocupacional", "oficio", "cotiza_pension", "regimen_salud",
  "dominio", "depto"
)
# Variables de hogar para el clasificador por persona (P), además de x_pers.
x_pers_hog <- c(
  "arriendo_pc_sobre_lp", "prop_ocupados_hogar", "tenencia_vivienda",
  "personas_hogar", "educ_max_adultos", "linea_pobreza"
)
# Preguntas por persona de test que no usaba ningún modelo (ni x_pers ni la
# base de 02): pagos en especie, segundo trabajo, búsqueda de más horas o de
# otro trabajo, último trabajo e ingresos de los desocupados. Solo entran en
# las versiones "extra" de 3.4 (V4e, Pe).
x_pers_extra <- c(
  "pago_especie_alimentos", "pago_especie_vivienda", "transporte_empresa",
  "pago_especie_otros", "viaticos_permanentes", "bonificaciones_anuales",
  "posicion_segundo_trabajo", "quiere_mas_horas", "diligencias_mas_horas",
  "disponible_mas_horas", "diligencias_cambiar_trabajo",
  "disponible_nuevo_trabajo",
  "primera_vez_busca_trabajo", "posicion_ultimo_trabajo",
  "ingreso_trabajo_desocupado", "ingreso_trabajo_desocupado_2"
)
cat_extra <- c("posicion_segundo_trabajo", "posicion_ultimo_trabajo")
# Arriendo imputado del hogar (01) per cápita sobre la línea: solo en Pa.
x_pers_imp <- "arr_imp_pc_lp"
pers <- bind_rows(
  pers_tr |>
    left_join(tibble(id = train$id, fold = fold), by = "id") |>
    mutate(base = "train"),
  pers_te |> mutate(
    base = "test", fold = NA_integer_, pobre = NA_integer_,
    Indigente = NA_integer_,
    Ingtot = NA_real_, ing_lab = NA_real_, ing_nolab = NA_real_
  )
)
pers <- mutate(
  pers,
  arr_imp_pc_lp = arriendo_imputado / personas_unidad_gasto / linea_pobreza
)
mat_pers <- pers |>
  select(all_of(c(x_pers, x_pers_hog, x_pers_extra, x_pers_imp))) |>
  mutate(across(
    all_of(c(cat_pers, "tenencia_vivienda", cat_extra)),
    ~ as.integer(factor(.x))
  ))
# Con códigos, matriz numérica; con onehot y nativo, data.frame con factores
# (categóricas nativas de xgboost).
mat_pers <- if (completo) {
  data.matrix(mat_pers)
} else {
  as.data.frame(mutate(
    mat_pers,
    across(all_of(c(cat_pers, "tenencia_vivienda", cat_extra)), factor)
  ))
}
z_pers <- z[match(pers$id, train$id)] # z del hogar de cada persona (NA en test)
pers <- pers |>
  select(
    id, base, fold, en_ug, jefe, ocup, pet, oficio, pobre, Indigente, Ingtot,
    ing_lab, ing_nolab,
    lp = linea_pobreza, n_ug = personas_unidad_gasto
  )
rm(pers_tr, pers_te)
invisible(gc())
col_v4 <- colnames(mat_pers) %in% x_pers # el ingreso predicho usa solo x_pers
col_p <- colnames(mat_pers) %in% c(x_pers, x_pers_hog) # P, P2 y PI
col_extra <- colnames(mat_pers) %in% x_pers_extra
col_imp <- colnames(mat_pers) %in% x_pers_imp

# Un modelo xgboost con parada temprana sobre el 10% de los hogares de
# entrenamiento (como en 13).
xgb_parada <- function(mat, etiqueta, ids, params, semilla) {
  set.seed(semilla)
  val <- ids %in% sample(unique(ids), round(0.1 * n_distinct(ids)))
  dtr <- xgb.DMatrix(
    mat[!val, , drop = FALSE],
    label = etiqueta[!val], missing = NA
  )
  dva <- xgb.DMatrix(
    mat[val, , drop = FALSE],
    label = etiqueta[val], missing = NA
  )
  xgb.train(
    params = params, data = dtr, nrounds = nrounds_max, evals = list(val = dva),
    early_stopping_rounds = parada, verbose = 0
  )
}
params_v4 <- list(
  objective = "reg:squarederror", eval_metric = "rmse", eta = 0.1,
  max_depth = 8, min_child_weight = 20, subsample = 0.8,
  colsample_bytree = 0.8, nthread = 1, seed = 123
)
params_p <- modifyList(params_base, list(eta = 0.1))

# Resúmenes por hogar del ingreso predicho por persona (`ing`, V4) y de una
# probabilidad por persona en log odds (`lo`: P, P2, PI).
resumir_ing <- function(pp) {
  pp |> summarize(
    .by = id, ing_pred_total = sum(ing[en_ug]),
    log_ing_pred_pc_lp = log((ing_pred_total + 1) / first(n_ug) / first(lp)),
    max_ing_pred_lp = max(c(ing[en_ug], 0)) / first(lp),
    prop_ing_jefe = if (ing_pred_total > 0) {
      sum(ing[jefe & en_ug]) /
        ing_pred_total
    } else {
      0
    }
  )
}
resumir_prob <- function(pp, prefijo) {
  pp |>
    summarize(
      .by = id, media = mean(lo), max = max(lo), min = min(lo),
      jefe = if (any(jefe)) lo[jefe][1] else mean(lo)
    ) |>
    rename_with(~ paste0(prefijo, "_", .x), -id)
}

# Personas con las que se entrena una tarea y personas que predice.
filas_ent <-
  function(tarea) which(pers$base == "train" & pers$fold %in% tarea$entren)
filas_pred <- function(tarea) {
  if (is.null(tarea$pred)) {
    which(pers$base == "test")
  } else {
    which(pers$base == "train" & pers$fold %in% tarea$pred)
  }
}
# V3: tasa de pobreza de cada oficio entre los ocupados de entrenamiento,
# suavizada hacia la media con peso 20 (ajustar_v3), y su media, mínimo y
# máximo entre los ocupados de cada hogar (predecir_v3). `oficio` permite
# usar otros valores (sección 15).
ajustar_v3 <- function(tarea) {
  i_ent <- filas_ent(tarea)
  ocup_ent <- i_ent[pers$ocup[i_ent]]
  media <- mean(pers$pobre[ocup_ent])
  tasas <- tibble(
    oficio = pers$oficio[ocup_ent], pobre = pers$pobre[ocup_ent]
  ) |>
    summarize(.by = oficio, n = n(), tasa = mean(pobre)) |>
    mutate(te = (n * tasa + 20 * media) / (n + 20))
  list(tasas = tasas, media = media)
}
predecir_v3 <- function(v3, filas, oficio = pers$oficio[filas]) {
  pers[filas, ] |>
    mutate(te = coalesce(
      v3$tasas$te[match(oficio, v3$tasas$oficio)], v3$media
    )) |>
    filter(ocup) |>
    summarize(
      .by = id, te_oficio_media = mean(te), te_oficio_min = min(te),
      te_oficio_max = max(te)
    )
}
# Une las variables por fold de un hogar: ingreso predicho, probabilidades por
# persona y V3 (los hogares sin ocupados quedan con la media de V3).
unir_tablas <- function(tablas, te_hogar, media) {
  reduce(tablas, left_join, by = "id") |>
    left_join(te_hogar, by = "id") |>
    mutate(across(starts_with("te_oficio"), ~ coalesce(.x, media)))
}
tarea_v34p <- function(tarea) {
  # V3, V4 (ingreso predicho) y P (clasificador por persona): V4 y P con
  # tarea_v4() y tarea_pc() de 3.4 y sus configuraciones originales
  # (conf_v4_original y conf_p_original, con las mismas semillas).
  v3 <- ajustar_v3(tarea)
  v4 <- tarea_v4(tarea, conf_v4_original)
  p <- tarea_pc(tarea, conf_p_original)
  list(
    hogar = unir_tablas(
      list(v4$hogar, p$hogar), predecir_v3(v3, filas_pred(tarea)), v3$media
    ),
    r2 = v4$r2
  )
}

tareas <- list()
for (f in 1:k) {
  tareas[[paste0("val_", f)]] <- list(
    entren = setdiff(1:k, f), pred = f,
    semilla = 100 + f
  )
  for (g in setdiff(1:k, f)) {
    tareas[[paste0("ent_", f, "_", g)]] <- list(
      entren = setdiff(1:k, c(f, g)),
      pred = g, semilla = 1000 + 10 * f + g
    )
  }
}
tareas[["test"]] <- list(entren = 1:k, pred = NULL, semilla = 999)

# P2: clasificador por persona con sus variables y las del hogar (base de 02,
# V1, V2, nuestras, N1 y N3; sin las que dependen de los folds).
vars_hog_p2 <- c(x_base, cols$V1, cols$V2, cols$nuestras, cols$N1, cols$N3)
if (completo) {
  hog_todos <- bind_rows(hog_tr, hog_te)
  mat_pers2 <- cbind(
    mat_pers[, col_p],
    data.matrix(hog_todos[match(pers$id, hog_todos$id), vars_hog_p2])
  )
  rm(hog_todos)
}
tarea_p2 <- function(tarea) {
  i_ent <- filas_ent(tarea)
  i_pred <- filas_pred(tarea)
  m <- xgb_parada(
    mat_pers2[i_ent, ], pers$pobre[i_ent], pers$id[i_ent], params_p,
    tarea$semilla + 2
  )
  p <- predict(m, xgb.DMatrix(mat_pers2[i_pred, ], missing = NA),
    iterationrange = c(1, xgb.attr(m, "best_iteration") + 1)
  )
  hogar <- pers[i_pred, ] |>
    mutate(lo = qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))) |>
    resumir_prob("p2_pers")
  list(hogar = hogar, iter_p2 = xgb.attr(m, "best_iteration"))
}

# Z: regresión de z = log(ingreso / línea) a nivel hogar con las variables que
# no dependen de los folds (usar V3, V4 o P exigiría un tercer nivel de CV
# anidada). Predice los hogares de `tarea$pred` (o test) con los de
# `tarea$entren`.
vars_z <- c(x_base, cols$V1, cols$V2, cols$nuestras, cols$N1, cols$N3)
mat_hog_tr <- data.matrix(hog_tr[vars_z])
mat_hog_te <- data.matrix(hog_te[vars_z])
params_z <- modifyList(
  params_base, list(objective = "reg:squarederror", eval_metric = "rmse")
)
tarea_z <- function(tarea) {
  i_ent <- which(fold %in% tarea$entren)
  m <- xgb_parada(
    mat_hog_tr[i_ent, ], z[i_ent], train$id[i_ent], params_z, tarea$semilla + 3
  )
  if (is.null(tarea$pred)) {
    ids <- test$id
    mat <- mat_hog_te
  } else {
    i_pred <- which(fold %in% tarea$pred)
    ids <- train$id[i_pred]
    mat <- mat_hog_tr[i_pred, ]
  }
  pred <- predict(
    m, xgb.DMatrix(mat, missing = NA),
    iterationrange = c(1, xgb.attr(m, "best_iteration") + 1)
  )
  list(
    hogar = tibble(id = ids, z_pred = pred),
    iter_z = xgb.attr(m, "best_iteration")
  )
}

# Corre (o lee) las tareas de una partición con la función `tarea_fun`.
correr_tareas <- function(tarea_fun, archivo, etiqueta) {
  if (!file.exists(archivo)) {
    marca(etiqueta, ":", length(tareas), "tareas en", nucleos - 1, "procesos")
    res <- en_paralelo(
      tareas, tarea_fun,
      nucleos = nucleos - 1, que = paste("tareas", etiqueta)
    )
    saveRDS(res, archivo)
    marca(etiqueta, "listo")
  }
  readRDS(archivo)
}
## 3.4 Versiones afinadas de los modelos auxiliares ---------------------------

# Ingreso predicho por persona configurable (V4t, V4h, V4c). `conf`: params,
# hogar (TRUE: también las variables del hogar x_pers_hog), componentes (TRUE:
# un modelo para el ingreso laboral y otro para el no laboral, que se suman) y
# sufijo de las columnas; desfase (de la semilla; 10 si no se da); extra
# (TRUE: también x_pers_extra); tweedie (TRUE:
# objetivo Tweedie sobre el ingreso en pesos, que predice la media; con el
# log del ingreso la predicción queda más cerca de la mediana y la suma del
# hogar se subestima; `params` trae tweedie_variance_power).
# ajustar_v4() estima los modelos (uno, o dos con componentes) y predecir_v4()
# arma las variables del hogar para las personas `filas`, con sus predictores
# o con otros (X, sección 15).
columnas_v4 <- function(conf) {
  (if (conf$hogar) col_p else col_v4) |
    (isTRUE(conf$extra) & col_extra)
}
ajustar_v4 <- function(tarea, conf) {
  i_ent <- filas_ent(tarea)
  i_lab <- i_ent[pers$pet[i_ent]]
  objetivos <- if (conf$componentes) {
    list(
      pers$ing_lab, pers$ing_nolab
    )
  } else {
    list(pers$Ingtot)
  }
  imap(objetivos, function(obj, j) {
    etiqueta <- if (isTRUE(conf$tweedie)) {
      pmax(
        obj[i_lab], 0
      )
    } else {
      log1p(obj[i_lab])
    }
    xgb_parada(
      mat_pers[i_lab, columnas_v4(conf)], etiqueta, pers$id[i_lab], conf$params,
      tarea$semilla + (conf$desfase %||% 10) + 10 * (j - 1)
    )
  })
}
predecir_v4 <- function(modelos, conf, filas, mat = mat_pers[filas, ]) {
  ing <- Reduce(`+`, map(modelos, function(m) {
    pred <- predict(m, xgb.DMatrix(mat[, columnas_v4(conf)], missing = NA),
      iterationrange = c(1, xgb.attr(m, "best_iteration") + 1)
    )
    pmax(if (isTRUE(conf$tweedie)) pred else expm1(pred), 0)
  }))
  pp <- pers[filas, ] |> mutate(ing = if_else(pet, ing, 0))
  hogar <- resumir_ing(pp)
  names(hogar)[-1] <- paste0(names(hogar)[-1], conf$sufijo)
  list(hogar = hogar, pp = pp)
}
tarea_v4 <- function(tarea, conf) {
  res <- predecir_v4(ajustar_v4(tarea, conf), conf, filas_pred(tarea))
  r2 <- if (!is.null(tarea$pred)) {
    with(
      res$pp[res$pp$pet, ], tibble(real = log1p(Ingtot), pred = log1p(ing))
    )
  }
  list(hogar = res$hogar, r2 = r2)
}
# R² de un ingreso predicho en las tareas de validación: por persona
# (log1p Ingtot, como r2_v4) y por hogar (z = log(ingreso / línea) sobre
# log_ing_pred_pc_lp, R² de una regresión lineal).
r2_validacion <- function(res, sufijo = "") {
  val <- res[paste0("val_", 1:k)]
  h <- bind_rows(map(val, "hogar"))
  x <- h[[paste0("log_ing_pred_pc_lp", sufijo)]][match(train$id, h$id)]
  p <- bind_rows(map(val, "r2"))
  c(
    r2_persona = with(p, 1 - sum((real - pred)^2) / sum((real - mean(real))^2)),
    r2_hogar = cor(x, z)^2
  )
}

# Afinamiento de un modelo por persona (ingreso predicho o clasificador): para
# cada fila de `grid` estima `tarea_fun` con la configuración conf_de(fila)
# solo en las 5 tareas de validación (un modelo por fold) y calcula
# metrica(res), un vector con nombre (R² o AUC por hogar). Guarda la tabla en
# dir_out/<archivo> o, si ya existe, la lee.
afinar <- function(archivo, grid, tarea_fun, metrica, que, conf_de) {
  archivo <- file.path(dir_out, archivo)
  conf_de <- as_mapper(conf_de)
  metrica <- as_mapper(metrica)
  if (!file.exists(archivo)) {
    tabla <- pmap_dfr(grid, function(...) {
      fila <- list(...)
      res <- en_paralelo(tareas[paste0("val_", 1:k)], tarea_fun,
        conf = conf_de(fila),
        nucleos = nucleos - 1, que = paste("el afinamiento:", que)
      )
      m <- metrica(res)
      marca(
        que, ":", paste(names(fila), fila, sep = " = ", collapse = "  "), " ",
        paste(names(m), sprintf("%.4f", m), sep = " = ", collapse = "  ")
      )
      as_tibble(c(fila, as.list(m)))
    })
    saveRDS(tabla, archivo)
  }
  readRDS(archivo)
}

# Ingreso predicho: por el R² por hogar, que es lo que usa el clasificador.
conf_v4 <- function(fila, componentes, sufijo) {
  list(
    hogar = fila$hogar, componentes = componentes, sufijo = sufijo,
    params = modifyList(params_v4, list(
      eta = 0.05, max_depth = fila$max_depth,
      min_child_weight = fila$min_child_weight
    ))
  )
}
grid_v4 <- expand_grid(
  hogar = c(FALSE, TRUE), max_depth = c(6, 8, 10), min_child_weight = c(20, 100)
)
afinado_v4 <- afinar(
  "afinado_v4.rds", grid_v4, tarea_v4, r2_validacion, "Ingreso predicho",
  ~ conf_v4(.x, componentes = FALSE, sufijo = "")
)
mejor_v4 <- function(d) slice_max(d, r2_hogar, n = 1, with_ties = FALSE)
confs_v4 <- list(
  t = conf_v4(mejor_v4(filter(afinado_v4, !hogar)), FALSE, "_t"),
  h = conf_v4(mejor_v4(filter(afinado_v4, hogar)), FALSE, "_h"),
  c = conf_v4(mejor_v4(afinado_v4), TRUE, "_c")
)
# V4e: como V4t, con las preguntas que no se usaban (x_pers_extra).
confs_v4$e <- modifyList(confs_v4$t, list(extra = TRUE, sufijo = "_e"))
# V4f: el ingreso predicho con x_pers_extra, afinado (V4e usa la configuración
# de V4t, que se afinó sin esas preguntas): variables del hogar, componentes
# y profundidad, por el R² por hogar en las 5 tareas de validación.
conf_v4f <- function(fila, sufijo) {
  list(
    hogar = fila$hogar, componentes = fila$componentes, extra = TRUE,
    sufijo = sufijo,
    params = modifyList(
      params_v4,
      list(eta = 0.05, max_depth = fila$max_depth, min_child_weight = 20)
    )
  )
}
grid_v4f <- expand_grid(
  hogar = c(FALSE, TRUE), componentes = c(FALSE, TRUE), max_depth = c(8, 10, 12)
)
afinado_v4f <- afinar(
  "afinado_v4f.rds", grid_v4f, tarea_v4, r2_validacion,
  "Ingreso predicho con preguntas no usadas",
  ~ conf_v4f(.x, sufijo = "")
)
confs_v4$f <- conf_v4f(mejor_v4(afinado_v4f), sufijo = "_f")
# V4w: como V4f, con objetivo Tweedie (ver tarea_v4); se escoge la potencia
# de la varianza (entre Poisson, 1, y gamma, 2) por el R² por hogar.
conf_v4w <- function(
  potencia, componentes = confs_v4$f$componentes, sufijo = "_w"
) {
  modifyList(confs_v4$f, list(
    tweedie = TRUE, sufijo = sufijo, componentes = componentes,
    params = modifyList(confs_v4$f$params, list(
      objective = "reg:tweedie", tweedie_variance_power = potencia,
      eval_metric = paste0("tweedie-nloglik@", potencia)
    ))
  ))
}
afinado_v4w <- afinar(
  "afinado_v4w.rds", tibble(potencia = c(1.2, 1.5, 1.8)), tarea_v4,
  r2_validacion,
  "Ingreso predicho Tweedie", ~ conf_v4w(.x$potencia, sufijo = "")
)
confs_v4$w <- conf_v4w(slice_max(
  afinado_v4w, r2_hogar,
  n = 1, with_ties = FALSE
)$potencia)
# V4x: la mejor potencia quedó en el borde de la grilla (1.2), así que se
# prueban potencias menores y mayores cerca de ella, por componentes (como
# V4w) y sobre el ingreso total. Si gana la misma configuración de V4w, no
# hay V4x.
grid_v4x <- expand_grid(
  potencia = c(1.05, 1.1, 1.2, 1.3), componentes = c(TRUE, FALSE)
) |>
  # ya está en afinado_v4w
  filter(!(potencia == 1.2 & componentes == confs_v4$f$componentes))
afinado_v4x <- afinar(
  "afinado_v4x.rds", grid_v4x, tarea_v4, r2_validacion,
  "Ingreso predicho Tweedie",
  ~ conf_v4w(.x$potencia, .x$componentes, sufijo = "")
)
afinado_v4x <- bind_rows(
  mutate(afinado_v4w, componentes = confs_v4$f$componentes), afinado_v4x
)
fila_v4x <- slice_max(afinado_v4x, r2_hogar, n = 1, with_ties = FALSE)
misma_v4w <- fila_v4x$potencia == confs_v4$w$params$tweedie_variance_power &&
  fila_v4x$componentes == confs_v4$w$componentes
if (!misma_v4w) {
  confs_v4$x <- conf_v4w(fila_v4x$potencia, fila_v4x$componentes, sufijo = "_x")
}

# Clasificador por persona configurable (Pt, Pe y cortes C). `conf`: params,
# extra (TRUE: también x_pers_extra), imputado (TRUE: también x_pers_imp),
# etiqueta (otra variable a predecir, p. ej. Indigente), desfase (de la
# semilla; 5 si no se da), corte (objetivo: ingreso del hogar menor
# que `corte` veces la línea; 1 = pobre) y prefijo de las columnas.
# Como V4: ajustar_pc() estima el modelo y predecir_pc() arma las variables
# del hogar.
columnas_pc <- function(conf) {
  col_p | (isTRUE(conf$extra) & col_extra) |
    (isTRUE(conf$imputado) & col_imp)
}
ajustar_pc <- function(tarea, conf) {
  i_ent <- filas_ent(tarea)
  etiqueta <- if (!is.null(conf$etiqueta)) {
    as.integer(pers[[conf$etiqueta]])
  } else if (conf$corte == 1) {
    pers$pobre
  } else {
    as.integer(z_pers < log(conf$corte))
  }
  xgb_parada(
    mat_pers[i_ent, columnas_pc(conf)], etiqueta[i_ent], pers$id[i_ent],
    conf$params, tarea$semilla + (conf$desfase %||% 5)
  )
}
predecir_pc <- function(m, conf, filas, mat = mat_pers[filas, ]) {
  p <- predict(
    m, xgb.DMatrix(mat[, columnas_pc(conf)], missing = NA),
    iterationrange = c(1, xgb.attr(m, "best_iteration") + 1)
  )
  pers[filas, ] |>
    mutate(lo = qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))) |>
    resumir_prob(conf$prefijo)
}
tarea_pc <- function(tarea, conf) {
  list(hogar = predecir_pc(
    ajustar_pc(tarea, conf), conf, filas_pred(tarea)
  ))
}
# AUC por hogar de una probabilidad por persona (promedio de log odds) en las
# tareas de validación.
auc_validacion <- function(res, prefijo) {
  h <- bind_rows(map(res[paste0("val_", 1:k)], "hogar"))
  calcular_auc(h[[paste0(prefijo, "_media")]][match(train$id, h$id)], y)
}

# Afinamiento del clasificador por persona (P usa params_p: eta 0.1, depth 6,
# mcw 50, sin afinar), con y sin x_pers_extra, por el AUC por hogar en las 5
# tareas de validación.
conf_p <- function(fila, corte, prefijo) {
  list(
    extra = fila$extra, corte = corte, prefijo = prefijo,
    params = modifyList(params_p, list(
      eta = 0.05, max_depth = fila$max_depth,
      min_child_weight = fila$min_child_weight
    ))
  )
}
grid_p <- expand_grid(
  extra = c(FALSE, TRUE), max_depth = c(6, 8, 10), min_child_weight = c(20, 50)
)
afinado_p <- afinar(
  "afinado_p.rds", grid_p, tarea_pc, ~ c(auc_hogar = auc_validacion(.x, "p")),
  "Clasificador por persona", ~ conf_p(.x, corte = 1, prefijo = "p")
)
mejor_p <- function(d) slice_max(d, auc_hogar, n = 1, with_ties = FALSE)
# Pt (afinado), Pe (afinado, con x_pers_extra) y cortes de 0.8 y 1.25 veces
# la línea (C080, C125) con la mejor configuración.
confs_p <- list(
  pt   = conf_p(mejor_p(filter(afinado_p, !extra)), 1, "pt"),
  pe   = conf_p(mejor_p(filter(afinado_p, extra)), 1, "pe"),
  c080 = conf_p(mejor_p(afinado_p), 0.8, "pc080"),
  c125 = conf_p(mejor_p(afinado_p), 1.25, "pc125")
)
# Pa: como Pe, más el arriendo imputado del hogar.
confs_p$pa <- modifyList(confs_p$pe, list(imputado = TRUE, prefijo = "pa"))

# Configuraciones originales de V4 y P (tarea_v34p, 3.3) y de PI: clasificador
# por persona de Indigente (solo train; como P, con sus mismas variables e
# hiperparámetros). `desfase`: lo que se suma a la semilla de la tarea.
conf_v4_original <- list(
  hogar = FALSE, componentes = FALSE, sufijo = "", params = params_v4,
  desfase = 0
)
conf_p_original <- list(
  extra = FALSE, corte = 1, prefijo = "p_pers", params = params_p, desfase = 1
)
conf_ind <- list(
  extra = FALSE, corte = 1, etiqueta = "Indigente", prefijo = "p_ind",
  params = params_p, desfase = 4
)

# IS: índices supervisados, logit de pobre sobre las variables de cada grupo
# (a nivel hogar, con los hogares de `tarea$entren`).
# ajustar_isup() estima un logit por grupo y predecir_isup() predice los
# hogares d_pred (también con variables permutadas, sección 15).
ajustar_isup <- function(tarea) {
  i_ent <- which(fold %in% tarea$entren)
  map(grupos_indice, ~ glm(
    reformulate(names(.x), "pobre"),
    family = binomial,
    data = mutate(hog_tr[i_ent, ], pobre = y[i_ent])
  ))
}
predecir_isup <- function(modelos, d_pred) {
  hogar <- tibble(id = d_pred$id)
  for (g in names(modelos)) {
    hogar[[paste0("idx_", g, "_sup")]] <- unname(predict(
      modelos[[g]],
      d_pred
    ))
  }
  hogar
}
tarea_isup <- function(tarea) {
  list(hogar = predecir_isup(
    ajustar_isup(tarea),
    if (is.null(tarea$pred)) hog_te else hog_tr[fold %in% tarea$pred, ]
  ))
}

# Usa la partición `semilla`: fija fold y calcula (o lee) V3/V4/P, P2 y Z
# (solo con códigos) y las variables de 3.4 (`extra`; con códigos, solo con la
# partición principal).
cambiar_particion <- function(semilla) {
  sufijo <- if (semilla == 123) "" else sprintf("_s%d", semilla)
  fold <<- particion(semilla)
  pers$fold <<- if_else(
    pers$base == "train", fold[match(pers$id, train$id)], NA_integer_
  )
  v34p <<- correr_tareas(
    tarea_v34p, file.path(dir_out, paste0("v34p", sufijo, ".rds")),
    paste("V3/V4/P", sufijo)
  )
  # P2 y Z solo con códigos (pruebas ya descartadas).
  v34p2 <<- if (completo) {
    correr_tareas(
      tarea_p2, file.path(dir_out, paste0("v34p2", sufijo, ".rds")),
      paste("P2", sufijo)
    )
  }
  vz <<- if (completo) {
    correr_tareas(
      tarea_z, file.path(dir_out, paste0("vz", sufijo, ".rds")),
      paste("Z", sufijo)
    )
  }
  # Las de 3.4 usan las configuraciones afinadas con la partición principal.
  extra <<- if (completo && semilla != 123) {
    list()
  } else {
    c(
      imap(confs_v4, ~ correr_tareas(
        function(t) tarea_v4(t, .x),
        file.path(dir_out, paste0("v4", .y, sufijo, ".rds")),
        paste0("Ingreso predicho V4", .y, sufijo)
      )),
      imap(confs_p, ~ correr_tareas(
        function(t) tarea_pc(t, .x),
        file.path(dir_out, paste0("persona_", .y, sufijo, ".rds")),
        paste("Clasificador por persona", .y, sufijo)
      )),
      list(
        ind = correr_tareas(
          function(t) tarea_pc(t, conf_ind),
          file.path(dir_out, paste0("indigencia", sufijo, ".rds")),
          paste("Indigencia PI", sufijo)
        ),
        isup = correr_tareas(
          tarea_isup, file.path(dir_out, paste0("indices_sup", sufijo, ".rds")),
          paste("Índices supervisados IS", sufijo)
        )
      )
    )
  }
  invisible(NULL)
}
cambiar_particion(123)
# R² fuera de muestra del ingreso predicho por persona (tareas de validación).
r2_v4 <- with(
  bind_rows(map(v34p[paste0("val_", 1:k)], "r2")),
  1 - sum((real - pred)^2) / sum((real - mean(real))^2)
)
marca(
  "R² fuera de muestra del ingreso por persona (log1p Ingtot):", round(r2_v4, 4)
)
r2_ingreso <- bind_rows(
  tibble(
    version = "V4", r2_persona = r2_v4,
    r2_hogar = r2_validacion(v34p)[["r2_hogar"]]
  ),
  imap_dfr(
    confs_v4,
    ~ c(
      version = paste0("V4", .y), as.list(r2_validacion(extra[[.y]], .x$sufijo))
    ) |> as_tibble()
  )
)
saveRDS(r2_ingreso, file.path(dir_out, "r2_ingreso.rds"))
marca("R² del ingreso predicho (validación):")
print(as.data.frame(r2_ingreso), digits = 4)
# AUC por hogar del clasificador por persona: P original, Pt, Pe y Pa.
auc_persona <- tibble(
  version = c("P", "Pt", "Pe", "Pa"),
  auc_hogar = c(
    auc_validacion(v34p, "p_pers"), auc_validacion(extra$pt, "pt"),
    auc_validacion(extra$pe, "pe"), auc_validacion(extra$pa, "pa")
  )
)
saveRDS(auc_persona, file.path(dir_out, "auc_persona.rds"))
marca("AUC por hogar del clasificador por persona (validación):")
print(as.data.frame(auc_persona), digits = 4)

# 4. Funciones de validación cruzada, combinación y predicción ----------------

# Matriz de predictores para unos hogares, con las variables por fold (V3, V4
# y P) de `tabla` y ing_pred_por_ocup.
armar <- function(base_hog, tabla, vars) {
  por_fold <- unlist(cols[c(
    "V3", "V4", "P", "P2", "Z", paste0("V4", sufijos_v4), "PI", "IS", "Pt",
    "Pe", "Pa", "C080", "C125"
  )])
  if (any(vars %in% c(
    por_fold, "ing_pred_por_ocup", cols$N4, cols$T, cols$A, "ing_arr_pc_lp_w"
  ))) {
    base_hog <- base_hog |>
      select(-any_of(por_fold)) |>
      left_join(tabla, by = "id") |>
      mutate(
        # Ingreso predicho por ocupado, relativo a la línea de pobreza.
        ing_pred_por_ocup = ing_pred_total / pmax(n_ocup, 1) / exp(log_lp),
        # N4
        ing_x_pmin = log_ing_pred_pc_lp * p_pers_min,
        ing_menos_arriendo = log_ing_pred_pc_lp - log_arriendo_pc_lp,
        ing_x_ocup_pc = log_ing_pred_pc_lp * ocup_pc,
        ing_x_educ_max = log_ing_pred_pc_lp * educ_max,
        ing_x_dependencia = log_ing_pred_pc_lp * dependencia,
        p_rango = p_pers_max - p_pers_min,
        p_jefe_menos_media = p_pers_jefe - p_pers_media,
        pmin_x_ocup_pc = p_pers_min * ocup_pc,
        arriendo_x_cuartos = log_arriendo_pc_lp * cuartos_pc,
        horas_x_educ_ocup = horas_pc * educ_ocup_media,
        # T
        log_ing_pred_total = log1p(ing_pred_total),
        log_horas_pc = log1p(horas_pc),
        log_cuartos_pc = log(cuartos_pc),
        log_educ_ocup_suma_pc = log1p(educ_ocup_suma_pc),
        log_beneficios_pc = log1p(beneficios_pc),
        ing_pred_pc_lp_2 = log_ing_pred_pc_lp^2,
        p_pers_min_2 = p_pers_min^2, arriendo_2 = log_arriendo_pc_lp^2,
        educ_max_2 = educ_max^2, dependencia_2 = dependencia^2
      )
    # ing_pred_por_ocup de los ingresos predichos de 3.4.
    for (s in intersect(
      paste0("_", sufijos_v4), sub("ing_pred_total", "", names(base_hog))
    )) {
      base_hog[[paste0(
        "ing_pred_por_ocup", s
      )]] <- base_hog[[paste0("ing_pred_total", s)]] / pmax(
        base_hog$n_ocup,
        1
      ) / exp(base_hog$log_lp)
    }
    # Ingreso predicho más arriendo imputado (A), como Ingpcug.
    for (s in intersect(
      c("", paste0("_", sufijos_v4)), sub("ing_pred_total", "", names(base_hog))
    )) {
      base_hog[[paste0("ing_arr_pc_lp", s)]] <- log((base_hog[[paste0(
        "ing_pred_total", s
      )]] + base_hog$arriendo_imputado + 1) /
        base_hog$n_ug / exp(base_hog$log_lp))
    }
  }
  for (v in intersect(
    vars, names(escala)
  )) {
    base_hog[[v]] <- (base_hog[[v]] - escala[[v]][["m"]]) / escala[[v]][["d"]]
  }
  codificar(base_hog[vars])
}
# Predictores en la codificación elegida (ver sección 1). En one-hot, las
# dummies se llaman <variable>__<código> (NA en todas si la categórica es NA).
codificar <- function(df) {
  cats <- intersect(names(df), categoricas)
  if (codificacion == "codigos" || !length(cats)) {
    return(data.matrix(df))
  }
  if (codificacion == "nativo") {
    for (v in cats) df[[v]] <- factor(df[[v]], levels = niveles_cat[[v]])
    return(as.data.frame(df))
  }
  dummies <- map(cats, function(v) {
    m <- sapply(niveles_cat[[v]], function(l) as.integer(df[[v]] == l))
    colnames(m) <- paste0(v, "__", niveles_cat[[v]])
    m
  })
  cbind(data.matrix(df[setdiff(names(df), cats)]), do.call(cbind, dummies))
}
# Variables por fold (V3, V4, P y P2) de una tarea.
tabla_tarea <- function(nombre) {
  t <- v34p[[nombre]]$hogar
  if (!is.null(v34p2)) t <- left_join(t, v34p2[[nombre]]$hogar, by = "id")
  if (!is.null(vz)) t <- left_join(t, vz[[nombre]]$hogar, by = "id")
  for (e in extra) t <- left_join(t, e[[nombre]]$hogar, by = "id")
  t
}
matrices <- function(f, vars) {
  i_ent <- which(fold != f)
  i_val <- which(fold == f)
  tabla_ent <- bind_rows(map(
    setdiff(1:k, f), ~ tabla_tarea(paste0("ent_", f, "_", .x))
  ))
  list(
    i_ent = i_ent, i_val = i_val,
    X_ent = armar(hog_tr[i_ent, ], tabla_ent, vars),
    X_val = armar(hog_tr[i_val, ], tabla_tarea(paste0("val_", f)), vars)
  )
}

# calcular_f1(), calcular_auc() y evaluar_umbral() están en funciones_comunes.R.
# F1 promedio de los folds por umbral (como los demás modelos por hogar) y el
# mejor.
f1_cv <- function(prob, umbrales = grid_umbral) {
  select(
    evaluar_umbral(prob, y, fold, umbrales), umbral, f1
  )
}
# Umbral honesto: el umbral de cada fold se escoge con los otros 4 (5.9).
umbral_honesto <- function(prob) {
  evaluar_umbral(
    prob, y, fold, grid_umbral
  )$f1_honesto
}

# Datos de entrenamiento de un modelo por hogar con los hogares `filas` de
# train (X: sus predictores): etiqueta pobre (o z si `regresion`), peso de la
# base aumentada por `pesos` y, con base "sin_ricos", sin los hogares ricos.
dmatrix_ent <- function(mat, filas, base, pesos = NULL, regresion = FALSE) {
  peso <- if (base == "aumentada") {
    ifelse(
      y == 1, sum(y == 0) / sum(y == 1), 1
    )
  } else {
    rep(1, length(y))
  }
  if (!is.null(pesos)) peso <- peso * pesos
  usar <- if (base == "sin_ricos") {
    train$hogar_rico == 0
  } else {
    rep(
      TRUE, length(y)
    )
  }
  ent <- usar[filas]
  xgb.DMatrix(mat[ent, , drop = FALSE],
    label = (if (regresion) z else y)[filas][ent],
    weight = peso[filas][ent], missing = NA
  )
}

# Validación cruzada de XGBoost a nivel hogar. Predicción fuera de muestra
# (OOF) de cada hogar con parada temprana en su fold. Los folds corren en
# paralelo. Guarda el resultado en dir_out/<nombre>.rds y, si ya existe, lo lee.
#   vars: variables. base: una de `bases`. params: hiperparámetros.
#   regresion = TRUE: predice z = log(ingreso / línea) en vez de pobre; la
#     probabilidad sale de un logit de pobre sobre la predicción (OOF).
#   pesos: peso de cada hogar de train en el entrenamiento (5.19), además del
#     de la base.
cv_hogar <- function(nombre, vars, base = "normal", params = params_base,
                     regresion = FALSE, nrounds = nrounds_max, pesos = NULL) {
  archivo <- file.path(dir_out, paste0(nombre, ".rds"))
  if (file.exists(archivo)) {
    return(invisible(readRDS(archivo)))
  }
  t0 <- Sys.time()
  if (regresion) {
    params <- modifyList(
      params, list(objective = "reg:squarederror", eval_metric = "rmse")
    )
  }
  res_folds <- en_paralelo(1:k, function(f) {
    m <- matrices(f, vars)
    dtr <- dmatrix_ent(m$X_ent, m$i_ent, base, pesos, regresion)
    dva <- xgb.DMatrix(
      m$X_val,
      label = (if (regresion) z else y)[m$i_val], missing = NA
    )
    b <- xgb.train(
      params = params, data = dtr, nrounds = nrounds, evals = list(val = dva),
      early_stopping_rounds = parada, verbose = 0
    )
    it <- xgb.attr(b, "best_iteration")
    list(
      i_val = m$i_val, pred = predict(b, dva, iterationrange = c(1, it + 1)),
      iter = it,
      ganancia = xgb.importance(model = b)
    )
  }, nucleos = min(nucleos, k), que = paste("un fold en", nombre))
  oof <- rep(NA_real_, length(y))
  for (r in res_folds) oof[r$i_val] <- r$pred
  prob <- oof
  metricas <- list()
  if (regresion) {
    metricas$r2 <- 1 - sum((z - oof)^2) / sum((z - mean(z))^2)
    metricas$rmse <- sqrt(mean((z - oof)^2))
    # Calibración: logit de pobre sobre la predicción (para combinar y para
    # usar los mismos umbrales que el clasificador).
    cal <- glm(y ~ oof, family = binomial)
    prob <- unname(predict(cal, type = "response"))
  }
  mejor <- f1_cv(prob)
  importancia <- bind_rows(map(res_folds, "ganancia")) |>
    # En one-hot, la ganancia de las dummies se suma en su variable.
    mutate(Feature = sub("__.*$", "", Feature)) |>
    summarize(ganancia = sum(Gain), .by = Feature) |>
    mutate(importancia = 100 * ganancia / sum(ganancia)) |>
    arrange(desc(importancia)) |>
    select(variable = Feature, importancia)
  out <- list(
    nombre = nombre, vars = vars, base = base, params = params,
    regresion = regresion,
    nrounds = nrounds, pesos = pesos,
    oof = oof, prob = prob, umbral = mejor$umbral, f1 = mejor$f1,
    auc = calcular_auc(prob, y), metricas = metricas,
    iters = map_int(res_folds, "iter"), importancia = importancia,
    cal = if (regresion) coef(cal) else NULL,
    minutos = as.numeric(difftime(Sys.time(), t0, units = "mins"))
  )
  saveRDS(out, archivo)
  marca(sprintf(
    "%-28s F1 = %.4f  umbral = %.2f  AUC = %.4f%s  iter = %d  (%.0f min)",
    nombre, out$f1, out$umbral, out$auc,
    if (regresion) {
      sprintf(
        "  R2 = %.4f  RMSE = %.4f", metricas$r2, metricas$rmse
      )
    } else {
      ""
    },
    round(mean(out$iters)), out$minutos
  ))
  invisible(out)
}

# Combinación de dos modelos: promedio de sus probabilidades fuera de muestra
# con peso w para el primero; w y umbral se escogen por F1.
combinar <- function(a, b, pesos = seq(0, 1, by = 0.1)) {
  tabla <- map_dfr(pesos, function(w) {
    m <- f1_cv(w * a$prob + (1 - w) * b$prob)
    tibble(w = w, umbral = m$umbral, f1 = m$f1)
  })
  mejor <- slice_max(tabla, f1, n = 1, with_ties = FALSE)
  marca(sprintf(
    "Combinación %s + %s: w = %.1f  umbral = %.2f  F1 = %.4f", a$nombre,
    b$nombre, mejor$w, mejor$umbral, mejor$f1
  ))
  list(
    a = a, b = b, tabla = tabla, w = mejor$w, umbral = mejor$umbral,
    f1 = mejor$f1
  )
}

# La mejor prueba (mayor F1) de una lista de resultados de cv_hogar().
mejor_de <- function(lista) lista[[which.max(map_dbl(lista, "f1"))]]

# Tabla de resumen de pruebas de cv_hogar(), con las de `refs` (modelos de
# referencia) primero. Con `nuevas`, también el % de la ganancia que se llevan
# esas variables (NA en las de referencia).
tabla_pruebas <- function(pruebas, refs = list(), nuevas = NULL) {
  todas <- c(refs, pruebas)
  t <- tibble(
    prueba = map_chr(todas, "nombre"),
    variables = map_int(todas, ~ length(.x$vars)),
    f1 = map_dbl(todas, "f1"),
    f1_honesto = map_dbl(todas, ~ umbral_honesto(.x$prob)),
    umbral = map_dbl(todas, "umbral")
  )
  if (!is.null(nuevas)) {
    t$ganancia_nuevas <- c(
      rep(NA, length(refs)),
      map_dbl(
        pruebas,
        ~ sum(.x$importancia$importancia[.x$importancia$variable %in% nuevas])
      )
    )
  }
  t
}

# Modelo final con todo train (nrounds = media de las mejores iteraciones de
# los folds × 1,1, como en 12) y probabilidad en test.
prob_test <- function(res) {
  tabla_te <- tabla_tarea("test")
  mat_ent <- armar(
    hog_tr, bind_rows(map(1:k, ~ tabla_tarea(paste0("val_", .x)))), res$vars
  )
  mat_te <- armar(hog_te, tabla_te, res$vars)
  dtr <- dmatrix_ent(mat_ent, seq_along(y), res$base, res$pesos, res$regresion)
  params <- modifyList(res$params, list(nthread = 1))
  b <- xgb.train(
    params = params, data = dtr, nrounds = round(mean(res$iters) * 1.1),
    verbose = 0
  )
  # Las variables por fold de train salen de las tareas de validación (fuera de
  # muestra) y las de test, de modelos con todo train (como en 13).
  p <- predict(b, xgb.DMatrix(mat_te, missing = NA))
  if (res$regresion) p <- plogis(res$cal[1] + res$cal[2] * p)
  p
}

# Submission (una fila por hogar) con formato Kaggle. `prefijo`: XGBH para
# XGBoost (y sus combinaciones), RFH para el random forest solo.
guardar_submission <- function(prob, umbral, nombre, prefijo = "XGBH") {
  sin_punto <- function(v) {
    gsub(
      ".", "", format(round(v, 2), scientific = FALSE),
      fixed = TRUE
    )
  }
  archivo <- paste0(
    prefijo, "_", if (!completo) paste0(codificacion, "_"), nombre,
    "_threshold_", sin_punto(umbral), ".csv"
  )
  dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
  if (!prueba) {
    write_csv(
      tibble(id = test$id, pobre = as.integer(prob > umbral)),
      here("02_outputs", "predictions", archivo)
    )
  }
  marca(
    "Submission:", archivo, "- % pobres en test:",
    round(100 * mean(prob > umbral), 1)
  )
}

# 5. Pruebas ------------------------------------------------------------------

# Conjuntos de variables. vars_h1 replica E6 de 13 (base + V1-V4).
vars_h0 <- x_base
vars_h1 <- c(x_base, cols$V1, cols$V2, cols$V3, cols$V4)
vars_h2 <- c(vars_h1, cols$nuestras)
vars_h3 <- c(vars_h2, cols$N1)
vars_h4 <- c(vars_h2, cols$N2)
vars_h5 <- c(vars_h2, cols$N3)
vars_h6 <- c(vars_h2, cols$N1, cols$N2, cols$N3)
vars_h7 <- c(vars_h6, cols$P)

# 5.1 Variables (base normal).
pruebas_h <- list(
  H0 = cv_hogar("H0_base01b", vars_h0),
  H1 = cv_hogar("H1_V1aV4", vars_h1),
  H2 = cv_hogar("H2_nuestras", vars_h2),
  H3 = cv_hogar("H3_N1_productos", vars_h3),
  H4 = cv_hogar("H4_N2_razones", vars_h4),
  H5 = cv_hogar("H5_N3_dominio", vars_h5),
  H6 = cv_hogar("H6_N1N2N3", vars_h6),
  H7 = cv_hogar("H7_stacking_P", vars_h7)
)
mejor_vars <- mejor_de(pruebas_h)
marca(
  "Mejor conjunto de variables:", mejor_vars$nombre, "- F1 =",
  round(mejor_vars$f1, 4)
)

# 5.2 Bases, con el mejor conjunto.
pruebas_b <- list(
  normal    = mejor_vars,
  aumentada = cv_hogar("B_aumentada", mejor_vars$vars, base = "aumentada"),
  sin_ricos = cv_hogar("B_sin_ricos", mejor_vars$vars, base = "sin_ricos")
)

# 5.3 Regresión del ingreso (log(ingreso / línea)): con las variables de
# H2 (sin no lineales) y con el mejor conjunto.
pruebas_r <- list(
  R_H2    = cv_hogar("R_H2", vars_h2, regresion = TRUE),
  R_mejor = cv_hogar("R_mejor", mejor_vars$vars, regresion = TRUE)
)

# 5.4 Afinamiento de hiperparámetros del clasificador, con el mejor conjunto.
pruebas_a <- pmap(grid_afinar, function(
  max_depth, min_child_weight, colsample_bytree
) {
  cv_hogar(
    sprintf(
      "A_depth%d_mcw%d_col%02d", max_depth, min_child_weight,
      round(10 * colsample_bytree)
    ),
    mejor_vars$vars,
    params = modifyList(params_base, list(
      max_depth = max_depth, min_child_weight = min_child_weight,
      colsample_bytree = colsample_bytree
    ))
  )
})
mejor_clf <- mejor_de(c(pruebas_h, pruebas_b, pruebas_a))
marca(
  "Mejor clasificador (5.1-5.4):", mejor_clf$nombre, "- F1 =",
  round(mejor_clf$f1, 4)
)

# 5.5 Grilla fina: eta más bajo (el LightGBM de 12 usa 0.02), con más
# rondas, sobre el mejor clasificador.
grid_fina <- expand_grid(
  eta = c(0.02, 0.03), max_depth = c(4, 6), colsample_bytree = c(0.6, 0.8)
)
pruebas_g <- pmap(grid_fina, function(eta, max_depth, colsample_bytree) {
  cv_hogar(
    sprintf(
      "G_eta%03d_depth%d_col%02d", round(100 * eta), max_depth,
      round(10 * colsample_bytree)
    ),
    mejor_clf$vars,
    base = mejor_clf$base, nrounds = 10000,
    params = modifyList(mejor_clf$params, list(
      eta = eta, max_depth = max_depth,
      colsample_bytree = colsample_bytree
    ))
  )
})

# 5.6 Stacking más rico (P2): el clasificador por persona también ve las
# variables del hogar. Se agrega a P o la reemplaza.
pruebas_p2 <- if (!completo) {
  list()
} else {
  list(
    H8 = cv_hogar(
      "H8_P_y_P2", c(mejor_clf$vars, cols$P2),
      base = mejor_clf$base,
      params = mejor_clf$params
    ),
    H9 = cv_hogar("H9_P2_sin_P", c(setdiff(mejor_clf$vars, cols$P), cols$P2),
      base = mejor_clf$base,
      params = mejor_clf$params
    )
  )
}

# 5.7 Selección de variables por importancia (ganancia del mejor clasificador):
# ¿un modelo más simple da lo mismo?
pruebas_s <- map(c(25, 50, 75), function(n) {
  cv_hogar(sprintf("S_top%d", n), head(mejor_clf$importancia$variable, n),
    base = mejor_clf$base, params = mejor_clf$params
  )
})

mejor_clf <- mejor_de(c(
  pruebas_h, pruebas_b, pruebas_a, pruebas_g, pruebas_p2, pruebas_s
))
marca("Mejor clasificador:", mejor_clf$nombre, "- F1 =", round(mejor_clf$f1, 4))

# 5.8 Combinación de modelos (promedio de probabilidades fuera de muestra, con
# pesos y umbral escogidos por F1):
#   - dos modelos: mejor clasificador + mejor regresión;
#   - tres modelos: + el modelo por persona (probabilidad del hogar = promedio
#     de los log odds de sus miembros, de P; es el enfoque de 04, 09 y 10).
mejor_reg <- mejor_de(pruebas_r)
lo_persona <- bind_rows(map(1:k, ~ tabla_tarea(paste0("val_", .x))))
p_persona <- plogis(lo_persona$p_pers_media[match(train$id, lo_persona$id)])
persona <- c(list(nombre = "persona_P", prob = p_persona), f1_cv(p_persona))
marca(sprintf(
  "Modelo por persona (P): F1 = %.4f  umbral = %.2f", persona$f1, persona$umbral
))
comb2 <- combinar(mejor_clf, mejor_reg)
pesos3 <- expand_grid(w1 = seq(0, 1, 0.1), w2 = seq(0, 1, 0.1)) |>
  filter(w1 + w2 <= 1 + 1e-9)
tabla3 <- pmap_dfr(pesos3, function(w1, w2) {
  m <-
    f1_cv(
      w1 * mejor_clf$prob + w2 * mejor_reg$prob + (1 - w1 - w2) * persona$prob
    )
  tibble(w1 = w1, w2 = w2, umbral = m$umbral, f1 = m$f1)
})
comb3 <- slice_max(tabla3, f1, n = 1, with_ties = FALSE)
marca(sprintf(
  paste0(
    "Combinación de 3 (clasificador %.1f, regresión %.1f, persona %.1f): ",
    "umbral = %.2f  F1 = %.4f"
  ),
  comb3$w1, comb3$w2, 1 - comb3$w1 - comb3$w2, comb3$umbral, comb3$f1
))
p_comb2 <- comb2$w * mejor_clf$prob + (1 - comb2$w) * mejor_reg$prob
p_comb3 <- comb3$w1 * mejor_clf$prob + comb3$w2 * mejor_reg$prob +
  (1 - comb3$w1 - comb3$w2) * persona$prob

# 5.9 Umbral honesto y umbral por zona. El F1 de arriba escoge el umbral con
# los mismos folds en que lo mide (optimista). Umbral honesto: el umbral de
# cada fold se escoge con los otros 4. Umbral por zona: uno para cabeceras y
# otro para el resto, escogidos también con los otros 4 folds.
# (umbral_honesto() está en la sección 4.)
zona <- hog_tr$cabecera
umbral_zona_honesto <- function(prob) {
  mean(map_dbl(1:k, function(f) {
    o <- fold != f
    # Punto de partida: el umbral común óptimo en los otros 4 folds.
    u0 <- grid_umbral[which.max(map_dbl(
      grid_umbral, ~ calcular_f1(as.integer(prob[o] > .x), y[o])
    ))]
    u <- c(u0, u0)
    for (zz in 0:1) { # una pasada por zona, con la otra fija
      f1z <- map_dbl(grid_umbral, function(uu) {
        uz <- if_else(
          zona[o] == 1, if (zz == 1) uu else u[2], if (zz == 0) uu else u[1]
        )
        calcular_f1(as.integer(prob[o] > uz), y[o])
      })
      u[zz + 1] <- grid_umbral[which.max(f1z)]
    }
    calcular_f1(
      as.integer(prob[fold == f] > if_else(zona[fold == f] == 1, u[2], u[1])),
      y[fold == f]
    )
  }))
}
honesto <- tibble(
  modelo = c(
    "H0 (base 01b)", mejor_clf$nombre, mejor_reg$nombre, "persona (P)",
    sprintf("combinación de 2 (w = %.1f)", comb2$w), "combinación de 3"
  ),
  f1_cv = c(
    pruebas_h$H0$f1, mejor_clf$f1, mejor_reg$f1, persona$f1, comb2$f1, comb3$f1
  ),
  prob = list(
    pruebas_h$H0$prob, mejor_clf$prob, mejor_reg$prob, persona$prob, p_comb2,
    p_comb3
  )
) |>
  mutate(
    f1_honesto = map_dbl(prob, umbral_honesto),
    f1_zona_honesto = map_dbl(prob, umbral_zona_honesto)
  ) |>
  select(-prob)
saveRDS(honesto, file.path(dir_out, "umbral_honesto.rds"))
marca("Umbral honesto y por zona:")
print(as.data.frame(honesto), digits = 4)

# 5.10 Ruido del F1: las mismas pruebas con otras 3 particiones en folds (cada
# una recalcula V3/V4/P/P2). Las diferencias entre modelos se comparan dentro
# de cada partición.
if (completo) {
  ruido <- map_dfr(c(456, 789, 2024), function(s) {
    cambiar_particion(s)
    r0 <- cv_hogar(sprintf("ruido_s%d_H0", s), vars_h0)
    r1 <- cv_hogar(sprintf("ruido_s%d_H1", s), vars_h1)
    r2 <- cv_hogar(sprintf("ruido_s%d_mejor", s), mejor_clf$vars,
      base = mejor_clf$base,
      params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    )
    tibble(particion = s, H0 = r0$f1, H1 = r1$f1, mejor = r2$f1)
  })
  cambiar_particion(123)
  ruido <- bind_rows(
    tibble(
      particion = 123, H0 = pruebas_h$H0$f1, H1 = pruebas_h$H1$f1,
      mejor = mejor_clf$f1
    ),
    ruido
  ) |>
    mutate(mejor_menos_H0 = mejor - H0, mejor_menos_H1 = mejor - H1)
  saveRDS(ruido, file.path(dir_out, "ruido.rds"))
  marca("Ruido del F1 entre particiones:")
  print(as.data.frame(ruido), digits = 4)
}

# 5.11 Eliminación de variables poco importantes con el mejor clasificador (sus
# hiperparámetros y su base): se dejan las variables que explican el 99,9%,
# 99,5% y 99% de su ganancia, y las 75, 60, 50, 40, 20, 10 y 5 con más ganancia.
# Las
# menos importantes son sobre todo conteos que repiten proporciones (n_of_*,
# n_ocup, n_formal...) e ingresos no laborales poco frecuentes (any_*).
imp_mejor <- mejor_clf$importancia
n_ganancia <- function(q) sum(cumsum(imp_mejor$importancia) <= q) + 1
cortes_sel <- c(
  ganancia999 = n_ganancia(99.9), ganancia995 = n_ganancia(99.5),
  ganancia99 = n_ganancia(99), top75 = 75, top60 = 60, top50 = 50, top40 = 40,
  # Muy pocas variables, para ver si reducir mucho el ruido ayuda.
  top20 = 20, top10 = 10, top5 = 5
)
pruebas_e <- imap(cortes_sel, function(n, nm) {
  cv_hogar(sprintf("E_%s_%dvars", nm, n), head(imp_mejor$variable, n),
    base = mejor_clf$base,
    params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
  )
})
seleccion <- tabla_pruebas(pruebas_e, refs = list(mejor_clf))
saveRDS(seleccion, file.path(dir_out, "seleccion.rds"))
marca("Eliminación de variables poco importantes:")
print(as.data.frame(seleccion), digits = 4)

# 5.12 Stacking de la regresión: la predicción de log(ingreso / línea) del
# hogar (z_pred, CV anidada) como variable del clasificador, en vez de solo
# promediar probabilidades (5.8). Los residuales no se pueden usar: requieren
# el ingreso real, que no está en test.
mejor_e <- mejor_de(pruebas_e)
pruebas_z <- list()
if (completo) {
  z_oof <- bind_rows(map(1:k, ~ vz[[paste0("val_", .x)]]$hogar))
  z_oof <- z_oof$z_pred[match(train$id, z_oof$id)]
  marca(sprintf(
    "Regresión para Z: R2 fuera de muestra = %.4f",
    1 - sum((z - z_oof)^2) / sum((z - mean(z))^2)
  ))
  pruebas_z <- list(
    H10 = cv_hogar("H10_mejor_mas_z", c(mejor_clf$vars, cols$Z),
      base = mejor_clf$base,
      params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    ),
    H11 = cv_hogar(paste0(mejor_e$nombre, "_mas_z"), c(mejor_e$vars, cols$Z),
      base = mejor_clf$base,
      params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    )
  )
  stack_z <- tabla_pruebas(
    pruebas_z,
    refs = list(mejor_clf, mejor_e), nuevas = cols$Z
  ) |> rename(importancia_z = ganancia_nuevas)
  saveRDS(stack_z, file.path(dir_out, "stacking_z.rds"))
  marca("Stacking de la regresión:")
  print(as.data.frame(stack_z), digits = 4)
}

# 5.13 Más combinaciones no lineales (N4) y transformaciones (T), sobre el
# mejor modelo con menos variables (5.11) y sobre el mejor clasificador.
pruebas_nl <- list()
if (completo) {
  pruebas_nl <- list(
    NL1 = cv_hogar(paste0(mejor_e$nombre, "_N4"), c(mejor_e$vars, cols$N4),
      base = mejor_clf$base,
      params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    ),
    NL2 = cv_hogar(paste0(mejor_e$nombre, "_T"), c(mejor_e$vars, cols$T),
      base = mejor_clf$base,
      params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    ),
    NL3 = cv_hogar(paste0(mejor_e$nombre, "_N4_T"), c(
      mejor_e$vars, cols$N4, cols$T
    ),
    base = mejor_clf$base,
    params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    ),
    NL4 = cv_hogar(paste0(mejor_clf$nombre, "_N4_T"), c(
      mejor_clf$vars, cols$N4, cols$T
    ),
    base = mejor_clf$base,
    params = mejor_clf$params, nrounds = mejor_clf$nrounds %||% nrounds_max
    )
  )
  no_lineales <- tabla_pruebas(
    pruebas_nl,
    refs = list(mejor_e, mejor_clf), nuevas = c(cols$N4, cols$T)
  )
  saveRDS(no_lineales, file.path(dir_out, "no_lineales.rds"))
  marca("Combinaciones no lineales y transformaciones:")
  print(as.data.frame(no_lineales), digits = 4)
  mejor_nl <- mejor_de(pruebas_nl)
}

# 5.14 Afinamiento para modelos con muy pocas variables (las 20, 10 y 5 con más
# ganancia). Con los hiperparámetros del mejor modelo, colsample_bytree = 0.6
# deja a cada árbol con 3-6 variables, y la profundidad y el mínimo por hoja
# óptimos pueden ser otros.
grid_pocas <- expand_grid(
  max_depth = c(3, 4, 6, 8), min_child_weight = c(10, 50),
  colsample_bytree = c(0.8, 1)
)
pocas <- map(c(20, 10, 5), function(n) {
  vars_n <- head(imp_mejor$variable, n)
  pmap(grid_pocas, function(max_depth, min_child_weight, colsample_bytree) {
    cv_hogar(
      sprintf(
        "F_top%d_depth%d_mcw%d_col%02d", n, max_depth, min_child_weight,
        round(10 * colsample_bytree)
      ),
      vars_n,
      base = mejor_clf$base, nrounds = 10000,
      params = modifyList(mejor_clf$params, list(
        max_depth = max_depth, min_child_weight = min_child_weight,
        colsample_bytree = colsample_bytree
      ))
    )
  })
})
pruebas_pocas <- flatten(pocas)
afinado_pocas <- tibble(
  prueba = map_chr(pruebas_pocas, "nombre"),
  variables = map_int(pruebas_pocas, ~ length(.x$vars)),
  max_depth = map_dbl(pruebas_pocas, ~ .x$params$max_depth),
  min_child_weight = map_dbl(pruebas_pocas, ~ .x$params$min_child_weight),
  colsample = map_dbl(pruebas_pocas, ~ .x$params$colsample_bytree),
  f1 = map_dbl(pruebas_pocas, "f1"), umbral = map_dbl(pruebas_pocas, "umbral"),
  iter = map_dbl(pruebas_pocas, ~ mean(.x$iters))
) |>
  arrange(variables, desc(f1))
mejores_pocas <- afinado_pocas |>
  slice_max(f1, n = 1, by = variables, with_ties = FALSE) |>
  mutate(f1_honesto = map_dbl(
    prueba,
    ~ umbral_honesto(pruebas_pocas[[which(map_chr(
      pruebas_pocas, "nombre"
    ) == .x)]]$prob)
  ))
saveRDS(
  list(todas = afinado_pocas, mejores = mejores_pocas),
  file.path(dir_out, "afinado_pocas.rds")
)
marca("Afinamiento con pocas variables (mejor de cada una):")
print(as.data.frame(mejores_pocas), digits = 4)

# 5.15-5.17: variables nuevas de 3.4 y 3.2 (ingreso predicho mejorado,
# indigencia e índices) sobre el mejor clasificador (`ref`, con sus
# hiperparámetros) y, los índices, también sobre el mejor de 20 variables
# (5.14), donde un resumen de muchas variables debería notarse más.
ref <- mejor_clf
ref20 <- mejor_de(keep(pruebas_pocas, ~ length(.x$vars) == 20))
# Prueba con la base, los hiperparámetros, las rondas y los pesos de `r`.
variante <- function(r, sufijo, vars, pesos = r$pesos) {
  cv_hogar(paste0(r$nombre, "_", sufijo), vars,
    base = r$base, params = r$params,
    nrounds = r$nrounds %||% nrounds_max, pesos = pesos
  )
}
con_v4 <- function(vars, s) {
  c(
    setdiff(vars, c(cols$V4, "ing_pred_por_ocup")), cols[[paste0("V4", s)]]
  )
}

# 5.15 Ingreso predicho mejorado: reemplaza a V4 (afinado, con variables del
# hogar, por componentes) y el mejor se agrega junto a V4.
pruebas_m <- map(
  c(t = "t", h = "h", c = "c"),
  ~ variante(ref, paste0("V4", .x), con_v4(ref$vars, .x))
)
s_v4 <- names(pruebas_m)[which.max(map_dbl(pruebas_m, "f1"))]
pruebas_m$mas <- variante(
  ref, paste0("V4_mas_V4", s_v4), c(ref$vars, cols[[paste0("V4", s_v4)]])
)

# 5.16 Indigencia (PI) y 5.17 índices (I sin supervisar, IS supervisados).
pruebas_pi <- list(PI = variante(ref, "PI", c(ref$vars, cols$PI)))
pruebas_i <- list(
  I    = variante(ref, "I", c(ref$vars, cols$I)),
  IS   = variante(ref, "IS", c(ref$vars, cols$IS)),
  I_IS = variante(ref, "I_IS", c(ref$vars, cols$I, cols$IS)),
  I20  = variante(ref20, "I", c(ref20$vars, cols$I)),
  IS20 = variante(ref20, "IS", c(ref20$vars, cols$IS))
)
# Todo junto: el mejor ingreso predicho (o V4 si ninguno mejora), PI, I e IS.
vars_m <- mejor_de(c(list(ref), pruebas_m))$vars
todo <- list(todo = variante(
  ref, "todo", unique(c(vars_m, cols$PI, cols$I, cols$IS))
))

nuevas <- c(cols$V4t, cols$V4h, cols$V4c, cols$PI, cols$I, cols$IS)
pruebas_nuevas <- c(pruebas_m, pruebas_pi, pruebas_i, todo)
mejoras <- tabla_pruebas(
  pruebas_nuevas,
  refs = list(ref, ref20), nuevas = nuevas
)
saveRDS(
  list(mejoras = mejoras, r2_ingreso = r2_ingreso, afinado_v4 = afinado_v4),
  file.path(dir_out, "mejoras.rds")
)
marca("Ingreso mejorado, indigencia e índices:")
print(as.data.frame(mejoras), digits = 4)
mejor_nuevo <- mejor_de(pruebas_nuevas)

# 5.18 Sobre el mejor de 5.15-5.17 (`ref2`): clasificador por persona afinado
# (Pt) y con las preguntas que no se usaban (Pe), ingreso predicho con esas
# preguntas (V4e) y probabilidades por persona de estar por debajo de 0.8 y
# 1.25 veces la línea (C), cada uno agregado a las variables de ref2.
ref2 <- mejor_nuevo
pruebas_q <- list(
  Pt   = variante(ref2, "Pt", c(ref2$vars, cols$Pt)),
  Pe   = variante(ref2, "Pe", c(ref2$vars, cols$Pe)),
  V4e  = variante(ref2, "V4e", c(ref2$vars, cols$V4e)),
  C    = variante(ref2, "C", c(ref2$vars, cols$C080, cols$C125))
)
# Todo junto: el mejor de Pt / Pe, V4e y los cortes.
p_mejor <- if (pruebas_q$Pe$f1 > pruebas_q$Pt$f1) cols$Pe else cols$Pt
pruebas_q$todo <- variante(
  ref2, "todo2", c(ref2$vars, p_mejor, cols$V4e, cols$C080, cols$C125)
)
nuevas2 <- c(cols$Pt, cols$Pe, cols$V4e, cols$C080, cols$C125)
mejoras2 <- tabla_pruebas(pruebas_q, refs = list(ref2), nuevas = nuevas2)
saveRDS(
  list(mejoras = mejoras2, afinado_p = afinado_p, auc_persona = auc_persona),
  file.path(dir_out, "mejoras2.rds")
)
marca("Clasificador por persona afinado, preguntas no usadas y cortes:")
print(as.data.frame(mejoras2), digits = 4)
pruebas_nuevas <- c(pruebas_nuevas, pruebas_q)
mejor_nuevo <- mejor_de(pruebas_nuevas)

# 5.19 Pesos según la distancia a la línea: el 82% de los errores está en
# hogares con ingreso a menos de 50% de la línea (sección 10). Más peso en el
# entrenamiento a esos hogares, 1 + a * exp(-|z| / b), para que los árboles
# gasten sus cortes en los casos difíciles. Usa el ingreso real solo para
# entrenar; la predicción de test no lo necesita.
ref3 <- mejor_nuevo
peso_linea <- function(a, b) 1 + a * exp(-abs(z) / b)
grid_peso <- expand_grid(a = c(1, 3), b = c(0.25, 0.5))
pruebas_w <- pmap(grid_peso, function(a, b) {
  variante(
    ref3, sprintf("peso_a%d_b%03d", a, round(100 * b)), ref3$vars,
    pesos = peso_linea(a, b)
  )
})
pruebas_nuevas <- c(pruebas_nuevas, pruebas_w)
mejor_nuevo <- mejor_de(pruebas_nuevas)

# 5.20 Regresión del ingreso con las variables del mejor modelo, sin pesos y
# con más peso cerca de la línea (en la regresión, un error en un hogar rico
# cuenta igual que uno al lado de la línea), y su combinación con el mejor
# clasificador. La regresión que mejor combina se usa en la sección 6.
pruebas_r2 <- list(
  R_todo = cv_hogar(paste0("R_", ref3$nombre), ref3$vars, regresion = TRUE),
  R_todo_peso = cv_hogar(paste0("R_", ref3$nombre, "_peso"), ref3$vars,
    regresion = TRUE,
    pesos = peso_linea(3, 0.25)
  )
)
regs <- c(list(R_mejor = mejor_reg), pruebas_r2)
combs_reg <- map(regs, ~ combinar(mejor_nuevo, .x))
reg_comb <- regs[[which.max(map_dbl(combs_reg, "f1"))]]
mejoras3 <- bind_rows(
  tibble(
    prueba = c(ref3$nombre, map_chr(pruebas_w, "nombre")),
    f1 = c(ref3$f1, map_dbl(pruebas_w, "f1")),
    f1_honesto = map_dbl(
      c(list(ref3$prob), map(pruebas_w, "prob")), umbral_honesto
    ),
    umbral = c(ref3$umbral, map_dbl(pruebas_w, "umbral"))
  ),
  tibble(
    prueba = map_chr(regs, "nombre"), f1 = map_dbl(regs, "f1"),
    f1_honesto = map_dbl(regs, ~ umbral_honesto(.x$prob)),
    umbral = map_dbl(regs, "umbral"),
    r2 = map_dbl(regs, ~ .x$metricas$r2)
  ),
  tibble(
    prueba = paste(mejor_nuevo$nombre, "+", map_chr(regs, "nombre")),
    f1 = map_dbl(combs_reg, "f1"), umbral = map_dbl(combs_reg, "umbral"),
    w = map_dbl(combs_reg, "w"),
    f1_honesto = map_dbl(
      combs_reg, ~ umbral_honesto(.x$w * .x$a$prob + (1 - .x$w) * .x$b$prob)
    )
  )
)
saveRDS(mejoras3, file.path(dir_out, "mejoras3.rds"))
marca("Pesos cerca de la línea y regresión:")
print(as.data.frame(mejoras3), digits = 4)

# 5.21 Ingreso predicho con las preguntas no usadas, afinado (V4f): en vez de
# V4e y junto a V4e.
ref4 <- mejor_nuevo
pruebas_v4f <- list(
  V4f     = variante(ref4, "V4f", c(setdiff(ref4$vars, cols$V4e), cols$V4f)),
  mas_V4f = variante(ref4, "mas_V4f", c(ref4$vars, cols$V4f))
)
pruebas_nuevas <- c(pruebas_nuevas, pruebas_v4f)
mejor_nuevo <- mejor_de(pruebas_nuevas)

# 5.22 Hiperparámetros del clasificador por hogar con el nuevo conjunto de
# variables (los de 5.4-5.5 se escogieron con las variables originales).
ref5 <- mejor_nuevo
grid_final <- expand_grid(
  max_depth = c(4, 6, 8), min_child_weight = c(10, 50),
  colsample_bytree = c(0.5, 0.8)
)
pruebas_af <- pmap(grid_final, function(
  max_depth, min_child_weight, colsample_bytree
) {
  cv_hogar(
    sprintf(
      "%s_af_depth%d_mcw%d_col%02d", ref5$nombre, max_depth, min_child_weight,
      round(10 * colsample_bytree)
    ),
    ref5$vars,
    base = ref5$base, nrounds = 10000, pesos = ref5$pesos,
    params = modifyList(ref5$params, list(
      max_depth = max_depth, min_child_weight = min_child_weight,
      colsample_bytree = colsample_bytree
    ))
  )
})
pruebas_nuevas <- c(pruebas_nuevas, pruebas_af)
mejor_nuevo <- mejor_de(pruebas_nuevas)

# 5.23 Selección de variables por ganancia en el mejor modelo nuevo.
ref6 <- mejor_nuevo
pruebas_s2 <- map(
  c(50, 75, 100),
  ~ variante(ref6, paste0("top", .x), head(ref6$importancia$variable, .x))
)
pruebas_nuevas <- c(pruebas_nuevas, pruebas_s2)
mejor_nuevo <- mejor_de(pruebas_nuevas)
pruebas4 <- c(list(ref4), pruebas_v4f, pruebas_af, pruebas_s2)
mejoras4 <- tabla_pruebas(pruebas4) |>
  mutate(
    max_depth = map_dbl(pruebas4, ~ .x$params$max_depth),
    min_child_weight = map_dbl(pruebas4, ~ .x$params$min_child_weight),
    colsample = map_dbl(pruebas4, ~ .x$params$colsample_bytree),
    .after = variables
  ) |>
  mutate(iter = map_dbl(pruebas4, ~ mean(.x$iters)))
saveRDS(
  list(mejoras = mejoras4, afinado_v4f = afinado_v4f, r2_ingreso = r2_ingreso),
  file.path(dir_out, "mejoras4.rds")
)
marca("Ingreso V4f, hiperparámetros y selección con el modelo nuevo:")
print(as.data.frame(mejoras4), digits = 4)

# 5.24 Arriendo imputado (A, Pa) e ingreso predicho Tweedie (V4w) sobre el
# mejor modelo (`ref7`). pobre = (ingresos de la unidad de gasto + arriendo
# imputado) / personas < línea: el modelo tenía el ingreso predicho y el
# arriendo por separado (y el arriendo mezclando lo que paga un arrendatario
# con lo que estima un propietario); A les da la suma hecha.
ref7 <- mejor_nuevo
pruebas_aw <- list(
  A   = variante(ref7, "A", c(ref7$vars, cols$A)),
  Pa  = variante(ref7, "Pa", c(ref7$vars, cols$Pa)),
  V4w = variante(ref7, "V4w", c(ref7$vars, cols$V4w, "ing_arr_pc_lp_w"))
)
pruebas_aw$todo3 <- variante(
  ref7, "todo3",
  unique(c(ref7$vars, cols$A, cols$Pa, cols$V4w, "ing_arr_pc_lp_w"))
)
nuevas5 <- c(cols$A, cols$Pa, cols$V4w, "ing_arr_pc_lp_w")
mejoras5 <- tabla_pruebas(c(list(ref7), pruebas_aw), nuevas = nuevas5)
saveRDS(
  list(
    mejoras = mejoras5, afinado_v4w = afinado_v4w, r2_ingreso = r2_ingreso,
    auc_persona = auc_persona
  ),
  file.path(dir_out, "mejoras5.rds")
)
marca("Arriendo imputado e ingreso Tweedie:")
print(as.data.frame(mejoras5), digits = 4)
pruebas_nuevas <- c(pruebas_nuevas, pruebas_aw)
mejor_nuevo <- mejor_de(pruebas_nuevas)

# 5.25 Tweedie afinado (V4x, en vez de V4w y junto a V4w) y regresión del
# ingreso con las variables del mejor modelo; la regresión que mejor combina
# con el mejor modelo pasa a ser `reg_comb` (sección 6).
ref8 <- mejor_nuevo
pruebas_x4 <- list()
if (!is.null(confs_v4$x) && any(ref8$vars %in% cols$V4w)) {
  pruebas_x4 <- list(
    V4x = variante(
      ref8, "V4x",
      c(setdiff(ref8$vars, c(cols$V4w, "ing_arr_pc_lp_w")), cols$V4x)
    ),
    mas_V4x = variante(ref8, "mas_V4x", c(ref8$vars, cols$V4x))
  )
  pruebas_nuevas <- c(pruebas_nuevas, pruebas_x4)
  mejor_nuevo <- mejor_de(pruebas_nuevas)
}
pruebas_r3 <- list(R_final = cv_hogar(
  paste0(
    "R_", mejor_nuevo$nombre
  ), mejor_nuevo$vars,
  regresion = TRUE,
  pesos = peso_linea(3, 0.25)
))
regs <- c(regs, pruebas_r3)
combs_reg <- map(regs, ~ combinar(mejor_nuevo, .x))
reg_comb <- regs[[which.max(map_dbl(combs_reg, "f1"))]]
mejoras6 <- tibble(
  prueba = c(
    ref8$nombre, map_chr(pruebas_x4, "nombre"),
    paste(mejor_nuevo$nombre, "+", map_chr(regs, "nombre"))
  ),
  f1 = c(ref8$f1, map_dbl(pruebas_x4, "f1"), map_dbl(combs_reg, "f1")),
  f1_honesto = c(
    umbral_honesto(ref8$prob), map_dbl(pruebas_x4, ~ umbral_honesto(.x$prob)),
    map_dbl(
      combs_reg, ~ umbral_honesto(.x$w * .x$a$prob + (1 - .x$w) * .x$b$prob)
    )
  ),
  w = c(NA, rep(NA, length(pruebas_x4)), map_dbl(combs_reg, "w"))
)
saveRDS(
  list(mejoras = mejoras6, afinado_v4x = afinado_v4x),
  file.path(dir_out, "mejoras6.rds")
)
marca("Tweedie afinado y regresión con el mejor modelo:")
print(as.data.frame(mejoras6), digits = 4)
marca("Mejor modelo:", mejor_nuevo$nombre, "- F1 =", round(mejor_nuevo$f1, 4))

# Resumen de todas las pruebas (el AUC se recalcula con las probabilidades).
todas <- c(
  pruebas_h, pruebas_b[-1], pruebas_r, pruebas_a, pruebas_g, pruebas_p2,
  pruebas_s, pruebas_e, pruebas_z, pruebas_nl, pruebas_pocas, pruebas_nuevas,
  pruebas_r2
)
resumen <- tibble(
  prueba = map_chr(todas, "nombre"), f1 = map_dbl(todas, "f1"),
  umbral = map_dbl(todas, "umbral"),
  auc = map_dbl(todas, ~ calcular_auc(.x$prob, y)),
  r2 = map_dbl(todas, ~ .x$metricas$r2 %||% NA),
  iter = map_dbl(todas, ~ mean(.x$iters)),
  variables = map_int(todas, ~ length(.x$vars)),
  minutos = map_dbl(todas, "minutos")
) |>
  arrange(desc(f1))
saveRDS(resumen, file.path(dir_out, "resumen.rds"))
print(as.data.frame(resumen), digits = 4)

# 6. Modelo final y submissions -----------------------------------------------

# Mejor clasificador y, si mejoran el F1, la mejor combinación.
p_clf <- prob_test(mejor_clf)
guardar_submission(p_clf, mejor_clf$umbral, mejor_clf$nombre)
# El mejor modelo con menos variables (5.11), si su F1 es mayor.
if (mejor_e$f1 > mejor_clf$f1) {
  guardar_submission(
    prob_test(mejor_e), mejor_e$umbral, mejor_e$nombre
  )
}
# Con combinaciones no lineales (5.13), si su F1 es el mayor.
if (completo && mejor_nl$f1 > max(mejor_e$f1, mejor_clf$f1)) {
  guardar_submission(
    prob_test(mejor_nl), mejor_nl$umbral, mejor_nl$nombre
  )
}
# Con las variables de 5.15-5.17, si su F1 es mayor; y su combinación con la
# regresión, si la mejora.
if (mejor_nuevo$f1 > max(mejor_e$f1, mejor_clf$f1)) {
  p_nuevo <- prob_test(mejor_nuevo)
  guardar_submission(p_nuevo, mejor_nuevo$umbral, mejor_nuevo$nombre)
  comb_nuevo <- combinar(mejor_nuevo, reg_comb)
  if (comb_nuevo$w < 1 && comb_nuevo$f1 > mejor_nuevo$f1) {
    guardar_submission(
      comb_nuevo$w * p_nuevo + (1 - comb_nuevo$w) * prob_test(reg_comb),
      comb_nuevo$umbral,
      sprintf(
        "comb_%s_%s_w%02d", mejor_nuevo$nombre, reg_comb$nombre,
        round(10 * comb_nuevo$w)
      )
    )
  }
}
# Para diversificar las submissions (F1 parecido, predicciones distintas): el
# mejor modelo con solo sus 75 variables de más ganancia (5.23).
top75 <- pruebas_s2[[2]]
guardar_submission(prob_test(top75), top75$umbral, top75$nombre)
if (max(comb2$f1, comb3$f1) > mejor_clf$f1) {
  p_reg <- prob_test(mejor_reg)
  if (comb3$f1 > comb2$f1) {
    lo_te <- tabla_tarea("test")
    p_per <- plogis(lo_te$p_pers_media[match(test$id, lo_te$id)])
    guardar_submission(
      comb3$w1 * p_clf + comb3$w2 * p_reg + (1 - comb3$w1 - comb3$w2) * p_per,
      comb3$umbral,
      sprintf(
        "comb3_w%02d_%02d_%02d", round(10 * comb3$w1), round(10 * comb3$w2),
        round(10 * (1 - comb3$w1 - comb3$w2))
      )
    )
  } else {
    guardar_submission(
      comb2$w * p_clf + (1 - comb2$w) * p_reg, comb2$umbral,
      sprintf(
        "comb_%s_%s_w%02d", mejor_clf$nombre, mejor_reg$nombre,
        round(10 * comb2$w)
      )
    )
  }
}

# 7. Naive bayes: con la base por hogar (curiosidad) y el de 04 combinado -----

if (completo) {
  # 7.1 Mismo enfoque que 04 (factores para categóricas y 0/1, normales para el
  # resto; laplace = 0), con las 94 variables de 02 y con el mejor conjunto.
  archivo_nb <- file.path(dir_out, "naive_bayes.rds")
  if (!file.exists(archivo_nb)) {
    cv_nb <- function(vars) {
      mat <- hog_tr[intersect(vars, names(hog_tr))] |>
        mutate(
          across(any_of(categoricas), factor),
          across(where(~ is.numeric(.x) && all(.x %in% c(0, 1, NA))), factor)
        )
      oof <- rep(NA_real_, length(y))
      for (f in 1:k) {
        if (any(vars %in% c(cols$V3, cols$V4, cols$P, "ing_pred_por_ocup"))) {
          m <- matrices(f, vars)
          x_e <- as.data.frame(m$X_ent)
          x_v <- as.data.frame(m$X_val)
          for (v in intersect(
            names(x_e), names(mat)[map_lgl(mat, is.factor)]
          )) {
            niv <- levels(mat[[v]])
            x_e[[v]] <- factor(x_e[[v]], levels = niv)
            x_v[[v]] <- factor(x_v[[v]], levels = niv)
          }
        } else {
          x_e <- mat[fold != f, ]
          x_v <- mat[fold == f, ]
        }
        modelo <- naiveBayes(x_e, factor(y[fold != f]))
        oof[fold == f] <- predict(modelo, x_v, type = "raw")[, "1"]
      }
      # Naive bayes sobrestima la pobreza: necesita umbrales más altos.
      # Umbrales por cuantiles: con tantas variables sus probabilidades se van a
      # los extremos (casi todas cerca de 0 o de 1).
      m <- f1_cv(oof, unique(quantile(oof, seq(0.5, 0.95, by = 0.0025))))
      list(f1 = m$f1, umbral = m$umbral, auc = calcular_auc(oof, y))
    }
    nb <- list(base01b = cv_nb(vars_h0), mejor = cv_nb(mejor_vars$vars))
    saveRDS(nb, archivo_nb)
    marca(sprintf(
      paste0(
        "Naive bayes: base 01b F1 = %.4f (umbral %.2f); %s F1 = %.4f ",
        "(umbral %.2f)"
      ),
      nb$base01b$f1, nb$base01b$umbral, mejor_vars$nombre, nb$mejor$f1,
      nb$mejor$umbral
    ))
  }

  # 7.2 El mejor naive bayes (04: por persona, xtodas, laplace = 0, promedio de
  # log odds por hogar) con los 5 folds por hogar, y combinación con el mejor
  # modelo. Sus probabilidades están muy descalibradas, así que antes de
  # combinar se calibran con un logit de pobre sobre sus log odds (fuera de
  # muestra), como la regresión.
  archivo_nb03 <- file.path(dir_out, "naive_bayes_03.rds")
  if (!file.exists(archivo_nb03)) {
    xtodas <- c(
      "anos", "anos_educ_imp", "mujer", "ocupado", "afiliado_salud",
      "regimen_salud",
      "actividad_sem_pasada", "log_meses_empresa", "tamano_empresa",
      "posicion_ocupacional",
      "horas_trabajo_semana", "arriendo_pc_sobre_lp", "prop_ocupados_hogar",
      "tenencia_vivienda", "recibio_arriendos_pensiones", "ayudas_instituciones"
    )
    cat_03 <- c(
      "regimen_salud", "actividad_sem_pasada", "tamano_empresa",
      "posicion_ocupacional",
      "tenencia_vivienda"
    )
    preparar_03 <- function(x) {
      x |> mutate(
        across(any_of(cat_03), factor),
        across(where(~ is.numeric(.x) && all(.x %in% c(0, 1, NA))), factor)
      )
    }
    p_tr <- read_rds(here("data", "train_clean.rds")) |>
      filter(id %in% train$id)
    p_te <- read_rds(here("data", "test_clean.rds")) |> filter(id %in% test$id)
    mat_tr <- preparar_03(select(p_tr, all_of(xtodas)))
    mat_te <- preparar_03(select(p_te, all_of(xtodas))) |>
      mutate(across(
        where(is.factor), ~ factor(.x, levels = levels(mat_tr[[cur_column()]]))
      ))
    y_p <- factor(p_tr$pobre)
    fold_p <- fold[match(p_tr$id, train$id)]
    # Log odds por persona -> promedio por hogar (como prob_hogar() de 04).
    lo_hogar <- function(modelo, mat, ids) {
      p <- pmin(pmax(predict(modelo, mat, type = "raw")[, "1"], 1e-6), 1 - 1e-6)
      tibble(id = ids, lo = qlogis(p)) |> summarize(lo = mean(lo), .by = id)
    }
    oof_nb <- bind_rows(en_paralelo(1:k, function(f) {
      m <- naiveBayes(mat_tr[fold_p != f, ], y_p[fold_p != f], laplace = 0)
      lo_hogar(m, mat_tr[fold_p == f, ], p_tr$id[fold_p == f])
    }, nucleos = min(nucleos, k), que = "naive bayes de 03"))
    lo_nb <- oof_nb$lo[match(train$id, oof_nb$id)]
    lo_nb_te <- lo_hogar(naiveBayes(mat_tr, y_p, laplace = 0), mat_te, p_te$id)
    saveRDS(
      list(lo = lo_nb, lo_test = lo_nb_te$lo[match(test$id, lo_nb_te$id)]),
      archivo_nb03
    )
    rm(p_tr, p_te, mat_tr, mat_te)
    invisible(gc())
  }
  nb03 <- readRDS(archivo_nb03)
  cal_nb <- glm(y ~ nb03$lo, family = binomial)
  p_nb <- unname(predict(cal_nb, type = "response"))
  nb <- c(list(nombre = "naive_bayes_03", prob = p_nb), f1_cv(p_nb))
  marca(sprintf(
    "Naive bayes de 03 (5 folds de main): F1 = %.4f  umbral = %.2f", nb$f1,
    nb$umbral
  ))
  # El mejor modelo hasta aquí (clasificador o combinación) y su combinación con
  # naive bayes, con peso w para el mejor modelo.
  candidatos <- list(
    clasificador = list(prob = mejor_clf$prob, f1 = mejor_clf$f1),
    comb2 = list(prob = p_comb2, f1 = comb2$f1),
    comb3 = list(prob = p_comb3, f1 = comb3$f1)
  )
  mejor_modelo <- names(candidatos)[which.max(map_dbl(candidatos, "f1"))]
  base_nb <- c(
    list(
      nombre = mejor_modelo, prob = candidatos[[mejor_modelo]]$prob
    ),
    f1 = candidatos[[mejor_modelo]]$f1
  )
  comb_nb <- combinar(base_nb, nb, pesos = seq(0.5, 1, by = 0.05))
  p_comb_nb <- comb_nb$w * base_nb$prob + (1 - comb_nb$w) * nb$prob
  saveRDS(
    list(
      nb_f1 = nb$f1, nb_umbral = nb$umbral, base = mejor_modelo,
      base_f1 = base_nb$f1,
      tabla = comb_nb$tabla, w = comb_nb$w, f1 = comb_nb$f1,
      f1_honesto_base = umbral_honesto(base_nb$prob),
      f1_honesto_comb = umbral_honesto(p_comb_nb)
    ),
    file.path(dir_out, "combinacion_nb.rds")
  )
  marca(sprintf(
    paste0(
      "Con naive bayes: %s F1 = %.4f (honesto %.4f) -> w = %.2f, F1 = %.4f ",
      "(honesto %.4f)"
    ),
    mejor_modelo, base_nb$f1, umbral_honesto(base_nb$prob), comb_nb$w,
    comb_nb$f1,
    umbral_honesto(p_comb_nb)
  ))
  # Submission de la combinación con naive bayes, solo si mejora el F1 honesto.
  if (
    comb_nb$w < 1 && umbral_honesto(p_comb_nb) > umbral_honesto(base_nb$prob)
  ) {
    p_base_te <- switch(mejor_modelo,
      clasificador = prob_test(mejor_clf),
      comb2 = comb2$w * prob_test(mejor_clf) +
        (1 - comb2$w) * prob_test(mejor_reg),
      comb3 = {
        lo_te <- tabla_tarea("test")
        comb3$w1 * prob_test(mejor_clf) + comb3$w2 * prob_test(mejor_reg) +
          (1 - comb3$w1 - comb3$w2) * plogis(lo_te$p_pers_media[match(
            test$id, lo_te$id
          )])
      }
    )
    p_nb_te <- plogis(coef(cal_nb)[1] + coef(cal_nb)[2] * nb03$lo_test)
    guardar_submission(
      comb_nb$w * p_base_te + (1 - comb_nb$w) * p_nb_te, comb_nb$umbral,
      sprintf("%s_con_NB_w%03d", mejor_modelo, round(100 * comb_nb$w))
    )
  }
}

# 8. Comparación de codificaciones de las categóricas -------------------------

# Lee el resumen de cada codificación que ya se corrió y compara las pruebas
# principales.
dirs <- c(
  codigos = "xgb_hogar", onehot = "xgb_hogar_onehot",
  nativo = "xgb_hogar_nativo"
)
resumenes <- imap_dfr(dirs, function(d, cod) {
  f <- here(
    "02_outputs", "pruebas",
    if (prueba) paste0(sub("xgb_hogar", "xgb_hogar_prueba", d)) else d,
    "resumen.rds"
  )
  if (file.exists(f)) mutate(readRDS(f), codificacion = cod) else NULL
})
if (nrow(resumenes)) {
  claves <- c(
    "H0_base01b", "H1_V1aV4", "H7_stacking_P", "E_top75_75vars",
    "E_top20_20vars",
    "E_top10_10vars", "E_top5_5vars"
  )
  comparacion <- resumenes |>
    mutate(prueba = if_else(
      str_detect(prueba, "^[AG]_"), "mejor A/G", prueba
    )) |>
    filter(prueba %in% c(claves, "mejor A/G")) |>
    summarize(f1 = max(f1), .by = c(prueba, codificacion)) |>
    pivot_wider(names_from = codificacion, values_from = f1)
  saveRDS(comparacion, file.path(dir_out, "comparacion_codificaciones.rds"))
  marca("Comparación de codificaciones (F1):")
  print(as.data.frame(comparacion), digits = 4)
}

# 9. Random forest por hogar y combinación con XGBoost ------------------------

# Bosque de probabilidad (ranger) con las variables del mejor XGBoost
# (`mejor_nuevo`), los mismos folds y las mismas variables por fold. Comete
# errores distintos a los de XGBoost, así que puede servir para combinar.
# Categóricas como factores (respect.unordered.factors = "order"); los NA los
# maneja ranger (na.action = "na.learn").
grid_rf <- expand_grid(mtry = c(15, 30), min.node.size = c(10, 50))
arboles_rf <- if (prueba) 50 else 500
cv_rf <- function(nombre, vars, mtry, min_nodo) {
  archivo <- file.path(dir_out, paste0(nombre, ".rds"))
  if (file.exists(archivo)) {
    return(readRDS(archivo))
  }
  t0 <- Sys.time()
  oof <- rep(NA_real_, length(y))
  for (f in 1:k) {
    m <- matrices(f, vars)
    rf <- ranger(
      x = as.data.frame(m$X_ent), y = factor(y[m$i_ent]), probability = TRUE,
      num.trees = arboles_rf, mtry = mtry, min.node.size = min_nodo,
      respect.unordered.factors = "order", num.threads = nucleos, seed = 123,
      verbose = FALSE
    )
    oof[m$i_val] <- predict(
      rf, as.data.frame(m$X_val),
      num.threads = nucleos
    )$predictions[, "1"]
    rm(rf)
    invisible(gc())
  }
  mejor <- f1_cv(oof)
  out <- list(
    nombre = nombre, vars = vars, mtry = mtry, min_nodo = min_nodo, prob = oof,
    umbral = mejor$umbral, f1 = mejor$f1, auc = calcular_auc(oof, y),
    minutos = as.numeric(difftime(Sys.time(), t0, units = "mins"))
  )
  saveRDS(out, archivo)
  marca(sprintf(
    "%-28s F1 = %.4f  umbral = %.2f  AUC = %.4f  (%.0f min)", nombre, out$f1,
    out$umbral, out$auc, out$minutos
  ))
  out
}
pruebas_rf <- pmap(grid_rf, function(mtry, min.node.size) {
  cv_rf(
    sprintf("RF_%s_mtry%d_nodo%d", mejor_nuevo$nombre, mtry, min.node.size),
    mejor_nuevo$vars, mtry, min.node.size
  )
})
mejor_rf <- mejor_de(pruebas_rf)
comb_rf <- combinar(mejor_nuevo, mejor_rf)
# Tres modelos: XGBoost, bosque y regresión.
tabla_rf3 <- pmap_dfr(pesos3, function(w1, w2) {
  m <- f1_cv(
    w1 * mejor_nuevo$prob + w2 * mejor_rf$prob + (1 - w1 - w2) * mejor_reg$prob
  )
  tibble(w1 = w1, w2 = w2, umbral = m$umbral, f1 = m$f1)
})
comb_rf3 <- slice_max(tabla_rf3, f1, n = 1, with_ties = FALSE)
p_comb_rf <- comb_rf$w * mejor_nuevo$prob + (1 - comb_rf$w) * mejor_rf$prob
p_comb_rf3 <- comb_rf3$w1 * mejor_nuevo$prob + comb_rf3$w2 * mejor_rf$prob +
  (1 - comb_rf3$w1 - comb_rf3$w2) * mejor_reg$prob
bosque <- tibble(
  modelo = c(
    mejor_nuevo$nombre,
    map_chr(pruebas_rf, ~ sprintf("RF mtry %d nodo %d", .x$mtry, .x$min_nodo)),
    sprintf("XGBoost + bosque (w = %.1f)", comb_rf$w),
    sprintf(
      "XGBoost %.1f + bosque %.1f + regresión %.1f", comb_rf3$w1, comb_rf3$w2,
      1 - comb_rf3$w1 - comb_rf3$w2
    )
  ),
  f1 = c(mejor_nuevo$f1, map_dbl(pruebas_rf, "f1"), comb_rf$f1, comb_rf3$f1),
  f1_honesto = map_dbl(
    c(
      list(mejor_nuevo$prob), map(pruebas_rf, "prob"),
      list(p_comb_rf, p_comb_rf3)
    ),
    umbral_honesto
  )
)
saveRDS(bosque, file.path(dir_out, "bosque.rds"))
marca("Random forest y combinación:")
print(as.data.frame(bosque), digits = 4)

# El bosque final se estima con todo train. Submission del bosque solo (para
# diversificar: es el modelo que más difiere de XGBoost) y de la mejor
# combinación con el bosque, si mejora el F1 honesto del XGBoost solo.
mat_ent <- armar(
  hog_tr, bind_rows(map(1:k, ~ tabla_tarea(paste0("val_", .x)))), mejor_rf$vars
)
mat_te <- armar(hog_te, tabla_tarea("test"), mejor_rf$vars)
rf <- ranger(
  x = as.data.frame(mat_ent), y = factor(y), probability = TRUE,
  num.trees = arboles_rf,
  mtry = mejor_rf$mtry, min.node.size = mejor_rf$min_nodo,
  respect.unordered.factors = "order",
  num.threads = nucleos, seed = 123, verbose = FALSE
)
p_rf_te <- predict(
  rf, as.data.frame(mat_te),
  num.threads = nucleos
)$predictions[, "1"]
rm(rf, mat_ent, mat_te)
invisible(gc())
guardar_submission(
  p_rf_te, mejor_rf$umbral,
  sprintf("mtry%d_nodo%d", mejor_rf$mtry, mejor_rf$min_nodo),
  prefijo = "RFH"
)
f1h_xgb <- umbral_honesto(mejor_nuevo$prob)
if (max(tail(bosque$f1_honesto, 2)) > f1h_xgb) {
  p_xgb_te <- prob_test(mejor_nuevo)
  if (tail(bosque$f1_honesto, 1) > tail(bosque$f1_honesto, 2)[1]) {
    guardar_submission(
      comb_rf3$w1 * p_xgb_te + comb_rf3$w2 * p_rf_te +
        (1 - comb_rf3$w1 - comb_rf3$w2) * prob_test(mejor_reg),
      comb_rf3$umbral,
      sprintf(
        "comb_%s_RF_R_w%02d_%02d", mejor_nuevo$nombre, round(10 * comb_rf3$w1),
        round(10 * comb_rf3$w2)
      )
    )
  } else {
    guardar_submission(
      comb_rf$w * p_xgb_te + (1 - comb_rf$w) * p_rf_te, comb_rf$umbral,
      sprintf(
        "comb_%s_RF_mtry%d_nodo%d_w%02d", mejor_nuevo$nombre, mejor_rf$mtry,
        mejor_rf$min_nodo, round(10 * comb_rf$w)
      )
    )
  }
}

# 10. Análisis de errores del mejor modelo ------------------------------------

# Dónde se equivoca el mejor XGBoost (predicción fuera de muestra con su
# umbral): por distancia a la línea (z = log(ingreso / línea)), ciudad, tamaño
# del hogar, ocupados, zona, tenencia y sexo del jefe. Si los errores se
# concentran cerca de la línea, son difíciles de separar con cualquier variable.
d_err <- tibble(
  real = y, pred = as.integer(mejor_nuevo$prob > mejor_nuevo$umbral), z = z,
  distancia = cut(z, c(-Inf, -1, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 1, Inf)),
  dominio = train$dominio, personas = pmin(train$n_ug, 6),
  ocupados = pmin(train$n_ocup, 3),
  cabecera = train$cabecera, tenencia = train$tenencia,
  jefe_mujer = train$jefe_mujer
) |>
  mutate(
    error = pred != real, fn = real == 1 & pred == 0, fp = real == 0 & pred == 1
  )
por_grupo <- function(g) {
  d_err |>
    summarize(
      .by = all_of(g), hogares = n(), pct_pobres = 100 * mean(real),
      pct_error = 100 * mean(error),
      falsos_neg = sum(fn), falsos_pos = sum(fp),
      f1 = calcular_f1(pred, real)
    ) |>
    mutate(
      pct_de_errores = 100 * (falsos_neg + falsos_pos) / sum(d_err$error),
      grupo = g, .before = 1
    ) |>
    rename(nivel = all_of(g)) |>
    mutate(nivel = as.character(nivel)) |>
    relocate(grupo, nivel) |>
    arrange(nivel)
}
errores <- map_dfr(
  c(
    "distancia", "personas", "ocupados", "cabecera", "tenencia", "jefe_mujer",
    "dominio"
  ),
  por_grupo
)
cerca <- tibble(
  banda = c(0.1, 0.25, 0.5),
  pct_hogares = map_dbl(banda, ~ 100 * mean(abs(z) < .x)),
  pct_errores = map_dbl(
    banda, ~ 100 * sum(d_err$error & abs(z) < .x) / sum(d_err$error)
  )
)
saveRDS(
  list(modelo = mejor_nuevo$nombre, errores = errores, cerca = cerca),
  file.path(dir_out, "errores.rds")
)
marca("Errores del mejor modelo", mejor_nuevo$nombre, "por grupo:")
print(as.data.frame(errores), digits = 3)
marca("Errores cerca de la línea (|log(ingreso / línea)| < banda):")
print(as.data.frame(cerca), digits = 3)

# 11. Modelo especialista para los hogares dudosos ----------------------------

# Segundo modelo, entrenado solo con los hogares a los que el mejor modelo
# (`mejor_nuevo`) da una probabilidad intermedia (su predicción fuera de
# muestra, p1, en `rango`), con las mismas variables más el log odds de p1.
# Vuelve a predecir esos hogares; los demás conservan p1. La zona se escoge
# con la probabilidad predicha, así que se aplica igual en test. Dos formas de
# usar su predicción: reemplazar p1 o promediarla con p1.
p1 <- mejor_nuevo$prob
con_p1 <- function(mat, p) {
  lo <- qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))
  if (is.data.frame(mat)) {
    mat$p1_lo <- lo
    mat
  } else {
    cbind(mat, p1_lo = lo)
  }
}
zonas <- list(z15_70 = c(0.15, 0.70), z10_85 = c(0.10, 0.85))
cv_especialista <- function(nombre, rango) {
  archivo <- file.path(dir_out, paste0(nombre, ".rds"))
  if (file.exists(archivo)) {
    return(readRDS(archivo))
  }
  dudoso <- p1 >= rango[1] & p1 <= rango[2]
  res <- en_paralelo(1:k, function(f) {
    m <- matrices(f, mejor_nuevo$vars)
    ent <- m$i_ent[dudoso[m$i_ent]]
    val <- m$i_val[dudoso[m$i_val]]
    dtr <- xgb.DMatrix(
      con_p1(m$X_ent[dudoso[m$i_ent], , drop = FALSE], p1[ent]),
      label = y[ent],
      missing = NA
    )
    dva <- xgb.DMatrix(
      con_p1(m$X_val[dudoso[m$i_val], , drop = FALSE], p1[val]),
      label = y[val],
      missing = NA
    )
    b <- xgb.train(
      params = mejor_nuevo$params, data = dtr, nrounds = 10000,
      evals = list(val = dva),
      early_stopping_rounds = parada, verbose = 0
    )
    it <- xgb.attr(b, "best_iteration")
    list(
      i = val, pred = predict(b, dva, iterationrange = c(1, it + 1)), iter = it
    )
  }, nucleos = min(nucleos, k), que = paste("el especialista", nombre))
  esp <- rep(NA_real_, length(y))
  for (r in res) esp[r$i] <- r$pred
  probs <- list(
    reemplazo = if_else(dudoso, esp, p1),
    promedio = if_else(dudoso, (esp + p1) / 2, p1)
  )
  out <- list(
    nombre = nombre, rango = rango, n_dudosos = sum(dudoso),
    iters = map_int(res, "iter"),
    pct_pobres_dudosos = 100 * mean(y[dudoso]),
    resultados = imap_dfr(probs, function(p, forma) {
      m <- f1_cv(p)
      tibble(
        forma = forma, f1 = m$f1, umbral = m$umbral,
        f1_honesto = umbral_honesto(p)
      )
    }), probs = probs
  )
  saveRDS(out, archivo)
  out
}
pruebas_esp <- imap(
  zonas,
  ~ cv_especialista(paste0("especialista_", mejor_nuevo$nombre, "_", .y), .x)
)
especialista <- imap_dfr(pruebas_esp, ~ mutate(
  .x$resultados,
  zona = .y, hogares = .x$n_dudosos,
  pct_pobres = .x$pct_pobres_dudosos, .before = 1
)) |>
  add_row(
    zona = "ninguna (mejor modelo)", forma = mejor_nuevo$nombre,
    f1 = mejor_nuevo$f1,
    umbral = mejor_nuevo$umbral, f1_honesto = umbral_honesto(p1), .before = 1
  )
saveRDS(especialista, file.path(dir_out, "especialista.rds"))
marca("Modelo especialista para los hogares dudosos:")
print(as.data.frame(especialista), digits = 4)

# Submission si el especialista mejora el F1 honesto del mejor modelo. El
# especialista final se estima con los hogares dudosos de todo train (p1 fuera
# de muestra); en test, la zona y p1 salen del mejor modelo estimado con todo
# train.
mejor_esp <- slice_max(
  filter(especialista, !is.na(hogares)), f1_honesto,
  n = 1, with_ties = FALSE
)
if (mejor_esp$f1_honesto > umbral_honesto(p1)) {
  rango <- zonas[[mejor_esp$zona]]
  dud <- p1 >= rango[1] & p1 <= rango[2]
  mat_ent <- armar(
    hog_tr, bind_rows(map(1:k, ~ tabla_tarea(paste0("val_", .x)))),
    mejor_nuevo$vars
  )
  mat_te <- armar(hog_te, tabla_tarea("test"), mejor_nuevo$vars)
  b <- xgb.train(
    params = mejor_nuevo$params,
    nrounds = round(mean(pruebas_esp[[mejor_esp$zona]]$iters) * 1.1),
    verbose = 0,
    data = xgb.DMatrix(
      con_p1(mat_ent[dud, , drop = FALSE], p1[dud]),
      label = y[dud],
      missing = NA
    )
  )
  p1_te <- prob_test(mejor_nuevo)
  dud_te <- p1_te >= rango[1] & p1_te <= rango[2]
  esp_te <- predict(
    b,
    xgb.DMatrix(
      con_p1(mat_te[dud_te, , drop = FALSE], p1_te[dud_te]),
      missing = NA
    )
  )
  p_te <- p1_te
  p_te[dud_te] <-
    if (mejor_esp$forma == "reemplazo") esp_te else (esp_te + p1_te[dud_te]) / 2
  guardar_submission(
    p_te, mejor_esp$umbral,
    sprintf(
      "%s_especialista_%s_%s", mejor_nuevo$nombre, mejor_esp$zona,
      mejor_esp$forma
    )
  )
}

# 12. Ruido: el mejor modelo con otras particiones en folds -------------------

# ¿La mejora de las preguntas no usadas depende de la partición? Con otras dos
# particiones (cada una recalcula todas las variables por fold, ~50 min) se
# estiman el mejor modelo antes de 5.15 (`ref`), el mejor antes de las
# preguntas no usadas (`ref2`) y el mejor modelo final, y se comparan dentro de
# cada partición. Solo con categóricas nativas (con códigos está 5.10).
if (!completo) {
  modelos_ruido <- list(
    antes_5_15 = ref, antes_preguntas = ref2, final = mejor_nuevo
  )
  f1_particion <- function(s) {
    imap_dfr(modelos_ruido, function(r, cual) {
      x <- if (s == 123) {
        r
      } else {
        cv_hogar(sprintf("ruido_s%d_%s", s, r$nombre), r$vars,
          base = r$base, params = r$params,
          nrounds = r$nrounds %||% nrounds_max, pesos = r$pesos
        )
      }
      tibble(
        particion = s, modelo = cual, prueba = r$nombre, f1 = x$f1,
        f1_honesto = umbral_honesto(x$prob)
      )
    })
  }
  ruido2 <- f1_particion(123)
  for (s in c(456, 789)) {
    cambiar_particion(s)
    ruido2 <- bind_rows(ruido2, f1_particion(s))
  }
  cambiar_particion(123)
  ruido2 <- ruido2 |>
    group_by(particion) |>
    mutate(
      mejora_sobre_antes_preguntas = f1 - f1[modelo == "antes_preguntas"]
    ) |>
    ungroup()
  saveRDS(ruido2, file.path(dir_out, "ruido_final.rds"))
  marca("F1 del mejor modelo con otras particiones:")
  print(as.data.frame(ruido2), digits = 4)
}

# 13. Combinación con el logit lasso (08) -------------------------------------

# El logit lasso con interacciones (08_logit_interacciones.R) usa la misma
# base por hogar y los mismos 5 folds, así que sus probabilidades fuera de
# muestra se pueden promediar con las de XGBoost. Es un modelo muy distinto
# (lineal en sus 1.159 columnas), así que comete errores distintos. Dos
# combinaciones:
# XGBoost + logit y XGBoost + regresión + logit, con pesos y umbral por F1.
archivo_logit <- here(
  "02_outputs", "logs", "logit_interacciones",
  if (prueba) "logit_oof_prueba.rds" else "logit_oof.rds"
)
if (file.exists(archivo_logit)) {
  lg <- readRDS(archivo_logit)
  stopifnot(all(train$id %in% lg$id_train), all(test$id %in% lg$id_test))
  logit <- c(
    list(
      nombre = "logit_main", prob = lg$prob_cv[match(train$id, lg$id_train)]
    ),
    f1_cv(lg$prob_cv[match(train$id, lg$id_train)])
  )
  comb_lg <- combinar(mejor_nuevo, logit, pesos = seq(0.5, 1, by = 0.05))
  pesos_lg3 <- expand_grid(w1 = seq(0.4, 1, 0.05), w2 = seq(0, 0.6, 0.05)) |>
    filter(w1 + w2 <= 1 + 1e-9)
  tabla_lg3 <- pmap_dfr(pesos_lg3, function(w1, w2) {
    m <-
      f1_cv(
        w1 * mejor_nuevo$prob + w2 * reg_comb$prob + (1 - w1 - w2) * logit$prob
      )
    tibble(w1 = w1, w2 = w2, umbral = m$umbral, f1 = m$f1)
  })
  comb_lg3 <- slice_max(tabla_lg3, f1, n = 1, with_ties = FALSE)
  # peso de XGBoost en XGBoost + regresión, con el mejor modelo actual
  w_reg <- combinar(mejor_nuevo, reg_comb)$w
  probs_lg <- list(
    xgboost = mejor_nuevo$prob, logit = logit$prob,
    xgboost_regresion = w_reg * mejor_nuevo$prob + (1 - w_reg) * reg_comb$prob,
    xgboost_logit = comb_lg$w * mejor_nuevo$prob + (1 - comb_lg$w) * logit$prob,
    xgboost_regresion_logit = comb_lg3$w1 * mejor_nuevo$prob +
      comb_lg3$w2 * reg_comb$prob +
      (1 - comb_lg3$w1 - comb_lg3$w2) * logit$prob
  )
  con_logit <- tibble(
    modelo = names(probs_lg),
    pesos = c(
      NA, NA, sprintf("%.2f / %.2f", w_reg, 1 - w_reg),
      sprintf("%.2f / %.2f", comb_lg$w, 1 - comb_lg$w),
      sprintf(
        "%.2f / %.2f / %.2f", comb_lg3$w1, comb_lg3$w2,
        1 - comb_lg3$w1 - comb_lg3$w2
      )
    ),
    f1 = map_dbl(probs_lg, ~ f1_cv(.x)$f1),
    umbral = map_dbl(probs_lg, ~ f1_cv(.x)$umbral),
    f1_honesto = map_dbl(probs_lg, umbral_honesto),
    auc = map_dbl(probs_lg, ~ calcular_auc(.x, y))
  )
  saveRDS(con_logit, file.path(dir_out, "combinacion_logit.rds"))
  marca("Combinación con el logit de main:")
  print(as.data.frame(con_logit), digits = 4)
  # Submission de la mejor combinación con el logit, si su F1 honesto es mayor
  # que el de XGBoost solo y que el de XGBoost + regresión.
  mejor_lg <- con_logit |>
    filter(str_detect(modelo, "logit$"), modelo != "logit") |>
    slice_max(f1_honesto, n = 1, with_ties = FALSE)
  if (mejor_lg$f1_honesto > max(con_logit$f1_honesto[con_logit$modelo %in% c(
    "xgboost", "xgboost_regresion"
  )])) {
    p_lg_te <- lg$prob_test[match(test$id, lg$id_test)]
    p_xgb_te <- prob_test(mejor_nuevo)
    if (mejor_lg$modelo == "xgboost_logit") {
      guardar_submission(
        comb_lg$w * p_xgb_te + (1 - comb_lg$w) * p_lg_te, mejor_lg$umbral,
        sprintf(
          "comb_%s_logit_w%03d", mejor_nuevo$nombre, round(100 * comb_lg$w)
        )
      )
    } else {
      guardar_submission(
        comb_lg3$w1 * p_xgb_te + comb_lg3$w2 * prob_test(reg_comb) +
          (1 - comb_lg3$w1 - comb_lg3$w2) * p_lg_te,
        mejor_lg$umbral, sprintf(
          "comb_%s_R_logit_w%03d_%03d", mejor_nuevo$nombre,
          round(100 * comb_lg3$w1), round(100 * comb_lg3$w2)
        )
      )
    }
  }
}

# 14. Importancia por permutación del mejor modelo ----------------------------

# ¿Cuánto cae el F1 si se rompe la relación de una variable con pobre? Para el
# mejor modelo (mejor_nuevo combinado con la regresión reg_comb, como en la
# sección 6): en cada fold se vuelven a estimar el clasificador y la regresión
# con los otros 4 (mismos hiperparámetros, base, pesos y rondas que en la CV,
# así que dan las mismas predicciones fuera de muestra; se verifica), se
# permutan al azar los valores de la variable entre los hogares del fold de
# validación y se mide el F1 con el umbral de la combinación. Importancia =
# F1 original - F1 con la variable permutada, promedio de los 5 folds y de 3
# permutaciones. Las variables muy correlacionadas (p. ej. las versiones del
# ingreso predicho) se reemplazan entre sí, y permutadas por separado pierden
# poca importancia; por eso también se permutan grupos (la misma permutación
# para todas sus columnas, que conserva la relación entre ellas).
ubicacion <- c("dominio", "depto", "cabecera", "log_lp")
grupos_imp <- list(
  "Ingreso predicho por persona (todas las versiones)" =
    c(
      cols$V4, "ing_pred_por_ocup", unlist(cols[paste0("V4", sufijos_v4)]),
      setdiff(cols$A, "arr_imp_pc_lp"), "ing_arr_pc_lp_w"
    ),
  "Probabilidad de ser pobre por persona (stacking)" = c(
    cols$P, cols$Pt, cols$Pe, cols$Pa
  ),
  "Probabilidad de ingreso menor a 0,8 y 1,25 líneas" = c(cols$C080, cols$C125),
  "Tasa de pobreza del oficio" = cols$V3,
  "Ubicación (dominio, departamento, zona, línea)" = ubicacion,
  "Base por hogar de 01b (sin ubicación)" = setdiff(x_base, ubicacion),
  "Otras construidas (expansión, mejor ocupado, productos, razones)" =
    c(
      cols$V1, cols$V2, cols$nuestras, cols$N1,
      setdiff(cols$N2, "ing_pred_por_ocup"), cols$N3, "arr_imp_pc_lp"
    )
)
grupo_de <- function(v) {
  g <- names(grupos_imp)[map_lgl(grupos_imp, ~ v %in% .x)]
  if (length(g)) g[1] else "Otras"
}
comb_final <- combinar(mejor_nuevo, reg_comb)
repeticiones <- 3
archivo_perm <- file.path(
  dir_out, paste0("importancia_permutacion_", mejor_nuevo$nombre, ".rds")
)
# Modelo de `res` estimado sin el fold f (como en cv_hogar(); también en 15).
modelo_fold <- function(res, f) {
  m <- matrices(f, res$vars)
  dtr <- dmatrix_ent(m$X_ent, m$i_ent, res$base, res$pesos, res$regresion)
  list(
    b = xgb.train(
      params = res$params, data = dtr, nrounds = res$iters[f] + 1, verbose = 0
    ),
    X_val = m$X_val, i_val = m$i_val, rango = c(1, res$iters[f] + 1)
  )
}
# Permuta las filas de las columnas `columnas` de X (data.frame o matriz).
permutar <- function(mat, columnas, perm) {
  for (v in intersect(columnas, colnames(mat))) {
    if (is.data.frame(mat)) {
      mat[[v]] <- mat[[v]][perm]
    } else {
      mat[
        ,
        v
      ] <- mat[perm, v]
    }
  }
  mat
}
if (!file.exists(archivo_perm)) {
  conjuntos <- c(as.list(set_names(mejor_nuevo$vars)), grupos_imp)
  t0 <- Sys.time()
  res_perm <- en_paralelo(1:k, function(f) {
    mc <- modelo_fold(mejor_nuevo, f)
    mr <- modelo_fold(reg_comb, f)
    stopifnot(identical(mc$i_val, mr$i_val))
    predecir <- function(x_c, x_r) {
      pc <- predict(
        mc$b, xgb.DMatrix(x_c, missing = NA),
        iterationrange = mc$rango
      )
      pr <- predict(
        mr$b, xgb.DMatrix(x_r, missing = NA),
        iterationrange = mr$rango
      )
      comb_final$w * pc +
        (1 - comb_final$w) * plogis(reg_comb$cal[1] + reg_comb$cal[2] * pr)
    }
    real <- y[mc$i_val]
    base <- predecir(mc$X_val, mr$X_val)
    # Debe reproducir la predicción fuera de muestra de la combinación.
    oof_comb <- comb_final$w * mejor_nuevo$prob[mc$i_val] +
      (1 - comb_final$w) * reg_comb$prob[mc$i_val]
    dif <- max(abs(base - oof_comb))
    f1_base <- calcular_f1(as.integer(base > comb_final$umbral), real)
    map_dfr(seq_len(repeticiones), function(r) {
      set.seed(1000 * r + f)
      perm <- sample(length(real))
      imap_dfr(conjuntos, function(columnas, nombre) {
        p <- predecir(
          permutar(mc$X_val, columnas, perm), permutar(mr$X_val, columnas, perm)
        )
        tibble(
          nombre = nombre, fold = f, repeticion = r,
          caida = f1_base - calcular_f1(as.integer(p > comb_final$umbral), real)
        )
      })
    }) |> mutate(f1_base = f1_base, dif_oof = dif)
    # 3: con 5 procesos el Mac (8 GB) se quedó sin memoria
  }, nucleos = 3, que = "la importancia por permutación")
  res_perm <- bind_rows(res_perm)
  if (max(res_perm$dif_oof) > 1e-4) {
    stop("Los modelos por fold no reproducen la predicción fuera de muestra")
  }
  saveRDS(res_perm, archivo_perm)
  marca(sprintf(
    "Importancia por permutación lista (%.0f min)",
    as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ))
}
res_perm <- readRDS(archivo_perm)
importancia_perm <- res_perm |>
  summarize(.by = nombre, de = sd(caida), caida = mean(caida)) |>
  mutate(
    tipo = if_else(nombre %in% names(grupos_imp), "grupo", "variable"),
    grupo = if_else(tipo == "grupo", nombre, map_chr(nombre, grupo_de))
  ) |>
  arrange(desc(caida))
# Ganancia de XGBoost (clasificador) por grupo, para comparar.
ganancia_grupo <- mejor_nuevo$importancia |>
  mutate(grupo = map_chr(variable, grupo_de)) |>
  summarize(.by = grupo, ganancia = sum(importancia), variables = n())
saveRDS(
  list(
    importancia = importancia_perm, ganancia_grupo = ganancia_grupo,
    f1_base = mean(distinct(res_perm, fold, f1_base)$f1_base),
    umbral = comb_final$umbral, w = comb_final$w
  ),
  file.path(dir_out, "importancia_permutacion.rds")
)
marca("Importancia por permutación (caída del F1):")
print(as.data.frame(head(importancia_perm, 30)), digits = 3)

# Las tablas para el documento las arma 15_comparacion_modelos.R (sección 7)
# con importancia_permutacion.rds.

# 15. Importancia de extremo a extremo por concepto ---------------------------

# La sección 14 permuta las variables a la entrada del modelo por hogar: si se
# permuta la educación del hogar, los modelos auxiliares (ingreso predicho,
# probabilidad por persona, cortes, tasa del oficio) siguen viendo la
# educación de cada persona y se quedan con el crédito. Aquí se permuta cada
# concepto en toda la cadena: sus variables de las personas (las que entran a
# los modelos auxiliares) y del hogar (las que entran al clasificador y a la
# regresión), se vuelven a calcular las variables de los modelos auxiliares y
# después la predicción final, y se mide la caída del F1 (con el umbral de la
# combinación). Tres formas:
#   total: se permutan las variables de las personas y las del hogar;
#   directa: solo las del hogar (los modelos auxiliares ven los datos reales);
#   vía modelos: solo las de las personas (efecto a través de los auxiliares).
# En cada fold se vuelven a estimar los modelos auxiliares de la tarea de
# validación y los dos modelos finales, con sus semillas (reproducen la
# predicción fuera de muestra; se verifica). Una permutación por fold (la
# misma en las tres formas); el promedio es de los 5 folds.
pers_beneficios <- c(
  "recibio_horas_extra", "recibio_primas", "recibio_bonificaciones",
  grep("^(prima_|subsidio_)", colnames(mat_pers), value = TRUE),
  "pago_especie_alimentos", "pago_especie_vivienda", "transporte_empresa",
  "pago_especie_otros", "viaticos_permanentes", "bonificaciones_anuales"
)
conceptos <- list(
  "Educación" = list(
    pers = c("max_educ", "anos_educ_imp", "educ_max_adultos"),
    hog = c(
      "educ_max", "educ_media_adultos", "educ_ocup_media", "educ_ocup_suma_pc",
      "prop_superior", "n_sin_educ",
      "n_ocup_superior", "superior_formal_pc", "jefe_educ", "educ_max_adultos",
      "cv_educ_adultos",
      "ocup_max_educ", "brecha_educ_jefe", "rel_dom_educ_max",
      "rel_dom_educ_media", "educ_adultos_x_ocup_pc",
      "educ_ocup_x_formal_pc", "jefe_educ_x_formal", "n_estudia"
    )
  ),
  "Ocupación y horas de trabajo" = list(
    pers = c(
      "ocupado", "desocupado", "inactivo", "actividad_sem_pasada",
      "horas_trabajo_semana",
      "horas_segundo_trabajo", "segundo_trabajo", "log_meses_empresa",
      "prop_ocupados_hogar"
    ),
    hog = c(
      "n_ocup", "ocup_pc", "ocup_por_adulto", "n_desocup", "desocup_pc",
      "n_inactivo_adulto", "horas_pc",
      "horas_formal_pc", "horas_informal_pc", "n_segundo_trab", "n_pareja_ocup",
      "n_hijos_ocup",
      "n_oficios_hogar", "max_log_meses", "jefe_ocup", "jefe_desocup",
      "jefe_log_meses", "ocup_max_horas",
      "cv_horas_ocupados", "dep_por_ocup", "horas_por_ocup", "rel_dom_horas_pc",
      "rel_dom_ocup_pc",
      "ocup_pc_x_log_lp", "horas_pc_x_formal", "educ_adultos_x_ocup_pc"
    )
  ),
  "Formalidad, posición y oficio" = list(
    pers = c(
      "posicion_ocupacional", "tamano_empresa", "oficio", "cotiza_pension"
    ),
    hog = c(
      "n_formal", "formal_pc", "prop_formal_ocup", "horas_formal_pc",
      "horas_informal_pc", "n_asalariado",
      "n_gobierno", "n_domestico", "n_cuenta_propia", "n_patron", "n_sin_pago",
      "n_jornalero",
      "n_empresa_sola", "n_empresa_peq", "n_empresa_grande",
      grep("^n_of_", x_base, value = TRUE),
      "jefe_formal", "jefe_posicion", "jefe_tamano", "jefe_oficio",
      "ocup_max_tamano", "ocup_posicion_educ",
      "superior_formal_pc", "educ_ocup_x_formal_pc", "horas_pc_x_formal",
      "jefe_educ_x_formal"
    )
  ),
  "Salud" = list(
    pers = c("regimen_salud", "afiliado_salud"),
    hog = c(
      "prop_contributivo", "prop_subsidiado", "n_especial", "n_no_afiliado",
      "jefe_regimen"
    )
  ),
  "Primas, subsidios y pagos en especie" = list(
    pers = pers_beneficios,
    hog = c(
      "beneficios_pc", "n_con_beneficios", "n_horas_extra",
      "beneficios_por_ocup"
    )
  ),
  "Ingresos no laborales" = list(
    pers = c(
      "recibio_arriendos_pensiones", "recibio_pension_jubilacion",
      "recibio_pension_alimenticia",
      "recibio_otros_ingresos",
      grep("^dinero_", colnames(mat_pers), value = TRUE),
      "ayudas_instituciones"
    ),
    hog = c(grep("^any_", x_base, value = TRUE), "n_fuentes_no_lab")
  ),
  "Búsqueda de trabajo e ingresos de desocupados" = list(
    pers = c(
      "posicion_segundo_trabajo", "quiere_mas_horas", "diligencias_mas_horas",
      "disponible_mas_horas",
      "diligencias_cambiar_trabajo", "disponible_nuevo_trabajo",
      "primera_vez_busca_trabajo",
      "posicion_ultimo_trabajo", "ingreso_trabajo_desocupado",
      "ingreso_trabajo_desocupado_2"
    ),
    hog = c("n_quiere_mas_horas")
  ),
  "Vivienda y arriendo" = list(
    pers = c("arriendo_pc_sobre_lp", "tenencia_vivienda", "arr_imp_pc_lp"),
    hog = c(
      "tenencia", "cuartos_pc", "pers_por_dormitorio", "log_arriendo_pc_lp",
      "arriendo_na",
      "arriendo_es_pagado", "log_cuota_pc_lp", "arriendo_x_log_lp",
      "rel_dom_log_arriendo", "arr_imp_pc_lp",
      "arriendo_imputado"
    )
  ),
  "Edad, sexo y tamaño del hogar" = list(
    pers = c("anos", "mujer", "parentesco_jefe_hogar", "personas_hogar"),
    hog = c(
      "n_ug", "log_n_ug", "n_menor6", "n_menor12", "n_menor18", "n_mayor65",
      "dependencia", "prop_menores",
      "prop_mujer_ug", "jefe_edad", "jefe_edad2", "jefe_mujer",
      "jefe_mujer_sola", "dep_por_ocup"
    )
  ),
  "Ubicación y línea de pobreza" = list(
    pers = c("dominio", "depto", "cabecera", "linea_pobreza"),
    hog = c(
      "dominio", "depto", "cabecera", "log_lp", "fex_c", "fex_dpto",
      "ocup_pc_x_log_lp", "arriendo_x_log_lp"
    )
  )
)
stopifnot(
  all(unlist(map(conceptos, "pers")) %in% colnames(mat_pers)),
  all(unlist(map(conceptos, "hog")) %in% names(hog_tr))
)
# Modelos auxiliares que usan el clasificador o la regresión (V4 y P
# originales siempre; V3 en predecir_final()).
vars_finales <- union(mejor_nuevo$vars, reg_comb$vars)
usa <- function(columnas) any(columnas %in% vars_finales)
confs_aux_v4 <- c(
  list(original = conf_v4_original),
  keep(confs_v4, ~ usa(cols[[paste0("V4", sub("_", "", .x$sufijo))]]))
)
confs_aux_p <- c(
  list(original = conf_p_original), if (usa(cols$PI)) list(ind = conf_ind),
  keep(confs_p, ~ usa(paste0(.x$prefijo, "_", resumen_p)))
)
# Los índices supervisados (IS) son logits del hogar: se recalculan con las
# variables del hogar permutadas. Los índices sin supervisar (I) quedan como
# están (no se recalculan).
usa_is <- usa(cols$IS)

# Cadena de modelos del fold f (secciones 15 y 16): estima los modelos
# auxiliares de la tarea de validación y toma los dos modelos finales.
# predecir_final() da la probabilidad combinada con predictores de las
# personas Xp (y su oficio) y del hogar hv, pasando por los modelos auxiliares.
cadena_fold <- function(f) {
  tarea <- tareas[[paste0("val_", f)]]
  filas <- filas_pred(tarea)
  v3 <- ajustar_v3(tarea)
  mods_v4 <- map(confs_aux_v4, ~ ajustar_v4(tarea, .x))
  mods_p <- map(confs_aux_p, ~ ajustar_pc(tarea, .x))
  mods_is <- if (usa_is) ajustar_isup(tarea)
  mc <- modelo_fold(mejor_nuevo, f)
  mr <- modelo_fold(reg_comb, f)
  mc$X_val <- mr$X_val <- NULL
  invisible(gc())
  real <- y[mc$i_val]
  predecir_final <- function(x_p, oficio, hv) {
    tablas <- c(
      map2(mods_v4, confs_aux_v4, ~ predecir_v4(.x, .y, filas, x_p)$hogar),
      map2(mods_p, confs_aux_p, ~ predecir_pc(.x, .y, filas, x_p)),
      if (usa_is) list(predecir_isup(mods_is, hv))
    )
    tabla <- unir_tablas(tablas, predecir_v3(v3, filas, oficio), v3$media)
    pc <- predict(
      mc$b, xgb.DMatrix(armar(hv, tabla, mejor_nuevo$vars), missing = NA),
      iterationrange = mc$rango
    )
    pr <- predict(
      mr$b, xgb.DMatrix(armar(hv, tabla, reg_comb$vars), missing = NA),
      iterationrange = mr$rango
    )
    comb_final$w * pc +
      (1 - comb_final$w) * plogis(reg_comb$cal[1] + reg_comb$cal[2] * pr)
  }
  f1_de <- function(p) calcular_f1(as.integer(p > comb_final$umbral), real)
  x_p0 <- mat_pers[filas, ]
  of0 <- pers$oficio[filas]
  hv0 <- hog_tr[mc$i_val, ]
  base <- predecir_final(x_p0, of0, hv0)
  # Diferencia con la predicción fuera de muestra (se verifica que sea ~0).
  oof_comb <- comb_final$w * mejor_nuevo$prob[mc$i_val] +
    (1 - comb_final$w) * reg_comb$prob[mc$i_val]
  dif <- max(abs(base - oof_comb))
  list(
    predecir_final = predecir_final, f1_de = f1_de, f1_base = f1_de(base),
    dif = dif,
    Xp0 = x_p0, of0 = of0, hv0 = hv0, n_pers = length(filas),
    n_hog = length(real)
  )
}

archivo_e2e <- file.path(
  dir_out, paste0("importancia_conceptos_", mejor_nuevo$nombre, ".rds")
)
if (!completo && !file.exists(archivo_e2e)) {
  t0 <- Sys.time()
  res_e2e <- en_paralelo(1:k, function(f) {
    cad <- cadena_fold(f)
    imap_dfr(conceptos, function(cpt, nombre) {
      set.seed(100 * f + match(nombre, names(conceptos)))
      perm_p <- sample(cad$n_pers)
      perm_h <- sample(cad$n_hog)
      x_p <- permutar(cad$Xp0, cpt$pers, perm_p)
      of <- if ("oficio" %in% cpt$pers) cad$of0[perm_p] else cad$of0
      hv <- permutar(cad$hv0, cpt$hog, perm_h)
      tibble(
        concepto = nombre, fold = f,
        total = cad$f1_base - cad$f1_de(cad$predecir_final(x_p, of, hv)),
        directa = cad$f1_base - cad$f1_de(cad$predecir_final(
          cad$Xp0, cad$of0, hv
        )),
        via_modelos = cad$f1_base - cad$f1_de(cad$predecir_final(
          x_p, of, cad$hv0
        ))
      )
    }) |> mutate(f1_base = cad$f1_base, dif_oof = cad$dif)
    # 2: cada proceso guarda ~14 modelos y los datos del fold
  }, nucleos = 2, que = "la importancia por concepto")
  res_e2e <- bind_rows(res_e2e)
  if (max(res_e2e$dif_oof) > 1e-4) {
    stop(paste0(
      "La cadena de modelos por fold no reproduce la predicción fuera de ",
      "muestra"
    ))
  }
  saveRDS(res_e2e, archivo_e2e)
  marca(sprintf(
    "Importancia por concepto lista (%.0f min)",
    as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ))
}
if (!completo) {
  importancia_conceptos <- readRDS(archivo_e2e) |>
    pivot_longer(
      c(total, directa, via_modelos),
      names_to = "forma", values_to = "caida"
    ) |>
    summarize(.by = c(concepto, forma), de = sd(caida), caida = mean(caida)) |>
    pivot_wider(names_from = forma, values_from = c(caida, de)) |>
    mutate(
      variables_personas = map_int(concepto, ~ length(conceptos[[.x]]$pers)),
      variables_hogar = map_int(
        concepto,
        ~ length(intersect(
          conceptos[[.x]]$hog, union(mejor_nuevo$vars, "arriendo_imputado")
        ))
      )
    ) |>
    arrange(desc(caida_total))
  saveRDS(
    list(importancia = importancia_conceptos, conceptos = conceptos),
    file.path(dir_out, "importancia_conceptos.rds")
  )
  marca("Importancia de extremo a extremo por concepto (caída del F1):")
  print(as.data.frame(importancia_conceptos), digits = 3)
}

# 16. Importancia de extremo a extremo por variable de la persona -------------

# Como la sección 15, pero una variable de la persona a la vez (no un
# concepto): se permuta en las personas y pasa por toda la cadena de modelos.
# Solo las variables de la persona de los 5 conceptos con mayor caída total
# (los demás caen 0,05 o menos en total). Las variables del hogar construidas
# con ella no se permutan: mide el efecto a través de los modelos auxiliares,
# que en los conceptos importantes es casi todo (subestima las que pesan sobre
# todo en el hogar, como la tenencia de la vivienda).
# Un fold a la vez: con 2 procesos la memoria no alcanza (swap) y la sección 15
# tardó 3 horas. Cada fold se guarda aparte y se retoma si se interrumpe.
if (!completo) {
  variables_e2e <- imap_dfr(
    conceptos[head(importancia_conceptos$concepto, 5)],
    ~ tibble(variable = .x$pers, concepto = .y)
  ) |>
    distinct(variable, .keep_all = TRUE)
  archivo_var <- file.path(
    dir_out, paste0("importancia_variables_e2e_", mejor_nuevo$nombre, ".rds")
  )
  t0 <- Sys.time()
  res_var <- map_dfr(1:k, function(f) {
    archivo_f <- sub("\\.rds$", paste0("_fold", f, ".rds"), archivo_var)
    if (file.exists(archivo_f)) {
      return(readRDS(archivo_f))
    }
    cad <- cadena_fold(f)
    r <- pmap_dfr(variables_e2e, function(variable, concepto) {
      set.seed(1000 * f + match(variable, variables_e2e$variable))
      perm_p <- sample(cad$n_pers)
      x_p <- permutar(cad$Xp0, variable, perm_p)
      of <- if (variable == "oficio") cad$of0[perm_p] else cad$of0
      tibble(
        variable = variable, concepto = concepto, fold = f,
        caida = cad$f1_base - cad$f1_de(cad$predecir_final(x_p, of, cad$hv0))
      )
    }) |> mutate(f1_base = cad$f1_base, dif_oof = cad$dif)
    if (max(r$dif_oof) > 1e-4) {
      stop(paste0(
        "La cadena de modelos por fold no reproduce la predicción fuera de ",
        "muestra"
      ))
    }
    saveRDS(r, archivo_f)
    marca(sprintf(
      "Importancia por variable: fold %d listo (%.0f min)", f,
      as.numeric(difftime(Sys.time(), t0, units = "mins"))
    ))
    rm(cad)
    invisible(gc())
    r
  })
  importancia_var_e2e <- res_var |>
    summarize(
      .by = c(variable, concepto), de = sd(caida), caida = mean(caida)
    ) |>
    arrange(desc(caida))
  saveRDS(
    importancia_var_e2e, file.path(dir_out, "importancia_variables_e2e.rds")
  )
  marca(paste0(
    "Importancia de extremo a extremo por variable de la persona (caída del ",
    "F1):"
  ))
  print(as.data.frame(importancia_var_e2e), digits = 3)
}

marca("=== TERMINADO")

# Notas -----------------------------------------------------------------------

# F1 de validación cruzada con los 5 folds por hogar. Diferencias de menos de
# ~0.001 entre modelos son ruido: con 4 particiones distintas en folds el F1
# del mismo modelo varía 0.0003 (sd). El umbral escogido con los mismos folds
# en que se mide el F1 es optimista en 0.0005 o menos (umbral honesto, 5.9).

# Nivel hogar: es el mayor salto. XGBoost por persona (10, x3) da 0.706; la
# base de 02 sola (H0) da 0.7528, igual al LightGBM de 12 (0.7530).

# Lo que más aportó, en orden:
#   - Modelos auxiliares por persona (V1-V4 de 13 y P): 0.7528 -> 0.7589. El
#     ingreso predicho (V4) y la probabilidad por persona (P) se llevan el 85%
#     de la ganancia.
#   - Categóricas nativas de xgboost en vez de códigos enteros: 0.7598 ->
#     0.7612 (sección 8). La ganancia viene de los modelos por persona (oficio
#     tiene 84 niveles); one-hot da casi lo mismo que nativo.
#   - Las 16 preguntas por persona que no usaba ningún modelo (x_pers_extra:
#     pagos en especie, ingresos de desocupados e inactivos, búsqueda de
#     trabajo): el ingreso predicho pasa de R² por persona 0.942 a 0.982 y el
#     F1 a 0.7715. Identifican a quién tiene ingreso sin estar ocupado. No son
#     fuga: están en test con la misma distribución que en train.
#   - Ingreso predicho con objetivo Tweedie (V4w): con el log del ingreso la
#     predicción se acerca a la mediana y la suma del hogar se subestima;
#     Tweedie predice la media. R² por hogar 0.629 -> 0.657 y F1 0.7772.

# Modelo final: XGBoost con V4w (eta 0.02, profundidad 8, min_child_weight
# 50, colsample 0.5) combinado 70/30 con la regresión del ingreso: F1 0.7775
# (honesto 0.7775), umbral 0.39. Le gana al modelo sin las preguntas no usadas
# por +0.015 en las tres particiones probadas.

# Lo que no aportó (dentro del ruido o peor): base aumentada (0.7536) o sin
# ricos; pesos mayores a los hogares cerca de la línea; variables no lineales
# (N1, N2, N4, T; solo N3 suma algo); índices (I, IS) e indigencia (PI);
# arriendo imputado (A, Pa: el modelo ya cruzaba ingreso, arriendo y
# tenencia); usar la regresión como variable (z_pred) en vez de combinarla;
# normalizar las variables; un umbral por zona; un modelo especialista para
# los hogares dudosos; combinar con random forest (0.7773), naive bayes o el
# logit de 08 (peso óptimo 0). Los hiperparámetros dan una superficie plana
# (±0.001), y dejar solo las 75 variables de más ganancia da lo mismo.

# Errores: el 30% está a menos de 10% de la línea (7% de los hogares) y el 82%
# a menos de 50%; justo al lado de la línea el modelo se equivoca en 40-46% de
# los hogares. La zona rural tiene más errores (15.5% frente a 9.1%) con un F1
# parecido; los hogares de 1-2 personas tienen el F1 más bajo (0.66-0.68).

# Importancia: permutando a la entrada del modelo (sección 14), el ingreso
# predicho es lo que más pesa (sin él el F1 cae 0.42), seguido de las
# probabilidades de ingreso bajo 0.8 y 1.25 líneas (0.061) y la probabilidad
# por persona (0.044). Permutando en toda la cadena (sección 15), lo que más
# pesa es formalidad, posición y oficio (0.226), ingresos no laborales (0.143),
# vivienda y arriendo (0.118) y ocupación y horas (0.117): las variables de
# las personas importan sobre todo a través de los modelos auxiliares, así que
# la sección 14 las subestima.

# Memoria: con 5 procesos en paralelo la sección 14 dejó al Mac (8 GB) sin
# memoria; se corre con 3. La sección 15 tarda ~3 horas.
