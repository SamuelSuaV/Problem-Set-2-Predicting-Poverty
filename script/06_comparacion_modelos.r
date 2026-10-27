###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   06_comparacion_modelos.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Comparar el mejor modelo de cada categoría con la misma
#                      evaluación (los 5 folds por hogar de 05b) y revisar si
#                      predicen distinto para distintas poblaciones (sexo,
#                      formalidad, zona, edad y educación del jefe, tamaño del
#                      hogar, menores y vivienda): errores de exclusión
#                      (pobres no detectados) y de inclusión (no pobres
#                      predichos pobres) por grupo.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Configuración: folds, umbrales, modelos y grupos
# 3. Predicciones fuera de muestra por hogar
#    3.1 Modelos por persona (03-05), con los 5 folds por hogar
#    3.2 Modelos por hogar (05b y logit de main)
# 4. F1 de cada modelo
# 5. Desempeño por grupo de población
# 6. Tablas y figura
# 7. Importancia de las variables del mejor modelo

################################################################################

# Input:  data/train_clean.rds (01); data/train_hogar.rds (01b);
#         output/pruebas/xgb_hogar_nativo/*.rds (05b con CODIFICACION=nativo);
#         output/pruebas/logit/logit_oof.rds (logit de main, ver 05b sección 13).
# Output: output/pruebas/comparacion/: predicciones fuera de muestra de los
#         modelos por persona (.rds) y resultados (resultados.rds);
#         output/tables/modelos_f1.tex , output/tables/equidad_mejor.tex ,
#         output/tables/equidad_brechas.tex ; output/tables/importancia_grupos.tex ,
#         importancia_variables.tex , importancia_directas.tex ,
#         importancia_conceptos.tex , importancia_variables_e2e.tex ;
#         output/figures/equidad_errores.png .
#
# Correr desde la raíz, después de 05b y del logit de main. Los modelos por
# persona tardan ~20 minutos y se guardan; si ya existen, se leen.

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos y gráficas.
  e1071,      # Naive bayes (03).
  rpart,      # CART (04).
  xgboost,    # XGBoost (05).
  parallel,   # Folds en paralelo (fork).
  kableExtra  # Exportar tablas a LaTeX.
)
# Variables y funciones comunes (categoricas, x1, preparar, prob_hogar,
# calcular_f1, evaluar_umbral).
source(here("script", "00_master_file.r"))

dir_out <- here("output", "pruebas", "comparacion")
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
dir_05b <- here("output", "pruebas", "xgb_hogar_nativo")

################################################################################

## 2. Configuración: folds, umbrales, modelos y grupos

# Hogares de train en el orden de train_hogar.rds y los 5 folds de main y 05b
# (una fila = un hogar). Todos los modelos se evalúan con estos folds; en
# 03-05 se usaron 10 folds, así que su F1 cambia un poco (se reporta también
# el de su script).
hog <- read_rds(here("data", "train_hogar.rds"))
y   <- hog$pobre
k   <- 5
set.seed(123)
fold <- sample(rep(1:k, length.out = nrow(hog)))
grid_umbral <- seq(0.10, 0.85, by = 0.01)
nucleos <- 5

# Grupos de población (a nivel hogar, con las variables del jefe de 01b).
grupos <- hog |>
  transmute(
    `Sexo del jefe` = if_else(jefe_mujer == 1, "Mujer", "Hombre"),
    `Empleo del jefe` = case_when(jefe_ocup != 1 ~ "No ocupado",
                                  jefe_formal == 1 ~ "Formal (cotiza a pensión)",
                                  .default = "Informal"),
    Zona = if_else(cabecera == 1, "Cabecera", "Resto (rural)"),
    `Edad del jefe` = cut(jefe_edad, c(-Inf, 29, 44, 64, Inf),
                          labels = c("Menos de 30", "30 a 44", "45 a 64", "65 o más")),
    `Educación del jefe` = case_when(jefe_educ <= 5 ~ "Primaria o menos",
                                     jefe_educ <= 11 ~ "Secundaria o media",
                                     .default = "Superior"),
    `Tamaño del hogar` = cut(n_ug, c(0, 2, 4, Inf), labels = c("1 o 2 personas", "3 o 4 personas", "5 o más")),
    Menores = if_else(n_menor18 > 0, "Con menores de 18", "Sin menores de 18"),
    Vivienda = case_when(tenencia == 3 ~ "Arriendo",
                         tenencia %in% c(1, 2) ~ "Propia",
                         .default = "Usufructo u otra")
  ) |>
  mutate(across(everything(), as.character))
# Orden de los grupos en tablas y figura.
orden_niveles <- c("Hombre", "Mujer", "Formal (cotiza a pensión)", "Informal", "No ocupado",
                   "Cabecera", "Resto (rural)", "Menos de 30", "30 a 44", "45 a 64", "65 o más",
                   "Primaria o menos", "Secundaria o media", "Superior", "1 o 2 personas",
                   "3 o 4 personas", "5 o más", "Con menores de 18", "Sin menores de 18",
                   "Propia", "Arriendo", "Usufructo u otra")
stopifnot(all(unlist(map(grupos, unique)) %in% orden_niveles))

################################################################################

## 3. Predicciones fuera de muestra por hogar

### 3.1 Modelos por persona (03-05), con los 5 folds por hogar ----------------

# Cada uno con su mejor configuración (la de su script) y la predicción por
# hogar de 03-05. Se estima con los hogares de 4 folds y predice el quinto.
pers <- read_rds(here("data", "train_clean.rds"))
stopifnot(all(pers$id %in% hog$id), all(hog$id %in% pers$id))
fold_p <- fold[match(pers$id, hog$id)]

# Variables: x1 (00_master_file.r; xtodas de 03) y x3 de 05.
x3 <- c(x1, "educ_max_adultos", "linea_pobreza", "cuartos_dormir",
        "prop_dependientes", "personas_hogar", "dominio", "depto", "cabecera")

# Corre `ajustar(f)` (devuelve la probabilidad por persona del fold f) en los
# 5 folds (`procesos` a la vez) y la resume por hogar (prob_hogar(), en el
# orden de los hogares de hog). Guarda el resultado y,
# si ya existe, lo lee.
oof_persona <- function(nombre, ajustar, procesos = min(nucleos, k)) {
  archivo <- file.path(dir_out, paste0(nombre, ".rds"))
  if (!file.exists(archivo)) {
    t0 <- Sys.time()
    res <- mclapply(1:k, ajustar, mc.cores = procesos, mc.preschedule = FALSE)
    if (any(map_lgl(res, ~ inherits(.x, "try-error")))) stop("Falló un fold de ", nombre)
    p <- rep(NA_real_, nrow(pers))
    for (f in 1:k) p[fold_p == f] <- res[[f]]
    h <- prob_hogar(p, pers$id)
    saveRDS(h$prob[match(hog$id, h$id)], archivo)
    cat(nombre, "listo en", round(difftime(Sys.time(), t0, units = "mins"), 1), "minutos\n")
  }
  readRDS(archivo)
}

# Naive bayes (03): x1, laplace = 0; factores para las categóricas, el texto
# y las 0/1.
x_nb <- preparar(select(pers, all_of(x1)), binarias = TRUE)
p_nb <- oof_persona("naive_bayes", function(f) {
  m <- naiveBayes(x_nb[fold_p != f, ], factor(pers$pobre[fold_p != f]), laplace = 0)
  predict(m, x_nb[fold_p == f, ], type = "raw")[, "1"]
})

# CART (04): x1, gini, cp = 2.5e-05, up-sampling de los pobres en las filas
# de entrenamiento (como sampling = "up" de caret: se repiten pobres al azar
# hasta igualar a los no pobres).
x_cart <- preparar(select(pers, all_of(x1)))
p_cart <- oof_persona("cart", function(f) {
  ent <- which(fold_p != f)
  pos <- ent[pers$pobre[ent] == 1]; neg <- ent[pers$pobre[ent] == 0]
  set.seed(123 + f)
  filas <- c(neg, pos, sample(pos, length(neg) - length(pos), replace = TRUE))
  d <- data.frame(pobre = factor(pers$pobre[filas]), x_cart[filas, ])
  m <- rpart(pobre ~ ., data = d, method = "class", parms = list(split = "gini"),
             control = rpart.control(cp = 2.5e-05, xval = 0))
  predict(m, x_cart[fold_p == f, ], type = "prob")[, "1"]
})

# XGBoost por persona (05): x3 con una dummy por nivel de las categóricas;
# 500 rondas, profundidad 6, eta 0.1, min_child_weight 50, colsample 0.8,
# subsample 0.63.
x_xgb <- pers |> select(all_of(x3)) |>
  mutate(across(any_of(categoricas), as.character), across(where(is.character), factor))
x_xgb <- model.matrix(~ . - 1, model.frame(~ ., x_xgb, na.action = na.pass),
                      contrasts.arg = map(keep(x_xgb, is.factor), ~ contrasts(.x, contrasts = FALSE)))
params_xgb <- list(objective = "binary:logistic", eta = 0.1, max_depth = 6, gamma = 0,
                   min_child_weight = 50, colsample_bytree = 0.8, subsample = 0.63,
                   nthread = 1, seed = 123)
p_xgb_pers <- oof_persona("xgboost_persona", function(f) {
  d <- xgb.DMatrix(x_xgb[fold_p != f, ], label = pers$pobre[fold_p != f], missing = NA)
  m <- xgb.train(params = params_xgb, data = d, nrounds = 500, verbose = 0)
  predict(m, xgb.DMatrix(x_xgb[fold_p == f, ], missing = NA))
}, procesos = 3)   # 3: cada proceso copia la matriz de su fold (8 GB de RAM)
rm(pers, x_nb, x_cart, x_xgb); invisible(gc())

### 3.2 Modelos por hogar (05b y logit de main) --------------------------------

# Predicciones fuera de muestra guardadas por 05b (mismos folds y mismo orden
# de los hogares). La combinación con la regresión usa el peso de 05b (0.7
# para el clasificador; sección 6 y 5.20).
leer_05b <- function(n) readRDS(file.path(dir_05b, paste0(n, ".rds")))$prob
f_sin   <- "G_eta002_depth6_col08_V4_mas_V4t_todo2_mas_V4f_af_depth8_mcw50_col05"
reg     <- leer_05b("R_G_eta002_depth6_col08_V4_mas_V4t_todo2_peso")
lg <- readRDS(here("output", "pruebas", "logit", "logit_oof.rds"))
stopifnot(identical(lg$id_train, hog$id))

# Modelos: categoría, nombre, script y F1 reportado por su script (03-05 con
# sus 10 folds; los demás con estos mismos 5 folds).
modelos <- tribble(
  ~categoria,            ~modelo,                                              ~script,                      ~f1_script, ~prob,
  "Naive Bayes",         "Por persona, 16 variables (x1), laplace 0",          "03_naive_bayes.r",           0.675,      p_nb,
  "CART",                "Por persona, x1, cp 2,5e-05, up-sampling",           "04_carts.r",                 0.670,      p_cart,
  "XGBoost por persona", "24 variables (x3), 500 rondas",                      "05_xgboost.r",               0.706,      p_xgb_pers,
  "Logit lasso",         "Por hogar, 1.159 columnas con interacciones (main)", "06_logit_interacciones.r",   0.7451,     lg$prob_cv,
  "XGBoost por hogar",   "Solo la base por hogar de 01b",                      "05b_xgboost_hogar.r",        NA,         leer_05b("H0_base01b"),
  "XGBoost por hogar",   "Mejor sin Tweedie, con la regresión del ingreso",    "05b_xgboost_hogar.r",        NA,         0.7 * leer_05b(f_sin) + 0.3 * reg,
  "Random forest",       "Por hogar, variables del mejor XGBoost",             "05b_xgboost_hogar.r",        NA,         leer_05b(paste0("RF_", f_sin, "_V4w_mtry30_nodo50")),
  "XGBoost por hogar",   "Mejor: Tweedie, con la regresión del ingreso",       "05b_xgboost_hogar.r",        NA,         0.7 * leer_05b(paste0(f_sin, "_V4w")) + 0.3 * reg
)

################################################################################

## 4. F1 de cada modelo

# Mejor umbral (F1 promedio de los folds, como 05b) y F1 honesto (el umbral
# de cada fold se escoge con los otros 4): evaluar_umbral() de 00_master_file.r.
modelos <- bind_cols(modelos, map_dfr(modelos$prob, ~ evaluar_umbral(.x, y, fold, grid_umbral)))
modelos <- mutate(modelos, pred = map2(prob, umbral, ~ as.integer(.x > .y)))
print(select(modelos, categoria, modelo, f1_script, f1, f1_honesto, umbral), n = Inf)

################################################################################

## 5. Desempeño por grupo de población

# En la focalización de un subsidio, un falso negativo es un hogar pobre que
# queda excluido (error de exclusión) y un falso positivo, un hogar no pobre
# que queda incluido (error de inclusión). Por grupo:
#   sensibilidad: % de los hogares pobres que el modelo detecta;
#   falsos positivos: % de los no pobres que predice pobres;
#   precisión: % de los predichos pobres que sí lo son;
#   predicho / real: % predicho pobre sobre % pobre (1 = no sobrestima ni
#     subestima la pobreza del grupo).
metricas <- function(pred, real) {
  tibble(hogares = length(real), pct_pobres = 100 * mean(real), pct_predicho = 100 * mean(pred),
         sensibilidad = 100 * sum(pred == 1 & real == 1) / sum(real == 1),
         falsos_positivos = 100 * sum(pred == 1 & real == 0) / sum(real == 0),
         precision = 100 * sum(pred == 1 & real == 1) / sum(pred == 1),
         f1 = calcular_f1(pred, real), predicho_sobre_real = mean(pred) / mean(real))
}
por_grupo <- modelos |>
  select(categoria, modelo, pred) |>
  mutate(tabla = map(pred, function(p) {
    imap_dfr(grupos, function(g, atributo) {
      tibble(nivel = g, pred = p, real = y) |>
        reframe(metricas(pred, real), .by = nivel) |>
        mutate(atributo = atributo, .before = 1)
    })
  })) |>
  select(-pred) |> unnest(tabla) |>
  arrange(match(modelo, modelos$modelo), match(atributo, names(grupos)), match(nivel, orden_niveles))

# Brechas por atributo: diferencia entre el nivel con mayor y menor
# sensibilidad, y con mayor y menor tasa de falsos positivos.
brechas <- por_grupo |>
  summarize(.by = c(categoria, modelo, atributo),
            brecha_sensibilidad = max(sensibilidad) - min(sensibilidad),
            menor_sensibilidad = nivel[which.min(sensibilidad)],
            brecha_falsos_positivos = max(falsos_positivos) - min(falsos_positivos),
            mayor_falsos_positivos = nivel[which.max(falsos_positivos)])
saveRDS(list(modelos = select(modelos, -prob, -pred), por_grupo = por_grupo, brechas = brechas),
        file.path(dir_out, "resultados.rds"))
mejor <- slice_max(modelos, f1, n = 1)$modelo
print(filter(por_grupo, modelo == mejor), n = Inf)
print(brechas, n = Inf)

################################################################################

## 6. Tablas y figura

dir.create(here("output", "tables"), showWarnings = FALSE)
dir.create(here("output", "figures"), showWarnings = FALSE)
coma <- function(x, d) formatC(x, format = "f", digits = d, decimal.mark = ",")

# F1 de cada modelo.
modelos |>
  arrange(f1) |>
  transmute(categoria, modelo, f1_script = if_else(is.na(f1_script), "", coma(f1_script, 3)),
            f1 = coma(f1, 4), f1_honesto = coma(f1_honesto, 4), umbral = coma(umbral, 2)) |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      col.names = c("Categoría", "Modelo", "$F_1$ en su script", "$F_1$ (5 folds)", "$F_1$ honesto", "Umbral"),
      caption = "Mejor modelo de cada categoría con la misma evaluación: 5 folds por hogar y F$_1$ por hogar. Naive Bayes, CART y XGBoost por persona se reestimaron con estos folds (en sus scripts usaron 10).",
      label = "modelos_f1") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position")) |>
  save_kable(here("output", "tables", "modelos_f1.tex"))

# Desempeño por grupo del mejor modelo.
filter(por_grupo, modelo == mejor) |>
  transmute(atributo, nivel, hogares = format(hogares, big.mark = ".", decimal.mark = ","),
            pct_pobres = coma(pct_pobres, 1), pct_predicho = coma(pct_predicho, 1),
            sensibilidad = coma(sensibilidad, 1), falsos_positivos = coma(falsos_positivos, 1),
            precision = coma(precision, 1), f1 = coma(f1, 3)) |>
  kbl(format = "latex", booktabs = TRUE, longtable = FALSE,
      col.names = c("Atributo", "Grupo", "Hogares", "\\% pobres", "\\% predicho", "Sensibilidad",
                    "Falsos pos.", "Precisión", "$F_1$"),
      escape = FALSE,
      caption = "Mejor modelo (XGBoost por hogar con Tweedie y regresión): desempeño por grupo de población, con las predicciones fuera de muestra de train. Sensibilidad: \\% de los hogares pobres detectados; falsos positivos: \\% de los no pobres predichos pobres; precisión: \\% de los predichos pobres que sí lo son.",
      label = "equidad_mejor") |>
  collapse_rows(columns = 1, latex_hline = "major", valign = "top") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position"), font_size = 9) |>
  save_kable(here("output", "tables", "equidad_mejor.tex"))

# Brechas de sensibilidad por atributo y modelo (puntos porcentuales).
orden_modelos <- arrange(modelos, f1)$modelo
brechas |>
  mutate(modelo = paste0(categoria, ": ", modelo),
         celda = paste0(coma(brecha_sensibilidad, 1), " (", menor_sensibilidad, ")")) |>
  select(atributo, modelo, celda) |>
  pivot_wider(names_from = atributo, values_from = celda) |>
  arrange(match(sub("^[^:]+: ", "", modelo), orden_modelos)) |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE, col.names = c("Modelo", names(grupos)),
      caption = "Brecha de sensibilidad por atributo: diferencia, en puntos porcentuales, entre el grupo con mayor y menor \\% de hogares pobres detectados (entre paréntesis, el grupo con menor sensibilidad).",
      label = "equidad_brechas") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position"), font_size = 8) |>
  save_kable(here("output", "tables", "equidad_brechas.tex"))

# Figura: sensibilidad y falsos positivos por grupo, para cada modelo.
figura <- por_grupo |>
  mutate(modelo = factor(paste0(categoria, ": ", modelo), levels = paste0(modelos$categoria, ": ", modelos$modelo)[order(modelos$f1)])) |>
  select(modelo, atributo, nivel, `Sensibilidad (%)` = sensibilidad, `Falsos positivos (%)` = falsos_positivos) |>
  pivot_longer(c(`Sensibilidad (%)`, `Falsos positivos (%)`), names_to = "medida", values_to = "valor") |>
  mutate(nivel = factor(nivel, levels = rev(orden_niveles)), atributo = factor(atributo, levels = names(grupos)),
         medida = factor(medida, levels = c("Sensibilidad (%)", "Falsos positivos (%)"))) |>
  ggplot(aes(x = valor, y = nivel, color = modelo)) +
  geom_point(size = 1.8, alpha = 0.85) +
  facet_grid(atributo ~ medida, scales = "free", space = "free_y") +
  labs(x = NULL, y = NULL, color = NULL) +
  theme_minimal(base_size = 9) +
  theme(legend.position = "bottom", legend.direction = "vertical", strip.text.y = element_text(angle = 0))
ggsave(here("output", "figures", "equidad_errores.png"), figura, width = 8, height = 10, dpi = 200)

################################################################################

## 7. Importancia de las variables del mejor modelo

# Tablas de la importancia por permutación que calcula la sección 14 de 05b
# (caída del F1 al permutar cada variable o grupo; ver ahí el método), con el
# nombre completo de cada variable. Tres tablas: por grupo, las 20 variables
# más importantes y las 15 más importantes de las que no salen de un modelo
# (base por hogar, construidas y ubicación); y la importancia por concepto de
# la sección 15 de 05b.
imp <- readRDS(file.path(dir_05b, "importancia_permutacion.rds"))
ganancia <- readRDS(file.path(dir_05b, paste0(f_sin, "_V4w.rds")))$importancia

# Nombre completo de las variables. Las que salen de modelos se nombran por
# su prefijo y su versión; las demás, con el diccionario. Si una variable no
# tiene nombre, queda su código.
version <- c("_t" = "afinado", "_e" = "con preguntas nuevas", "_f" = "preguntas nuevas, afinado", "_w" = "Tweedie")
estadistico <- c(media = "promedio", max = "máximo", min = "mínimo", jefe = "del jefe")
nombres <- c(
  tenencia = "Tenencia de la vivienda", cuartos_pc = "Cuartos por persona",
  dominio = "Ciudad (dominio geográfico)", depto = "Departamento", cabecera = "Cabecera municipal",
  log_lp = "Línea de pobreza (log)", n_ug = "Personas en la unidad de gasto",
  beneficios_pc = "Primas, subsidios y bonificaciones por persona",
  beneficios_por_ocup = "Primas, subsidios y bonificaciones por ocupado",
  educ_adultos_x_ocup_pc = "Educación máxima de los adultos × ocupados por persona",
  rel_dom_log_arriendo = "Arriendo por persona relativo a la mediana de la ciudad",
  jefe_oficio = "Oficio del jefe", jefe_edad = "Edad del jefe", jefe_educ = "Años de educación del jefe",
  jefe_regimen = "Régimen de salud del jefe", jefe_posicion = "Posición ocupacional del jefe",
  jefe_tamano = "Tamaño de la empresa del jefe", jefe_formal = "Jefe formal (cotiza a pensión)",
  log_arriendo_pc_lp = "Arriendo (pagado o estimado) por persona sobre la línea (log)",
  arriendo_x_log_lp = "Arriendo por persona sobre la línea × línea (log)",
  arriendo_na = "Sin arriendo pagado ni estimado",
  arr_imp_pc_lp = "Arriendo imputado por persona sobre la línea",
  cv_educ_adultos = "Coeficiente de variación de la educación de los adultos",
  cv_horas_ocupados = "Coeficiente de variación de las horas de los ocupados",
  educ_ocup_x_formal_pc = "Educación de los ocupados por persona × formales por persona",
  educ_ocup_suma_pc = "Años de educación de los ocupados por persona",
  educ_max_adultos = "Educación máxima de los adultos", educ_media_adultos = "Educación promedio de los adultos",
  educ_max = "Educación máxima del hogar", educ_ocup_media = "Educación promedio de los ocupados",
  prop_contributivo = "Proporción en régimen contributivo de salud",
  prop_subsidiado = "Proporción en régimen subsidiado de salud",
  max_log_meses = "Mayor antigüedad laboral de los ocupados (log de meses)",
  ocup_max_educ = "Educación del ocupado más educado", ocup_max_horas = "Horas del ocupado que más trabaja",
  ocup_max_tamano = "Mayor tamaño de empresa de los ocupados",
  ocup_posicion_educ = "Posición ocupacional del ocupado más educado",
  ocup_pc = "Ocupados por persona", ocup_por_adulto = "Ocupados por adulto", horas_pc = "Horas trabajadas por persona",
  horas_formal_pc = "Horas formales por persona", horas_informal_pc = "Horas informales por persona",
  formal_pc = "Ocupados formales por persona", prop_formal_ocup = "Proporción de ocupados formales",
  dependencia = "Tasa de dependencia", prop_menores = "Proporción de menores de 18",
  pers_por_dormitorio = "Personas por dormitorio", fex_c = "Factor de expansión",
  fex_dpto = "Factor de expansión departamental",
  horas_pc_x_formal = "Horas por persona × proporción de ocupados formales",
  ocup_pc_x_log_lp = "Ocupados por persona × línea (log)", jefe_educ_x_formal = "Educación del jefe × jefe formal",
  dep_por_ocup = "Dependientes por ocupado", horas_por_ocup = "Horas por ocupado",
  brecha_educ_jefe = "Educación máxima de los adultos menos la del jefe",
  rel_dom_educ_max = "Educación máxima relativa a la mediana de la ciudad",
  rel_dom_horas_pc = "Horas por persona relativas a la mediana de la ciudad",
  rel_dom_ocup_pc = "Ocupados por persona relativos a la mediana de la ciudad",
  rel_dom_educ_media = "Educación promedio de los adultos relativa a la mediana de la ciudad",
  n_ocup = "Número de ocupados", n_formal = "Número de ocupados formales", desocup_pc = "Desocupados por persona"
)
nombre_completo <- function(v) {
  s <- str_extract(v, "_(t|e|f|w)$"); base <- if_else(is.na(s), v, str_remove(v, "_(t|e|f|w)$"))
  ver <- if_else(is.na(s), "original", version[coalesce(s, "_t")])
  est <- estadistico[str_extract(v, "(media|max|min|jefe)$")]
  case_when(
    base == "ing_arr_pc_lp"      ~ paste0("Ingreso predicho + arriendo imputado, por persona sobre la línea (log; ", ver, ")"),
    base == "log_ing_pred_pc_lp" ~ paste0("Ingreso predicho por persona sobre la línea (log; ", ver, ")"),
    base == "ing_pred_total"     ~ paste0("Ingreso predicho del hogar (", ver, ")"),
    base == "max_ing_pred_lp"    ~ paste0("Mayor ingreso predicho de un miembro sobre la línea (", ver, ")"),
    base == "prop_ing_jefe"      ~ paste0("Participación del jefe en el ingreso predicho (", ver, ")"),
    base == "ing_pred_por_ocup"  ~ paste0("Ingreso predicho por ocupado sobre la línea (", ver, ")"),
    str_starts(v, "pe_")     ~ paste0("Probabilidad de ser pobre por persona, con preguntas nuevas (", est, ")"),
    str_starts(v, "p_pers_") ~ paste0("Probabilidad de ser pobre por persona (", est, ")"),
    str_starts(v, "pc080_")  ~ paste0("Probabilidad de ingreso menor a 0,8 líneas (", est, ")"),
    str_starts(v, "pc125_")  ~ paste0("Probabilidad de ingreso menor a 1,25 líneas (", est, ")"),
    str_starts(v, "te_oficio_") ~ paste0("Tasa de pobreza del oficio de los ocupados (", est, ")"),
    v %in% names(nombres) ~ nombres[v],
    .default = v)
}
codigo_tex <- function(v) paste0("\\texttt{", gsub("_", "\\\\_", v), "}")
variables_imp <- imp$importancia |> filter(tipo == "variable") |>
  left_join(select(ganancia, nombre = variable, ganancia = importancia), by = "nombre") |>
  mutate(completo = nombre_completo(nombre), ganancia = coalesce(ganancia, 0))
desde_modelo <- c("Ingreso predicho por persona (todas las versiones)", "Probabilidad de ser pobre por persona (stacking)",
                  "Probabilidad de ingreso menor a 0,8 y 1,25 líneas", "Tasa de pobreza del oficio")
tabla_vars <- function(d) {
  transmute(d, completo, codigo = codigo_tex(nombre), caida = coma(caida, 4), de = coma(de, 4), ganancia = coma(ganancia, 1))
}

imp$importancia |> filter(tipo == "grupo") |>
  left_join(imp$ganancia_grupo, by = "grupo") |>
  transmute(grupo, variables = coalesce(variables, 0L), caida = coma(caida, 4), de = coma(de, 4),
            ganancia = coma(coalesce(ganancia, 0), 1)) |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      col.names = c("Grupo de variables", "Variables", "Caída del $F_1$", "D.E.", "Ganancia (\\%)"),
      caption = "Importancia por permutación de grupos de variables en el mejor modelo (XGBoost con Tweedie y regresión, $F_1$ 0,778): caída del $F_1$ al permutar a la vez todas las variables del grupo entre los hogares del fold de validación (promedio de 5 folds y 3 permutaciones; D.E.: desviación estándar). Ganancia: participación del grupo en la ganancia de los árboles del clasificador.",
      label = "importancia_grupos") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position")) |>
  save_kable(here("output", "tables", "importancia_grupos.tex"))

variables_imp |> head(20) |> tabla_vars() |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      col.names = c("Variable", "Código", "Caída del $F_1$", "D.E.", "Ganancia (\\%)"),
      caption = "Las 20 variables con mayor importancia por permutación en el mejor modelo (una variable a la vez; promedio de 5 folds y 3 permutaciones).",
      label = "importancia_variables") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position"), font_size = 9) |>
  save_kable(here("output", "tables", "importancia_variables.tex"))

variables_imp |> filter(!grupo %in% desde_modelo) |> head(15) |> tabla_vars() |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      col.names = c("Variable", "Código", "Caída del $F_1$", "D.E.", "Ganancia (\\%)"),
      caption = "Las 15 variables más importantes que no salen de un modelo (base por hogar, construidas y ubicación), con importancia por permutación en el mejor modelo.",
      label = "importancia_directas") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position"), font_size = 9) |>
  save_kable(here("output", "tables", "importancia_directas.tex"))
print(select(head(filter(variables_imp, !grupo %in% desde_modelo), 15), completo, caida, ganancia), n = Inf)

# Importancia de extremo a extremo por concepto (sección 15 de 05b): cada
# concepto se permuta en las personas (entrada de los modelos auxiliares) y en
# el hogar, y la perturbación pasa por toda la cadena de modelos.
conceptos_imp <- readRDS(file.path(dir_05b, "importancia_conceptos.rds"))$importancia
conceptos_imp |>
  transmute(concepto, variables = paste0(variables_personas, " / ", variables_hogar),
            total = coma(caida_total, 3), directa = coma(caida_directa, 3), via = coma(caida_via_modelos, 3),
            de = coma(de_total, 3)) |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      col.names = c("Concepto", "Variables (persona / hogar)", "Total", "Directa", "Vía modelos auxiliares", "D.E. total"),
      caption = "Importancia de extremo a extremo por concepto en el mejor modelo: caída del $F_1$ al permutar las variables del concepto. Total: en las personas (entrada de los modelos auxiliares) y en el hogar; directa: solo en el hogar; vía modelos auxiliares: solo en las personas. La perturbación pasa por toda la cadena de modelos (promedio de 5 folds).",
      label = "importancia_conceptos") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position")) |>
  save_kable(here("output", "tables", "importancia_conceptos.tex"))

# Importancia de extremo a extremo por variable de la persona (sección 16 de
# 05b): una variable a la vez, permutada en las personas y pasando por toda la
# cadena de modelos (sin permutar los agregados del hogar construidos con ella).
nombres_persona <- c(
  posicion_ocupacional = "Posición ocupacional (P6430)",
  recibio_arriendos_pensiones = "Recibió arriendos o pensiones el mes pasado (P7495)",
  ingreso_trabajo_desocupado_2 = "Inactivo: recibió ingresos por trabajo el mes pasado (P7472)",
  ingreso_trabajo_desocupado = "Desocupado: recibió ingresos por trabajo el mes pasado (P7422)",
  arriendo_pc_sobre_lp = "Arriendo (pagado o estimado) por persona sobre la línea (log)",
  horas_trabajo_semana = "Horas trabajadas a la semana (P6800)", ocupado = "Ocupado",
  prop_ocupados_hogar = "Proporción de ocupados en el hogar", oficio = "Oficio",
  recibio_otros_ingresos = "Recibió dinero de hogares, instituciones, intereses o cesantías (P7505)",
  dinero_cesantias = "Recibió cesantías (P7510s6)", log_meses_empresa = "Meses en la empresa (log)",
  dinero_intereses_dividendos = "Recibió intereses, dividendos o utilidades (P7510s5)",
  tenencia_vivienda = "Tenencia de la vivienda (P5090)",
  dinero_hogares_pais = "Recibió dinero de otros hogares en el país (P7510s1)"
)
readRDS(file.path(dir_05b, "importancia_variables_e2e.rds")) |>
  head(15) |>
  transmute(completo = coalesce(nombres_persona[variable], variable), concepto,
            caida = coma(caida, 3), de = coma(de, 3)) |>
  kbl(format = "latex", booktabs = TRUE, escape = FALSE,
      col.names = c("Variable", "Concepto", "Caída del $F_1$", "D.E."),
      caption = "Las 15 variables de la persona con mayor importancia de extremo a extremo en el mejor modelo: caída del $F_1$ al permutar una variable de las personas, que pasa por toda la cadena de modelos (promedio de 5 folds). Se probaron las 36 variables de la persona de los 5 conceptos con mayor caída total; los agregados del hogar construidos con cada variable no se permutan.",
      label = "importancia_variables_e2e") |>
  kable_styling(latex_options = c("scale_down", "HOLD_position"), font_size = 9) |>
  save_kable(here("output", "tables", "importancia_variables_e2e.tex"))

### NOTAS:

# Resultados (corrida del 2026-10-25).

# F1 con los 5 folds por hogar: naive bayes 0.675, CART 0.669, XGBoost por
# persona 0.705 (iguales a los de sus scripts con 10 folds: el número de folds
# casi no importa), logit lasso de main 0.745, XGBoost con solo la base por
# hogar 0.753, mejor XGBoost sin Tweedie + regresión 0.774, random forest por
# hogar 0.775 y mejor modelo (XGBoost con Tweedie + regresión) 0.778. En todos
# el F1 honesto es igual al F1 con el mejor umbral (diferencias < 0.001).

# Desempeño por grupo del mejor modelo (sensibilidad = % de pobres
# detectados; falsos positivos = % de no pobres predichos pobres):
# - Los pobres de los grupos con poca pobreza se detectan menos: jefe formal
#   (sensibilidad 65%, frente a 80% informal y 85% no ocupado), jefe con
#   educación superior (65%), hogares de 1 o 2 personas (67%) y sin menores
#   (65%). Ahí el modelo subestima la pobreza (predicho / real 0.93-0.98): un
#   hogar pobre con jefe formal o educado se parece a los no pobres.
# - En los grupos con mucha pobreza pasa lo contrario: más falsos positivos y
#   pobreza sobrestimada (predicho / real 1.09-1.11): zona rural (falsos
#   positivos 12.6% frente a 6.2% en cabeceras, con la misma sensibilidad),
#   jefe informal (10.1%), primaria o menos (10.8%), 5 o más personas (12.1%).
# - Sexo del jefe: los hogares con jefa tienen más sensibilidad (82.7% frente
#   a 78.9%) y algo más de falsos positivos (7.4% frente a 6.2%); la brecha es
#   pequeña y va a favor de los hogares con jefa. Con los modelos por persona
#   era de 1-2 puntos.
# - Es el patrón esperable cuando los grupos tienen tasas de pobreza muy
#   distintas: con una sola regla no se pueden igualar a la vez la
#   sensibilidad y los falsos positivos entre grupos. Los modelos mejores
#   reducen casi todas las brechas: la de falsos positivos entre zonas pasa
#   de 27.8 puntos (naive bayes) a 6.5 (mejor modelo), y la de sensibilidad
#   por empleo del jefe de 33.4 a 19.7.
