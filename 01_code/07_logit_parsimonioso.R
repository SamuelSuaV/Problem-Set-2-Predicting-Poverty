###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   07_logit_parsimonioso.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Construir la curva de F1 frente al número de variables
#                      de un logit a nivel hogar y escoger un modelo corto
#                      (la competencia pide usar el mínimo de variables).
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Ranking de variables por orden de entrada al lasso
# 4. Curva de F1 frente al número de variables
# 5. Tabla y figura
# 6. Modelo escogido y predicción en test

# Input:  Bases a nivel hogar creadas en 02_bases_hogar.R; train_hogar.rds ,
#         test_hogar.rds .
# Output: 02_outputs/tables/logit_parsimonia.tex ;
#         02_outputs/figures/logit_f1_vs_variables.png ;
#         02_outputs/predictions/logit_vars_<k>_threshold_<c>.csv .
#
# Con el argumento --prueba (Rscript 01_code/07_logit_parsimonioso.R --prueba)
# corre con el 10% de los hogares de train y no guarda nada; sirve para
# verificar que el script corre.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos y gráficas.
  glmnet, # Lasso para ordenar las variables.
  knitr # Exportar la tabla a LaTeX.
)
# Funciones comunes (calcular_f1, marca, ...).
source(here("01_code", "functions", "funciones_comunes.R"), encoding = "UTF-8")

prueba <- "--prueba" %in% commandArgs(trailingOnly = TRUE)

# 2. Cargar datos -------------------------------------------------------------

train <- read_rds(here("data", "train_hogar.rds"))
test <- read_rds(here("data", "test_hogar.rds"))

if (prueba) {
  set.seed(1)
  train <- slice_sample(train, prop = 0.1)
  cat("MODO PRUEBA:", nrow(train), "hogares de train; no se guarda nada\n")
}
y <- train$pobre

# Categóricas con códigos DANE: entran como factor (las mismas de 06).
categoricas <- c(
  "dominio", "depto", "tenencia", "jefe_posicion",
  "jefe_regimen", "jefe_tamano", "jefe_oficio"
)

# Todas las variables de la base a nivel hogar son candidatas.
x_base <- setdiff(names(train), c("id", "pobre"))

# Mismos folds que 06 (una fila = un hogar).
set.seed(123)
k <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

grid_umbral <- seq(0.2, 0.5, by = 0.01)

# 3. Ranking de variables por orden de entrada al lasso -----------------------

# Lasso con todas las variables como efectos principales, sin interacciones,
# sobre todo train. Cada variable cuenta como una aunque sea un factor con
# varias dummies: entra en el lambda más grande en el que alguna de sus
# columnas tiene coeficiente distinto de 0.
datos_rank <- train |>
  select(all_of(x_base)) |>
  mutate(across(all_of(categoricas), factor))
mat_rank <- model.matrix(~., datos_rank)
variable_col <- x_base[attr(mat_rank, "assign")[-1]] # variable de cada columna
mat_rank <- mat_rank[, -1]

lasso <- glmnet(mat_rank, y, family = "binomial", alpha = 1, nlambda = 200)

# Primer lambda (el más grande) en que cada columna es distinta de 0.
beta <- as.matrix(lasso$beta)
entrada_col <- apply(beta != 0, 1, function(b) if (any(b)) which(b)[1] else Inf)

# Desempate entre variables que entran en el mismo lambda: el mayor
# coeficiente estandarizado (|beta| x desviación estándar de la columna) en
# ese lambda.
sd_col <- apply(mat_rank, 2, sd)
ranking <- tibble(variable = variable_col, entrada = entrada_col) |>
  summarize(entrada = min(entrada), .by = variable)
ranking$peso <- map_dbl(seq_len(nrow(ranking)), function(i) {
  if (!is.finite(ranking$entrada[i])) {
    return(0)
  }
  cols <- which(variable_col == ranking$variable[i])
  max(abs(beta[cols, ranking$entrada[i]]) * sd_col[cols])
})
ranking <- ranking |>
  arrange(entrada, desc(peso)) |>
  mutate(
    orden = row_number(),
    lambda_entrada = if_else(is.finite(entrada),
      lasso$lambda[pmin(entrada, length(lasso$lambda))], NA_real_
    )
  )
cat(
  "Variables que nunca entran al lasso:", sum(!is.finite(ranking$entrada)), "\n"
)
print(ranking |> select(orden, variable, lambda_entrada), n = Inf)
rm(mat_rank, datos_rank, beta)

# 4. Curva de F1 frente al número de variables --------------------------------

# glm() falla en predict() si un factor tiene un nivel que no vio al estimar.
# Los niveles con menos de 30 hogares en train se agrupan en "otro" (en train
# y en test, y también los niveles de test que no están en train). Si "otro"
# queda con menos de 30 hogares en train, se une al nivel más común.
agrupar_niveles <- function(train, test, vars, minimo = 30) {
  for (v in vars) {
    conteo <- table(as.character(train[[v]]))
    comunes <- names(conteo)[conteo >= minimo]
    tr <- if_else(as.character(train[[v]]) %in% comunes,
      as.character(train[[v]]),
      "otro"
    )
    te <- if_else(as.character(test[[v]]) %in% comunes, as.character(test[[v]]),
      "otro"
    )
    if (sum(tr == "otro") < minimo) {
      modal <- names(which.max(table(tr[tr != "otro"])))
      tr[tr == "otro"] <- modal
      te[te == "otro"] <- modal
    }
    train[[v]] <- factor(tr)
    test[[v]] <- factor(te, levels = levels(train[[v]]))
  }
  list(train = train, test = test)
}
bases <- agrupar_niveles(train, test, categoricas)
train_glm <- bases$train
test_glm <- bases$test
rm(bases)

# F1 de validación cruzada de un logit sin penalizar con las variables vars:
# F1 promedio de los folds para cada umbral y el mejor umbral, como en 06.
cv_glm <- function(vars) {
  formula <- reformulate(vars, response = "pobre")
  map_dfr(1:k, function(i) {
    modelo <- glm(formula, family = binomial, data = train_glm[fold != i, ])
    p <- predict(modelo, train_glm[fold == i, ], type = "response")
    real <- y[fold == i]
    tibble(
      umbral = grid_umbral,
      f1 = map_dbl(grid_umbral, ~ calcular_f1(as.integer(p > .x), real))
    )
  }) |>
    summarize(f1 = mean(f1), .by = umbral) |>
    slice_max(f1, n = 1, with_ties = FALSE)
}

n_vars <- c(3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, length(x_base))
curva <- map_dfr(n_vars, function(n) {
  mejor <- cv_glm(ranking$variable[1:n])
  cat("k =", n, "- F1 =", round(mejor$f1, 4), "- umbral =", mejor$umbral, "\n")
  tibble(k = n, f1 = mejor$f1, umbral = mejor$umbral)
})

# Variables que se agregan en cada paso de la curva.
curva <- curva |>
  mutate(
    desde = lag(k, default = 0) + 1,
    variables = map2_chr(desde, k, function(a, b) {
      if (b == length(x_base) && b - a + 1 > 10) {
        paste0("Todas las demás (", b - a + 1, ")")
      } else {
        paste(ranking$variable[a:b], collapse = ", ")
      }
    })
  ) |>
  select(k, variables, f1, umbral)
print(curva, n = Inf, width = Inf)

# 5. Tabla y figura -----------------------------------------------------------

f1_completo <- 0.7417 # Lasso con interacciones de 06 (F1 de CV).

if (!prueba) {
  tabla <- kable(curva,
    format = "latex", booktabs = TRUE,
    digits = c(0, 0, 4, 2),
    col.names = c("k", "Variables agregadas", "F1 (CV)", "Umbral"),
    align = c("r", "p{9cm}", "r", "r"),
    caption = paste0(
      "F1 de validación cruzada de un logit sin penalizar con las k ",
      "primeras variables del ranking del lasso"
    ),
    label = "logit_parsimonia"
  )
  writeLines(
    enc2utf8(tabla), here("02_outputs", "tables", "logit_parsimonia.tex"),
    useBytes = TRUE
  )

  grafica <- ggplot(curva, aes(k, f1)) +
    geom_hline(
      yintercept = f1_completo, linetype = "dashed", color = "firebrick"
    ) +
    annotate("text",
      x = max(curva$k), y = f1_completo, vjust = -0.5, hjust = 1,
      label = "Lasso con interacciones (06): 0,7417", color = "firebrick"
    ) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks = curva$k) +
    labs(x = "Número de variables (k)", y = "F1 de validación cruzada") +
    theme_minimal()
  ggsave(here("02_outputs", "figures", "logit_f1_vs_variables.png"), grafica,
    width = 8, height = 5
  )
}

# 6. Modelo escogido y predicción en test -------------------------------------

# El menor k con F1 a menos de 0,01 del modelo completo; si ninguno llega, el
# menor a menos de 0,02; si tampoco, el de mayor F1.
elegido <- curva |>
  filter(f1 >= f1_completo - 0.01) |>
  slice_min(k, n = 1)
if (nrow(elegido) == 0) {
  cat(
    "Ningún k llega a", f1_completo - 0.01,
    "; se usa el menor a menos de 0,02\n"
  )
  elegido <- curva |>
    filter(f1 >= f1_completo - 0.02) |>
    slice_min(k, n = 1)
}
if (nrow(elegido) == 0) {
  cat("Ningún k llega a", f1_completo - 0.02, "; se usa el de mayor F1\n")
  elegido <- curva |> slice_max(f1, n = 1, with_ties = FALSE)
}
vars_elegidas <- ranking$variable[1:elegido$k]
cat(
  "Modelo escogido: k =", elegido$k, "- F1 =", round(elegido$f1, 4),
  "- umbral =", elegido$umbral, "\n"
)
cat("Variables:", paste(vars_elegidas, collapse = ", "), "\n")

# Estimación con todo train y predicción en test.
modelo <- glm(reformulate(vars_elegidas, response = "pobre"),
  family = binomial,
  data = train_glm
)
prob_test <- predict(modelo, test_glm, type = "response")
predicciones <- tibble(
  id = test$id, pobre = as.integer(prob_test > elegido$umbral)
)
nrow(predicciones) # una fila por hogar de test
mean(predicciones$pobre) # proporción predicha de pobres (en train es 0,20)

# Nombre con formato Kaggle, sin punto decimal (umbral 0.36 -> 036).
sin_punto <- function(v) gsub("[.-]", "", as.character(v))
archivo <- paste0(
  "logit_vars_", elegido$k, "_threshold_",
  sin_punto(round(elegido$umbral, 2)), ".csv"
)
if (!prueba) {
  dir.create(here("02_outputs", "predictions"), showWarnings = FALSE)
  predicciones |> write_csv(here("02_outputs", "predictions", archivo))
}
cat("Submission:", archivo, "\n")

# Notas -----------------------------------------------------------------------

# Curva (logit sin penalizar, F1 de CV con el mejor umbral de cada k):
#   k =  3: 0.5848 (0.27)   k = 15: 0.6970 (0.33)   k = 40: 0.7273 (0.35)
#   k =  5: 0.6351 (0.33)   k = 20: 0.7039 (0.34)   k = 50: 0.7313 (0.34)
#   k =  8: 0.6591 (0.33)   k = 25: 0.7106 (0.36)   k = 94: 0.7357 (0.35)
#   k = 10: 0.6732 (0.35)   k = 30: 0.7157 (0.35)
#   k = 12: 0.6739 (0.34)
# Las 3 primeras variables (arriendo, régimen contributivo y menores) ya dan
# 0.58; con 15 se llega a 0.70 y con 40 a 0.727. De 50 a 94 variables el F1
# sube menos de 0.005.

# Modelo escogido con la regla (menor k con F1 >= 0.7417 - 0.01 = 0.7317):
# k = 94, es decir, todas las variables (F1 0.7357, umbral 0.35). k = 50 queda
# en 0.7313, a 0.0004 del corte, así que la regla no logra un modelo corto:
# sin interacciones, un logit con pocas variables no alcanza al lasso de 06.
# Si se acepta perder 0.015 de F1, k = 40 (0.7273) es un buen modelo corto.

# Ranking (orden de entrada al lasso sin interacciones): log_arriendo_pc_lp,
# prop_contributivo, n_menor18, educ_ocup_suma_pc, jefe_regimen, horas_pc,
# n_formal, n_con_beneficios, n_fuentes_no_lab, pers_por_dormitorio,
# any_arriendos, n_asalariado, prop_menores, tenencia, arriendo_na, n_desocup,
# n_ocup_superior, prop_superior, n_jornalero, educ_media_adultos, dominio,
# cuartos_pc, cabecera, n_empresa_grande, arriendo_es_pagado, ... (la lista
# completa sale en el log). Cuatro variables nunca entran al lasso:
# horas_formal_pc, horas_informal_pc, n_cuenta_propia y log_cuota_pc_lp.

# Preguntas de la encuesta de las que salen las variables del modelo escogido
# (todas las de train_hogar; descripción en la tabla variables_hogar de
# documentation/guia_variables.Rmd):
#   Ubicación: dominio (Dominio), depto (Depto), cabecera (Clase),
#     log_lp (Lp, línea de pobreza).
#   Demografía: n_ug, log_n_ug (Npersug, P6050 parentesco); n_menor6,
#     n_menor12, n_menor18, n_mayor65, dependencia, prop_menores (P6040 edad);
#     prop_mujer_ug (P6020 sexo).
#   Trabajo: n_ocup, ocup_pc, ocup_por_adulto, jefe_ocup (Oc); n_desocup,
#     desocup_pc, jefe_desocup (Des); n_inactivo_adulto (Ina, P6040);
#     horas_pc (P6800, P7045); horas_formal_pc, horas_informal_pc (P6800,
#     P7045, P6920); n_segundo_trab (P7040); n_quiere_mas_horas (P7090);
#     n_oficios_hogar, n_estudia (P6240); n_pareja_ocup, n_hijos_ocup (P6050).
#   Calidad del empleo: n_formal, formal_pc, prop_formal_ocup, jefe_formal
#     (P6920 cotiza a pensión); n_asalariado, n_gobierno, n_domestico,
#     n_cuenta_propia, n_patron, n_sin_pago, n_jornalero, jefe_posicion
#     (P6430); n_empresa_sola, n_empresa_peq, n_empresa_grande, jefe_tamano
#     (P6870); beneficios_pc, n_con_beneficios (P6545, P6580, P6585s1-s4,
#     P6630s1-s4, P6630s6); n_horas_extra (P6510); max_log_meses,
#     jefe_log_meses (P6426); n_of_*, jefe_oficio (Oficio).
#   Salud: prop_contributivo, prop_subsidiado, n_especial, n_no_afiliado,
#     jefe_regimen (P6090, P6100).
#   Educación: educ_max, educ_media_adultos, educ_ocup_media,
#     educ_ocup_suma_pc, prop_superior, n_sin_educ, n_ocup_superior,
#     superior_formal_pc, jefe_educ (P6210, P6210s1).
#   Jefe: jefe_edad, jefe_edad2 (P6040); jefe_mujer, jefe_mujer_sola (P6020,
#     P6050).
#   Ingresos no laborales: any_pension (P7500s2, P6920), any_arriendos
#     (P7495), any_hogares_pais (P7510s1), any_exterior (P7510s2), any_ayudas
#     (P7510s3), any_intereses (P7510s5), any_cesantias (P7510s6),
#     any_pension_alim (P7500s3), any_otros_ing (P7505), n_fuentes_no_lab.
#   Vivienda: tenencia (P5090); cuartos_pc (P5000, Nper); pers_por_dormitorio
#     (P5010, Nper); log_arriendo_pc_lp, arriendo_na, arriendo_es_pagado
#     (P5130, P5140, Nper, Lp); log_cuota_pc_lp (P5100).

# Los factores con niveles de menos de 30 hogares se agrupan en "otro" (sobre
# todo jefe_oficio). Con k grande hay variables colineales (n_fuentes_no_lab
# es la suma de cinco any_*; depto está anidado en dominio): glm les da
# coeficiente NA y predict las ignora, sin afectar el F1.
