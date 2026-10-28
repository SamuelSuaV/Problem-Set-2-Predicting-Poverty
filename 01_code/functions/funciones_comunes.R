###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   funciones_comunes.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Variables y funciones que comparten los scripts de
#                      modelos (04 a 15). Cada uno lo carga con
#                      source(here("01_code", "functions",
#                      "funciones_comunes.R")) después de cargar sus
#                      librerías (usa tidyverse) y antes de definir sus
#                      propias variables (los modelos por hogar definen su
#                      propio `categoricas`).
###############################################################################

# Estructura:
# 1. Variables comunes
# 2. Funciones de evaluación
# 3. Utilidades

# Input:  ninguno.
# Output: ninguno (define objetos en la sesión).

# 1. Variables comunes --------------------------------------------------------

# Categóricas por persona con códigos DANE. En naive bayes y CART pasan a
# factor y en XGBoost por persona a dummies; si quedan numéricas, los modelos
# las tratan como si los códigos estuvieran ordenados.
categoricas <- c(
  "max_educ", "regimen_salud", "actividad_sem_pasada",
  "cotiza_pension", "tamano_empresa", "oficio",
  "posicion_ocupacional", "tenencia_vivienda"
)

# Mejor conjunto de variables de naive bayes (04), base de CART (05) y XGBoost
# por persona (05).
x1 <- c(
  "anos", "anos_educ_imp", "mujer", "ocupado", "afiliado_salud",
  "regimen_salud", "actividad_sem_pasada", "log_meses_empresa",
  "tamano_empresa", "posicion_ocupacional", "horas_trabajo_semana",
  "arriendo_pc_sobre_lp", "prop_ocupados_hogar", "tenencia_vivienda",
  "recibio_arriendos_pensiones", "ayudas_instituciones"
)

# Categóricas y texto a factor, para que naive bayes y CART las traten como
# categorías. binarias = TRUE (naive bayes): también las 0/1, que si no
# naiveBayes() modela como normales. Se aplica igual a train y a test.
preparar <- function(x, binarias = FALSE) {
  x <- x |>
    mutate(
      across(any_of(categoricas), factor), across(where(is.character), factor)
    )
  if (binarias) {
    x <- mutate(
      x, across(where(~ is.numeric(.x) && all(.x %in% c(0, 1, NA))), factor)
    )
  }
  x
}

# 2. Funciones de evaluación --------------------------------------------------

# F1 a partir de predicciones y valores reales (0/1 o lógicos):
# 2 VP / (predichos positivos + positivos reales).
calcular_f1 <- function(pred, real) {
  vp <- sum(pred == 1 & real == 1)
  2 * vp / (sum(pred == 1) + sum(real == 1))
}

# Probabilidad de cada hogar a partir de la de sus miembros: promedio de los
# log odds, devuelto a probabilidad (pesan más los miembros con señal fuerte;
# da mejor F1 que promediar probabilidades). Recorte para que p = 0 o 1 no den
# log odds infinitos. Devuelve una fila por hogar (id, prob).
prob_hogar <- function(p, id) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  tibble(id = id, prob = p) |>
    summarize(prob = plogis(mean(qlogis(prob))), .by = id)
}

# F1 por umbral (filas) y fold (columnas), con una predicción por hogar.
tabla_f1 <- function(prob, real, fold, umbrales) {
  sapply(sort(unique(fold)), function(f) {
    map_dbl(
      umbrales, ~ calcular_f1(as.integer(prob[fold == f] > .x), real[fold == f])
    )
  })
}

# Mejor umbral (el de mayor F1 promedio de los folds) y F1 honesto: el umbral
# de cada fold se escoge con los demás folds, así que no usa sus etiquetas.
evaluar_umbral <- function(prob, real, fold, umbrales) {
  t <- tabla_f1(prob, real, fold, umbrales)
  i <- which.max(rowMeans(t))
  tibble(
    umbral = umbrales[i], f1 = rowMeans(t)[i],
    f1_honesto = mean(map_dbl(
      seq_len(ncol(t)),
      function(f) t[which.max(rowMeans(t[, -f, drop = FALSE])), f]
    ))
  )
}

# AUC (probabilidad de que un pobre tenga mayor puntaje que un no pobre).
calcular_auc <- function(puntaje, real) {
  r <- rank(puntaje)
  n1 <- as.numeric(sum(real == 1)) # numeric: n1 * n0 desborda los enteros
  n0 <- as.numeric(sum(real == 0))
  (sum(r[real == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# 3. Utilidades ---------------------------------------------------------------

# Avance con la hora (07 la reemplaza por una que también escribe en su log).
marca <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "-", ..., "\n")

# Escribir un .tex en UTF-8 (en Windows writeLines pasaría a CP1252).
escribir_tex <- function(x, archivo) {
  writeLines(
    enc2utf8(x), archivo,
    useBytes = TRUE
  )
}
