###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   05_lpm.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar un modelo lineal de probabilidad (LPM) a nivel
#                      hogar con las mismas variables y folds que los demás
#                      modelos, con efectos principales y con la especificación
#                      de 06, y comparar con pesos por clase.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos y armar las matrices de diseño
# 3. Funciones: F1 y validación cruzada
# 4. Especificaciones y pesos por clase
# 5. Tabla, modelos finales y submissions

# Input:  Bases a nivel hogar creadas en 02_bases_hogar.R; train_hogar.rds ,
#         test_hogar.rds .
# Output: 02_outputs/tables/lpm.tex ;
#         02_outputs/predictions/LPM_<spec>_threshold_<c>.csv (y la versión _w);
#         02_outputs/logs/lpm/resultados.rds .
#
# Correr con source(..., encoding = "UTF-8"). Con prueba <- TRUE antes del
# source() usa el 10% de los hogares y no guarda nada.

# 1. Cargar librerías ---------------------------------------------------------

library(pacman)
p_load(
  here, # Rutas relativas a la raíz del proyecto.
  tidyverse, # Manipulación de datos.
  knitr # Exportar la tabla a LaTeX.
)
# Funciones comunes (calcular_f1, marca, ...).
source(here("01_code", "functions", "funciones_comunes.R"), encoding = "UTF-8")

prueba <- exists("prueba") && isTRUE(prueba)
dir_out <- here("02_outputs", "logs", "lpm")
dir.create(dir_out, showWarnings = FALSE, recursive = TRUE)
inicio_total <- Sys.time()

# 2. Cargar datos y armar las matrices de diseño ------------------------------

train <- read_rds(here("data", "train_hogar.rds"))
test <- read_rds(here("data", "test_hogar.rds"))
if (prueba) {
  set.seed(1)
  train <- slice_sample(train, prop = 0.1)
  cat("MODO PRUEBA:", nrow(train), "hogares de train; no se guarda nada\n")
}
y <- train$pobre

# Mismos folds que 06 y 12.
set.seed(123)
k <- 5
fold <- sample(rep(1:k, length.out = nrow(train)))

categoricas <- c(
  "dominio", "depto", "tenencia", "jefe_posicion",
  "jefe_regimen", "jefe_tamano", "jefe_oficio"
)
x_base <- setdiff(names(train), c("id", "pobre"))

# (a) Efectos principales. (b) La fórmula de 06: interacciones de a pares
# entre x_clave, dominio x cuatro continuas y cuadrados.
x_clave <- c(
  "log_arriendo_pc_lp", "horas_pc", "ocup_pc", "prop_contributivo",
  "prop_subsidiado", "prop_menores", "log_n_ug", "educ_media_adultos",
  "jefe_educ", "cuartos_pc", "formal_pc", "beneficios_pc",
  "dependencia", "n_fuentes_no_lab", "cabecera", "any_ayudas",
  "educ_ocup_media", "horas_formal_pc"
)
formulas <- list(
  principales = paste("~", paste(x_base, collapse = " + ")) |> as.formula(),
  interacciones = paste(
    "~", paste(x_base, collapse = " + "),
    "+ (", paste(x_clave, collapse = " + "), ")^2",
    paste0(
      "+ dominio:(log_arriendo_pc_lp + horas_pc + prop_contributivo + ",
      "educ_media_adultos)"
    ),
    "+ I(log_arriendo_pc_lp^2) + I(horas_pc^2) + I(ocup_pc^2)",
    "+ I(educ_media_adultos^2) + I(prop_menores^2)"
  ) |> as.formula()
)

# Las matrices se arman con train y test juntos (como en 06) para que los
# factores tengan los mismos niveles: así lm no falla con niveles que no vio.
todo <- bind_rows(
  train |> select(all_of(x_base)),
  test |> select(all_of(x_base))
) |>
  mutate(across(all_of(categoricas), factor))
matrices <- map(formulas, function(f) {
  mat <- model.matrix(f, todo)[, -1]
  list(
    tr = mat[seq_len(nrow(train)), ],
    te = mat[nrow(train) + seq_len(nrow(test)), ]
  )
})
rm(todo)
iwalk(matrices, ~ cat("Especificación", .y, ":", ncol(.x$tr), "columnas\n"))

peso_pobre <- sum(y == 0) / sum(y == 1)
pesos <- if_else(y == 1, peso_pobre, 1)

# 3. Funciones: F1 y validación cruzada ---------------------------------------

grid_umbral <- seq(0.1, 0.6, by = 0.01)
resumir_f1 <- function(prob) {
  f <- map_dfr(1:k, function(i) {
    tibble(
      umbral = grid_umbral,
      f1 = map_dbl(
        grid_umbral,
        ~ calcular_f1(as.integer(prob[fold == i] > .x), y[fold == i])
      )
    )
  }) |>
    summarize(f1 = mean(f1), .by = umbral)
  mejor <- slice_max(f, f1, n = 1, with_ties = FALSE)
  tibble(
    umbral = mejor$umbral, f1 = mejor$f1,
    f1_05 = f$f1[abs(f$umbral - 0.5) < 1e-9]
  )
}

# MCO (con pesos si se dan). La "probabilidad" es el valor ajustado; puede
# salir de [0, 1] y no se recorta. Los coeficientes NA (columnas colineales,
# p. ej. depto anidado en dominio) cuentan como 0.
estimar_lpm <- function(mat, y, w = NULL) {
  modelo <- lm(y ~ mat, weights = w)
  coalesce(coef(modelo), 0)
}
predecir_lpm <- function(b, mat) drop(cbind(1, mat) %*% b)

cv_lpm <- function(spec, con_pesos = FALSE) {
  mat <- matrices[[spec]]$tr
  oof <- numeric(length(y))
  for (f in 1:k) {
    i_tr <- which(fold != f)
    b <- estimar_lpm(mat[i_tr, ], y[i_tr], if (con_pesos) pesos[i_tr] else NULL)
    oof[fold == f] <- predecir_lpm(b, mat[fold == f, ])
  }
  r <- resumir_f1(oof) |>
    mutate(
      especificacion = spec, pesos = con_pesos, columnas = ncol(mat),
      fuera_01 = mean(oof < 0 | oof > 1), .before = 1
    )
  marca(
    spec, "- pesos", con_pesos, "- F1", round(r$f1, 4), "- umbral", r$umbral,
    "- fuera de [0,1]:", round(100 * r$fuera_01, 1), "%"
  )
  list(resumen = r, oof = oof)
}

# 4. Especificaciones y pesos por clase ---------------------------------------

res_sin <- map(names(formulas), cv_lpm) |> set_names(names(formulas))
mejor_spec <- names(res_sin)[which.max(map_dbl(res_sin, ~ .x$resumen$f1))]
cat("Mejor especificación:", mejor_spec, "\n")
res_con <- map(names(formulas), ~ cv_lpm(.x, con_pesos = TRUE)) |>
  set_names(names(formulas))
tabla <- bind_rows(unname(c(map(res_sin, "resumen"), map(res_con, "resumen"))))
print(tabla, width = Inf)

# 5. Tabla, modelos finales y submissions -------------------------------------

if (!prueba) {
  tabla |>
    mutate(
      especificacion = recode(especificacion,
        principales = "Efectos principales",
        interacciones = "Interacciones y cuadrados (06)"
      ),
      pesos = if_else(pesos, "Sí", "No"), fuera_01 = 100 * fuera_01
    ) |>
    select(especificacion, pesos, columnas, umbral, f1, f1_05, fuera_01) |>
    kable(
      format = "latex", booktabs = TRUE, digits = c(0, 0, 0, 2, 4, 4, 1),
      col.names = c(
        "Especificación", "Pesos por clase", "Columnas", "Umbral",
        "F1 (CV)", "F1 (umbral 0,5)", "\\% fuera de [0,1]"
      ),
      escape = FALSE,
      caption = paste0(
        "Modelo lineal de probabilidad: F1 de validación cruzada y ",
        "predicciones fuera de [0,1]"
      ),
      label = "lpm"
    ) |>
    escribir_tex(here("02_outputs", "tables", "lpm.tex"))
}

sin_punto <- function(v) gsub("[.]", "", as.character(v))
verificar <- function(s) {
  stopifnot(
    nrow(s) == nrow(test), identical(names(s), c("id", "pobre")),
    all(s$pobre %in% c(0, 1)), !anyNA(s), !anyDuplicated(s$id),
    setequal(s$id, test$id)
  )
  mean(s$pobre)
}
final <- function(r, con_pesos, sufijo = "") {
  b <- estimar_lpm(
    matrices[[r$especificacion]]$tr, y,
    if (con_pesos) pesos else NULL
  )
  prob <- predecir_lpm(b, matrices[[r$especificacion]]$te)
  s <- tibble(id = test$id, pobre = as.integer(prob > r$umbral))
  prop <- verificar(s)
  archivo <- paste0(
    "LPM_", r$especificacion, "_threshold_", sin_punto(r$umbral), sufijo, ".csv"
  )
  if (!prueba) write_csv(s, here("02_outputs", "predictions", archivo))
  cat(
    "Submission:", archivo, "- F1 de CV", round(r$f1, 4),
    "- proporción de pobres",
    round(prop, 4), "\n"
  )
  list(archivo = archivo, prop = prop)
}
final_sin <- final(res_sin[[mejor_spec]]$resumen, FALSE)
final_con <- final(res_con[[mejor_spec]]$resumen, TRUE, "_w")

if (!prueba) {
  write_rds(
    list(
      tabla = tabla, mejor_spec = mejor_spec,
      mejor = res_sin[[mejor_spec]]$resumen,
      mejor_w = res_con[[mejor_spec]]$resumen,
      oof = res_sin[[mejor_spec]]$oof,
      submissions = c(final_sin$archivo, final_con$archivo),
      prop = c(final_sin$prop, final_con$prop)
    ),
    file.path(dir_out, "resultados.rds")
  )
}
marca(
  "Tiempo total:", round(difftime(Sys.time(), inicio_total, units = "mins"), 1),
  "minutos"
)

# Notas -----------------------------------------------------------------------

# Efectos principales: F1 0.7068 (umbral 0.38; 0.635 con 0.5). Con las
# interacciones y cuadrados de 06: 0.7250 (umbral 0.38; 0.683 con 0.5),
# frente a 0.7417 del logit lasso con la misma fórmula. El LPM pierde ~0.017:
# la relación lineal en la probabilidad no captura que el efecto de cada
# variable es más fuerte cerca del umbral de pobreza.

# Predicciones fuera de [0, 1]: 23% (principales) y 27% (interacciones) de
# las predicciones fuera de muestra. Con interacciones, 25.8% son negativas y
# 1.2% mayores que 1. No se recortan: solo importa el orden respecto al
# umbral.

# Con pesos por clase: 0.7206 y 0.7279 (+0.014 y +0.003), con umbral 0.6, el
# borde de la grilla. Los pesos suben las predicciones; con la grilla hasta
# 0.6 el umbral óptimo podría estar por encima. En MCO los pesos cambian las
# pendientes, no solo el nivel, y por eso pueden mejorar el orden.

# Tiempo: ~19 minutos (cada lm con 497 columnas toma ~1 minuto).
