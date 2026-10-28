# Prueba: CART con todas las variables disponibles + interacciones, para medir
# la importancia de las variables (reducción de impureza). Base aumentada;
# mismos folds por hogar que 04 y 05.
suppressMessages({
  library(here)
  library(tidyverse)
  library(rpart)
  library(caret)
})

train <- read_rds(here("data", "train_clean_augmented.rds"))
y <- factor(train$pobre, levels = c(1, 0), labels = c("pobre", "no_pobre"))
id <- train$id

# Una versión por concepto: se excluyen identificadores (id, orden), pesos de la
# encuesta (fex_c, fex_dpto), variables solo de train y recodificaciones de
# otras variables (dummies educ_*/salud_*/vivienda_*, anos2, anos_educ,
# meses_empresa, max_grado_escolar, tasa_dependencia, cruces, edad_trabajar),
# que repartirían la importancia entre copias de la misma información.
binarias <- c(
  "afiliado_salud", "recibio_horas_extra", "recibio_primas",
  "recibio_bonificaciones",
  "subsidio_alimentacion", "subsidio_transporte", "subsidio_familiar",
  "subsidio_educativo",
  "pago_especie_alimentos", "pago_especie_vivienda", "transporte_empresa",
  "pago_especie_otros",
  "prima_servicios", "prima_navidad", "prima_vacaciones",
  "viaticos_permanentes",
  "bonificaciones_anuales", "segundo_trabajo", "quiere_mas_horas",
  "diligencias_mas_horas",
  "disponible_mas_horas", "diligencias_cambiar_trabajo",
  "disponible_nuevo_trabajo",
  "primera_vez_busca_trabajo", "ingreso_trabajo_desocupado",
  "ingreso_trabajo_desocupado_2",
  "recibio_arriendos_pensiones", "recibio_pension_jubilacion",
  "recibio_pension_alimenticia",
  "recibio_otros_ingresos", "dinero_hogares_pais", "dinero_hogares_exterior",
  "ayudas_instituciones", "dinero_intereses_dividendos", "dinero_cesantias",
  "dinero_otras_fuentes", "ocupado", "desocupado", "inactivo", "mujer",
  "cabecera"
)
factores <- c(
  "parentesco_jefe_hogar", "regimen_salud", "max_educ", "actividad_sem_pasada",
  "oficio", "posicion_ocupacional", "tamano_empresa", "cotiza_pension",
  "posicion_segundo_trabajo", "posicion_ultimo_trabajo", "tenencia_vivienda",
  "dominio", "depto"
)
numericas <- c(
  "anos", "anos_educ_imp", "educ_max_adultos", "cv_educ_adultos",
  "log_meses_empresa", "horas_trabajo_semana", "horas_segundo_trabajo",
  "cv_horas_ocupados",
  "cuartos_hogar", "cuartos_dormir", "cuota_amortizacion", "arriendo_estimado",
  "arriendo_pagado", "arriendo_pc_sobre_lp", "personas_hogar",
  "personas_unidad_gasto",
  "linea_indigencia", "linea_pobreza", "prop_edad_trabajar", "prop_mujer",
  "prop_ocupados_hogar", "prop_dependientes"
)

x <- train |>
  select(all_of(c(binarias, factores, numericas))) |>
  mutate(across(all_of(factores), ~ factor(as.character(.x))))

# Interacciones: productos entre pares de 12 numéricas centrales. Otras formas
# funcionales: razones y diferencias (las transformaciones monótonas, como log
# o cuadrado, no cambian los cortes de un árbol).
centrales <- c(
  "anos", "anos_educ_imp", "educ_max_adultos", "horas_trabajo_semana",
  "log_meses_empresa", "personas_hogar", "cuartos_dormir",
  "arriendo_pc_sobre_lp",
  "prop_ocupados_hogar", "prop_dependientes", "prop_mujer", "linea_pobreza"
)
for (p in combn(
  centrales, 2,
  simplify = FALSE
)) {
  x[[paste(p, collapse = "_x_")]] <- x[[p[1]]] * x[[p[2]]]
}
x <- x |> mutate(
  personas_por_dormitorio = personas_hogar / cuartos_dormir,
  personas_por_cuarto = personas_hogar / cuartos_hogar,
  brecha_educ_hogar = educ_max_adultos - anos_educ_imp,
  ocupados_por_dependiente = prop_ocupados_hogar / pmax(
    prop_dependientes, 1 / personas_hogar
  )
)
x <- as.data.frame(x)
rm(train)
invisible(gc())

set.seed(123)
k <- 10
hogares <- unique(id)
fold <- sample(rep(1:k, length.out = length(hogares)))[match(id, hogares)]
index <- map(1:k, ~ which(fold != .x)) |> setNames(sprintf("Fold%02d", 1:k))
cat("Filas:", nrow(x), "- columnas:", ncol(x), "\n")
