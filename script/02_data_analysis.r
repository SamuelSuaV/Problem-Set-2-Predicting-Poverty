###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   02_data_analysis.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Entender la muestra de entrenamiento usando estadísticas
#                      descriptivas y figuras.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Estadísticas descriptivas
# 4. Visualización de datos

################################################################################

# Input:  Bases limpias creadas en 01_data_cleaning.r; train_clean.rds ,
#         test_clean.rds .
# Output: output/tables/descriptivas.tex , output/figures/ingreso_dummies.png ,
#         output/figures/ingreso_continuas.png .

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  ggplot2,    # Visualización de datos.
  kableExtra  # Exportar tablas a LaTeX.
)

################################################################################

## 2. Cargar datos
train <- read_rds(here("data", "train_clean.rds"))

################################################################################

## 3. Estadísticas descriptivas

# ocupado viene como 1 o NA: NA = no ocupado.
train <- train |> mutate(ocupado = replace_na(ocupado, 0))

dummies   <- c("mujer", "afiliado_salud", "ocupado")
categoricas <- c("max_educ", "Estrato1", "tamano_empresa", "tenencia_vivienda")
continuas <- c("anos", "meses_empresa", "horas_trabajo_semana", "cuartos_hogar",
               "cuartos_dormir", "arriendo_estimado", "prop_edad_trabajar")
# En la tabla, max_educ y tenencia_vivienda entran como dummies por categoría
# (la media de un código no se interpreta). Estrato1 y tamano_empresa son
# ordinales y se dejan.
dummies_categorias <- grep("^(educ|vivienda)_", names(train), value = TRUE)
variables <- c(dummies, dummies_categorias, "Estrato1", "tamano_empresa",
               continuas)

# Medias por grupo y prueba t de diferencia de medias (pobres vs. no pobres).
descriptivas <- map_dfr(variables, \(v) {
  prueba <- t.test(train[[v]] ~ train$pobre)
  tibble(
    variable   = v,
    no_pobres  = prueba$estimate[[1]],
    pobres     = prueba$estimate[[2]],
    diferencia = pobres - no_pobres,
    p_valor    = prueba$p.value
  )
})

descriptivas |>
  kbl(format = "latex", booktabs = TRUE, digits = 3,
      col.names = c("Variable", "No pobres", "Pobres", "Diferencia", "p-valor"),
      caption = "Estadísticas descriptivas por condición de pobreza",
      label = "descriptivas") |>
  save_kable(here("output", "tables", "descriptivas.tex"))

################################################################################

## 4. Visualización de datos

# Lp varía por dominio; se grafica la mediana como referencia.
linea_pobreza <- median(train$linea_pobreza)
# Ingreso muy sesgado: el eje y se corta en el percentil 99.
tope_y <- quantile(train$ingreso, 0.99)

# Barras: ingreso promedio por categoría (dummies y categóricas).
grafica_barras <- train |>
  select(ingreso, all_of(c(dummies, categoricas))) |>
  pivot_longer(-ingreso, names_to = "variable", values_to = "valor") |>
  filter(!is.na(valor)) |>
  summarise(ingreso = mean(ingreso), .by = c(variable, valor)) |>
  ggplot(aes(x = factor(valor), y = ingreso)) +
  geom_col(fill = "steelblue") +
  geom_hline(yintercept = linea_pobreza, linetype = "dashed", color = "red") +
  facet_wrap(~ variable, scales = "free_x") +
  scale_y_continuous(labels = scales::comma) +
  labs(x = NULL, y = "Ingreso per cápita promedio") +
  theme_minimal()

ggsave(here("output", "figures", "ingreso_dummies.png"), grafica_barras,
       width = 10, height = 6)

# Scatter: ingreso vs. variables continuas.
grafica_scatter <- train |>
  select(ingreso, all_of(continuas)) |>
  pivot_longer(-ingreso, names_to = "variable", values_to = "valor") |>
  filter(!is.na(valor)) |>
  ggplot(aes(x = valor, y = ingreso)) +
  geom_point(alpha = 0.05, size = 0.3) +
  geom_hline(yintercept = linea_pobreza, linetype = "dashed", color = "red") +
  facet_wrap(~ variable, scales = "free_x") +
  coord_cartesian(ylim = c(0, tope_y)) +
  scale_y_continuous(labels = scales::comma) +
  labs(x = NULL, y = "Ingreso per cápita") +
  theme_minimal()

ggsave(here("output", "figures", "ingreso_continuas.png"), grafica_scatter,
       width = 10, height = 6)
