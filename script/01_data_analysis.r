###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   01_data_wrangling.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Entender la muestra de entrenamiento usando estadísticas
#                      descriptivas y figuras.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Estadísticas descriptivas
# 4. Figuras

################################################################################

# Input:  Dataframes crudos de hogares e individuos; train_hogares.csv ,
#         train_personas.csv .
# Output: 

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  srvyr,      # Diseño y análisis de encuestas.
  broom,      # Salida ordenada de pruebas estadísticas.
  kableExtra  # Exportar tablas a LaTeX.
)

################################################################################

## 2. Cargar datos
raw_test_hogares  <- read_csv(here("data", "test_hogares.csv"))
raw_test_personas <- read_csv(here("data", "test_personas.csv"))

################################################################################

## 3. Estadísticas descriptivas

