###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   00_rundirectory.R
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Correr todos los scripts en orden, desde los CSV crudos
#                      hasta la comparación de modelos.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Scripts en orden
# 3. Correr los scripts

# Input:  Los 4 CSV crudos de data/ ({train,test}_{hogares,personas}.csv).
# Output: Los de cada script: bases en data/, tablas y figuras en 02_outputs/ y
#         submissions en 02_outputs/predictions/.
#
# Correr desde la raíz: Rscript 01_code/00_rundirectory.R
# Necesita los 4 CSV de Kaggle en data/ (ver README).
# Completo tarda muchas horas (11, el modelo final, es casi todo el tiempo;
# guarda cada prueba y retoma lo que ya estimó). Con PRUEBA=1
# (PRUEBA=1 Rscript 01_code/00_rundirectory.R) los scripts de modelos corren en
# su modo de prueba: el 10% de los hogares, pocas iteraciones y sin
# submissions, para verificar en poco tiempo que todo corre. 01, 02 y 03
# corren completos porque los demás usan sus bases.
#
# Las funciones comunes de los modelos están en funciones_comunes.R. El
# documento de resultados (latex/) y las presentaciones se compilan aparte.

# 1. Cargar librerías ---------------------------------------------------------

if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
library(pacman)
p_load(
  here # Rutas relativas a la raíz del proyecto.
)

# 2. Scripts en orden ---------------------------------------------------------

scripts <- c(
  "01_data_cleaning", # CSV crudos -> bases limpias por persona
  "02_bases_hogar", # bases por hogar (una fila por hogar)
  "03_data_analysis", # estadísticas descriptivas y figuras
  "04_naive_bayes",
  "05_lpm",
  "06_logit",
  "07_logit_parsimonioso",
  "08_logit_interacciones", # guarda logit_oof.rds, que usan 11 y 15
  "09_carts",
  "10_xgboost_persona",
  "11_xgboost_hogar", # modelo final
  "12_lightgbm",
  "13_lightgbm_variables",
  "14_lightgbm_ingreso",
  "15_comparacion_modelos", # compara los modelos anteriores
  "16_figuras_presentacion" # figuras y datos de las presentaciones
)

# 3. Correr los scripts -------------------------------------------------------

# Cada script corre en un proceso de R aparte: no hereda los objetos del
# anterior y libera su memoria al terminar.
# Si uno falla, se detiene. Se carga con source(..., encoding = "UTF-8") para
# que las tildes salgan bien también en Windows. En modo de prueba se activan
# las tres formas que usan los scripts: prueba <- TRUE antes del source() (05
# y 12-14), el argumento --prueba (06-08) y la variable de entorno PRUEBA=1,
# que los procesos heredan (04, 09, 10 y 11).
prueba <- Sys.getenv("PRUEBA") == "1"
rscript <- file.path(R.home("bin"), "Rscript")
inicio <- Sys.time()
for (s in scripts) {
  cat(format(Sys.time(), "%H:%M:%S"), "-", s, if (prueba) "(prueba)", "\n")
  codigo <- sprintf(
    "%ssource('%s', encoding = 'UTF-8')",
    if (prueba) "prueba <- TRUE; " else "", here("01_code", paste0(s, ".R"))
  )
  estado <- system2(rscript, c("-e", shQuote(codigo), if (prueba) "--prueba"))
  if (estado != 0) stop(s, " falló (código ", estado, ")")
}
cat(
  "Listo:", length(scripts), "scripts en",
  round(as.numeric(difftime(Sys.time(), inicio, units = "mins"))), "minutos\n"
)
