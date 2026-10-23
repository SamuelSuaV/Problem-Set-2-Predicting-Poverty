###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   03_naive_bayes.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar un modelo naive bayes para predecir pobreza y
#                      evaluarlo con F1 por validación cruzada.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Validación cruzada
# 4. Grid search de laplace y umbral
# 5. Predicción en test

################################################################################

# Input:  Bases limpias creadas en 01_data_cleaning.r; train_clean.rds (o
#         train_clean_augmented.rds), test_clean.rds .
# Output: F1 por fold y promedio y resultados del grid search (en consola);
#         output/submissions/NB_<base>_laplace_<l>_threshold_<c>.csv .

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  e1071       # Modelos naive bayes.
)

################################################################################

## 2. Cargar datos

# Base de entrenamiento:
#   "train_clean": base original.
#   "train_clean_augmented": hogares pobres replicados hasta balancear las
#     clases (ver 01). Las copias conservan el id y quedan en el mismo fold; al
#     validar, prob_hogar() las junta en un solo hogar.
#   "train_clean_sin_ricos": se entrena sin el 10% de hogares más ricos, pero
#     se valida con todos los hogares (si los ricos salieran también de la
#     validación, el F1 se mediría sobre una población distinta a la de test).
#     Por eso se carga train_clean y se usa hogar_rico.
base <- "train_clean"
train <- read_rds(here("data", paste0(if (base == "train_clean_sin_ricos") "train_clean" else base, ".rds")))
test <- read_rds(here("data", "test_clean.rds"))

# Filas con las que se entrenan los modelos (en la validación cruzada y en el
# modelo final). La validación usa siempre todas las filas del fold.
entrenar <- if (base == "train_clean_sin_ricos") train$hogar_rico == 0 else rep(TRUE, nrow(train))


y <- factor(train$pobre)

# Definimos varios conjuntos de covariables

# Nombres de las dummies de educación (se usan dentro de select con all_of).
x_educ <- c("educ_preescolar", "educ_primaria", "educ_secundaria",
            "educ_media", "educ_superior", "educ_no_sabe")

# Variables como factor para que naive bayes las trate como categóricas y no
# como normales: las categóricas con códigos DANE, el texto y las dummies 0/1.
# Se aplica igual a train y a test, si no predict() no reconoce los tipos.
categoricas <- c("max_educ", "regimen_salud", "actividad_sem_pasada", "cotiza_pension", "tamano_empresa", "oficio", "posicion_ocupacional", "tenencia_vivienda")
preparar <- function(x) {
  x |>
    mutate(across(any_of(categoricas), factor),
           across(where(is.character), factor),
           across(where(~ is.numeric(.x) && all(.x %in% c(0, 1, NA))), factor))
}

# Educación
x1 <- train |> select(anos, educ_ninguno) # Iniciamos con edad y educ. minima
x2 <- train |> select(anos, anos2, educ_ninguno) # Añadimos edad al cuadrado por no linealidad
x3 <- train |> select(anos, anos2, educ_ninguno, educ_preescolar) # Añadimos educ. preescolar
x4 <- train |> select(anos, anos2, educ_preescolar) # Añadimos educ. preescolar
x5 <- train |> select(anos, anos2, all_of(x_educ)) # Añadimos todas las variables de educación
x6 <- train |> select(anos, anos2, anos_educ_imp) # Cambiamos por años continuos de educ.

# Tomamos el mejor modelo (con todas las dummies de educ.) y añadimos otras variables de individuo

# Genero
x7 <- train |> select(anos, anos2, all_of(x_educ), mujer) # Añadimos variable de genero
x8 <- train |> select(anos, anos2, educ_ninguno, mujer) # Revisamos si eliminar vars. de educ mejora el desempeño al combinar con genero.

# Empleo
x9 <- train |> select(anos, anos2, all_of(x_educ), ocupado) # Añadimos variable de empleo

# Salud
x10 <- train |> select(anos, anos2, all_of(x_educ), afiliado_salud) # Añadimos variable de afiliacion
x11 <- train |> select(anos, anos2, all_of(x_educ), afiliado_salud, regimen_salud) # Probamos si añadir regimen de salud mejora el desempeño

# Actividades
x12 <- train |> select(anos, anos2, all_of(x_educ), actividad_sem_pasada) # Probamos si añadir actividad de la semana pasada mejora el desempeño

# Pension
x13 <- train |> select(anos, anos2, all_of(x_educ), cotiza_pension) # Probamos si añadir cotizacion a pension mejora el desempeño

# Variables de empresa
x14 <- train |> select(anos, anos2, all_of(x_educ), meses_empresa) # Probamos si añadir meses de empresa mejora el desempeño
x15 <- train |> select(anos, anos2, all_of(x_educ), tamano_empresa) # Probamos si añadir tamaño de empresa mejora el desempeño
x16 <- train |> select(anos, anos2, all_of(x_educ), oficio) # Probamos si añadir oficio mejora el desempeño
x17 <- train |> select(anos, anos2, all_of(x_educ), posicion_ocupacional) # Probamos si añadir posicion ocupacional mejora el desempeño
x18 <- train |> select(anos, anos2, all_of(x_educ), horas_trabajo_semana) # Probamos si añadir horas de trabajo mejora el desempeño

# Pagos
x19 <- train |> select(anos, anos2, all_of(x_educ), recibio_horas_extra)
x20 <- train |> select(anos, anos2, all_of(x_educ), recibio_primas)
x21 <- train |> select(anos, anos2, all_of(x_educ), recibio_bonificaciones)

# Segundo trabajo y busqueda de trabajo
x22 <- train |> select(anos, anos2, all_of(x_educ), segundo_trabajo)

# Otros ingresos
# poner ayudas instituciones

# Vivienda
# poner tenencia vivienda
# poner prop_mujeres
# poner prop_edad_trabajar

# Todas las variables anteriores que vale la pena añadir al modelo de educacion
xtodas <- train |> select(anos, anos_educ_imp,
                          mujer, ocupado, afiliado_salud, regimen_salud,
                          actividad_sem_pasada, log_meses_empresa, tamano_empresa,
                          posicion_ocupacional, horas_trabajo_semana, arriendo_pc_sobre_lp,
                          prop_ocupados_hogar, tenencia_vivienda, recibio_arriendos_pensiones,
                          ayudas_instituciones) # Probamos todas

set.seed(123) # Para reproducibilidad

# Folds por hogar: todas las personas de un hogar quedan en el mismo fold,
# porque comparten el valor de pobre.
k       <- 10
hogares <- unique(train$id)
fold    <- sample(rep(1:k, length.out = length(hogares)))
fold    <- fold[match(train$id, hogares)]


################################################################################

## 3. Validación cruzada

# Probabilidad de que cada hogar sea pobre: promedio de los log odds
# (log(p / (1 - p))) de sus miembros, devuelto a probabilidad. En esa escala
# pesan más los miembros con señal fuerte, y da mejor F1 que promediar
# probabilidades. El hogar se predice pobre si pasa del umbral.
prob_hogar <- function(modelo, x, id, umbral = 0.5) {
  # Recorte para que p = 0 o p = 1 no den log odds infinitos.
  p <- pmin(pmax(predict(modelo, x, type = "raw")[, "1"], 1e-6), 1 - 1e-6)
  tibble(id = id, prob = p) |>
    summarize(prob = plogis(mean(qlogis(prob))), .by = id) |>
    mutate(pred = as.integer(prob > umbral))
}

# F1 a partir de predicciones y valores reales (0/1).
calcular_f1 <- function(pred, real) {
  vp        <- sum(pred == 1 & real == 1)
  precision <- vp / sum(pred == 1)
  recall    <- vp / sum(real == 1)
  2 * precision * recall / (precision + recall)
}

# F1 por validación cruzada de naive bayes con covariables x. Se predice
# igual que en test: se promedian los log odds de los miembros de cada
# hogar y el F1 se calcula con una predicción por hogar.
cv_naive_bayes <- function(x, y, fold, id = train$id) {
  x <- preparar(x)

  k  <- max(fold)
  f1 <- numeric(k)
  for (i in 1:k) {
    prueba <- fold == i
    modelo <- naiveBayes(x[!prueba & entrenar, ], y[!prueba & entrenar])

    # Todos los miembros de un hogar comparten pobre: se toma uno por hogar.
    hogar <- prob_hogar(modelo, x[prueba, ], id[prueba]) |>
      left_join(distinct(tibble(id = id[prueba], real = y[prueba])), by = "id")
    f1[i] <- calcular_f1(hogar$pred, hogar$real)
    cat("Fold", i, "- F1:", round(f1[i], 4), "\n")
  }
  # Promedio entre folds (cada F1 ya usa las probabilidades promediadas).
  cat("F1 promedio de los", k, "folds:", round(mean(f1), 4), "\n")
  invisible(f1)
}

# f1_x1 <- cv_naive_bayes(x1, y, fold)
# f1_x2 <- cv_naive_bayes(x2, y, fold)
# f1_x3 <- cv_naive_bayes(x3, y, fold)
# f1_x4 <- cv_naive_bayes(x4, y, fold)
# f1_x5 <- cv_naive_bayes(x5, y, fold)
## f1_x6 <- cv_naive_bayes(x6, y, fold)
# f1_x7 <- cv_naive_bayes(x7, y, fold)
# f1_x8 <- cv_naive_bayes(x8, y, fold)
# f1_x9 <- cv_naive_bayes(x9, y, fold)
# f1_x10 <- cv_naive_bayes(x10, y, fold)
# f1_x11 <- cv_naive_bayes(x11, y, fold)
# f1_x12 <- cv_naive_bayes(x12, y, fold)
# f1_x13 <- cv_naive_bayes(x13, y, fold)
# f1_x14 <- cv_naive_bayes(x14, y, fold)
# f1_x15 <- cv_naive_bayes(x15, y, fold)
# f1_x16 <- cv_naive_bayes(x16, y, fold)
# f1_x17 <- cv_naive_bayes(x17, y, fold)
# f1_x18 <- cv_naive_bayes(x18, y, fold)
# f1_x19 <- cv_naive_bayes(x19, y, fold)
# f1_x20 <- cv_naive_bayes(x20, y, fold)
# f1_x21 <- cv_naive_bayes(x21, y, fold)
# f1_x22 <- cv_naive_bayes(x22, y, fold)

f1_xtodas <- cv_naive_bayes(xtodas, y, fold)

################################################################################

## 4. Grid search de laplace y umbral

# laplace: suavizado de las categóricas en naiveBayes() (0 = sin suavizado).
# umbral (c): probabilidad promedio del hogar desde la que se predice pobre.
# Por cada laplace se estima el modelo una vez por fold; todos los umbrales se
# evalúan con esas mismas probabilidades, así que probar muchos casi no cuesta.
grid_laplace <- c(0, 0.5, 1, 2, 5, 10, 50)
grid_umbral  <- seq(0.1, 0.6, by = 0.01)

# F1 promedio de los folds para cada combinación de laplace y umbral.
grid_search <- function(x, y, fold, id = train$id) {
  x <- preparar(x)
  # Un proceso por valor de laplace (en paralelo).
  resultados <- parallel::mclapply(grid_laplace, function(l) {
    map_dfr(1:max(fold), function(i) {
      prueba <- fold == i
      modelo <- naiveBayes(x[!prueba & entrenar, ], y[!prueba & entrenar], laplace = l)
      hogar  <- prob_hogar(modelo, x[prueba, ], id[prueba]) |>
        left_join(distinct(tibble(id = id[prueba], real = y[prueba])), by = "id")
      tibble(laplace = l, fold = i, umbral = grid_umbral,
             f1 = map_dbl(grid_umbral,
                          ~ calcular_f1(as.integer(hogar$prob > .x), hogar$real)))
    })
  }, mc.cores = min(length(grid_laplace), parallel::detectCores() - 2))
  bind_rows(resultados) |> summarize(f1 = mean(f1), .by = c(laplace, umbral))
}

grid_xtodas <- grid_search(xtodas, y, fold)

# Mejor umbral para cada laplace, y la mejor combinación.
grid_xtodas |> slice_max(f1, n = 1, by = laplace, with_ties = FALSE) |> print()
# Desempate: entre las combinaciones a menos de 0.0005 del mejor F1 (ruido),
# se toma el menor laplace (el modelo con menos suavizado).
mejor <- grid_xtodas |>
  filter(f1 >= max(f1) - 0.0005) |>
  arrange(laplace, desc(f1)) |>
  slice(1)
cat("Mejor: laplace =", mejor$laplace, "- umbral =", mejor$umbral,
    "- F1 =", round(mejor$f1, 4), "\n")

################################################################################

## 5. Predicción en test

# Mejor modelo (xtodas) con los hiperparámetros del grid search y el mismo
# tratamiento de variables que en la validación cruzada.
modelo_xtodas <- naiveBayes(preparar(xtodas)[entrenar, ], y[entrenar], laplace = mejor$laplace)
# Una predicción por hogar: promedio de los log odds de sus miembros.
predicciones <- prob_hogar(modelo_xtodas,
                           preparar(test |> select(all_of(names(xtodas)))),
                           test$id, umbral = mejor$umbral)
nrow(predicciones)  # una fila por hogar de test

predicciones <- predicciones |> select(id, pred) |> rename(pobre = pred)

# Nombre con formato Kaggle: modelo_hiperparámetro_valor, sin punto decimal
# (p. ej. NB_laplace_0_threshold_045.csv; con otra base, NB_aug_... o
# NB_sinricos_...). Sale de los valores del grid search.
sin_punto <- function(v) gsub(".", "", as.character(round(v, 2)), fixed = TRUE)
etiqueta_base <- c(train_clean = "", train_clean_augmented = "aug_",
                   train_clean_sin_ricos = "sinricos_")[[base]]
archivo <- paste0("NB_", etiqueta_base, "laplace_", sin_punto(mejor$laplace),
                  "_threshold_", sin_punto(mejor$umbral), ".csv")
dir.create(here("output", "submissions"), showWarnings = FALSE)
predicciones |> write_csv(here("output", "submissions", archivo))

### NOTAS:

# El modelo de naive bayes requiere mucho poder de computo cuando hay muchas variables.
# Pasar de educación a educación al cuadrado mejora mucho el desempeño
# Tener ninguna educación y preescolar a la vez empeora mucho el desempeño. No se aporta mucha información adicional.
# La educación preescolar por si sola provee menos info. util que la educación mínima.

# Parece que el efecto de sesgo que puede causar poner variables relacionadas de educ.
# es menor al efecto de sesgo de no tener esta información. 

# El modelo empeora al tener educación y genero. Posiblemente por el sesgo que causa
# tener variables relacionadas. Pero reducir las vars. de educ. no ayuda.

# Las variables de genero, formalidad ayudan poco. Ocupado ayuda mucho.

# Regimen salud explica mucho más que afiliación a salud.

# Cotiza pension solo se pregunta a los ocupados y mide informnación similar a las variables
# de salud y de actividad de la semana pasada. Por eso no ayuda  a mejorar el desempeño.

# Con otras variables de pagos del trabajo, el efecto adicional es bajo, dado que ya hay otros
# indicadores de la calidad del empleo.

# Augmentation no ayuda en naive bayes. Probablemente porque afecta las distribuciones.

# Los cruces entre variables categoricas que proveen info. similar no son muy utiles.
# Puede ser porque terminan teniendo muchos niveles y pocos datos por nivel.
# Tipos de variables: naiveBayes() trata toda variable numérica como normal, aunque
# sean códigos de una categórica. Las categóricas y las dummies deben ir como factor
# (preparar()), y con el mismo tipo en train y en test o predict() falla.

# NA: naive bayes ignora las variables en NA. Si el NA tiene significado hay que
# recodificarlo: pasar ocupado de NA a 0 (no ocupado) subió el F1 de 0.29 a 0.41.

# Predecir por hogar: pobre es del hogar, así que se combinan las probabilidades de
# los miembros. Promediar los log odds da mejor F1 que promediar probabilidades
# (0.671 vs 0.660, mejor en los 10 folds). Ponderar solo adultos o solo el jefe
# empeora: los niños también informan (hogares pobres tienen más dependientes).

# Umbral: 0.5 no es el óptimo. Las probabilidades de naive bayes están
# descalibradas (sobrestiman la pobreza por contar varias veces la misma
# evidencia), así que el umbral se escoge por validación cruzada. Para comparar
# conjuntos de variables hay que usar el mejor umbral de cada uno: con 0.5,
# cotiza_pension como factor parecía mucho peor que como numérica y no lo era.

# Laplace casi no cambia el F1 (quinta decimal): con 500 mil observaciones los
# conteos por nivel son grandes y el suavizado no los mueve.

# Medidas repetidas: tener las dummies educ_*, max_educ y anos_educ_imp a la vez
# cuenta tres veces la educación. Dejar solo anos_educ_imp subió el F1 de 0.590 a
# 0.612, y quitar anos2 (repite anos) lo subió a 0.623. Corrige lo que se pensaba
# arriba sobre las variables relacionadas de educación.

# Residualizar las variables (x12, x13) para quitar la correlación empeora: elimina
# la correlación lineal pero no la dependencia, y vuelve continuas las dummies.

# Variables asimétricas: la normal les queda mal. Los arriendos crudos empeoran el
# modelo, pero el arriendo per cápita sobre la línea de pobreza en log
# (arriendo_pc_sobre_lp) es la variable que más aporta. log_meses_empresa (NA para
# no ocupados) es mejor que meses_empresa con 0 para los no ocupados.

# Variables del hogar: arriendo_pc_sobre_lp y prop_ocupados_hogar aportan mucho.
# La educación del jefe y la máxima del hogar empeoran porque repiten anos_educ_imp.
# Personas del hogar, cuartos y línea de pobreza también empeoran.

# Otros ingresos y segundo trabajo aportan poco. Lo más útil es
# recibio_arriendos_pensiones; ingreso_trabajo_desocupado ayuda apenas.

# Ubicación (dominio, depto, cabecera) empeora en los 10 folds, aunque la pobreza
# varía mucho entre regiones: esa información ya está en las variables de cada
# persona (régimen de salud, educación, arriendo) y naive bayes la cuenta otra vez.
# En modelos de árboles puede ser distinto.
