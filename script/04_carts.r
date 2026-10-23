###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   04_carts.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Estimar árboles de clasificación (CART) para predecir
#                      pobreza: elegir cp y umbral por validación cruzada con
#                      caret (F1 por hogar), con y sin up-sampling, y predecir
#                      en test.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos y conjuntos de variables
# 3. Funciones de validación cruzada y predicción
# 4. Pruebas
# 5. Modelo final: importancia de las variables y predicción en test

################################################################################

# Input:  Bases limpias creadas en 01_data_cleaning.r; train_clean.rds (o
#         train_clean_augmented.rds), test_clean.rds .
# Output: F1 de validación por cp y umbral, importancia de las variables (en
#         consola); output/submissions/CART_<base_><up_>cp_<cp>_threshold_<c>.csv .

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse,  # Manipulación de datos.
  rpart,      # Árboles de decisión (CART).
  caret       # Validación cruzada y fine tuning.
)

################################################################################

## 2. Cargar datos y conjuntos de variables

# Base de entrenamiento:
#   "train_clean": base original.
#   "train_clean_augmented": hogares pobres replicados hasta balancear las
#     clases (ver 01). Las copias conservan el id y quedan en el mismo fold; al
#     validar, prob_hogar() las junta en un solo hogar. No combinarla con
#     up-sampling, porque ya está balanceada.
#   "train_clean_sin_ricos": se entrena sin el 10% de hogares más ricos, pero
#     se valida con todos los hogares (si los ricos salieran también de la
#     validación, el F1 se mediría sobre una población distinta a la de test).
#     Por eso se carga train_clean y se usa hogar_rico.
base  <- "train_clean"
train <- read_rds(here("data", paste0(if (base == "train_clean_sin_ricos") "train_clean" else base, ".rds")))
test  <- read_rds(here("data", "test_clean.rds"))

# Filas con las que se entrenan los modelos (en la validación cruzada y en el
# modelo final). La validación usa siempre todas las filas del fold.
entrenar <- if (base == "train_clean_sin_ricos") train$hogar_rico == 0 else rep(TRUE, nrow(train))

# caret necesita niveles con nombres válidos para devolver probabilidades
# (classProbs). La clase positiva va primero.
y <- factor(train$pobre, levels = c(1, 0), labels = c("pobre", "no_pobre"))

# Categóricas con códigos DANE como factor: si quedan numéricas, el árbol las
# corta como si los códigos estuvieran ordenados (p. ej. régimen 1 < 2 < 3).
# El texto (dominio, depto) también pasa a factor.
categoricas <- c("max_educ", "regimen_salud", "actividad_sem_pasada",
                 "cotiza_pension", "tamano_empresa", "oficio",
                 "posicion_ocupacional", "tenencia_vivienda")
preparar <- function(x) {
  x |> mutate(across(any_of(categoricas), factor),
              across(where(is.character), factor))
}

# Conjuntos de variables (vectores de nombres). Para probar uno nuevo, se
# define aquí y se agrega una línea en la sección 4.

# Mejor conjunto de naive bayes (03_naive_bayes.r).
x1 <- c("anos", "anos_educ_imp", "mujer", "ocupado", "afiliado_salud",
        "regimen_salud", "actividad_sem_pasada", "log_meses_empresa",
        "tamano_empresa", "posicion_ocupacional", "horas_trabajo_semana",
        "arriendo_pc_sobre_lp", "prop_ocupados_hogar", "tenencia_vivienda",
        "recibio_arriendos_pensiones", "ayudas_instituciones")

# Ubicación: en naive bayes empeoraba (repite información), pero un árbol no
# suma evidencia y puede usarla para separar casos que las demás no separan.
x2 <- c(x1, "dominio", "depto", "cabecera")

# Variables del hogar que en naive bayes empeoraban por ser asimétricas o
# repetidas; a un árbol no le afecta la forma de la distribución.
x3 <- c(x2, "personas_hogar", "cuartos_dormir", "prop_edad_trabajar",
        "prop_mujer", "max_educ")

################################################################################

## 3. Funciones de validación cruzada y predicción

set.seed(123) # Para reproducibilidad

# Folds por hogar (los mismos del 03): todas las personas de un hogar quedan en
# el mismo fold. caret recibe, para cada fold, las filas de entrenamiento
# (index, sin los hogares que no se usan para entrenar) y las de validación
# (index_out, todas las del fold).
k       <- 10
hogares <- unique(train$id)
fold    <- sample(rep(1:k, length.out = length(hogares)))
fold    <- fold[match(train$id, hogares)]
index     <- map(1:k, ~ which(fold != .x & entrenar)) |> setNames(sprintf("Fold%02d", 1:k))
index_out <- map(1:k, ~ which(fold == .x)) |> setNames(sprintf("Fold%02d", 1:k))

# Grillas. cp: penalidad por hoja, relativa al error del árbol sin cortes. Con
# 500 mil observaciones los mejores árboles son grandes (cp bajos). caret
# estima un solo árbol por fold (el de menor cp) y poda para los demás, así
# que más valores de cp casi no cuestan. Umbral: probabilidad del hogar desde
# la que se predice pobre.
grid_cp     <- 10^seq(-5, -2, length.out = 16)
grid_umbral <- seq(0.10, 0.85, by = 0.01)
nombres_f1  <- sprintf("F1_%03d", round(100 * grid_umbral))  # F1_045 = 0.45

# pobre es del hogar y la submission es una fila por hogar. Como en el 03, la
# probabilidad del hogar es el promedio de los log odds de sus miembros.
prob_hogar <- function(p, id) {
  # Recorte: las hojas puras dan p = 0 o p = 1 y log odds infinitos.
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  tibble(id = id, prob = p) |>
    summarize(prob = plogis(mean(qlogis(prob))), .by = id)
}

# F1 a partir de predicciones y valores reales (0/1 o lógicos).
calcular_f1 <- function(pred, real) {
  vp        <- sum(pred == 1 & real == 1)
  precision <- vp / sum(pred == 1)
  recall    <- vp / sum(real == 1)
  2 * precision * recall / (precision + recall)
}

# summaryFunction de caret: F1 por hogar de un fold de validación, uno por
# umbral. rowIndex es la fila en train, con la que se recupera el id.
f1_hogar <- function(data, lev = NULL, model = NULL) {
  id    <- train$id[data$rowIndex]
  hogar <- prob_hogar(data$pobre, id) |>
    left_join(distinct(tibble(id = id, real = data$obs == "pobre")), by = "id")
  f1 <- map_dbl(grid_umbral, ~ calcular_f1(as.integer(hogar$prob > .x), hogar$real))
  setNames(f1, nombres_f1)
}

# selectionFunction de caret: el cp cuyo mejor umbral da el mayor F1 promedio.
# Así el modelo final (finalModel) se estima con ese cp.
elegir_cp <- function(x, metric, maximize) {
  which.max(apply(x[nombres_f1], 1, max, na.rm = TRUE))
}

# Fine tuning de cp y umbral por validación cruzada para un conjunto de
# variables. rebalanceo = "up" hace up-sampling solo en la parte de
# entrenamiento de cada fold (el fold de validación conserva la distribución
# original). Imprime el mejor resultado y devuelve el modelo, el F1 por cp y
# la mejor combinación.
cv_arbol <- function(variables, rebalanceo = NULL) {
  # data.frame y no tibble: caret les pone nombres de fila a los predictores.
  x <- as.data.frame(preparar(select(train, all_of(variables))))

  # Semilla en cada llamada: el up-sampling es aleatorio y así cada prueba da
  # el mismo resultado sin importar qué se corrió antes.
  set.seed(123)
  modelo <- train(
    x = x, y = y, method = "rpart",
    metric = "F1_050",  # caret exige una métrica; elegir_cp decide.
    tuneGrid = data.frame(cp = grid_cp),
    trControl = trainControl(method = "cv", index = index, indexOut = index_out,
                             classProbs = TRUE, summaryFunction = f1_hogar,
                             selectionFunction = elegir_cp,
                             sampling = rebalanceo, returnData = FALSE),
    parms = list(split = "gini")
  )

  # Mejor umbral y F1 para cada cp.
  por_cp <- modelo$results |>
    select(cp, all_of(nombres_f1)) |>
    pivot_longer(-cp, names_to = "umbral", values_to = "f1") |>
    mutate(umbral = as.numeric(sub("F1_", "", umbral)) / 100) |>
    slice_max(f1, n = 1, by = cp, with_ties = FALSE)
  mejor <- slice_max(por_cp, f1, n = 1, with_ties = FALSE)

  # caret estima el modelo final con todas las filas. Si se entrena sin una
  # parte de la base (train_clean_sin_ricos), se vuelve a estimar solo con las
  # filas de entrenamiento y el cp elegido.
  if (!all(entrenar)) {
    set.seed(123)
    modelo <- train(
      x = x[entrenar, , drop = FALSE], y = y[entrenar], method = "rpart",
      tuneGrid = data.frame(cp = mejor$cp),
      trControl = trainControl(method = "none", classProbs = TRUE, sampling = rebalanceo),
      parms = list(split = "gini")
    )
  }

  cat(length(variables), "variables -", base, "-", rebalanceo %||% "sin rebalanceo",
      "- mejor cp =", signif(mejor$cp, 3), "- umbral =", mejor$umbral,
      "- F1 =", round(mejor$f1, 4), "\n")
  invisible(list(modelo = modelo, por_cp = por_cp, mejor = mejor,
                 variables = variables, rebalanceo = rebalanceo))
}

# Predicción en test de un resultado de cv_arbol(): una fila por hogar, con el
# cp y el umbral elegidos. Guarda la submission con formato Kaggle
# (modelo_hiperparámetro_valor, sin punto decimal).
predecir_test <- function(resultado) {
  x_train <- preparar(select(train, all_of(resultado$variables)))
  # Test con los mismos niveles que train (un nivel que no está en train queda NA).
  x_test <- preparar(select(test, all_of(resultado$variables))) |>
    mutate(across(where(is.factor), ~ factor(.x, levels = levels(x_train[[cur_column()]]))))

  # na.action = na.pass: sin esto predict() descarta las filas con NA y las
  # predicciones quedan desalineadas con test$id.
  p <- predict(resultado$modelo, newdata = as.data.frame(x_test), type = "prob",
               na.action = na.pass)[, "pobre"]
  predicciones <- prob_hogar(p, test$id) |>
    transmute(id, pobre = as.integer(prob > resultado$mejor$umbral))

  sin_punto <- function(v) gsub(".", "", format(signif(v, 2), scientific = FALSE), fixed = TRUE)
  etiqueta_base <- c(train_clean = "", train_clean_augmented = "aug_",
                     train_clean_sin_ricos = "sinricos_")[[base]]
  archivo <- paste0("CART_", etiqueta_base,
                    if (identical(resultado$rebalanceo, "up")) "up_" else "",
                    "cp_", sin_punto(resultado$mejor$cp),
                    "_threshold_", sin_punto(resultado$mejor$umbral), ".csv")
  dir.create(here("output", "submissions"), showWarnings = FALSE)
  write_csv(predicciones, here("output", "submissions", archivo))
  cat("Guardado:", archivo, "-", nrow(predicciones), "hogares\n")
  invisible(predicciones)
}

################################################################################

## 4. Pruebas

# Cada prueba tarda unos 5 minutos (10 folds con la base completa).
# cv_x1    <- cv_arbol(x1)        # F1 = 0.648 (cp = 2.5e-05, umbral = 0.26)
cv_x1_up <- cv_arbol(x1, "up")    # F1 = 0.670 (cp = 2.5e-05, umbral = 0.61)
# cv_x2_up <- cv_arbol(x2, "up")  # F1 = 0.667 (cp = 6.3e-05, umbral = 0.61)
# cv_x3_up <- cv_arbol(x3, "up")  # F1 = 0.659 (cp = 6.3e-05, umbral = 0.57)

# F1 por cp de una prueba (para ver la curva de complejidad):
# print(cv_x1_up$por_cp)

################################################################################

## 5. Modelo final: importancia de las variables y predicción en test

final <- cv_x1_up

# Importancia por reducción de la impureza (Gini) en el árbol final,
# normalizada para que sume 100. Mide su uso en la construcción del árbol, no
# su aporte al F1; con variables correlacionadas, una puede ocultar a otra.
importancia <- final$modelo$finalModel$variable.importance
tibble(variable = names(importancia),
       importancia = 100 * importancia / sum(importancia)) |>
  print(n = Inf)

predecir_test(final)

### NOTAS:

# Up-sampling (dentro de cada fold) mejora el F1 en todos los cp: 0.648 -> 0.670
# con x1. Con clases balanceadas el umbral óptimo sube de ~0.26 a ~0.57.

# Los mejores árboles son grandes (cp ~ 2.5e-05): con 500 mil observaciones el
# árbol puede crecer mucho antes de sobreajustar.

# Con x1 el árbol (0.670) queda casi igual al mejor naive bayes (0.675).

# Agregar ubicación (x2: 0.667) o variables del hogar (x3: 0.659) no mejora un
# solo árbol: con más variables candidatas el árbol hace cortes más ruidosos.
# Puede ser distinto en un random forest, que promedia muchos árboles.
# arriendo_pc_sobre_lp y prop_ocupados_hogar son las variables más importantes,
# igual que en naive bayes.
