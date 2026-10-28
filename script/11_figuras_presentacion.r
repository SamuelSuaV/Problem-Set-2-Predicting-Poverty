###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   11_figuras_presentacion.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Figuras y datos de las presentaciones (presentacion/):
#                      predictores del mejor modelo, nube 3D de hogares con el
#                      resultado del modelo final, métricas de clasificación
#                      (matriz de confusión, ROC-AUC) de los 8 modelos de 06,
#                      F1 según el umbral, errores según la distancia a la
#                      línea y F1 según el número de variables por algoritmo.
###############################################################################

# Estructura:
# 1. Cargar librerías y datos
# 2. Predictores del mejor modelo
# 3. Nube 3D de hogares
# 4. Métricas de clasificación de los 8 modelos
# 5. F1, precisión y recall según el umbral (modelo final)
# 6. Errores según la distancia a la línea (modelo final)
# 7. F1 según el número de variables, por algoritmo

################################################################################

# Input:  data/train_hogar.rds (01b); data/train_clean.rds (01);
#         ps2_predicciones_fuera_de_muestra/predicciones_oof.csv y modelos.csv
#         (probabilidades fuera de muestra de los 8 modelos de 06, con los 5
#         folds por hogar); ps2_predicciones_fuera_de_muestra/originales/
#         H0_base01b.rds (importancia por ganancia de la base por hogar).
# Output: output/figures/pres_predictores_cat.png , pres_predictores_cont.png ,
#         pres_3d.png , pres_3d.html (con su carpeta pres_3d_files/);
#         output/tables/presentacion/*.dat (tablas que lee pgfplots en las
#         slides); output/pruebas/f1_vs_k/*.rds (una por algoritmo y k; si ya
#         existe, se lee).
#
# Correr desde la raíz. Las secciones 1-6 tardan un par de minutos; la 7,
# ~1 hora (5 algoritmos x 9 tamaños x 5 folds), y retoma si se interrumpe.

################################################################################

## 1. Cargar librerías y datos

library(pacman)
p_load(
  here,        # Rutas relativas a la raíz del proyecto.
  tidyverse,   # Manipulación de datos y gráficas.
  e1071,       # Naive bayes.
  rpart,       # CART.
  ranger,      # Random forest.
  xgboost,     # XGBoost.
  plot3D,      # Nube 3D estática.
  plotly,      # Nube 3D interactiva.
  htmlwidgets  # Guardar la nube interactiva en HTML.
)
# calcular_f1() y evaluar_umbral().
source(here("script", "00_master_file.r"))

dir_fig <- here("output", "figures")
dir_dat <- here("output", "tables", "presentacion")
dir_k   <- here("output", "pruebas", "f1_vs_k")
dir_oof <- here("ps2_predicciones_fuera_de_muestra")
walk(c(dir_dat, dir_k), dir.create, showWarnings = FALSE, recursive = TRUE)

# Tabla de texto para pgfplots (\addplot table): separada por espacios, sin
# comillas; el texto no puede tener espacios.
guardar_dat <- function(x, nombre) {
  write.table(x, file.path(dir_dat, paste0(nombre, ".dat")), quote = FALSE,
              row.names = FALSE, sep = " ")
}

# Colores de las presentaciones.
col_azul <- "#0B3C5D"; col_azulclaro <- "#328CC1"; col_dorado <- "#D9B310"
col_rojo <- "#B23A48"; col_verde <- "#2E8B57"; col_gris <- "#6B7280"

hog <- read_rds(here("data", "train_hogar.rds"))
oof <- read_csv(file.path(dir_oof, "predicciones_oof.csv"), show_col_types = FALSE)
modelos <- read_csv(file.path(dir_oof, "modelos.csv"), show_col_types = FALSE)
stopifnot(identical(as.character(hog$id), oof$id), identical(as.integer(hog$pobre), as.integer(oof$pobre)))
y    <- oof$pobre
fold <- oof$fold
grid_umbral <- seq(0.10, 0.85, by = 0.01)

# Del archivo de personas: ingreso per cápita de la unidad de gasto y línea
# (iguales para todos los miembros), y preguntas de la persona resumidas por
# hogar.
pers <- read_rds(here("data", "train_clean.rds")) |>
  select(id, ingreso, linea_pobreza, ingreso_trabajo_desocupado_2,
         recibio_arriendos_pensiones, ocupado, personas_unidad_gasto)
por_hogar <- pers |>
  summarize(.by = id,
            ingreso = first(ingreso), linea = first(linea_pobreza),
            inactivo_con_ingreso = as.integer(any(ingreso_trabajo_desocupado_2 == 1, na.rm = TRUE)),
            arriendos_pensiones  = as.integer(any(recibio_arriendos_pensiones == 1, na.rm = TRUE)),
            personas = first(personas_unidad_gasto))
rm(pers); invisible(gc())
hog <- left_join(hog, por_hogar, by = "id")
stopifnot(!anyNA(hog$ingreso))
hog <- mutate(hog, r = ingreso / linea)   # ingreso sobre la línea (< 1 = pobre)

# Resultado del modelo final con su umbral (0,39).
u_final <- modelos$umbral[modelos$columna == "prob_modelo_final"]
pred_final <- as.integer(oof$prob_modelo_final > u_final)
resultado <- case_when(y == 1 & pred_final == 1 ~ "Pobre detectado",
                       y == 1 & pred_final == 0 ~ "Pobre excluido",
                       y == 0 & pred_final == 1 ~ "No pobre incluido",
                       TRUE                     ~ "No pobre bien clasificado")

################################################################################

## 2. Predictores del mejor modelo

# Las variables con más importancia de extremo a extremo (sección 16 de 05b):
# posición ocupacional, haber recibido arriendos o pensiones, ingresos por
# trabajo de los inactivos, arriendo por persona, horas y ocupados. Se
# muestran a nivel hogar, con el % de hogares pobres de cada grupo.
tasa_general <- 100 * mean(y)

posiciones <- c("0" = "Jefe no ocupado", "1" = "Empleado particular", "2" = "Empleado del gobierno",
                "3" = "Empleado doméstico", "4" = "Cuenta propia", "5" = "Patrón o empleador",
                "6" = "Familiar sin pago", "7" = "Sin pago (otros hogares)", "8" = "Jornalero o peón",
                "9" = "Otro")
tenencias <- c("1" = "Propia, pagada", "2" = "Propia, pagando", "3" = "En arriendo",
               "4" = "Usufructo", "5" = "Posesión sin título", "6" = "Otra")
cat_largo <- bind_rows(
  tibble(panel = "Posición ocupacional del jefe (P6430)",
         grupo = posiciones[as.character(hog$jefe_posicion)], pobre = y),
  tibble(panel = "Tenencia de la vivienda (P5090)",
         grupo = tenencias[as.character(hog$tenencia)], pobre = y),
  tibble(panel = "Algún miembro recibe arriendos o pensiones (P7495)",
         grupo = if_else(hog$arriendos_pensiones == 1, "Sí", "No"), pobre = y),
  tibble(panel = "Algún inactivo recibió ingresos por trabajo (P7472)",
         grupo = if_else(hog$inactivo_con_ingreso == 1, "Sí", "No"), pobre = y),
  tibble(panel = "Jefe cotiza a pensión (P6920)",
         grupo = if_else(hog$jefe_formal == 1, "Sí", "No"), pobre = y)
) |>
  summarize(.by = c(panel, grupo), hogares = n(), pct_pobres = 100 * mean(pobre)) |>
  filter(hogares >= 100) |>
  mutate(etiqueta = sprintf("%.0f%%", pct_pobres),
         grupo = fct_reorder(paste(panel, grupo, sep = "___"), pct_pobres))

g_cat <- ggplot(cat_largo, aes(pct_pobres, grupo)) +
  geom_col(fill = col_azulclaro, width = 0.7) +
  geom_vline(xintercept = tasa_general, linetype = "dashed", color = col_rojo) +
  geom_text(aes(label = etiqueta), hjust = -0.15, size = 3, color = col_azul) +
  scale_y_discrete(labels = function(x) sub("^.*___", "", x)) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  facet_wrap(~ panel, scales = "free_y", ncol = 2) +
  labs(x = "% de hogares pobres (línea punteada: 20% de todos los hogares)", y = NULL) +
  theme_minimal(base_size = 10) +
  theme(strip.text = element_text(face = "bold", color = col_azul, hjust = 0),
        panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())
ggsave(file.path(dir_fig, "pres_predictores_cat.png"), g_cat, width = 10, height = 6.2, dpi = 200)

# Continuas: % de hogares pobres por grupo de igual tamaño (quintiles o
# deciles; los empates en 0 quedan en un grupo).
binned <- function(x, nombre, n = 10) {
  tibble(x = x, pobre = y) |>
    mutate(grupo = ntile(x, n)) |>
    summarize(.by = grupo, x = median(x), pct_pobres = 100 * mean(pobre), hogares = n()) |>
    summarize(.by = x, pct_pobres = weighted.mean(pct_pobres, hogares), hogares = sum(hogares)) |>
    mutate(variable = nombre)
}
cont <- bind_rows(
  binned(exp(hog$log_arriendo_pc_lp), "Arriendo por persona / línea"),
  binned(hog$horas_pc, "Horas trabajadas por persona (semana)"),
  binned(hog$ocup_pc, "Proporción de ocupados del hogar"),
  binned(hog$cuartos_pc, "Cuartos por persona")
)
g_cont <- ggplot(cont, aes(x, pct_pobres)) +
  geom_hline(yintercept = tasa_general, linetype = "dashed", color = col_rojo) +
  geom_line(color = col_azulclaro, linewidth = 0.8) +
  geom_point(aes(size = hogares), color = col_azul, show.legend = FALSE) +
  scale_size_area(max_size = 3) +
  facet_wrap(~ variable, scales = "free_x", ncol = 2) +
  labs(x = "Mediana de cada decil de hogares", y = "% de hogares pobres") +
  theme_minimal(base_size = 10) +
  theme(strip.text = element_text(face = "bold", color = col_azul, hjust = 0),
        panel.grid.minor = element_blank())
ggsave(file.path(dir_fig, "pres_predictores_cont.png"), g_cont, width = 10, height = 6.2, dpi = 200)

################################################################################

## 3. Nube 3D de hogares

# x: log del ingreso sobre la línea (0 = en la línea); y: proporción de
# ocupados; z: personas de la unidad de gasto. Color: resultado del modelo
# final. Muestra de 8.000 hogares (los errores con más peso para que se vean:
# todos los de la muestra proporcional, más hasta completar).
niveles_res <- c("Pobre detectado", "Pobre excluido", "No pobre incluido", "No pobre bien clasificado")
colores_res <- c(col_azul, col_rojo, col_dorado, "#B8C4CE")
set.seed(123)
nube <- tibble(x = pmin(pmax(log(pmax(hog$r, 0.02)), -3), 3), yy = hog$ocup_pc,
               z = pmin(hog$personas, 10), resultado = factor(resultado, niveles_res)) |>
  slice_sample(n = 8000) |>
  mutate(z_j = z + runif(n(), -0.3, 0.3), yy_j = pmin(pmax(yy + runif(n(), -0.02, 0.02), 0), 1)) |>
  arrange(desc(resultado))   # los errores se dibujan encima

png(file.path(dir_fig, "pres_3d.png"), width = 2000, height = 1650, res = 220)
par(mar = c(4.5, 0.5, 0.5, 0.5), xpd = NA)
scatter3D(nube$x, nube$yy_j, nube$z_j, colvar = as.integer(nube$resultado),
          col = colores_res, pch = 16, cex = 0.45, bty = "g", ticktype = "detailed",
          theta = 40, phi = 15, colkey = FALSE, expand = 0.8, cex.lab = 1.1,
          xlab = "log(ingreso / línea)", ylab = "Proporción de ocupados", zlab = "Personas")
legend("bottom", inset = c(0, -0.12), legend = niveles_res, col = colores_res, pch = 16,
       bty = "n", horiz = TRUE, cex = 0.95, pt.cex = 1.5)
dev.off()

p3d <- plot_ly(nube, x = ~x, y = ~yy_j, z = ~z_j, color = ~resultado, colors = colores_res,
               type = "scatter3d", mode = "markers", marker = list(size = 2)) |>
  layout(scene = list(xaxis = list(title = "log(ingreso / línea)"),
                      yaxis = list(title = "Proporción de ocupados"),
                      zaxis = list(title = "Personas")))
saveWidget(p3d, file.path(dir_fig, "pres_3d.html"), selfcontained = FALSE, libdir = "pres_3d_files",
           title = "Hogares: resultado del modelo final")

# Resumen por resultado (para el texto de la slide).
resumen_3d <- tibble(resultado = factor(resultado, niveles_res), r = hog$r, ocup = hog$ocup_pc,
                     personas = hog$personas) |>
  summarize(.by = resultado, hogares = n(), mediana_r = median(r), media_ocup = mean(ocup),
            media_personas = mean(personas), pct_cerca = 100 * mean(abs(log(pmax(r, 1e-3))) < log(1.25))) |>
  arrange(resultado)
print(resumen_3d)
guardar_dat(mutate(resumen_3d, resultado = gsub(" ", "_", resultado)), "resumen_3d")

################################################################################

## 4. Métricas de clasificación de los 8 modelos

# Con el umbral de cada modelo (el de 06). Precisión, recall y accuracy con
# todos los hogares; F1 como en 06 (promedio de los 5 folds); AUC: probabilidad
# de que un pobre tenga mayor puntaje que un no pobre.
auc <- function(p, y) {
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (as.numeric(n1) * n0)
}
claves <- c(prob_naive_bayes = "nb", prob_cart = "cart", prob_xgboost_persona = "xgbpers",
            prob_logit_lasso = "logit", prob_xgboost_hogar_base = "xgbbase",
            prob_xgboost_hogar_sin_tweedie = "xgbsintw", prob_random_forest_hogar = "rf",
            prob_modelo_final = "final")
metricas <- map_dfr(seq_len(nrow(modelos)), function(i) {
  p <- oof[[modelos$columna[i]]]; pred <- as.integer(p > modelos$umbral[i])
  vp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1); vn <- sum(pred == 0 & y == 0)
  tibble(modelo = claves[modelos$columna[i]], umbral = modelos$umbral[i],
         vp = vp, fp = fp, fn = fn, vn = vn,
         precision = vp / (vp + fp), recall = vp / (vp + fn), accuracy = (vp + vn) / length(y),
         especificidad = vn / (vn + fp),
         f1 = mean(map_dbl(1:5, ~ calcular_f1(pred[fold == .x], y[fold == .x]))),
         auc = auc(p, y))
})
print(metricas, width = Inf)
stopifnot(all(abs(metricas$f1 - modelos$f1) < 1e-4))
guardar_dat(metricas, "metricas")

# Curva ROC de cada modelo, reducida a ~300 puntos.
iwalk(claves, function(clave, columna) {
  p <- oof[[columna]]; o <- order(p, decreasing = TRUE)
  tpr <- cumsum(y[o] == 1) / sum(y == 1); fpr <- cumsum(y[o] == 0) / sum(y == 0)
  i <- unique(c(1, round(seq(1, length(o), length.out = 300)), length(o)))
  guardar_dat(tibble(fpr = c(0, fpr[i]), tpr = c(0, tpr[i])), paste0("roc_", clave))
})

################################################################################

## 5. F1, precisión y recall según el umbral (modelo final)

umbral_final <- map_dfr(seq(0.05, 0.95, by = 0.01), function(u) {
  pred <- as.integer(oof$prob_modelo_final > u)
  tibble(umbral = u,
         f1 = mean(map_dbl(1:5, ~ calcular_f1(pred[fold == .x], y[fold == .x]))),
         precision = sum(pred == 1 & y == 1) / max(sum(pred == 1), 1),
         recall = sum(pred == 1 & y == 1) / sum(y == 1))
})
guardar_dat(umbral_final, "umbral_final")

################################################################################

## 6. Errores según la distancia a la línea (modelo final)

# Grupos de ingreso / línea. Debajo de 1 el error posible es excluir a un
# pobre (falso negativo); encima, incluir a un no pobre (falso positivo).
cortes <- c(0, 0.25, 0.5, 0.75, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, Inf)
etiquetas <- c("0-0,25", "0,25-0,5", "0,5-0,75", "0,75-0,9", "0,9-1", "1-1,1", "1,1-1,25",
               "1,25-1,5", "1,5-2", "2-3", "3+")
errores_linea <- tibble(grupo = cut(hog$r, cortes, labels = etiquetas, right = FALSE),
                        y = y, pred = pred_final) |>
  summarize(.by = grupo, hogares = n(), errores = sum(pred != y),
            fn = sum(y == 1 & pred == 0), fp = sum(y == 0 & pred == 1)) |>
  arrange(grupo) |>
  mutate(posicion = row_number(), pct_hogares = 100 * hogares / sum(hogares),
         pct_errores = 100 * errores / sum(errores), tasa_error = 100 * errores / hogares,
         grupo = as.character(grupo))
print(errores_linea)
guardar_dat(errores_linea, "errores_linea")

################################################################################

## 7. F1 según el número de variables, por algoritmo

# Todos con la base por hogar de 01b (94 variables) y las k primeras según la
# ganancia del XGBoost de esa base (H0 de 05b), el mismo orden para todos.
# Mismos 5 folds y umbral escogido por F1 promedio (evaluar_umbral()). No
# usan los modelos por persona (ingreso predicho, probabilidades), que el
# modelo final y el random forest de 06 sí usan.
imp_h0 <- read_rds(file.path(dir_oof, "originales", "H0_base01b.rds"))$importancia
vars_base <- setdiff(names(hog), c("id", "pobre", "ingreso", "linea", "inactivo_con_ingreso",
                                   "arriendos_pensiones", "personas", "r"))
orden <- c(intersect(imp_h0$variable, vars_base), setdiff(vars_base, imp_h0$variable))
stopifnot(length(orden) == 94)
guardar_dat(tibble(posicion = seq_along(orden), variable = gsub("_", "-", orden)), "orden_variables")

cat_base <- c("dominio", "depto", "jefe_posicion", "jefe_regimen", "jefe_tamano", "jefe_oficio", "tenencia")
X <- hog |> select(all_of(orden)) |> mutate(across(any_of(cat_base), ~ factor(as.character(.x))))
tamanos <- c(3, 5, 10, 15, 20, 30, 40, 60, 94)
hilos <- max(1, parallel::detectCores() - 1)

# Probabilidad fuera de muestra de un algoritmo con las variables `v`.
ajustar <- list(
  logit = function(v, ent, val) {
    d <- data.frame(pobre = y[ent], X[ent, v, drop = FALSE])
    m <- suppressWarnings(glm(pobre ~ ., data = d, family = binomial))
    # Niveles de las categóricas que no aparecen en entrenamiento: al nivel más común.
    nv <- X[val, v, drop = FALSE]
    for (c in intersect(v, cat_base)) {
      lv <- m$xlevels[[c]]
      nv[[c]] <- factor(if_else(nv[[c]] %in% lv, as.character(nv[[c]]), names(which.max(table(d[[c]])))), lv)
    }
    suppressWarnings(predict(m, nv, type = "response"))
  },
  naive_bayes = function(v, ent, val) {
    m <- naiveBayes(X[ent, v, drop = FALSE], factor(y[ent]), laplace = 0)
    predict(m, X[val, v, drop = FALSE], type = "raw")[, "1"]
  },
  cart = function(v, ent, val) {
    d <- data.frame(pobre = factor(y[ent]), X[ent, v, drop = FALSE])
    m <- rpart(pobre ~ ., data = d, method = "class",
               control = rpart.control(cp = 1e-4, minbucket = 50, xval = 0))
    predict(m, X[val, v, drop = FALSE], type = "prob")[, "1"]
  },
  random_forest = function(v, ent, val) {
    d <- data.frame(pobre = factor(y[ent]), X[ent, v, drop = FALSE])
    m <- ranger(pobre ~ ., data = d, num.trees = 300, mtry = max(1, ceiling(length(v) / 5)),
                min.node.size = 50, probability = TRUE, respect.unordered.factors = "order",
                num.threads = hilos, seed = 123)
    predict(m, X[val, v, drop = FALSE])$predictions[, "1"]
  },
  xgboost = function(v, ent, val) {
    dtr <- xgb.DMatrix(X[ent, v, drop = FALSE], label = y[ent])
    dva <- xgb.DMatrix(X[val, v, drop = FALSE], label = y[val])
    params <- list(objective = "binary:logistic", eval_metric = "logloss", eta = 0.05,
                   max_depth = 6, min_child_weight = 50, subsample = 0.8, colsample_bytree = 0.8,
                   nthread = hilos, seed = 123)
    b <- xgb.train(params = params, data = dtr, nrounds = 5000, evals = list(val = dva),
                   early_stopping_rounds = 100, verbose = 0)
    predict(b, dva, iterationrange = c(1, xgb.attr(b, "best_iteration") + 1))
  }
)

f1_k <- expand_grid(algoritmo = names(ajustar), k = tamanos) |>
  pmap_dfr(function(algoritmo, k) {
    archivo <- file.path(dir_k, sprintf("%s_k%02d.rds", algoritmo, k))
    if (!file.exists(archivo)) {
      t0 <- Sys.time()
      v <- orden[seq_len(k)]
      p <- rep(NA_real_, length(y))
      for (f in 1:5) p[fold == f] <- ajustar[[algoritmo]](v, which(fold != f), which(fold == f))
      ev <- evaluar_umbral(p, y, fold, grid_umbral)
      saveRDS(list(prob = p, umbral = ev$umbral, f1 = ev$f1, auc = auc(p, y)), archivo)
      cat(sprintf("%-14s k = %2d  F1 = %.4f  umbral = %.2f  (%.1f min)\n", algoritmo, k, ev$f1,
                  ev$umbral, as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    }
    r <- readRDS(archivo)
    tibble(algoritmo = algoritmo, k = k, f1 = r$f1, umbral = r$umbral, auc = r$auc)
  })
print(f1_k, n = Inf)
# Una columna por algoritmo para pgfplots.
guardar_dat(select(pivot_wider(f1_k, id_cols = k, names_from = algoritmo, values_from = f1), k, everything()), "f1_k")
