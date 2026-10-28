# Grid search del CART: cp, minbucket, split, maxdepth, maxsurrogate y umbral,
# con F1 por hogar. Guarda cada combinación al terminar y, si se interrumpe,
# retoma desde la última guardada. Correr desde la raíz del repo:
#   T=02_outputs/pruebas/top9 Rscript 01_code/pruebas/top9/noche.R
carpeta <- Sys.getenv("T")
source(file.path("01_code", "pruebas", "top9", "datos.R"))
log_msg <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), ..., "\n",
    file = file.path(carpeta, "progreso.log"), append = TRUE
  )
}

grid_cp <- 10^seq(-6, -2, length.out = 17)
grid_umbral <- seq(0.10, 0.90, by = 0.01)
nombres_f1 <- sprintf("F1_%03d", round(100 * grid_umbral))

prob_hogar <- function(p, id) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  tibble(id = id, prob = p) |>
    summarize(prob = plogis(mean(qlogis(prob))), .by = id)
}
calcular_f1 <- function(pred, real) {
  vp <- sum(pred == 1 & real == 1)
  precision <- vp / sum(pred == 1)
  recall <- vp / sum(real == 1)
  2 * precision * recall / (precision + recall)
}
# Las copias de la base aumentada tienen el mismo id y la misma predicción:
# prob_hogar() las junta y el F1 se calcula sobre los hogares originales.
f1_hogar <- function(data, lev = NULL, model = NULL) {
  ids <- id[data$rowIndex]
  h <- prob_hogar(data$pobre, ids) |>
    left_join(distinct(tibble(id = ids, real = data$obs == "pobre")), by = "id")
  setNames(
    map_dbl(grid_umbral, ~ calcular_f1(as.integer(h$prob > .x), h$real)),
    nombres_f1
  )
}
elegir_cp <- function(x, metric, maximize) {
  which.max(apply(
    x[nombres_f1], 1, max,
    na.rm = TRUE
  ))
}

# Importancia por reducción de la impureza, normalizada a 100: la de rpart
# (cortes principales + sustitutos) y solo cortes principales.
normalizar <- function(v) 100 * v / sum(v)
imp_principal <- function(fit) {
  fr <- fit$frame
  internos <- fr$var != "<leaf>"
  filas <- (1 + fr$ncompete + fr$nsurrogate)[internos]
  inicio <- cumsum(c(1, head(filas, -1)))
  s <- fit$splits[inicio, , drop = FALSE]
  normalizar(tapply(s[, "improve"], rownames(s), sum))
}

combos <- expand_grid(
  minbucket = c(7, 25, 100, 400), split = c("gini", "information"),
  maxdepth = c(30, 15), maxsurrogate = c(5, 0)
)
archivo <- file.path(carpeta, "resultados.rds")
resultados <- if (file.exists(archivo)) readRDS(archivo) else list()

for (j in seq_len(nrow(combos))) {
  cmb <- combos[j, ]
  nombre <- with(
    cmb,
    paste0("mb", minbucket, "_", split, "_md", maxdepth, "_ms", maxsurrogate)
  )
  if (nombre %in% names(resultados)) next
  t0 <- Sys.time()
  out <- tryCatch(
    {
      set.seed(123)
      modelo <- train(
        x = x, y = y, method = "rpart", metric = "F1_050",
        tuneGrid = data.frame(cp = grid_cp),
        trControl = trainControl(
          method = "cv", index = index, classProbs = TRUE,
          summaryFunction = f1_hogar, selectionFunction = elegir_cp,
          returnData = FALSE, allowParallel = FALSE
        ),
        parms = list(split = cmb$split),
        control = rpart.control(
          minbucket = cmb$minbucket, maxdepth = cmb$maxdepth,
          maxsurrogate = cmb$maxsurrogate
        )
      )
      por_cp <- modelo$results |>
        select(cp, all_of(nombres_f1)) |>
        pivot_longer(-cp, names_to = "umbral", values_to = "f1") |>
        mutate(umbral = as.numeric(sub("F1_", "", umbral)) / 100)
      b <- slice_max(por_cp, f1, n = 1, with_ties = FALSE)
      fit <- modelo$finalModel
      list(
        combo = cmb, cp = b$cp, umbral = b$umbral, f1 = b$f1, por_cp = por_cp,
        hojas = sum(fit$frame$var == "<leaf>"),
        imp_rpart = normalizar(fit$variable.importance),
        imp_principal = imp_principal(fit),
        minutos = as.numeric(difftime(Sys.time(), t0, units = "mins"))
      )
    },
    error = function(e) list(combo = cmb, f1 = NA, error = conditionMessage(e))
  )
  resultados[[nombre]] <- out
  saveRDS(resultados, archivo)
  log_msg(
    nombre, "- F1 =", round(out$f1, 4), "- cp =", signif(out$cp %||% NA, 3),
    "- umbral =", out$umbral %||% NA, "- hojas =", out$hojas %||% NA,
    "- min =", round(out$minutos %||% NA, 1), out$error %||% ""
  )
  if (exists("modelo")) rm(modelo)
  invisible(gc())
}
log_msg("=== TERMINADO")
