# Base normalizada: los mejores modelos de 05b (categóricas nativas) con las
# variables numéricas de la base por hogar estandarizadas (NORMALIZAR=1), y
# comparación con la base sin normalizar. Los modelos por persona (V4, P) no
# usan la base por hogar, así que su v34p.rds se copia de xgb_hogar_nativo/.
# Correr desde la raíz:  Rscript output/pruebas/normalizada.r
Sys.setenv(NORMALIZAR = "1", CODIFICACION = "nativo")
codigo <- readLines("script/05b_xgboost_hogar.r")
eval(parse(text = codigo[seq_len(grep("^## 5. Pruebas", codigo) - 1)]))

umbral_honesto <- function(prob) {
  tabla <- sapply(1:k, function(f) map_dbl(grid_umbral, ~ calcular_f1(as.integer(prob[fold == f] > .x), y[fold == f])))
  mean(map_dbl(1:k, function(f) {
    u <- grid_umbral[which.max(rowMeans(tabla[, -f, drop = FALSE]))]
    calcular_f1(as.integer(prob[fold == f] > u), y[fold == f])
  }))
}
orig <- function(n) readRDS(here("output", "pruebas", "xgb_hogar_nativo", paste0(n, ".rds")))
modelos <- c("G_eta002_depth6_col08", "E_top75_75vars", "R_mejor")
norm <- map(set_names(modelos), function(n) {
  o <- orig(n)
  cv_hogar(n, o$vars, base = o$base, params = o$params, regresion = o$regresion,
           nrounds = o$nrounds %||% nrounds_max)
})
comb_o <- combinar(orig("G_eta002_depth6_col08"), orig("R_mejor"))
comb_n <- combinar(norm$G_eta002_depth6_col08, norm$R_mejor)
tabla <- tibble(modelo = c(modelos, "combinación clasificador + regresión"),
                f1_original = c(map_dbl(modelos, ~ orig(.x)$f1), comb_o$f1),
                f1_normalizada = c(map_dbl(norm, "f1"), comb_n$f1),
                honesto_original = c(map_dbl(modelos, ~ umbral_honesto(orig(.x)$prob)),
                                     umbral_honesto(comb_o$w * orig("G_eta002_depth6_col08")$prob + (1 - comb_o$w) * orig("R_mejor")$prob)),
                honesto_normalizada = c(map_dbl(norm, ~ umbral_honesto(.x$prob)),
                                        umbral_honesto(comb_n$w * norm$G_eta002_depth6_col08$prob + (1 - comb_n$w) * norm$R_mejor$prob))) |>
  mutate(diferencia = f1_normalizada - f1_original,
         max_dif_prob = c(map_dbl(modelos, ~ max(abs(norm[[.x]]$prob - orig(.x)$prob))), NA))
saveRDS(tabla, file.path(dir_out, "comparacion_normalizada.rds"))
marca("Base normalizada frente a la original:"); print(as.data.frame(tabla), digits = 4)
