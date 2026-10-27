# Corre XGBoost (grid_xgb) con x2 y x3 en las tres bases, con los folds en
# paralelo. Carga las funciones de script/05_xgboost.r (secciones 1 a 3, sin
# las pruebas) y guarda cada resultado al terminar. Correr desde la raíz:
#   Rscript output/pruebas/xgb/conjuntos.r
codigo <- readLines("script/05_xgboost.r")
eval(parse(text = codigo[seq_len(grep("^## 4. Pruebas", codigo) - 1)]))

conjuntos <- list(x2 = x2, x3 = x3)
for (b in c("train_clean", "train_clean_augmented", "train_clean_sin_ricos")) {
  for (n in names(conjuntos)) {
    archivo <- file.path("output/pruebas/xgb", paste0("conjuntos_", n, "_", b, ".rds"))
    if (file.exists(archivo)) next
    t0 <- Sys.time()
    r  <- cv_xgb(conjuntos[[n]], base = b, grid = grid_xgb, nombre = n)
    predecir_test(r)
    saveRDS(list(por_combinacion = r$por_combinacion, mejor = r$mejor,
                 importancia = importancia_xgb(r),
                 minutos = as.numeric(difftime(Sys.time(), t0, units = "mins"))), archivo)
    cat("=== listo", n, b, "-", round(as.numeric(difftime(Sys.time(), t0, units = "mins"))), "min\n")
    rm(r); invisible(gc())
  }
}
cat("=== TERMINADO\n")
