# Corre la grilla ampliada de XGBoost con x1 en las tres bases. Carga las
# funciones de script/05_xgboost.r (secciones 1 a 3, sin las pruebas) y guarda
# cada resultado al terminar. Correr desde la raíz del repo:
#   Rscript output/pruebas/xgb/ampliada.r
codigo <- readLines("script/05_xgboost.r")
eval(parse(text = codigo[seq_len(grep("^## 4. Pruebas", codigo) - 1)]))

for (b in c("train_clean", "train_clean_augmented", "train_clean_sin_ricos")) {
  archivo <- file.path("output/pruebas/xgb", paste0("ampliada_", b, ".rds"))
  if (file.exists(archivo)) next
  t0 <- Sys.time()
  r  <- cv_xgb(x1, base = b, grid = grid_ampliada)
  predecir_test(r)
  saveRDS(list(por_combinacion = r$por_combinacion, mejor = r$mejor,
               importancia = importancia_xgb(r),
               minutos = as.numeric(difftime(Sys.time(), t0, units = "mins"))), archivo)
  cat("=== listo", b, "-", round(as.numeric(difftime(Sys.time(), t0, units = "mins"))), "min\n")
  rm(r); invisible(gc())
}
cat("=== TERMINADO\n")
