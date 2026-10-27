#!/bin/zsh
# Corre 05b con las tres codificaciones de las categóricas, una tras otra.
cd "$(dirname "$0")/../.."
for c in codigos nativo onehot; do
  CODIFICACION=$c caffeinate -dis Rscript script/05b_xgboost_hogar.r > output/pruebas/xgb_hogar_$c.log 2>&1
  echo "$(date) - $c terminó (código $?)" >> output/pruebas/codificaciones.log
done
