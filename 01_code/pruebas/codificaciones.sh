#!/bin/zsh
# Corre 05b con las tres codificaciones de las categóricas, una tras otra.
cd "$(dirname "$0")/../.."
for c in codigos nativo onehot; do
  CODIFICACION=$c caffeinate -dis Rscript 01_code/11_xgboost_hogar.R > 02_outputs/pruebas/xgb_hogar_$c.log 2>&1
  echo "$(date) - $c terminó (código $?)" >> 02_outputs/pruebas/codificaciones.log
done
