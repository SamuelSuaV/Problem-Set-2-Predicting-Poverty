# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Problem Set 2 de Big Data & Machine Learning Applied in Economics: predecir qué hogares de Colombia están por debajo de la línea de pobreza usando la GEIH 2018 (DANE). Autores: Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle. Código, comentarios y documentación en español.

## Reglas sobre variables

- **Todo cambio a una variable (renombrar, recodificar, crear, eliminar) se hace en `script/01_data_cleaning.r`**, nunca en los scripts de análisis. Excepción: las variables agregadas a nivel hogar se crean en `agregar_hogar()` de `script/01b_bases_hogar.r`.
- **Después de cualquier cambio a variables, actualizar `documentation/guia_variables.Rmd`** (la guía de variables para el equipo). Su último chunk compara la guía con `data/train_clean.rds` y avisa si hay variables sin documentar o que ya no existen.
- Cualquier edición debe aplicarse igual a train y test: va dentro de la función `editar()` de `01_data_cleaning.r`, que se llama sobre ambas bases.

## Comandos

R con `pacman::p_load` y `here` (rutas relativas a la raíz del repo). Correr desde la raíz:

```sh
Rscript script/01_data_cleaning.r    # CSV crudos -> data/train_clean.rds, data/test_clean.rds, data/train_clean_augmented.rds
Rscript script/02_data_analysis.r    # tabla en output/tables/, figuras en output/figures/
Rscript script/01b_bases_hogar.r      # bases por persona -> data/train_hogar.rds, data/test_hogar.rds (una fila por hogar)
Rscript script/03_naive_bayes.r      # validación cruzada, grid search y submission en output/submissions/
Rscript script/04_logit.r            # logit lasso a nivel hogar: CV de lambda y umbral y submission (~20 min)
Rscript script/05_logit_parsimonioso.r   # curva F1 vs número de variables, tabla, figura y submission (~25 min)
Rscript script/06_logit_interacciones.r  # lasso con más interacciones y splines (~55 min)
Rscript -e "source('script/07_boosting.r', encoding = 'UTF-8')"   # LightGBM: grilla + random search en paralelo, submission e importancia (~30 min)
Rscript -e 'rmarkdown::render("documentation/guia_variables.Rmd")'   # compila la guía y verifica que esté al día
cd latex && latexmk -pdf resultados.tex   # documento de resultados
```

Flujo: 01 → 02 → LaTeX, 01 → 03 y 01 → 01b → 04, 05, 06 y 07. 02, 03 y 01b leen los `.rds` de 01 y 04 a 07 leen los de 01b, así que 01 (y 01b) deben correrse primero cada vez que cambian las variables. `latex/resultados.tex` no tiene números de tablas copiados: hace `\input{../output/tables/descriptivas.tex}` y carga las figuras de `../output/figures/`, así que basta con correr 02 y recompilar. Se referencia la tabla con `\ref{tab:descriptivas}` (el `label` está en el `kbl()` de 02). Los números que sí están escritos en el texto (tamaño de muestra, % de pobres, mediana de la línea de pobreza, percentil 99 del ingreso) hay que actualizarlos a mano si cambia la limpieza. `script/00_master_file.r` existe pero está vacío.

Correr 01 con `Rscript` (sesión nueva): ocupa ~1,4 GB, y en una sesión interactiva que ya tiene cargados los objetos de 03 se puede quedar sin memoria (el Mac tiene 8 GB). 03 tarda varios minutos: `predict()` de `e1071` calcula fila por fila; cada validación cruzada de 10 folds toma ~1-2 min por conjunto de variables.

## Datos

- `data/` tiene los 4 CSV crudos (`{train,test}_{hogares,personas}.csv`) y los `.rds` limpios. `*.csv` y `*.rds` están en `.gitignore` (los `.rds` pesan cientos de MB).
- Diccionario del DANE: `documentation/ddi-documentation-spanish-608.pdf`. Ahí están las etiquetas y códigos de cada pregunta `P####`.
- La unidad de observación de las bases limpias es la **persona**; las variables del hogar se repiten para cada persona del mismo `id`. `pobre` es del hogar (todos los miembros comparten el valor) y la submission de Kaggle es una fila por hogar (`id`, `pobre`).

## Decisiones de limpieza que no son obvias

- `read_csv(..., guess_max = Inf)`: sin esto, columnas vacías en las primeras 1000 filas se leen como lógicas y se pierden datos.
- Merge: `left_join(personas, hogares, by = c("id", "Clase", "Dominio", "Fex_c", "Depto", "Fex_dpto"), relationship = "many-to-one")`. Esas columnas son idénticas en ambas tablas, así que van como llaves para no generar `.x`/`.y`.
- **Variable objetivo**: `pobre = ingreso < linea_pobreza`, donde `ingreso` es `Ingpcug` (ingreso per cápita de la unidad de gasto con arriendo imputado). Se verificó que se cumple en el 100% de los hogares. También `Indigente = ingreso < linea_indigencia`.
- Solo existen en train: `ingreso`, `pobre` (`Pobre` en el CSV), `Indigente`, `Npobres`, `Nindigentes`, `Estrato1`. No sirven como predictores. Las demás preguntas de montos de ingreso (`P####s#a#`, etc.) y agregados (`Impa`…`Ingtot`, `Ingtotug`, `Ingtotugarr`) se eliminan de train porque no están en test.
- Preguntas sí/no (vector `si_no`): 1 = sí, 0 = no, `NA` = no sabe (el 9 original) o pregunta no aplicada. `sexo` → `mujer`, `clase` → `cabecera`.
- Categóricas `max_educ`, `regimen_salud`, `tenencia_vivienda`: se conservan con los códigos del DANE y además tienen dummies por categoría (lista `categorias`), incluida "no sabe".
- `ocupado` es 1/0 (el `NA` original = no ocupado se pasó a 0), pero `edad_trabajar`, `desocupado` e `inactivo` siguen siendo 1/`NA`.
- `NA` casi siempre significa que la pregunta no aplica por los saltos del cuestionario (p. ej. preguntas de trabajo solo para ocupados); no reemplazar por 0 sin revisar qué significa.
- `arriendo_estimado`, `arriendo_pagado`, `cuota_amortizacion`: 98 (no sabe) y 99 (no informa) → `NA`. Los valores atípicos (arriendos de hasta $600M) se dejan a propósito.
- Se eliminan de train los 2 hogares con `cuartos_hogar` 98 y 43 (error de digitación). **Nunca se eliminan filas de test**: hay que predecir todas.
- `train_clean_augmented`: `train_clean` con los hogares pobres replicados (3 o 4 copias, el residuo al azar con `set.seed(123)`) hasta tener tantos hogares pobres como no pobres. Se arma al final de 01 desde `train_clean` ya editada, así que hereda cualquier cambio de variables sin tocarla aparte. Las copias conservan el `id` (los folds por hogar las dejan juntas). Para evaluar, validar siempre sobre hogares originales, no sobre la base aumentada.

## Modelo naive bayes (`script/03_naive_bayes.r`)

- **Tipos:** `naiveBayes()` (e1071) modela toda columna numérica como normal y solo los factores como categóricas, según el tipo de R, no según los valores. `preparar()` convierte a factor las variables del vector `categoricas`, el texto y toda columna 0/1. Se aplica igual a train y a test antes de estimar y de predecir; si test queda numérico, `predict()` da *type mismatch*. Al agregar una categórica con códigos DANE, agregarla a `categoricas`.
- **Folds por hogar** (`fold`, 10 folds, `set.seed(123)`): todas las personas de un `id` quedan en el mismo fold.
- **Predicción por hogar:** `prob_hogar()` predice la probabilidad de cada persona, la promedia en log odds dentro del `id` y compara con el umbral. El F1 de `cv_naive_bayes()` y del grid search se calcula con una predicción por hogar, igual que la predicción de test.
- **Grid search** (sección 4): `laplace` × `umbral`; por cada `laplace` se estima una vez por fold y se evalúan todos los umbrales con esas probabilidades. `laplace` casi no cambia el F1 (desempate: el menor). Las probabilidades de naive bayes están descalibradas (sobrestiman la pobreza), por eso el umbral óptimo queda cerca de 0,45-0,5 y no en ~F1/2.
- **Nombre de la submission:** formato Kaggle `modelo_hiperparámetro_valor` sin punto decimal (`NB_laplace_0_threshold_045.csv`); se arma solo con los valores del grid search. `*.csv` está en `.gitignore`.
- **Selección de variables:** naive bayes cuenta dos veces la información de variables correlacionadas; repetir una medida (dummies `educ_*` + `max_educ` + `anos_educ_imp`, o `anos` + `anos2`) empeora el F1. Numéricas muy asimétricas (arriendos) le quedan mal a la normal: usar logaritmos (`arriendo_pc_sobre_lp`, `log_meses_empresa`). Comparar conjuntos de variables con su mejor umbral, no con 0,5.

## Modelo logit (`script/04_logit.r`)

- **Nivel hogar:** usa `train_hogar`/`test_hogar` (`script/01b_bases_hogar.r`, `agregar_hogar()`): una fila por hogar con conteos y proporciones de la unidad de gasto (parentesco 6-8 fuera), sin `NA` (lo que no aplica cuenta como 0). Una variable nueva del hogar se agrega en `agregar_hogar()` y en la tabla `variables_hogar` de la guía.
- **Modelo:** lasso (`glmnet`, `alpha = 1`) con todas las variables de la base, interacciones de a pares entre las del vector `x_clave`, `dominio ×` cuatro continuas y algunos cuadrados. La matriz se arma con train y test juntos para que los factores (`categoricas`) tengan los mismos niveles (test no tiene Bogotá).
- **Selección:** 5 folds (`set.seed(123)`); `keep = TRUE` da las predicciones fuera de muestra de cada lambda y se busca el par lambda × umbral con mejor F1 promedio. El lambda óptimo queda casi en el mínimo de la grilla: la ganancia viene de las interacciones, no de la penalización. Umbral óptimo ≈ 0,34-0,37.
- **Resultados de referencia (F1 CV):** naive bayes 0,671; logit sin penalizar por bloques hasta 0,738; lasso con interacciones 0,7415. Las notas al final de 04 tienen el detalle.
- **05 (`Rscript script/05_logit_parsimonioso.r`):** ordena las variables por su entrada a un lasso sin interacciones y estima logits sin penalizar con las k primeras. F1 CV: 3 variables 0,585, 15 → 0,697, 40 → 0,727, 50 → 0,731, todas (94) → 0,736. Ningún k corto llega a 0,7317 (0,7417 − 0,01), así que la regla escoge las 94. Guarda `output/tables/logit_parsimonia.tex` y `output/figures/logit_f1_vs_variables.png`. Los niveles de factor con menos de 30 hogares se agrupan en "otro" para que `predict()` de `glm` no falle.
- **06 (`Rscript script/06_logit_interacciones.r`):** igual que 04 con un `x_clave` de 36 variables, `ns(df = 4)` en 8 continuas, `dominio ×` 10 variables, `tenencia:log_arriendo_pc_lp` y `jefe_posicion:horas_pc` (1.159 columnas). F1 CV 0,7451 con lambda 5,4e-05 (ya no en el borde), umbral 0,37 y 794 coeficientes distintos de 0.

## Modelo boosting (`script/07_boosting.r`)

- **Comando:** `Rscript -e "source('script/07_boosting.r', encoding = 'UTF-8')"` (~30 min con 32 núcleos). Con `prueba <- TRUE;` antes del `source()` corre 2 configuraciones con 50 iteraciones y no guarda nada.
- **Level playing field:** las mismas 94 variables de `train_hogar` que el logit (sin interacciones ni splines) y los mismos 5 folds de 04. Las categóricas pasan a códigos enteros con niveles armados con train y test juntos y van como `categorical_feature`.
- **Búsqueda:** grilla de clase (18: `learning_rate` × `num_leaves` × `max_depth`) + random search de 30 (`set.seed(123)`). `lgb.cv` con `nrounds = 5000` y early stopping (100) en logloss; las probabilidades fuera de muestra salen de los boosters de cada fold en la mejor iteración, y el umbral se escoge por F1 en `seq(0.1, 0.6, 0.01)`. Sin ajuste de desbalance: se maneja con el umbral. Corre en un clúster PSOCK de 8 procesos × 4 hilos; cada configuración se guarda en `output/logs/boosting/config_XX.rds`.
- **Modelo final:** `lgb.train` con todo train, `nrounds` = mejor iteración × 1,1. En `output/logs/boosting/` quedan `grid.rds`, `oof.rds`, `importancia.rds` y `modelo_final.txt`, para el diagnóstico sin volver a estimar.
- **Resultado:** F1 CV 0,7537 (umbral 0,37; lr 0,02, 127 hojas, profundidad 6, 1313 iteraciones), por encima del logit 06 (0,7451). La superficie es plana: casi todas las configuraciones dan entre 0,749 y 0,754. Variables con más ganancia: `log_arriendo_pc_lp`, `horas_pc`, `prop_contributivo` y `educ_ocup_suma_pc` (casi la mitad del total).

## Windows

- **Windows Server (sin UTF-8 nativo en R):** para que las tildes salgan bien en figuras, correr con `Rscript -e "source('script/05_logit_parsimonioso.r', encoding = 'UTF-8')"`. Con ese `source()`, `writeLines()` pasa el texto a CP1252: para escribir un `.tex` en UTF-8 usar `writeLines(enc2utf8(x), archivo, useBytes = TRUE)` (así lo hace 07; 05 todavía no).

## Estilo

Los scripts siguen una plantilla: encabezado con proyecto, nombre del script, autores, propósito, estructura numerada, input y output; secciones `## N. Título` separadas por líneas de `#`. Mantener la plantilla y actualizar la estructura/input/output del encabezado al agregar secciones.
