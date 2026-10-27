# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Problem Set 2 de Big Data & Machine Learning Applied in Economics: predecir qué hogares de Colombia están por debajo de la línea de pobreza usando la GEIH 2018 (DANE). Autores: Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle. Código, comentarios y documentación en español.

## Reglas sobre variables

- **Todo cambio a una variable (renombrar, recodificar, crear, eliminar) se hace en `script/01_data_cleaning.r`**, nunca en los scripts de análisis.
- **Después de cualquier cambio a variables, actualizar `documentation/guia_variables.Rmd`** (la guía de variables para el equipo). Su último chunk compara la guía con `data/train_clean.rds` y avisa si hay variables sin documentar o que ya no existen.
- Cualquier edición debe aplicarse igual a train y test: va dentro de la función `editar()` de `01_data_cleaning.r`, que se llama sobre ambas bases. `editar()` va por pasos en orden (cada uno puede usar lo creado antes): 5.1 recodificar, 5.2 dummies de `categorias`, 5.3 nuevas variables de la persona, 5.4 nuevas variables del hogar (`mutate(..., .by = id)`), 5.5 pasar las 0/1 a entero (debe quedar al final).

## Comandos

R con `pacman::p_load` y `here` (rutas relativas a la raíz del repo). Correr desde la raíz:

```sh
Rscript script/01_data_cleaning.r    # CSV crudos -> data/train_clean.rds, data/test_clean.rds, data/train_clean_augmented.rds, data/train_clean_sin_ricos.rds
Rscript script/02_data_analysis.r    # tabla en output/tables/, figuras en output/figures/
Rscript script/03_naive_bayes.r      # validación cruzada, grid search y submission en output/submissions/
Rscript script/04_carts.r            # CART con caret: cp y umbral por validación cruzada, submission
Rscript script/05_xgboost.r          # XGBoost con caret (modelo propio xgb_caret): hiperparámetros y umbral, submission
Rscript script/05b_xgboost_hogar.r    # XGBoost a nivel hogar (5 folds de main). PRUEBA=1: 10% de los hogares, guarda en output/pruebas/xgb_hogar_prueba_<cod>/ (probar ahí antes de una corrida larga). CODIFICACION=nativo (por defecto) | onehot | codigos; NORMALIZAR=1 estandariza la base. Cada combinación guarda en output/pruebas/xgb_hogar[_<cod>][_normalizada]/ y retoma. La sección 13 combina con el logit de main: antes correr output/pruebas/logit/06_logit_interacciones_main.r (copia del 06 de main que guarda sus probabilidades en logit_oof.rds)
Rscript script/06_comparacion_modelos.r   # mejor modelo de cada categoría con los 5 folds por hogar y desempeño por grupo de población (equidad): tablas output/tables/modelos_f1.tex, equidad_*.tex y figura equidad_errores.png. Después de 05b y del logit
Rscript -e 'rmarkdown::render("documentation/guia_variables.Rmd")'   # compila la guía y verifica que esté al día
cd latex && latexmk -pdf resultados.tex   # documento de resultados
cd latex && latexmk -pdf estructura_modelo_slide.tex   # diagrama simplificado del modelo final para slides (el comando para pasarlo a PNG de 600 dpi está en su encabezado)
```

Flujo: 01 → 02 → LaTeX, y 01 → 03 / 04 / 05. 02 a 05 leen los `.rds` de 01, así que 01 debe correrse primero cada vez que cambian las variables. `latex/resultados.tex` no tiene números de tablas copiados: hace `\input{../output/tables/descriptivas.tex}` y carga las figuras de `../output/figures/`, así que basta con correr 02 y recompilar. Se referencia la tabla con `\ref{tab:descriptivas}` (el `label` está en el `kbl()` de 02). Los números que sí están escritos en el texto (tamaño de muestra, % de pobres, mediana de la línea de pobreza, percentil 99 del ingreso) hay que actualizarlos a mano si cambia la limpieza. Lo mismo la sección "Modelos" (Cuadro `tab:modelos` y los cuadros de pruebas de 05b): hiperparámetros, umbral, F1 y nombre de la submission se copiaron a mano de las corridas, y también los hiperparámetros del diagrama `fig:estructura` (TikZ dentro de `resultados.tex`; su versión para slides es `latex/estructura_modelo_slide.tex`). Las tablas de importancia por permutación (calculada en la sección 14 de 05b; las tablas `importancia_grupos.tex`, `importancia_variables.tex` e `importancia_directas.tex`, con el nombre completo de cada variable, las arma la sección 7 de 06; también `importancia_conceptos.tex`, de la sección 15 de 05b) y de comparación y equidad (06) sí se generan solas. `script/00_master_file.r` tiene las variables y funciones comunes (`categoricas`, `x1`, `preparar()`, `calcular_f1()`, `prob_hogar()`, `evaluar_umbral()`); 03, 04, 05, 05b y 06 lo cargan con `source()`: si se cambia una, cambia en todos. `output/pruebas/` tiene pruebas exploratorias que no están en los scripts (cada una con `datos.r` y `noche.r`; se corren con `T=output/pruebas/<prueba> Rscript output/pruebas/<prueba>/noche.r`, guardan cada combinación en `resultados.rds` y retoman si se interrumpen).

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
- `arriendo_imputado`: el arriendo que el DANE suma como ingreso a los hogares que no pagan arriendo (`Ingpcug = (Ingtotug + arriendo imputado) / Npersug`). Regla reconstruida (coincide en el 98,4% de los hogares de train): propia pagada o usufructo → `arriendo_estimado`; propia pagando → `arriendo_estimado − cuota_amortizacion` (mínimo 0; 0 si la cuota es NA); demás → 0. `arriendo_pc_sobre_lp` en cambio mezcla arriendo pagado (gasto) y estimado.
- Se eliminan de train los 2 hogares con `cuartos_hogar` 98 y 43 (error de digitación). **Nunca se eliminan filas de test**: hay que predecir todas.
- `train_clean_augmented`: `train_clean` con los hogares pobres replicados (3 o 4 copias, el residuo al azar con `set.seed(123)`) hasta tener tantos hogares pobres como no pobres. Se arma al final de 01 desde `train_clean` ya editada, así que hereda cualquier cambio de variables sin tocarla aparte. Las copias conservan el `id` (los folds por hogar las dejan juntas). Para evaluar, validar siempre sobre hogares originales, no sobre la base aumentada.
- `hogar_rico` (solo train, como `ingreso`): hogar en el 10% de mayor ingreso per cápita, calculado sobre hogares y sin factores de expansión. `train_clean_sin_ricos` es `train_clean` sin esos hogares. **No se usa como una base más**: si los ricos salen también de la validación, el F1 se mide sobre una población distinta a la de test. En 03, 04 y 05 la opción `base = "train_clean_sin_ricos"` carga `train_clean`, entrena sin los ricos (vector `entrenar`; en caret, `index` sin ellos e `indexOut` con todo el fold) y valida con todos los hogares; el modelo final se vuelve a estimar solo con los no ricos (caret lo estimaría con todas las filas).
- **Bases en los scripts de modelos:** 03 y 04 eligen la base con la variable `base` al cargar los datos; 05 la recibe por prueba (`cv_xgb(x, base = ...)`, vector `bases`). La base aumentada conserva el orden de los hogares de `train_clean`, así que los folds son los mismos con cualquier base. Las submissions llevan `aug_` o `sinricos_` en el nombre.

## XGBoost por hogar (`script/05b_xgboost_hogar.r`)

- **Retoma:** cada prueba o sección guarda su `.rds` en `output/pruebas/xgb_hogar_<cod>/` y se salta si el archivo ya existe; para volver a calcular algo, borrar su `.rds`. Con todo guardado, llegar al final tarda ~15 min. `marca()` escribe el avance en `progreso.log` de esa carpeta; correrlo en segundo plano con `caffeinate -dis Rscript script/05b_xgboost_hogar.r > output/pruebas/<nombre>.log 2>&1`.
- **Importancia:** la sección 14 permuta las columnas a la entrada del modelo por hogar (solo efecto directo: los modelos auxiliares por persona siguen viendo los datos reales y se quedan con el crédito). Las secciones 15 (por concepto, lista `conceptos`) y 16 (por variable de la persona, solo los 5 conceptos con mayor caída) permutan en toda la cadena con `cadena_fold(f)`, que vuelve a estimar los modelos auxiliares del fold y verifica que reproduce la predicción fuera de muestra.
- **Memoria:** el proceso principal ocupa ~5 GB y `mclapply` copia el proceso por núcleo; en el Mac de 8 GB, 2 núcleos llevaron la sección 15 a swap (187 min). La sección 16 va un fold a la vez y guarda cada fold (`*_fold<f>.rds`); el fold completo tarda ~1 h.
- Para ver páginas del PDF no hay `pdftoppm`: usar `gs -sDEVICE=png16m -r200 -dFirstPage=N -dLastPage=N`.

## Modelo naive bayes (`script/03_naive_bayes.r`)

- **Tipos:** `naiveBayes()` (e1071) modela toda columna numérica como normal y solo los factores como categóricas, según el tipo de R, no según los valores. `preparar()` convierte a factor las variables del vector `categoricas`, el texto y toda columna 0/1. Se aplica igual a train y a test antes de estimar y de predecir; si test queda numérico, `predict()` da *type mismatch*. Al agregar una categórica con códigos DANE, agregarla a `categoricas`.
- **Folds por hogar** (`fold`, 10 folds, `set.seed(123)`): todas las personas de un `id` quedan en el mismo fold.
- **Predicción por hogar:** `prob_hogar()` predice la probabilidad de cada persona, la promedia en log odds dentro del `id` y compara con el umbral. El F1 de `cv_naive_bayes()` y del grid search se calcula con una predicción por hogar, igual que la predicción de test.
- **Grid search** (sección 4): `laplace` × `umbral`; por cada `laplace` se estima una vez por fold y se evalúan todos los umbrales con esas probabilidades. `laplace` casi no cambia el F1 (desempate: el menor). Las probabilidades de naive bayes están descalibradas (sobrestiman la pobreza), por eso el umbral óptimo queda cerca de 0,45-0,5 y no en ~F1/2.
- **Nombre de la submission:** formato Kaggle `modelo_hiperparámetro_valor` sin punto decimal (`NB_laplace_0_threshold_045.csv`); se arma solo con los valores del grid search. `*.csv` está en `.gitignore`.
- **Selección de variables:** naive bayes cuenta dos veces la información de variables correlacionadas; repetir una medida (dummies `educ_*` + `max_educ` + `anos_educ_imp`, o `anos` + `anos2`) empeora el F1. Numéricas muy asimétricas (arriendos) le quedan mal a la normal: usar logaritmos (`arriendo_pc_sobre_lp`, `log_meses_empresa`). Comparar conjuntos de variables con su mejor umbral, no con 0,5.

## Estilo

Los scripts siguen una plantilla: encabezado con proyecto, nombre del script, autores, propósito, estructura numerada, input y output; secciones `## N. Título` separadas por líneas de `#`. Mantener la plantilla y actualizar la estructura/input/output del encabezado al agregar secciones.
