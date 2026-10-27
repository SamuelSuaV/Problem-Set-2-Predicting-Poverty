# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Problem Set 2 de Big Data & Machine Learning Applied in Economics: predecir qué hogares de Colombia están por debajo de la línea de pobreza usando la GEIH 2018 (DANE). Autores: Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle. Código, comentarios y documentación en español.

## Reglas sobre variables

- **Todo cambio a una variable (renombrar, recodificar, crear, eliminar) se hace en `script/01_data_cleaning.r`**, nunca en los scripts de análisis. Excepción: las variables agregadas a nivel hogar se crean en `agregar_hogar()` de `script/01b_bases_hogar.r`.
- **Después de cualquier cambio a variables, actualizar `documentation/guia_variables.Rmd`** (la guía de variables para el equipo). Su último chunk compara la guía con `data/train_clean.rds` y avisa si hay variables sin documentar o que ya no existen.
- Cualquier edición debe aplicarse igual a train y test: va dentro de la función `editar()` de `01_data_cleaning.r`, que se llama sobre ambas bases. `editar()` va por pasos en orden (cada uno puede usar lo creado antes): 5.1 recodificar, 5.2 dummies de `categorias`, 5.3 nuevas variables de la persona, 5.4 nuevas variables del hogar (`mutate(..., .by = id)`), 5.5 pasar las 0/1 a entero (debe quedar al final).

## Comandos

R con `pacman::p_load` y `here` (rutas relativas a la raíz del repo). Correr desde la raíz:

```sh
Rscript script/01_data_cleaning.r    # CSV crudos -> data/train_clean.rds, data/test_clean.rds, data/train_clean_augmented.rds, data/train_clean_sin_ricos.rds
Rscript script/02_data_analysis.r    # tabla en output/tables/, figuras en output/figures/
Rscript script/01b_bases_hogar.r      # bases por persona -> data/train_hogar.rds, data/test_hogar.rds (una fila por hogar)
Rscript script/03_naive_bayes.r      # validación cruzada, grid search y submission en output/submissions/
Rscript script/04_carts.r            # CART con caret: cp y umbral por validación cruzada, submission
Rscript script/05_xgboost.r          # XGBoost con caret (modelo propio xgb_caret): hiperparámetros y umbral, submission
Rscript script/05b_xgboost_hogar.r    # XGBoost a nivel hogar (5 folds de main). PRUEBA=1: 10% de los hogares, guarda en output/pruebas/xgb_hogar_prueba_<cod>/ (probar ahí antes de una corrida larga). CODIFICACION=nativo (por defecto) | onehot | codigos; NORMALIZAR=1 estandariza la base. Cada combinación guarda en output/pruebas/xgb_hogar[_<cod>][_normalizada]/ y retoma. La sección 13 combina con el logit de main: antes correr output/pruebas/logit/06_logit_interacciones_main.r (copia del 06 de main que guarda sus probabilidades en logit_oof.rds)
Rscript script/06_comparacion_modelos.r   # mejor modelo de cada categoría con los 5 folds por hogar y desempeño por grupo de población (equidad): tablas output/tables/modelos_f1.tex, equidad_*.tex y figura equidad_errores.png. Después de 05b y del logit
Rscript script/04_logit.r            # logit lasso a nivel hogar: CV de lambda y umbral y submission (~20 min)
Rscript script/05_logit_parsimonioso.r   # curva F1 vs número de variables, tabla, figura y submission (~25 min)
Rscript script/06_logit_interacciones.r  # lasso con más interacciones y splines (~55 min)
Rscript -e "source('script/07_boosting.r', encoding = 'UTF-8')"   # LightGBM: grilla + random search en paralelo, submission e importancia (~30 min)
Rscript -e "source('script/07b_boosting_variables.r', encoding = 'UTF-8')"   # LightGBM con variables nuevas e ingreso predicho + ensamble con logit 06 (~52 min)
Rscript -e "source('script/07c_boosting_ingreso.r', encoding = 'UTF-8')"     # LightGBM con variables auxiliares predichas (CV anidada), reajuste y semillas (~65 min, 20 núcleos)
Rscript -e "source('script/10_lpm.r', encoding = 'UTF-8')"                   # modelo lineal de probabilidad (~19 min)
Rscript -e 'rmarkdown::render("documentation/guia_variables.Rmd")'   # compila la guía y verifica que esté al día
cd latex && latexmk -pdf resultados.tex   # documento de resultados
cd latex && latexmk -pdf estructura_modelo_slide.tex   # diagrama simplificado del modelo final para slides (el comando para pasarlo a PNG de 600 dpi está en su encabezado)
```

Hay dos series de scripts con números que se repiten: la de Samuel (`04_carts`, `05_xgboost`, `05b_xgboost_hogar`, `06_comparacion_modelos`) y la de logit y LightGBM (`04_logit`, `05_logit_parsimonioso`, `06_logit_interacciones`, `07*`, `10_lpm`). Los nombres no chocan, pero hay que referirse a cada script por su nombre completo.

Flujo: 01 → 02 → LaTeX; 01 → 03, 04_carts y 05_xgboost; 01 → 01b → 04_logit, 05_logit_parsimonioso, 06_logit_interacciones, 07, 07b, 07c, 10_lpm y 05b_xgboost_hogar → 06_comparacion_modelos. 02, 03, 04_carts, 05_xgboost y 01b leen los `.rds` de 01, y los modelos por hogar leen los de 01b, así que 01 (y 01b) deben correrse primero cada vez que cambian las variables. `latex/resultados.tex` no tiene números de tablas copiados: hace `\input{../output/tables/descriptivas.tex}` y carga las figuras de `../output/figures/`, así que basta con correr 02 y recompilar. Se referencia la tabla con `\ref{tab:descriptivas}` (el `label` está en el `kbl()` de 02). Los números que sí están escritos en el texto (tamaño de muestra, % de pobres, mediana de la línea de pobreza, percentil 99 del ingreso) hay que actualizarlos a mano si cambia la limpieza. Lo mismo la sección "Modelos" (Cuadro `tab:modelos` y los cuadros de pruebas de 05b): hiperparámetros, umbral, F1 y nombre de la submission se copiaron a mano de las corridas, y también los hiperparámetros del diagrama `fig:estructura` (TikZ dentro de `resultados.tex`; su versión para slides es `latex/estructura_modelo_slide.tex`). Las tablas de importancia por permutación (calculada en la sección 14 de 05b; las tablas `importancia_grupos.tex`, `importancia_variables.tex` e `importancia_directas.tex`, con el nombre completo de cada variable, las arma la sección 7 de 06; también `importancia_conceptos.tex` e `importancia_variables_e2e.tex`, de las secciones 15 y 16 de 05b) y de comparación y equidad (06) sí se generan solas. `script/00_master_file.r` tiene las variables y funciones comunes (`categoricas`, `x1`, `preparar()`, `calcular_f1()`, `prob_hogar()`, `evaluar_umbral()`); 03, 04, 05, 05b y 06 lo cargan con `source()`: si se cambia una, cambia en todos. `output/pruebas/` tiene pruebas exploratorias que no están en los scripts (cada una con `datos.r` y `noche.r`; se corren con `T=output/pruebas/<prueba> Rscript output/pruebas/<prueba>/noche.r`, guardan cada combinación en `resultados.rds` y retoman si se interrumpen).

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

## Boosting con variables nuevas (`script/07b_boosting_variables.r`)

- **Comando:** `Rscript -e "source('script/07b_boosting_variables.r', encoding = 'UTF-8')"` (~52 min; casi todo es el logit 06, que corre en paralelo con `callr::r_bg`). Con `prueba <- TRUE;` usa el 10% de los hogares y no guarda nada.
- **Hiperparámetros fijos** de la mejor configuración de 07 y los mismos 5 folds. Cada fold se estima por separado (las variables cambian por fold), así que la base E0 da 0,7530 y no 0,7537. Comparar siempre contra E0.
- **Bloques:** V1 factor de expansión; V2 el mejor ocupado del hogar; V3 target encoding del oficio (tasa de pobreza suavizada con peso 20); V4 ingreso predicho con un LightGBM de regresión de `log1p(Ingtot)` a nivel persona, con variables que están en test. `Ingtot` se lee del CSV crudo de personas de train y solo es variable objetivo del modelo auxiliar.
- **Fuga:** V3 y V4 usan `pobre` e `Ingtot` y se estiman con validación cruzada anidada. Para el fold f, los hogares de entrenamiento reciben la variable de modelos que no vieron ni f ni su propio fold, y los de f la de un modelo con los otros 4. El modelo final usa en train las variables fuera de muestra de 5 folds y en test un modelo con todo train. V1 y V2 podrían pasar a `agregar_hogar()`; V3 y V4 tienen que quedarse en el script del modelo.
- **Resultado:** solo el ingreso predicho sirve. E4 (+V4) da 0,7594 y E6 0,7596, frente a 0,7530 de E0; V1-V3 aportan menos de 0,001. El ensamble con el logit 06 (w = 0,8) da 0,7604. El R² del modelo de personas es 0,94 y el AUC del ingreso predicho solo es 0,921. Se generaron S2 (E6) y S4 (ensamble con V4), que dependen de que se permita usar el ingreso de train.

## Boosting con variables auxiliares (`script/07c_boosting_ingreso.r`)

- **Comando:** `Rscript -e "source('script/07c_boosting_ingreso.r', encoding = 'UTF-8')"` (~65 min con 20 núcleos; `prueba <- TRUE;` para probar con el 10%).
- **Paso 0 (fórmula del DANE):** Ingpcug = Ingtotugarr / Npersug e Ingtotug = suma de Ingtot de la unidad de gasto (100%); Ingtot = laboral (Impa, Isa, Ie, Imdi y sus imputados) + no laboral (Iof*). Arriendo imputado = Ingtotugarr − Ingtotug: P5130 en propia pagada y usufructo, P5130 − P5100 en propia pagando, 0 en el resto y si P5130 es 98/99 (coincide en el 98,4%).
- **Variables auxiliares** (todas con CV anidada): V4 ingreso de las personas (07b); V5 V4 + arriendo imputado con la regla; V6 ingreso laboral y no laboral por persona con las variables de su hogar; V7 regresión de log(Ingpcug / Lp) a nivel hogar; V8 regresión del estrato.
- **Errores de anidar predicciones:** (1) fuga de la etiqueta, controlada con CV anidada (ningún modelo auxiliar usa la etiqueta del fold que se evalúa); (2) error distinto en train y test, controlado dando a test el promedio de los 5 modelos de los folds (no un modelo con todo train), para que tenga el mismo error que las variables fuera de muestra con que aprendió el clasificador. La diferencia resultó mínima (proporción de pobres 24,08% vs. 24,03%). Cambiar solo las semillas de los modelos auxiliares mueve el F1 ~0,001: por eso solo cuentan mejoras mayores a 0,001.
- **Resultado:** F0 (= E4) 0,7584; V5 no aporta; V6 +0,0015; V7 +0,0013; V8 +0,0003; F5 (V6 + V7) 0,7623; reajuste (lr 0,02, 31 hojas) 0,7629; promedio de 5 semillas 0,7622. `log_ing6arr_pc_lp` (63%) e `inghog_pred` (20%) dominan la ganancia.

## Modelo lineal de probabilidad (`script/10_lpm.r`)

- **Comando:** `Rscript -e "source('script/10_lpm.r', encoding = 'UTF-8')"` (~19 min). Usa exactamente las variables de `train_hogar` (sin variables predichas) y los 5 folds de 04.
- `lm` con la matriz armada con train y test juntos (como en 04) para no fallar con niveles nuevos. Genera dos submissions: la mejor configuración y la misma con pesos por clase (pobres = #no pobres / #pobres), con sufijo `_w`. Reporta F1 con el mejor umbral y con 0,5, y guarda sus resultados en `output/logs/lpm/resultados.rds`.
- **Resultado:** efectos principales 0,7068; fórmula de 04 0,7250 (con pesos 0,7279). 27% de las predicciones quedan fuera de [0,1], casi todas negativas. Con pesos, el umbral óptimo queda en 0,6, el borde de la grilla. La submission en Kaggle es `LPM_interacciones_threshold_038.csv`.

## Windows

- **Windows Server (sin UTF-8 nativo en R):** para que las tildes salgan bien en figuras, correr con `Rscript -e "source('script/05_logit_parsimonioso.r', encoding = 'UTF-8')"`. Con ese `source()`, `writeLines()` pasa el texto a CP1252: para escribir un `.tex` en UTF-8 usar `writeLines(enc2utf8(x), archivo, useBytes = TRUE)` (así lo hacen 05, 07 y 07b).

## Estilo

Los scripts siguen una plantilla: encabezado con proyecto, nombre del script, autores, propósito, estructura numerada, input y output; secciones `## N. Título` separadas por líneas de `#`. Mantener la plantilla y actualizar la estructura/input/output del encabezado al agregar secciones.
