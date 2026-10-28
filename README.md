# Predicting Poverty: ¿quién es pobre en Colombia?

Predicción de los hogares por debajo de la línea de pobreza monetaria en
Colombia con la Gran Encuesta Integrada de Hogares (GEIH 2018, DANE) para
la competencia [Uniandes BDML 2026-20 PS2](https://www.kaggle.com/competitions/uniandes-bdml-2026-20-ps-2/overview). El modelo
final es un XGBoost a nivel hogar alimentado por modelos auxiliares por persona
(el ingreso predicho de cada miembro, con pérdida Tweedie), combinado con una
regresión del ingreso del hogar: F1 de validación cruzada 0,7775 , F1 de ranking 
público de Kaggle: 0,77 , F1 de ranking privado de Kaggle: 0.78.

**Autores:** Maria Jose Perez, Juan Manuel Lozano y Samuel Suárez Valle.
**Curso:** Big Data y Machine Learning Aplicado a la Economía, Universidad de
los Andes, 2026.

## Replicación

1. Descargar los cuatro archivos de la
   [competencia en Kaggle](https://www.kaggle.com/competitions/uniandes-bdml-2026-20-ps-2/data) y ponerlos en
   `data/`: `train_hogares.csv`, `train_personas.csv`, `test_hogares.csv` y
   `test_personas.csv`.
2. Desde la raíz del repositorio, con una sesión de R vacía:

```sh
Rscript 01_code/00_rundirectory.R
```

`00_rundirectory.R` corre todos los scripts en orden, cada uno en un proceso de
R aparte, y se detiene si alguno falla. Instala los paquetes que falten con
`pacman::p_load()`. Las rutas son relativas a la raíz (paquete `here`).

**Tiempo.** La corrida completa tarda muchas horas, casi todas en
`11_xgboost_hogar.R` (validación cruzada anidada e importancia por
permutación). Ese script y `15_comparacion_modelos.R` guardan cada resultado al
terminarlo y, si se vuelven a correr, solo estiman lo que falta. Se probó en un
Mac de 8 GB de RAM.

**Modo de prueba.** Para verificar en unos 20 minutos que todo corre, existe
un modo de prueba: los modelos usan el 10% de los hogares y pocas iteraciones,
y no escriben predicciones. Viene **apagado**; para activarlo:

```sh
PRUEBA=1 Rscript 01_code/00_rundirectory.R
```

## Estructura

```
.
├── 01_code/
│   ├── 00_rundirectory.R       # corre todo en orden
│   ├── 01_ ... 16_*.R          # un script por tarea (ver abajo)
│   ├── functions/
│   │   └── funciones_comunes.R # F1, probabilidad por hogar, umbral, etc.
│   └── pruebas/                # pruebas exploratorias que no entran al flujo principal
├── 02_outputs/
│   ├── figures/                # figuras (.png)
│   ├── tables/                 # tablas (.tex)
│   ├── predictions/            # predicciones para Kaggle (.csv)
│   ├── logs/                   # resultados intermedios de los modelos por hogar
│   └── pruebas/                # resultados guardados de 11, 15 y 16 (para retomar)
├── data/                       # CSV crudos (no se versionan) y bases limpias (.rds)
├── documentation/              # enunciado, diccionario del DANE y guía de variables
├── latex/                      # documento de resultados
└── presentacion/               # presentaciones (beamer) y su narrativa
```

## Scripts

| Script | Qué hace |
| :--- | :--- |
| `01_data_cleaning.R` | Une personas y hogares, renombra y recodifica variables y crea la variable objetivo y las variables nuevas. Arma también la base con clases balanceadas y la base sin el 10% más rico. |
| `02_bases_hogar.R` | Agrega las personas a una fila por hogar (94 variables). |
| `03_data_analysis.R` | Estadísticas descriptivas por condición de pobreza y figuras. |
| `04_naive_bayes.R` | Naive bayes por persona; la probabilidad del hogar es el promedio de los log odds de sus miembros. |
| `05_lpm.R` | Modelo lineal de probabilidad por hogar. |
| `06_logit.R` | Logit lasso por hogar con interacciones. |
| `07_logit_parsimonioso.R` | F1 frente al número de variables de un logit. |
| `08_logit_interacciones.R` | Logit lasso con más interacciones y splines. |
| `09_carts.R` | Árbol de clasificación (CART) por persona. |
| `10_xgboost_persona.R` | XGBoost por persona. |
| `11_xgboost_hogar.R` | **Modelo final:** XGBoost por hogar con modelos auxiliares por persona (validación cruzada anidada), combinación con la regresión del ingreso, análisis de errores e importancia por permutación. |
| `12_lightgbm.R` | LightGBM por hogar. |
| `13_lightgbm_variables.R` | LightGBM con variables nuevas e ingreso predicho. |
| `14_lightgbm_ingreso.R` | LightGBM con modelos auxiliares del ingreso. |
| `15_comparacion_modelos.R` | Compara el mejor modelo de cada familia con los mismos folds y el desempeño por grupo de población. |
| `16_figuras_presentacion.R` | Figuras y datos de las presentaciones: predictores, nube 3D, matriz de confusión y ROC de los 8 modelos, F1 según el umbral y según el número de variables. |

Todos los modelos escogen sus hiperparámetros y el umbral de clasificación por
F1 en validación cruzada, con folds por hogar: todas las personas de un hogar
quedan en el mismo fold, porque la pobreza es del hogar. Las notas al final de
cada script resumen sus resultados.

## Outputs

- `02_outputs/tables/`: descriptivas (`descriptivas.tex`), comparación de
  modelos (`modelos_f1.tex`), equidad por grupo (`equidad_*.tex`), importancia
  de las variables (`importancia_*.tex`) y tablas de cada familia de modelos.
- `02_outputs/figures/`: ingreso por variable, F1 del logit por número de
  variables, importancia y errores por grupo; las que empiezan por `pres_` son
  de las presentaciones (16).
- `02_outputs/predictions/`: una predicción por modelo con el formato de
  Kaggle (`id`, `pobre`; una fila por hogar de test). El nombre dice el modelo
  y sus hiperparámetros, p. ej. `NB_laplace_0_threshold_045.csv`. Se generan al
  correr los scripts y no se versionan.
- `02_outputs/tables/presentacion/`: datos (`.dat`) de las gráficas de las
  presentaciones (16).
- `latex/resultados.tex` lee las tablas y figuras de `02_outputs/`; se compila
  con `cd latex && latexmk -pdf resultados.tex`. Las presentaciones, con
  `cd presentacion && latexmk -pdf best_equipo_04.tex compare_equipo_04.tex`.

## Software

R 4.6.1. Paquetes (se instalan con `pacman::p_load()`): tidyverse 2.0.0,
here 1.0.2, e1071 1.7.17, rpart 4.1.27, caret 7.0.1, xgboost 3.2.1.1,
lightgbm 4.7.0, ranger 0.18.0, glmnet 5.1, doParallel 1.0.17, callr 3.8.0,
knitr 1.52, kableExtra 1.4.1, plot3D y plotly. xgboost debe ser 3.x: `10_xgboost_persona.R`
define su propio modelo para caret porque el `xgbTree` de caret no funciona con
esa versión.
