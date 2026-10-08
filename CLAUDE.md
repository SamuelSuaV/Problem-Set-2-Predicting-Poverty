# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Problem Set 2 de Big Data & Machine Learning Applied in Economics: predecir qué hogares de Colombia están por debajo de la línea de pobreza usando la GEIH 2018 (DANE). Autores: Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle. Código, comentarios y documentación en español.

## Reglas sobre variables

- **Todo cambio a una variable (renombrar, recodificar, crear, eliminar) se hace en `script/01_data_cleaning.r`**, nunca en los scripts de análisis.
- **Después de cualquier cambio a variables, actualizar `documentation/guia_variables.Rmd`** (la guía de variables para el equipo). Su último chunk compara la guía con `data/train_clean.rds` y avisa si hay variables sin documentar o que ya no existen.
- Cualquier edición debe aplicarse igual a train y test: va dentro de la función `editar()` de `01_data_cleaning.r`, que se llama sobre ambas bases.

## Comandos

R con `pacman::p_load` y `here` (rutas relativas a la raíz del repo). Correr desde la raíz:

```sh
Rscript script/01_data_cleaning.r    # CSV crudos -> data/train_clean.rds, data/test_clean.rds
Rscript script/02_data_analysis.r    # tabla en output/tables/, figuras en output/figures/
Rscript -e 'rmarkdown::render("documentation/guia_variables.Rmd")'   # compila la guía y verifica que esté al día
cd latex && latexmk -pdf resultados.tex   # documento de resultados
```

Flujo: 01 → 02 → LaTeX. 02 lee los `.rds` de 01, así que 01 debe correrse primero cada vez que cambian las variables. `latex/resultados.tex` no tiene números de tablas copiados: hace `\input{../output/tables/descriptivas.tex}` y carga las figuras de `../output/figures/`, así que basta con correr 02 y recompilar. Se referencia la tabla con `\ref{tab:descriptivas}` (el `label` está en el `kbl()` de 02). Los números que sí están escritos en el texto (tamaño de muestra, % de pobres, mediana de la línea de pobreza, percentil 99 del ingreso) hay que actualizarlos a mano si cambia la limpieza. `script/00_master_file.r` existe pero está vacío.

## Datos

- `data/` tiene los 4 CSV crudos (`{train,test}_{hogares,personas}.csv`) y los `.rds` limpios. `*.csv` y `*.rds` están en `.gitignore` (los `.rds` pesan cientos de MB).
- Diccionario del DANE: `documentation/ddi-documentation-spanish-608.pdf`. Ahí están las etiquetas y códigos de cada pregunta `P####`.
- La unidad de observación de las bases limpias es la **persona**; las variables del hogar se repiten para cada persona del mismo `id`.

## Decisiones de limpieza que no son obvias

- `read_csv(..., guess_max = Inf)`: sin esto, columnas vacías en las primeras 1000 filas se leen como lógicas y se pierden datos.
- Merge: `left_join(personas, hogares, by = c("id", "Clase", "Dominio", "Fex_c", "Depto", "Fex_dpto"), relationship = "many-to-one")`. Esas columnas son idénticas en ambas tablas, así que van como llaves para no generar `.x`/`.y`.
- **Variable objetivo**: `pobre = ingreso < linea_pobreza`, donde `ingreso` es `Ingpcug` (ingreso per cápita de la unidad de gasto con arriendo imputado). Se verificó que se cumple en el 100% de los hogares. También `Indigente = ingreso < linea_indigencia`.
- Solo existen en train: `ingreso`, `pobre` (`Pobre` en el CSV), `Indigente`, `Npobres`, `Nindigentes`, `Estrato1`. No sirven como predictores. Las demás preguntas de montos de ingreso (`P####s#a#`, etc.) y agregados (`Impa`…`Ingtot`, `Ingtotug`, `Ingtotugarr`) se eliminan de train porque no están en test.
- Preguntas sí/no (vector `si_no`): 1 = sí, 0 = no, `NA` = no sabe (el 9 original) o pregunta no aplicada. `sexo` → `mujer`, `clase` → `cabecera`.
- Categóricas `max_educ`, `regimen_salud`, `tenencia_vivienda`: se conservan con los códigos del DANE y además tienen dummies por categoría (lista `categorias`), incluida "no sabe".
- `NA` casi siempre significa que la pregunta no aplica por los saltos del cuestionario (p. ej. preguntas de trabajo solo para ocupados); no reemplazar por 0 sin revisar qué significa.
- `arriendo_estimado`, `arriendo_pagado`, `cuota_amortizacion`: 98 (no sabe) y 99 (no informa) → `NA`. Los valores atípicos (arriendos de hasta $600M) se dejan a propósito.
- Se eliminan de train los 2 hogares con `cuartos_hogar` 98 y 43 (error de digitación). **Nunca se eliminan filas de test**: hay que predecir todas.

## Estilo

Los scripts siguen una plantilla: encabezado con proyecto, nombre del script, autores, propósito, estructura numerada, input y output; secciones `## N. Título` separadas por líneas de `#`. Mantener la plantilla y actualizar la estructura/input/output del encabezado al agregar secciones.
