# ¿Por qué XGBoost? Comparación de modelos para predecir la pobreza en Colombia

**Equipo 04:** Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
Big Data y Machine Learning Aplicado a la Economía, Universidad de los Andes.

Este documento acompaña la presentación [`compare_equipo_04.tex`](compare_equipo_04.tex), el segundo set de diapositivas. El primero, [`best_equipo_04.tex`](best_equipo_04.tex), presenta el modelo ganador en detalle. Aquí contamos por qué ese modelo sobresalió, cómo se compara con los demás algoritmos que probamos y qué tradeoffs implica elegirlo para una política pública.

Los números salen de:
- [`output/tables/modelos_f1.tex`](../output/tables/modelos_f1.tex), [`equidad_brechas.tex`](../output/tables/equidad_brechas.tex), [`descriptivas.tex`](../output/tables/descriptivas.tex) y las tablas de importancia;
- [`latex/resultados.tex`](../latex/resultados.tex);
- las notas de los scripts de logit de la rama `main` (`04_logit.r`, `05_logit_parsimonioso.r` y `06_logit_interacciones.r`).

---

## 1. El modelo que elegimos

**XGBoost por hogar.** Es un clasificador de 163 variables con categóricas nativas y estos hiperparámetros: `eta` 0,02, `max_depth` 8, `min_child_weight` 50, `colsample_bytree` 0,5 y `subsample` 0,8. Lo alimentan modelos auxiliares por persona, estimados con validación cruzada anidada:
- el ingreso de cada miembro, con pérdida **Tweedie**;
- la probabilidad de que cada miembro sea pobre.

El clasificador se combina 70/30 con una regresión XGBoost del ingreso del hogar, y el umbral es 0,39.

| Métrica | Valor |
| :--- | :---: |
| F1 de validación cruzada (5 folds por hogar) | **0,7775** |
| F1 con umbral honesto | 0,7775 |
| Kaggle, leaderboard público (20% de test) | 0,77 (puesto 4) |
| Kaggle, leaderboard privado (80% de test) | **0,78 (puesto 1, ▲3)** |

Capturas: [`img/kaggle_publico.png`](img/kaggle_publico.png) y [`img/kaggle_privado.png`](img/kaggle_privado.png).

**El arranque de la historia.** En el leaderboard público estábamos en el puesto 4; en el privado quedamos en el **puesto 1**, tras subir tres lugares. El leaderboard público usa solo el 20% de test y tiene una desviación estándar de cerca de 0,006, así que no distingue mejoras de 0,001. Por eso nunca escogimos modelos por el leaderboard público, sino por la validación cruzada:
- 5 folds por hogar;
- validación cruzada anidada sin fuga;
- F1 con umbral honesto;
- ruido medido con otras particiones.

Los equipos que optimizaron contra el leaderboard público se sobreajustaron a ese 20%. Nuestro modelo generalizó mejor.

---

## 2. Datos

### 2.1 Limpieza

- **Muestra:** GEIH 2018 (DANE), con **543.103 personas en 164.958 hogares**. Son pobres el 20,0% de los hogares y el 25,1% de las personas.
- **Merge:** personas y hogares se unen por las columnas que comparten (`id`, `Clase`, `Dominio`, `Fex_c`, `Depto`, `Fex_dpto`), en una relación de muchos a uno.
- **Variable objetivo:** `pobre` = ingreso per cápita de la unidad de gasto con arriendo imputado (`Ingpcug`) < línea de pobreza. Verificamos que se cumple en el 100% de los hogares.
- **Montos de ingreso:** no están en test, así que se eliminan de train. El modelo tiene que reconstruir el ingreso.
- **`NA` = "la pregunta no aplica"** por los saltos del cuestionario (por ejemplo, las preguntas de trabajo solo se hacen a los ocupados). No se reemplaza por 0 sin revisar qué significa.
- **Preguntas sí/no:** se pasan a 1/0, y el "no sabe" queda como `NA`. Las categóricas conservan su código DANE **y** tienen una dummy por categoría.
- **Arriendos:** los códigos 98 y 99 de `arriendo_estimado`, `arriendo_pagado` y `cuota_amortizacion` pasan a `NA`. Los valores atípicos se dejan a propósito.
- **Hogares eliminados:** 2 de train, con un número de cuartos imposible (98 y 43). En test no se elimina nada.
- **`arriendo_imputado`:** reconstruye la regla del DANE y coincide en el 98,4% de los hogares.
- **Base por hogar** (`01b_bases_hogar.r`): una fila por hogar con sumas, proporciones y máximos de sus miembros.

### 2.2 Renombramiento de variables

Las preguntas del DANE vienen con códigos (`P####`). Las renombramos para que el equipo y quien lea los resultados las entienda sin el diccionario. La guía completa está en [`documentation/guia_variables.Rmd`](../documentation/guia_variables.Rmd).

| Código DANE | Nombre en las bases | Contenido |
| :--- | :--- | :--- |
| `P6020` | `mujer` | 1 = mujer, 0 = hombre |
| `P6040` | `anos` | Años cumplidos |
| `P6210` | `max_educ` | Nivel educativo más alto alcanzado |
| `P6100` | `regimen_salud` | Contributivo, especial o subsidiado |
| `P6430` | `posicion_ocupacional` | Empleado particular, gobierno, cuenta propia… |
| `P6800` | `horas_trabajo_semana` | Horas que trabaja normalmente |
| `P6870` | `tamano_empresa` | Personas que trabajan en la empresa |
| `P6920` | `cotiza_pension` | Cotiza a pensión (formalidad) |
| `P7495` | `recibio_arriendos_pensiones` | Recibió arriendos o pensiones el mes pasado |
| `P5090` | `tenencia_vivienda` | Propia, arriendo, usufructo, sin título… |
| `Clase` | `cabecera` | 1 = cabecera municipal |
| Creada | `arriendo_imputado` | Regla del DANE reconstruida |

### 2.3 Estadísticas descriptivas

| Variable (persona) | No pobres | Pobres | Diferencia |
| :--- | :---: | :---: | :---: |
| Ocupado | 0,508 | 0,304 | −0,204 |
| Educación superior | 0,295 | 0,092 | −0,203 |
| Primaria | 0,232 | 0,343 | 0,111 |
| Vivienda propia pagada | 0,427 | 0,294 | −0,133 |
| Vivienda sin título | 0,032 | 0,115 | 0,082 |
| Tamaño de empresa (escala 0–9) | 2,15 | 0,64 | −1,51 |
| Meses en la empresa | 43,8 | 25,5 | −18,3 |
| Edad | 35,7 | 27,3 | −8,3 |
| Proporción del hogar en edad de trabajar | 0,856 | 0,731 | −0,125 |
| Arriendo estimado (miles de pesos) | 600 | 278 | −322 |

Todas las diferencias tienen un p-valor < 0,001 (prueba *t*). La tabla completa está en [`descriptivas.tex`](../output/tables/descriptivas.tex) y las figuras en [`ingreso_continuas.png`](../output/figures/ingreso_continuas.png) e [`ingreso_dummies.png`](../output/figures/ingreso_dummies.png).

**Lo que anticipan los datos sobre los modelos:**
1. Lo que separa a los pobres es **del hogar**: cuántos trabajan, cuántos dependen de ellos y qué vivienda tienen. Por eso los modelos que agregan la información por hogar le ganan a los que modelan a cada persona.
2. Las relaciones son **no lineales y con umbrales**: el tamaño de la empresa, la edad, el arriendo y la interacción entre ocupados y educación. Los árboles encuentran esas formas solos, mientras que un modelo lineal necesita que se las construyan a mano.

---

## 3. Comparación de algoritmos

### 3.1 Panorama: el mejor modelo de cada categoría

Todos los modelos se evalúan igual: los 5 folds por hogar de `05b` y el F1 por hogar. Naive Bayes, CART y XGBoost por persona se volvieron a estimar con esos folds; en sus scripts usaban 10.

| Escalón | Categoría | Especificación | F1 | F1 honesto | Umbral |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Por persona | CART | 16 variables, `cp` 2,5e-05, up-sampling | 0,6685 | 0,6679 | 0,62 |
| | Naive Bayes | 16 variables, `laplace` 0 | 0,6755 | 0,6752 | 0,42 |
| | XGBoost | 24 variables, 500 rondas | 0,7055 | 0,7055 | 0,32 |
| Por hogar | **Logit lasso** | 1.159 columnas con interacciones y splines | 0,7451 | 0,7449 | 0,37 |
| | XGBoost | Solo la base por hogar de 01b | 0,7534 | 0,7532 | 0,36 |
| Por hogar + modelos anidados | XGBoost | Sin Tweedie, con la regresión del ingreso | 0,7738 | 0,7736 | 0,38 |
| | **Random forest** | `ranger`, 500 árboles, `mtry` 30, variables del mejor XGBoost | 0,7752 | 0,7750 | 0,40 |
| | **XGBoost final** | Tweedie + regresión del ingreso (70/30) | **0,7775** | **0,7775** | 0,39 |

Otros puntajes de Kaggle del equipo: logit lasso 0,74 y LightGBM por hogar 0,75 (F1 CV 0,7537).

**Lectura:** el salto lo dan la **unidad de análisis** (de persona a hogar, cerca de +0,04) y la **información** (los modelos anidados que reconstruyen el ingreso, cerca de +0,02), más que el algoritmo. Con la misma información, random forest y XGBoost quedan a 0,002 de distancia.

### 3.2 La frontera entre precisión y complejidad

Si ordenamos los modelos por complejidad, o por cuánta interpretabilidad se pierde, la frontera es **cóncava**:

| Modelo | F1 |
| :--- | :---: |
| Logit sin penalizar con 15 variables | 0,697 |
| Logit lasso con interacciones | 0,745 |
| XGBoost con la base por hogar | 0,753 |
| Random forest y XGBoost con modelos anidados | 0,775–0,778 |

Cada punto adicional de F1 cuesta más complejidad. El XGBoost final le gana al logit lasso por **0,032**, a cambio de 26 modelos anidados por cada variable que depende del ingreso o de la etiqueta.

### 3.3 Top 3: drivers, resultados y limitaciones

| | **XGBoost final** | **Random forest** | **Logit lasso** |
| :--- | :--- | :--- | :--- |
| F1 / umbral | **0,7775** / 0,39 | 0,7752 / 0,40 | 0,7451 / 0,37 |
| Cómo funciona | Árboles secuenciales en los que cada uno corrige el error del anterior; stacking con modelos por persona | 500 árboles profundos en paralelo (`mtry` 30) que se promedian | Regresión logística penalizada (L1) con interacciones de a pares y splines |
| Drivers | Ingreso predicho por persona (permutar el grupo resta 0,42 de F1). De extremo a extremo, por pregunta: posición ocupacional P6430 (−0,208), arriendos/pensiones P7495 (−0,058), ingresos por trabajo de inactivos P7472 (−0,056), arriendo por persona (−0,038), horas P6800 (−0,021) | Las mismas variables que el XGBoost; la importancia propia del bosque no se calculó. Depende de la misma cadena de modelos anidados | Por bloques: demografía 0,468 → + trabajo 0,547 → + calidad del empleo 0,617 → + salud y educación 0,665 → + vivienda e ingresos no laborales 0,726 → + ubicación 0,730 → + interacciones 0,738. Variables más predictivas solas (AUC): arriendo por persona sobre la línea 0,79, proporción en régimen contributivo 0,76, horas por persona 0,74 |
| Fortalezas | Mejor F1; estable entre particiones (DE 0,0003); puesto 1 en el leaderboard privado | Casi el mismo F1 y menos sensible a los hiperparámetros | Coeficientes con signo y magnitud; auditable; sin modelos anidados que mantener |
| Limitaciones | Caja negra; 26 modelos por variable; costoso de mantener y de explicar | Caja negra; combinado con XGBoost no suma (0,7773): se equivoca en los mismos hogares | 0,03 menos de F1; 794 coeficientes no nulos, así que se interpreta por bloques y no coeficiente por coeficiente |

El logit y el XGBoost tienen una **correlación de 0,92** en log odds, y el peso óptimo del logit en una combinación es 0: los dos se equivocan en los mismos hogares, los que están cerca de la línea. Los tres modelos chocan con el mismo límite: el 30% de los errores está a menos del 10% de la línea de pobreza.

Fuentes de importancia: [`importancia_grupos.tex`](../output/tables/importancia_grupos.tex), [`importancia_variables_e2e.tex`](../output/tables/importancia_variables_e2e.tex) (secciones 14–16 de [`05b_xgboost_hogar.r`](../script/05b_xgboost_hogar.r)).

### 3.4 Cuánto cuesta en F1 un modelo que se puede leer

Logit sin penalizar con las *k* primeras variables, en el orden en que entran al lasso (`05_logit_parsimonioso.r` de `main`):

| *k* | Variables agregadas | F1 (CV) |
| :---: | :--- | :---: |
| 3 | `log_arriendo_pc_lp`, `prop_contributivo`, `n_menor18` | 0,5848 |
| 5 | `educ_ocup_suma_pc`, `jefe_regimen` | 0,6351 |
| 8 | `horas_pc`, `n_formal`, `n_con_beneficios` | 0,6591 |
| 10 | `n_fuentes_no_lab`, `pers_por_dormitorio` | 0,6732 |
| 15 | `any_arriendos`, `n_asalariado`, `prop_menores`, `tenencia`, `arriendo_na` | 0,6970 |
| 20 | Desocupados, ocupados con educación superior, jornaleros, educación media de los adultos | 0,7039 |
| 25 | Dominio, cuartos por persona, cabecera, empresa grande, arriendo pagado | 0,7106 |
| 30 | Ocupados, régimen especial, sin pago, oficios del hogar, pensión | 0,7157 |
| 40 | Patrón, quiere más horas, inactivos, posición del jefe, remesas, dependencia, depto… | 0,7273 |
| 50 | Características del jefe (oficio, sexo, educación, meses, tamaño de empresa) | 0,7313 |
| 94 | Todas | 0,7357 |
| — | Lasso con interacciones (1.159 columnas) | 0,7451 |

- Con **15 variables** el logit llega a 0,697, y con **40** a 0,727: a 0,05 del XGBoost.
- Las interacciones del lasso suman 0,01 más. El resto de la distancia la cubre la **información anidada** (el ingreso reconstruido por persona), no la forma funcional.
- Una tabla de 15 a 40 coeficientes se puede publicar, discutir y auditar.

### 3.5 Equidad: ¿a quién excluye cada modelo?

Brecha de sensibilidad, en puntos porcentuales, entre el grupo mejor y el peor detectado (predicciones fuera de muestra de train):

| Atributo | Naive Bayes | Logit lasso | Random forest | XGBoost final |
| :--- | :---: | :---: | :---: | :---: |
| Sexo del jefe | 2,0 | 4,2 | 4,4 | 3,8 |
| Empleo del jefe | 33,4 | 22,2 | 21,4 | 19,7 |
| Zona | 12,9 | 0,0 | 0,2 | 0,8 |
| Edad del jefe | 9,3 | 7,0 | 6,9 | 6,9 |
| Educación del jefe | 30,6 | 19,3 | 17,7 | 18,0 |
| Tamaño del hogar | 38,8 | 25,7 | 21,8 | 22,4 |
| Menores | 31,4 | 23,8 | 19,1 | 19,9 |
| Vivienda | 19,9 | 7,2 | 6,2 | 6,4 |

- Los modelos más precisos **reducen** casi todas las brechas. Por ejemplo, la brecha de falsos positivos entre zonas pasa de 27,8 puntos con naive Bayes a 6,5 con el XGBoost final. Entre los top 3, las brechas son parecidas.
- Todos los modelos excluyen más a los pobres con jefe formal o con educación superior y a los hogares pequeños o sin menores. Estos son los pobres "que no lo parecen".
- **Sexo del jefe:** la única brecha que crece con los modelos por hogar, de 1–2 a unos 4 puntos. Siempre favorece a los hogares con jefa, que se detectan mejor.
- Con una sola regla de decisión no se pueden igualar a la vez la sensibilidad y los falsos positivos entre grupos con tasas de pobreza distintas (Kleinberg, Mullainathan y Raghavan, 2017).

Figura completa por grupo: [`equidad_errores.png`](../output/figures/equidad_errores.png).

---

## 4. Conclusión: interpretabilidad vs. flexibilidad

| Criterio | Logit | XGBoost |
| :--- | :---: | :---: |
| F1 | 0,745 | **0,778** |
| Se puede explicar a un beneficiario | sí | no |
| Auditable por un ente de control | sí | difícil |
| Costo de mantener y reentrenar | bajo | alto (26 modelos anidados por variable) |
| Riesgo de manipulación | mayor (reglas visibles) | menor |
| Captura no linealidades | a mano | sola |

Un instrumento de focalización tiene que ser **creíble**. Los hogares, los alcaldes y los entes de control tienen que poder entender por qué un hogar entra o sale de un programa. Un XGBoost con 163 variables y 26 modelos anidados predice mejor, pero difícilmente es accesible para quien diseña la política. Además, la evidencia sobre el SISBÉN muestra que la credibilidad y la manipulación del instrumento son problemas reales (Camacho y Conover, 2011). Para decisiones de alto impacto, la literatura también sugiere preferir modelos interpretables cuando la pérdida de precisión es pequeña (Rudin, 2019).

### Recomendación de política

1. **Instrumento oficial: un logit parsimonioso**, de 15 a 40 variables del módulo laboral, la vivienda y la composición del hogar. Se puede publicar, discutir y auditar, y pierde entre 0,05 y 0,08 de F1 frente al XGBoost.
2. **XGBoost como auditor:** señala los hogares que el logit excluiría y que probablemente son pobres (los pobres "que no lo parecen": jefe formal o educado, hogares pequeños) para verificarlos en campo antes de excluirlos.
3. **El umbral es una decisión de política:** debe fijarse según el costo social de excluir a un pobre frente al de incluir a un no pobre, no según el F1.

### Para la discusión

- ¿Cuántos puntos de F1 vale la transparencia?
- ¿Quién responde por el error de una caja negra en un programa social?
- ¿Es aceptable que el instrumento oficial sea menos preciso si es más legítimo?

---

## 5. Cómo replicar

```sh
Rscript script/01_data_cleaning.r                      # bases limpias
Rscript output/pruebas/hogar/01b_bases_hogar_main.r    # bases por hogar
Rscript script/05b_xgboost_hogar.r                     # XGBoost final, random forest y permutaciones
Rscript output/pruebas/logit/06_logit_interacciones_main.r   # logit lasso (copia del 06 de main)
Rscript script/06_comparacion_modelos.r                # modelos_f1.tex y tablas de equidad
cd presentacion && latexmk -pdf compare_equipo_04.tex  # presentación
```

---

## Referencias

- Angulo, R., Díaz, Y. y Pardo, R. (2016). The Colombian Multidimensional Poverty Index: Measuring poverty in a public policy context. *Social Indicators Research*, 127(1), 1–38.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Camacho, A. y Conover, E. (2011). Manipulation of social program eligibility. *American Economic Journal: Economic Policy*, 3(2), 41–65.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- DANE (2019). Pobreza monetaria y multidimensional en Colombia 2018. Boletín técnico.
- Grosh, M. y Baker, J. (1995). Proxy means tests for targeting social programs. LSMS Working Paper 118, Banco Mundial.
- Kleinberg, J., Ludwig, J., Mullainathan, S. y Obermeyer, Z. (2015). Prediction policy problems. *American Economic Review*, 105(5), 491–495.
- Kleinberg, J., Mullainathan, S. y Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. *ITCS*.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society B*, 58(1), 267–288.
