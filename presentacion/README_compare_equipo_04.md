# ¿Por qué XGBoost? Comparación de modelos para predecir la pobreza en Colombia

**Equipo 04:** Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
Big Data y Machine Learning Aplicado a la Economía, Universidad de los Andes.

Este documento acompaña la presentación [`compare_equipo_04.tex`](compare_equipo_04.tex), el segundo set de diapositivas. El primero, [`best_equipo_04.tex`](best_equipo_04.tex), presenta el modelo ganador en detalle. Aquí contamos cómo se compara ese modelo con los demás algoritmos que probamos y qué tradeoffs implica elegirlo para una política pública.

Los números salen de:
- las predicciones fuera de muestra de los 8 modelos (`ps2_predicciones_fuera_de_muestra/predicciones_oof.csv`, mismas 5 particiones por hogar) y de [`script/11_figuras_presentacion.r`](../script/11_figuras_presentacion.r), que calcula las matrices de confusión, el ROC-AUC, las métricas, la nube 3D y las curvas de F1 por número de variables (`output/tables/presentacion/*.dat`);
- [`output/tables/modelos_f1.tex`](../output/tables/modelos_f1.tex) y [`equidad_brechas.tex`](../output/tables/equidad_brechas.tex) (script `06`);
- [`latex/resultados.tex`](../latex/resultados.tex).

---

## 1. El modelo que elegimos

**XGBoost por hogar.** Es un clasificador de 163 variables con categóricas nativas: la base por hogar más los resúmenes de modelos por persona (ingreso con pérdida **Tweedie** y probabilidad de ser pobre, estimados con validación cruzada anidada). Hiperparámetros: `eta` 0,02, `max_depth` 8, `min_child_weight` 50, `colsample_bytree` 0,5 y `subsample` 0,8. Se combina 70/30 con una regresión del ingreso del hogar, con umbral 0,39.

| Métrica | Valor |
| :--- | :---: |
| F1 de validación cruzada (5 folds por hogar) | **0,7775** |
| Kaggle, puntaje público (20% de test) | 0,77 (puesto 4) |
| Kaggle, puntaje privado (80% de test) | **0,78 (puesto 1)** |

Capturas: [`img/kaggle_publico.png`](img/kaggle_publico.png) y [`img/kaggle_privado.png`](img/kaggle_privado.png).

---

## 2. Datos

### 2.1 Lo que hicimos distinto

GEIH 2018: 543.103 personas en 164.958 hogares, 20% de hogares pobres. Los montos de ingreso no están en test, así que hay que reconstruirlos. Cuatro decisiones nos separaron de un pipeline estándar:

1. **El hogar como unidad.** Una fila por hogar con sumas, proporciones y los datos del jefe, porque la pobreza se define en la unidad de gasto.
2. **La regla del DANE, reconstruida.** El arriendo imputado (lo que el DANE suma como ingreso a quien no paga arriendo) se reconstruyó con la tenencia, el arriendo estimado y la cuota de amortización. Coincide en el 98,4% de los hogares de train.
3. **El ingreso como objetivo intermedio.** Modelos por persona predicen el ingreso laboral y no laboral (pérdida Tweedie) y se suman en el hogar, sin usar nunca la etiqueta del fold evaluado.
4. **Preguntas añadidas en una segunda revisión.** Al revisar el cuestionario agregamos preguntas que habíamos dejado por fuera al principio: pagos en especie, búsqueda de trabajo e ingresos de desocupados e inactivos. Identifican quién recibe ingresos sin estar ocupado: el $R^2$ del ingreso por persona sube de 0,94 a 0,98.

**Limpieza (resumen).** Merge personas–hogares muchos a uno; `pobre` = ingreso per cápita con arriendo imputado < línea (se cumple en el 100% de los hogares); `NA` = "la pregunta no aplica" por los saltos del cuestionario; sí/no a 1/0; las categóricas conservan su código DANE y una dummy por categoría; se eliminan 2 hogares de train con cuartos imposibles (en test, ninguno). Las preguntas `P####` se renombraron (P6430 → `posicion_ocupacional`, P7495 → `recibio_arriendos_pensiones`, P5090 → `tenencia_vivienda`, etc.); la guía completa está en [`documentation/guia_variables.Rmd`](../documentation/guia_variables.Rmd).

### 2.2 Descriptivas comparativas: dónde caen los aciertos y los errores

La nube 3D ([`pres_3d.png`](../output/figures/pres_3d.png); versión interactiva que se puede rotar, aparte del PDF: [`pres_3d.html`](../output/figures/pres_3d.html)) muestra 8.000 hogares de train al azar en tres ejes:

- **x:** log(ingreso / línea), donde 0 es la línea de pobreza (ingreso real, que solo existe en train);
- **y:** proporción de ocupados del hogar;
- **z:** personas en la unidad de gasto.

El color es el resultado del modelo final fuera de muestra. Para que los errores se vean en un salón, los hogares bien clasificados (pobres detectados y no pobres bien clasificados) van en gris, y los dos errores resaltan encima: pobres excluidos en rojo y no pobres incluidos en amarillo.

| Resultado | Hogares | Ingreso / línea (mediana) | Prop. ocupados | Personas | % a menos de 25% de la línea |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Pobre detectado | 26.635 | 0,64 | 0,28 | 4,3 | 27 |
| **Pobre excluido** | 6.389 | 0,85 | 0,45 | 3,3 | **60** |
| **No pobre incluido** | 8.854 | 1,22 | 0,35 | 3,8 | **53** |
| No pobre bien clasificado | 123.080 | 2,63 | 0,56 | 3,0 | 7 |

- Los errores se apiñan junto a la línea: el 60% de los pobres excluidos y el 53% de los no pobres incluidos están a menos de 25% de ella.
- Un pobre **excluido** se parece a un no pobre: más ocupados y un hogar más pequeño que los pobres detectados.
- Un no pobre **incluido** se parece a un pobre: menos ocupados y un hogar más grande que los no pobres bien clasificados.

Esto anticipa la comparación: todos los modelos chocan con los mismos hogares cerca de la línea, y lo que los separa es cuántos hogares "lejos" de ella clasifican bien.

---

## 3. Comparación de algoritmos

### 3.1 Panorama: el mejor modelo de cada categoría

Todos los modelos se evalúan igual: los mismos 5 folds por hogar y el F1 por hogar. Naive Bayes, CART y XGBoost por persona se volvieron a estimar con esos folds (en sus scripts usaban 10).

| Escalón | Categoría | Especificación | F1 | Umbral |
| :--- | :--- | :--- | :---: | :---: |
| Por persona | CART | 16 variables, `cp` 2,5e-05, up-sampling | 0,6685 | 0,62 |
| | Naive Bayes | 16 variables, `laplace` 0 | 0,6755 | 0,42 |
| | XGBoost | 24 variables, 500 rondas | 0,7055 | 0,32 |
| Por hogar | **Logit lasso** | 1.159 columnas con interacciones | 0,7451 | 0,37 |
| | XGBoost | Solo la base por hogar | 0,7534 | 0,36 |
| Por hogar + modelos anidados | XGBoost | Sin Tweedie, con la regresión del ingreso | 0,7738 | 0,38 |
| | **Random forest** | `ranger`, 500 árboles, `mtry` 30, variables del mejor XGBoost | 0,7752 | 0,40 |
| | **XGBoost final** | Tweedie + regresión del ingreso (70/30) | **0,7775** | 0,39 |

Otros puntajes de Kaggle del equipo: logit lasso 0,74 y LightGBM por hogar 0,75.

**Lectura:** el salto lo dan la **unidad de análisis** (de persona a hogar, cerca de +0,04) y la **información** (los modelos anidados que reconstruyen el ingreso, cerca de +0,02), más que el algoritmo.

### 3.2 La frontera: precisión vs. interpretabilidad

En la gráfica, el eje x ordena los modelos por **pérdida de interpretabilidad**: logit con 15 variables, CART, naive Bayes, logit lasso, XGBoost por persona, XGBoost con la base por hogar, random forest y XGBoost final. La frontera es **cóncava**: de 0,697 (logit con 15 variables) a 0,745 (lasso) a 0,778 (XGBoost). El XGBoost final le gana al logit lasso por **0,032**, a cambio de 26 modelos anidados por cada variable que depende del ingreso o de la etiqueta. Random forest y XGBoost están a 0,002 entre sí: con la misma información, el algoritmo casi no importa.

### 3.3 Top 3: cómo funcionan, resultados y limitaciones

| | **XGBoost final** | **Random forest** | **Logit lasso** |
| :--- | :--- | :--- | :--- |
| F1 / AUC | **0,7775** / 0,959 | 0,7752 / 0,958 | 0,7451 / 0,946 |
| Cómo funciona | Árboles secuenciales en los que cada uno corrige el error del anterior; stacking con modelos por persona | 500 árboles profundos en paralelo (`mtry` 30) que se promedian | Regresión logística penalizada (L1) con interacciones |
| Qué lo mueve | Ingreso predicho por persona; posición ocupacional | Las mismas variables que XGBoost | Arriendo por persona, afiliación contributiva, horas |
| Fortalezas | Mejor F1; estable entre particiones; puesto 1 en Kaggle privado | Casi el mismo F1 y menos sensible a los hiperparámetros | Coeficientes con signo y magnitud; auditable |
| Limitaciones | Caja negra; 26 modelos por variable; costoso de mantener y explicar | Caja negra; combinado con XGBoost no suma (0,7773): mismos errores | 0,03 menos de F1; 794 coeficientes no nulos, así que se lee por bloques |

**Detalle de "qué lo mueve"** (no va en la slide):
- *XGBoost:* permutar el ingreso predicho resta 0,42 de F1. De extremo a extremo, por pregunta: posición ocupacional P6430 (−0,208), arriendos/pensiones P7495 (−0,058), ingresos por trabajo de inactivos P7472 (−0,056), arriendo por persona (−0,038), horas P6800 (−0,021) ([`importancia_variables_e2e.tex`](../output/tables/importancia_variables_e2e.tex)).
- *Random forest:* usa la misma cadena de modelos anidados; su importancia propia no se calculó.
- *Logit:* por bloques, demografía 0,468 → + trabajo 0,547 → + calidad del empleo 0,617 → + salud y educación 0,665 → + vivienda e ingresos no laborales 0,726 → + interacciones 0,738. Las variables con mayor AUC solas: arriendo por persona sobre la línea (0,79), proporción en régimen contributivo (0,76) y horas por persona (0,74).

El logit y el XGBoost tienen una **correlación de 0,92** en log odds, y el peso óptimo del logit en una combinación es 0: los dos se equivocan en los mismos hogares, los que están cerca de la línea.

### 3.4 Cuánto cuesta en F1 un modelo que se puede leer

Para comparar los algoritmos con la misma información, cada uno se entrena con las *k* variables más importantes de la **base por hogar** (94 variables; el orden es el de la ganancia del XGBoost de esa base, igual para todos), con los mismos 5 folds y el umbral que maximiza el F1 (sección 7 de `11_figuras_presentacion.r`):

- **Logit:** `glm` binomial sin penalizar.
- **Naive Bayes:** `e1071`, normales para las numéricas.
- **CART:** `rpart`, `cp` 1e-4, mínimo 50 hogares por hoja.
- **Random forest:** `ranger`, 300 árboles, `mtry` = k/5, mínimo 50 hogares por hoja.
- **XGBoost:** `eta` 0,05, profundidad 6, `min_child_weight` 50, parada temprana.

| *k* | Logit | Naive Bayes | CART | Random forest | XGBoost |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 3 | 0,632 | 0,633 | 0,640 | 0,653 | 0,655 |
| 5 | 0,652 | 0,650 | 0,671 | 0,681 | 0,685 |
| 10 | 0,685 | 0,615 | 0,694 | 0,713 | 0,720 |
| 15 | 0,700 | 0,634 | 0,695 | 0,722 | 0,732 |
| 20 | 0,704 | 0,621 | 0,694 | 0,725 | 0,735 |
| 30 | 0,722 | 0,630 | 0,698 | 0,736 | 0,749 |
| 40 | 0,726 | 0,613 | 0,698 | 0,736 | 0,749 |
| 60 | 0,732 | 0,616 | 0,698 | 0,736 | 0,752 |
| 94 | 0,736 | 0,590 | 0,698 | 0,737 | 0,754 |

Las 5 primeras variables son `prop_contributivo`, `horas_pc`, `log_arriendo_pc_lp`, `educ_ocup_suma_pc` y `tenencia` (`orden_variables.dat`). Dos verificaciones: el XGBoost con las 94 da 0,7536, lo mismo que la base por hogar de `05b` (0,7534), y el logit con las 94 da 0,7357, igual que la curva de `main`.

- **Cóncavas:** con 15 variables el logit ya llega a 0,70 y XGBoost a 0,73. Desde 30 variables, los árboles casi no suben.
- **Con la misma información, el algoritmo vale cerca de 0,02:** con las 94 variables, logit 0,736 y random forest 0,737 frente a XGBoost 0,754. Los modelos anidados agregan otros 0,02 (XGBoost final 0,7775).
- **CART** se estanca en 0,698 desde 30 variables: un solo árbol no aprovecha más información.
- **Naive Bayes empeora** con más variables (0,65 con 5 → 0,59 con 94): supone independencia y cuenta dos veces la información de las variables correlacionadas. Su umbral óptimo se va al borde de la grilla (0,85) porque sus probabilidades quedan descalibradas.

La flecha punteada lleva del XGBoost con las 94 variables de la base (0,754) al XGBoost final de 163 variables (0,7775), que además usa los modelos anidados por persona: ese tramo es lo que aporta la información anidada, no el algoritmo. (El random forest completo, 0,7752, quedaría casi en el mismo punto, así que no se dibuja.) Esta curva reemplaza la del logit de la rama `main` (que usaba el orden del lasso), para que todos los algoritmos usen el mismo orden de variables.

### 3.5 Matriz de confusión de los tres mejores

Hogares de train (33.024 pobres y 131.934 no pobres), predicciones fuera de muestra con el umbral de cada modelo; porcentaje de la fila.

| Modelo | Pobre detectado | **Pobre excluido** | **No pobre incluido** | No pobre bien clasificado |
| :--- | :---: | :---: | :---: | :---: |
| XGBoost final | 26.635 (80,7%) | 6.389 (19,3%) | 8.854 (6,7%) | 123.080 (93,3%) |
| Random forest | 26.591 (80,5%) | 6.433 (19,5%) | 8.993 (6,8%) | 122.941 (93,2%) |
| Logit lasso | 26.155 (79,2%) | 6.869 (20,8%) | 11.026 (8,4%) | 120.908 (91,6%) |

Frente al logit, el XGBoost excluye **480 pobres menos** e incluye **2.172 no pobres menos**. XGBoost y random forest son casi idénticos.

### 3.6 Métricas de los 8 modelos

| Modelo | Precisión | Recall | Accuracy | F1 | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CART (persona) | 0,642 | 0,698 | 0,862 | 0,669 | 0,895 |
| Naive Bayes (persona) | 0,630 | 0,729 | 0,860 | 0,676 | 0,903 |
| XGBoost (persona) | 0,656 | 0,763 | 0,872 | 0,706 | 0,929 |
| Logit lasso | 0,703 | 0,792 | 0,892 | 0,745 | 0,946 |
| XGBoost base hogar | 0,707 | 0,807 | 0,894 | 0,753 | 0,951 |
| XGBoost sin Tweedie | 0,739 | 0,812 | 0,905 | 0,774 | 0,958 |
| Random forest | 0,747 | 0,805 | 0,906 | 0,775 | 0,958 |
| **XGBoost final** | **0,751** | 0,807 | **0,908** | **0,778** | **0,959** |

Precisión, recall y accuracy con todos los hogares; F1 como promedio de los 5 folds (`metricas.dat`).

- **Precisión:** de los hogares predichos pobres, cuántos lo son. **Recall:** de los pobres, cuántos se detectan.
- Entre los modelos por hogar el recall es casi igual (cerca de 0,80): la ganancia está en la **precisión** (0,70 → 0,75), es decir, en incluir menos no pobres.
- La accuracy engaña: decir "nadie es pobre" ya da 0,80.

### 3.7 Curvas ROC

Las curvas ROC (`roc_*.dat`) muestran el recall frente a la tasa de falsos positivos para todos los umbrales posibles. El **AUC** es la probabilidad de que un hogar pobre reciba un puntaje mayor que uno no pobre, y no depende del umbral. Los tres mejores casi empatan (0,959, 0,958 y 0,946): **ordenan** a los hogares casi igual de bien, y la diferencia en F1 se concentra en la zona del umbral. La slide muestra la zona de falsos positivos entre 0 y 0,4, donde están los umbrales que se usarían en la práctica.

### 3.8 Equidad: ¿a quién excluye cada modelo?

Brecha de recall, en puntos porcentuales, entre el grupo mejor y el peor detectado:

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

- Los modelos más precisos **reducen** casi todas las brechas; entre los tres mejores son parecidas.
- Todos excluyen más a los pobres con jefe formal o con educación superior y a los hogares pequeños o sin menores: los pobres "que no lo parecen".
- **Sexo del jefe:** la única brecha que crece con los modelos por hogar, a unos 4 puntos, siempre a favor de los hogares con jefa.
- Con una sola regla de decisión no se pueden igualar a la vez la sensibilidad y los falsos positivos entre grupos con tasas de pobreza distintas (Kleinberg, Mullainathan y Raghavan, 2017).

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

Un instrumento de focalización tiene que ser **creíble**: los hogares, los alcaldes y los entes de control tienen que poder entender por qué un hogar entra o sale de un programa. Un XGBoost con 163 variables y 26 modelos anidados predice mejor, pero difícilmente es accesible para quien diseña la política. La evidencia sobre el SISBÉN muestra que la credibilidad y la manipulación del instrumento son problemas reales (Camacho y Conover, 2011), y para decisiones de alto impacto la literatura sugiere preferir modelos interpretables cuando la pérdida de precisión es pequeña (Rudin, 2019).

**Recomendación de política**

1. **Instrumento oficial: un logit parsimonioso**, de 15 a 40 variables del módulo laboral, la vivienda y la composición del hogar. Se puede publicar, discutir y auditar.
2. **XGBoost como auditor:** señala los hogares que el logit excluiría y que probablemente son pobres (jefe formal o educado, hogares pequeños), para verificarlos en campo antes de excluirlos.
3. **El umbral es una decisión de política:** debe fijarse según el costo social de excluir a un pobre frente al de incluir a un no pobre, no según el F1.

**Para la discusión**

- ¿Cuántos puntos de F1 vale la transparencia?
- ¿Quién responde por el error de una caja negra en un programa social?

---

## 5. Cómo replicar

```sh
Rscript script/01_data_cleaning.r                      # bases limpias
Rscript script/01b_bases_hogar.r                      # bases por hogar
Rscript script/05b_xgboost_hogar.r                     # XGBoost final, random forest y permutaciones
Rscript output/pruebas/logit/06_logit_interacciones_main.r   # logit lasso (copia del 06 de main)
Rscript script/06_comparacion_modelos.r                # modelos_f1.tex y tablas de equidad
Rscript script/11_figuras_presentacion.r               # 3D, métricas, ROC y F1 vs. k (lee ps2_predicciones_fuera_de_muestra/)
cd presentacion && latexmk -pdf compare_equipo_04.tex  # presentación
```

---

## Referencias

- Angulo, R., Díaz, Y. y Pardo, R. (2016). The Colombian Multidimensional Poverty Index: Measuring poverty in a public policy context. *Social Indicators Research, 127*(1), 1–38.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
- Camacho, A. y Conover, E. (2011). Manipulation of social program eligibility. *American Economic Journal: Economic Policy, 3*(2), 41–65.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD* (pp. 785–794).
- DANE (2019). *Pobreza monetaria y multidimensional en Colombia 2018* [Boletín técnico].
- Grosh, M. y Baker, J. (1995). *Proxy means tests for targeting social programs* (LSMS Working Paper 118). Banco Mundial.
- Kleinberg, J., Ludwig, J., Mullainathan, S. y Obermeyer, Z. (2015). Prediction policy problems. *American Economic Review, 105*(5), 491–495.
- Kleinberg, J., Mullainathan, S. y Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. En *8th Innovations in Theoretical Computer Science Conference (ITCS)*.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence, 1*(5), 206–215.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B, 58*(1), 267–288.
