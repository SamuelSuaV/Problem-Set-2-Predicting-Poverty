# ¿Quién es pobre en Colombia? Predicción de la pobreza monetaria con la GEIH 2018

**Equipo 04:** Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
Big Data y Machine Learning Aplicado a la Economía, Universidad de los Andes.

Este documento acompaña la presentación [`best_equipo_04.tex`](best_equipo_04.tex) y tiene la narrativa y el sustento técnico completos. Los números salen de [`latex/resultados.tex`](../latex/resultados.tex), de las tablas de [`output/tables/`](../output/tables/) y de [`script/11_figuras_presentacion.r`](../script/11_figuras_presentacion.r), que arma las figuras nuevas y los datos de las gráficas (`output/tables/presentacion/*.dat`).

---

## 1. Overview

### 1.1 Qué alimenta el modelo

El modelo final es una cadena de seis componentes:

1. **Base por hogar.** 89 agregados de los miembros: trabajo (ocupados, formales, horas, posición), demografía (tamaño, menores, dependencia), vivienda (tenencia, cuartos, arriendo por persona sobre la línea), ingresos no laborales (si algún miembro recibe pensión, arriendos, remesas, ayudas) y ubicación y línea.
2. **Modelos por persona,** estimados con validación cruzada anidada (sección 3.2):
   - ingreso laboral y no laboral de cada miembro, con pérdida **Tweedie**;
   - probabilidad de ser pobre;
   - probabilidad de que el ingreso del hogar quede por debajo de 0,8 y de 1,25 líneas;
   - tasa de pobreza del oficio.

   Sus predicciones se resumen por hogar: suma, mínimo, máximo, promedio y el valor del jefe.
3. **Ingreso predicho + arriendo imputado, por persona y sobre la línea.** Es la fórmula con la que el DANE define la pobreza, aplicada al ingreso predicho.
4. **Clasificador XGBoost por hogar** con las 163 variables anteriores (categóricas nativas).
5. **Regresión XGBoost** de $z = \log(\text{ingreso}/\text{línea})$, con más peso a los hogares cerca de la línea.
6. **Combinación:** $p = 0{,}7\,p_{\text{clasif}} + 0{,}3\,p_{\text{regr}}$. Un hogar es pobre si $p > 0{,}39$.

| Métrica | Valor |
| :--- | :---: |
| **F1 de validación cruzada** (5 folds por hogar, datos públicos de train) | **0,7775** |
| Kaggle, puntaje público (20% de test) | 0,77 |
| Kaggle, puntaje privado (80% de test) | **0,78** |

### 1.2 El mensaje

Con la IA, combinar variables está al alcance de cualquier equipo: interacciones, logaritmos, índices y agregados se generan en minutos. Por eso todos los grupos llegaban a F1 parecidos con los mismos datos públicos, y en nuestro caso las rondas de "más combinaciones" no sumaron nada (sección 3.5). Lo que nos diferenció fue:

- la **validación**: folds por hogar, validación cruzada anidada sin fuga, la misma partición para comparar todo y otras particiones para confirmar;
- el **uso de las herramientas**: categóricas nativas, pérdida Tweedie, stacking, parada temprana y pesos cerca de la línea;
- la **intuición económica**, que no es escoger variables sino decidir **qué objeto económico predecir y cómo se mide** (sección 2.2).

---

## 2. Datos

### 2.1 Muestra y variable objetivo

Usamos la muestra de entrenamiento de la GEIH 2018 (DANE): **543.103 personas en 164.958 hogares**. Un hogar es pobre cuando su ingreso per cápita de la unidad de gasto, **con arriendo imputado** (`Ingpcug`), es menor que la línea de pobreza de su dominio. Verificamos que esa regla se cumple en el 100% de los hogares. Son pobres el **20,0% de los hogares** y el 25,1% de las personas. La mediana de la línea de pobreza es $280.072.

Ningún monto de ingreso está en test. El modelo tiene que **reconstruir el ingreso** a partir de las preguntas que sí están: empleo, vivienda, demografía y fuentes de ingreso.

### 2.2 Intuición económica: qué predecir, no qué variables meter

La intuición económica no está en la lista de variables (cualquier equipo puede meterlas todas), sino en entender cómo se construye la pobreza monetaria y modelar eso:

| Hecho económico | Decisión de modelación | Efecto |
| :--- | :--- | :--- |
| La pobreza se mide en la **unidad de gasto**, no en la persona. | Una fila por hogar: los ingresos de los miembros se suman y se reparten. | F1 0,706 → 0,753, el mayor salto del proyecto. |
| La regla es una **fórmula**: (ingreso + arriendo imputado) / personas < línea (MESEP, DANE y DNP, 2012). | Predecir el **ingreso** de cada persona y reconstruir la regla, en vez de predecir solo la etiqueta. El arriendo imputado se reconstruyó con tenencia, arriendo estimado y cuota (coincide en el 98,4% de los hogares). | El ingreso predicho resume casi todo: sin él, el F1 cae a 0,35. |
| El ingreso tiene muchos ceros y una cola larga, y para sumar a los miembros se necesita la **media**, no la mediana. | Pérdida Tweedie para el ingreso por persona. La cuadrática sobre el logaritmo predice la mediana y subestima el ingreso del hogar. | $R^2$ del ingreso del hogar 0,63 → 0,66; F1 +0,004. |
| Hay **ingresos sin empleo**: pensiones, arriendos, trabajos ocasionales de inactivos y desocupados. | Usar las preguntas del cuestionario que identifican quién recibe ingresos sin estar ocupado. | $R^2$ del ingreso por persona 0,94 → 0,98; F1 +0,009. |
| La clasificación solo es difícil **cerca de la línea**. | Más peso a esos hogares en la regresión del ingreso. | Entra en la combinación 70/30 (sección 3.4). |

### 2.3 Los predictores que importan

Las figuras nuevas ([`pres_predictores_cat.png`](../output/figures/pres_predictores_cat.png) y [`pres_predictores_cont.png`](../output/figures/pres_predictores_cont.png), sección 2 de `11_figuras_presentacion.r`) muestran el % de hogares pobres según las preguntas con mayor importancia en el modelo (sección 3.5), todas a nivel hogar.

- **Posición ocupacional del jefe (P6430):** va de 1% de hogares pobres (empleado del gobierno) a 27% (cuenta propia) y 34% (familiar sin pago). Es la pregunta individual más importante del modelo.
- **Algún miembro recibe arriendos o pensiones (P7495):** 4% de pobres si sí, frente a 24% si no.
- **Jefe cotiza a pensión:** 6% frente a 26%.
- **Tenencia de la vivienda:** de 10% (propia, pagando) a 48% (posesión sin título).
- **Ingresos por trabajo de inactivos (P7472):** casi no cambia la tasa de pobreza (21% frente a 20%). Su importancia en el modelo no viene de la respuesta, sino de que la pregunta solo se les hace a los inactivos: ayuda a identificar quién es inactivo y qué ingresos tiene.
- **Continuas:** la pobreza cae rápido con el arriendo por persona sobre la línea, los cuartos por persona, las horas trabajadas por persona y la proporción de ocupados, y después se aplana. Las horas y los ocupados no son monótonos: los hogares con cero horas (muchos pensionados) son menos pobres que los que trabajan pocas horas. Son relaciones no lineales y con umbrales, que es la forma que aprovechan los árboles.

### 2.4 El pivote: de la persona al hogar

Cada nivel de la gráfica del pivote es el mejor modelo hasta ese momento, con los mismos 5 folds por hogar:

| Nivel | Qué se agregó | F1 |
| :--- | :--- | :---: |
| **Por persona, promediado** | XGBoost con 24 variables de cada persona. La probabilidad del hogar es el promedio de los log odds de sus miembros. Es el techo de la fase 1. | 0,7055 |
| **Por hogar: agregados de los miembros** | Una fila por hogar con 94 variables: cuántos ocupados y formales hay, horas por persona, educación de los adultos, dependencia, hacinamiento, arriendo por persona sobre la línea y los datos del jefe. Con las mismas herramientas, el F1 sube 0,05. | 0,7534 |
| **+ ingreso predicho y stacking** | Un modelo por persona predice el ingreso de cada miembro y se suma en el hogar; la probabilidad de ser pobre por persona se resume (mínimo, máximo, promedio, jefe). Los dos con validación cruzada anidada. Además, `eta` 0,02. | 0,7598 |
| **+ categóricas nativas** | Dominio, departamento, oficio y posición ocupacional entran como categorías, no como códigos ordenados. Ayuda sobre todo a los modelos por persona (el oficio tiene 84 niveles). | 0,7612 |
| **+ ingresos sin estar ocupado** | Preguntas que habíamos dejado por fuera y añadimos al revisar el cuestionario: pagos en especie, posición en el segundo y en el último trabajo, búsqueda de más horas o de otro trabajo, e ingresos de desocupados e inactivos. Entran al modelo del ingreso por persona ($R^2$ 0,94 → 0,98). Incluye usar el ingreso afinado junto al original (0,7622). | 0,7714 |
| **+ ingreso afinado y arriendo imputado** | El ingreso predicho con las preguntas nuevas, afinado; profundidad 8, `min_child_weight` 50 y `colsample` 0,5 (0,7733); y el ingreso predicho más el arriendo imputado, por persona y sobre la línea (la fórmula del DANE). | 0,7741 |
| **+ ingreso con Tweedie** | El ingreso por persona se estima con pérdida Tweedie, laboral y no laboral por separado. | 0,7772 |
| **+ regresión del ingreso (70/30)** | Combinación con una regresión del ingreso del hogar (sección 3.4). | **0,7775** |

Después del pivote, cada escalón agrega **información sobre el ingreso del hogar**, no combinaciones de variables.

---

## 3. Mejor modelo

### 3.1 Estructura (tres niveles)

Diagrama: [`output/figures/estructura_modelo_slide.png`](../output/figures/estructura_modelo_slide.png) (fuente: [`latex/estructura_modelo_slide.tex`](../latex/estructura_modelo_slide.tex)).

1. **Modelos auxiliares por persona** (XGBoost): el ingreso de cada miembro (cinco versiones, la última con Tweedie), la probabilidad de ser pobre, la probabilidad de que el ingreso del hogar quede por debajo de 0,8 y 1,25 líneas, y la tasa de pobreza del oficio.
2. **Resúmenes por hogar:** suma, máximo, mínimo, el valor del jefe y el ingreso por ocupado.
3. **Modelos por hogar:** el clasificador y la regresión, combinados 70/30.

### 3.2 Validación: cómo se mide sin hacer trampa

*Esta parte ya no tiene slide propia (se da por entendida en la presentación); queda aquí como referencia por si preguntan.*

- **5 folds por hogar.** Todos los miembros de un hogar quedan en el mismo fold. Se entrena con 4 folds y se mide con el quinto, cinco veces. Todos los modelos del equipo usan los mismos folds, así que cualquier mejora se compara contra la misma partición.
- **Validación cruzada anidada.** El ingreso predicho, la probabilidad por persona y la tasa del oficio usan la etiqueta. Si se estimaran con todos los datos, el modelo vería la respuesta del hogar que se evalúa. Por eso, dentro de los 4 folds de entrenamiento se repite la partición: cada parte se predice con modelos entrenados en las otras tres, y el fold evaluado se predice con modelos entrenados en los cuatro. El fold de validación nunca entra en el entrenamiento de ningún nivel. Son 26 modelos por variable.
- **Otras particiones.** El modelo final se volvió a estimar con otras dos particiones en folds (recalculando toda la cadena anidada): 0,7784 y 0,7767. La desviación estándar del F1 entre particiones es de unos 0,0003, así que las diferencias de menos de 0,001 entre modelos son ruido.
- **Sin fuga.** Las preguntas añadidas tienen la misma frecuencia y distribución en train y en test.

**Parada temprana.** XGBoost agrega árboles uno a uno, y cada árbol corrige el error de los anteriores. Al principio la log-loss baja en entrenamiento y en validación; después solo baja en entrenamiento (el modelo empieza a memorizar). El entrenamiento para cuando la log-loss del fold de validación lleva **100 árboles seguidos sin mejorar**, y se usa el árbol con la mejor log-loss: entre 613 y 869 según el fold. *Nota de transparencia:* el número de árboles se decide con el mismo fold que se evalúa. Eso introduce un optimismo pequeño en el F1 de validación cruzada. No cambia las conclusiones: el F1 se mantiene con otras particiones, y el puntaje en test de Kaggle (0,78) es consistente con el de validación cruzada.

**Umbral.** El modelo entrega una probabilidad, y el umbral la convierte en una decisión. Subirlo aumenta la precisión y reduce el recall. Para cada umbral de una grilla de 0,10 a 0,85 se calcula el F1 de cada fold, y se escoge el que maximiza el promedio: **0,39**. Con 0,5 el F1 sería 0,765 (precisión 0,80, recall 0,73); con 0,39 es 0,7775 (precisión 0,75, recall 0,81). No es 0,5 porque solo el 20% de los hogares es pobre y el F1 premia encontrarlos (datos en `output/tables/presentacion/umbral_final.dat`).

### 3.3 Hiperparámetros: qué controla cada uno

| Parámetro | Valor | Qué controla | Intuición |
| :--- | :---: | :--- | :--- |
| `eta` | 0,02 | Tamaño del paso | Cada árbol corrige solo el 2% del error que queda. Pasos cortos necesitan más árboles, pero sobreajustan menos. |
| `max_depth` | 8 | Profundidad de cada árbol | Hasta 8 preguntas por rama: interacciones de hasta 8 variables (por ejemplo, ingreso predicho × tenencia × tamaño del hogar). |
| `min_child_weight` | 50 | Evidencia mínima por hoja | Una hoja necesita suficiente peso (en la log-loss, unos 50 hogares con duda): no se crean reglas para casos aislados. |
| `colsample_bytree` | 0,5 | Variables por árbol | Cada árbol ve la mitad de las variables: obliga a usar señales distintas y descorrelaciona los árboles. |
| `subsample` | 0,8 | Hogares por árbol | Cada árbol ve el 80% de los hogares: más azar, menos varianza. |
| Árboles | 613–869 | Número de árboles | Los fija la parada temprana. |
| `gamma` / `lambda` / `alpha` | 0 / 1 / 0 | Penalizaciones (poda, L2, L1) | Fijas desde iteraciones anteriores, para aislar el efecto de las variables y pérdidas nuevas. |

La grilla final (profundidad 4, 6 u 8; `min_child_weight` 10 o 50; `colsample` 0,5 u 0,8) da una superficie plana: el F1 va de 0,7723 a 0,7733. **Los hiperparámetros no fueron la fuente de la ganancia.** La regresión por hogar usa `eta` 0,05 y profundidad 6.

### 3.4 Funciones de pérdida y pesos de la combinación

**Pérdida Tweedie en el ingreso por persona** (`reg:tweedie`, potencia $\rho = 1{,}2$; un modelo para el ingreso laboral y otro para el no laboral):

$$L(y,\mu) = -\frac{y\,\mu^{1-\rho}}{1-\rho} + \frac{\mu^{2-\rho}}{2-\rho}, \qquad 1 < \rho < 2$$

Con error cuadrático sobre $\log y$, la predicción de cada persona se acerca a la **mediana**, y al sumar a los miembros el ingreso del hogar queda subestimado. La Tweedie (entre la Poisson y la Gamma) admite masa en cero y una cola larga, y predice la **media** en pesos, que es la que importa al sumar. El $R^2$ del ingreso del hogar pasa de 0,63 a 0,66 y el F1 de 0,7733 a 0,7772.

**Cómo se calculan los pesos 70/30.** La combinación tiene cuatro pasos (secciones 5.19, 5.20 y 6 de [`05b_xgboost_hogar.r`](../script/05b_xgboost_hogar.r)):

1. **La regresión.** Un XGBoost con error cuadrático predice $z_i = \log(\text{ingreso}_i/\text{línea}_i)$ con las mismas variables del clasificador. Un hogar es pobre si $z < 0$. Cada hogar pesa en el entrenamiento

   $$w_i = 1 + a\,e^{-|z_i|/b}, \qquad a = 3,\ b = 0{,}25.$$

   Un hogar en la línea pesa 4; uno al doble de la línea ($z \approx 0{,}69$) pesa 1,2; uno muy lejos, 1. Se probaron $a \in \{1, 3\}$ y $b \in \{0{,}25;\ 0{,}5\}$ y se escogió la que mejor combina. La razón: en una regresión sin pesos, un error de 50% en un hogar rico cuenta igual que uno en un hogar junto a la línea, aunque solo el segundo cambia la clasificación.
2. **De predicción a probabilidad.** La predicción fuera de muestra $\hat z_i$ se convierte en probabilidad con un logit de `pobre` sobre $\hat z$, estimado con las predicciones fuera de muestra: $p_{\text{regr},i} = \Lambda(-0{,}159 - 5{,}689\,\hat z_i)$. Así queda en la misma escala que la probabilidad del clasificador. Sola, la regresión tiene un F1 de 0,769.
3. **El promedio.** Para cada peso $w \in \{0;\ 0{,}1;\ \ldots;\ 1\}$ se calcula $p_i = w\,p_{\text{clasif},i} + (1-w)\,p_{\text{regr},i}$.
4. **La elección.** Para cada $w$ se busca el umbral que maximiza el F1 promedio de los 5 folds, y se escoge el par $(w, \text{umbral})$ con mayor F1: $w = 0{,}7$ y umbral 0,39, F1 0,7775 (el clasificador solo: 0,7772).

La regresión aporta poco (+0,0003) porque se equivoca en casi los mismos hogares que el clasificador, pero ve el problema de otra forma (cuánto falta para la línea, no solo de qué lado está). Los mismos pesos aplicados **al clasificador** no ayudaron (0,7712 frente a 0,7715).

**Stacking.** El ingreso predicho y la probabilidad por persona entran como variables del clasificador. En cada ronda se probó una nueva versión del ingreso predicho como variable adicional; XGBoost escogía las versiones útiles y la validación cruzada indicaba cuándo una dejaba de aportar.

### 3.5 Importancia de las variables y qué sumó

**Permutación por grupo.** En cada fold se reestiman el clasificador y la regresión con los otros cuatro, y se desordenan al azar los valores de un grupo de variables entre los hogares del fold de validación. La importancia es la caída del F1 con el umbral de 0,39 (5 folds × 3 permutaciones). La slide muestra el F1 después de permutar cada grupo, partiendo de 0,7775:

| Grupo | Variables | Caída del F1 | F1 después |
| :--- | :---: | :---: | :---: |
| Ingreso predicho por persona (todas las versiones) | 26 | **0,4232** | 0,354 |
| Probabilidad de ingreso menor a 0,8 y 1,25 líneas | 8 | 0,0607 | 0,717 |
| Probabilidad de ser pobre por persona (stacking) | 8 | 0,0437 | 0,734 |
| Base por hogar (sin ubicación) | 89 | 0,0097 | 0,768 |
| Otras construidas | 24 | 0,0051 | 0,772 |
| Ubicación (dominio, departamento, zona, línea) | 4 | 0,0022 | 0,775 |
| Tasa de pobreza del oficio | 3 | 0,0005 | 0,777 |

La variable individual más importante es `ing_arr_pc_lp_w`, la fórmula del DANE con el ingreso predicho (−0,043). De las 163 variables, 21 tienen una importancia nula o negativa.

**Por concepto, de extremo a extremo.** Se permutan las variables del concepto en las personas y en el hogar, y se recalcula toda la cadena. La slide separa la importancia **directa** (solo las variables del hogar, a la izquierda) de la que pasa **por los modelos por persona** (a la derecha):

| Concepto | Total | Directa | Vía modelos por persona |
| :--- | :---: | :---: | :---: |
| Formalidad, posición y oficio | **0,226** | 0,002 | 0,226 |
| Ingresos no laborales | 0,143 | 0,001 | 0,137 |
| Vivienda y arriendo | 0,118 | **0,037** | 0,045 |
| Ocupación y horas de trabajo | 0,117 | 0,005 | 0,122 |
| Búsqueda de trabajo e ingresos de desocupados | 0,070 | 0,000 | 0,070 |
| Edad, sexo y tamaño del hogar | 0,050 | **0,035** | 0,023 |
| Ubicación y línea de pobreza | 0,024 | 0,009 | 0,017 |
| Educación | 0,021 | 0,006 | 0,024 |
| Primas, subsidios y pagos en especie | 0,012 | 0,005 | 0,008 |
| Salud | 0,009 | 0,001 | 0,010 |

El trabajo entra casi todo a través del ingreso predicho. Solo la vivienda y el tamaño del hogar entran directo, porque la fórmula suma el arriendo imputado y divide entre personas. Las importancias no se suman entre conceptos: están correlacionados.

**Por qué importan estos conceptos: la teoría detrás.** La importancia por concepto coincide con lo que predice la literatura sobre ingreso y pobreza, y ayuda a entender por qué el modelo detecta mejor a unos hogares que a otros:

| Concepto (caída del F1) | Teoría | Qué implica para el modelo |
| :--- | :--- | :--- |
| Formalidad, posición y oficio (−0,226) | **Lewis (1954):** en una economía dual, el sector de subsistencia (informal) absorbe un exceso de trabajadores en tareas de baja productividad, cuyos ingresos se quedan en el nivel de subsistencia. | La posición ocupacional ubica al hogar en uno de los dos sectores. Por eso, al quitar este concepto el modelo pierde la señal que más separa a pobres de no pobres. |
| Educación (−0,021) | **Becker (1964):** el capital humano es un activo que aumenta la productividad y abre trabajos especializados, de mayor valor agregado y más formales. | La educación determina el ingreso de largo plazo y el grado de formalización **a través** del empleo. Su información ya está en la posición, el oficio y la formalidad, y por eso pesa poco sola. |
| Ingresos no laborales (−0,143) | **Deaton (1997):** los hogares vulnerables no consumen ni sobreviven con un salario mensual estable; su bienestar depende de amortiguar choques con ingresos no laborales (remesas, transferencias, pensiones, ayudas). | Por eso esas preguntas entran a los modelos por persona: identifican quién tiene ingresos sin estar ocupado. |
| Ingresos irregulares (errores cerca de la línea) | **Collins, Morduch, Rutherford y Ruthven (2009):** los hogares pobres manejan flujos de ingreso irregulares con crédito informal, ahorro y deudas que están pagando constantemente. | Un mes de encuesta no refleja el ingreso permanente del hogar: cerca de la línea, buena parte del error es irreducible. |

**Quién se detecta mejor.** Los pobres informales, con poca educación y en hogares grandes encajan en el sector de subsistencia de Lewis, y el modelo los detecta bien (recall de 80–89%). Los pobres con jefe formal o con educación superior "no lo parecen" según Becker, y el modelo los excluye más (recall de 65%; sección 4.2).

**Qué sumó y qué restó.** Cada barra de la slide compara una prueba con el mejor modelo de ese momento (cuadros de `resultados.tex`):

| Prueba | Cambio en el F1 |
| :--- | :---: |
| Ingresos sin estar ocupado (preguntas nuevas en el ingreso predicho) | +0,0092 |
| Ingreso predicho y stacking (H0 → H7) | +0,0061 |
| Pérdida Tweedie | +0,0031 |
| Ingreso afinado con las preguntas nuevas | +0,0016 |
| Categóricas nativas | +0,0014 |
| Arriendo imputado (regla del DANE) | +0,0008 |
| Regresión del ingreso (70/30) | +0,0003 |
| Más peso cerca de la línea en el clasificador | −0,0003 |
| Modelo especialista para los hogares dudosos | −0,0007 |
| Índices de variables débiles | −0,0007 |
| Predicción de la regresión como variable | −0,0008 |
| Logaritmos y cuadrados | −0,0010 |
| Base aumentada (pobres replicados) | −0,0062 |

Sumó todo lo que mejora la **medición del ingreso** del hogar. Restó, o no cambió nada, transformar las mismas variables, rebalancear la muestra y agregar modelos cerca de la línea.

Fuentes: [`importancia_grupos.tex`](../output/tables/importancia_grupos.tex), [`importancia_conceptos.tex`](../output/tables/importancia_conceptos.tex), [`importancia_variables.tex`](../output/tables/importancia_variables.tex) (secciones 14 y 15 de `05b_xgboost_hogar.r`).

### 3.6 Diagnósticos: dónde se equivoca

Tasa de error del modelo final según el ingreso del hogar sobre la línea (sección 6 de `11_figuras_presentacion.r`, `errores_linea.dat`):

| Ingreso / línea | % de los hogares | Error | Tipo |
| :---: | :---: | :---: | :--- |
| 0–0,5 | 5,6 | 6–8% | pobres excluidos |
| 0,5–0,75 | 6,5 | 14% | pobres excluidos |
| 0,75–0,9 | 4,7 | 26% | pobres excluidos |
| 0,9–1 | 3,3 | **41%** | pobres excluidos |
| 1–1,1 | 3,5 | **43%** | no pobres incluidos |
| 1,1–1,25 | 4,9 | 28% | no pobres incluidos |
| 1,25–1,5 | 7,9 | 15% | no pobres incluidos |
| 1,5–2 | 13,5 | 6% | no pobres incluidos |
| 2 o más | 50,1 | 0,4–1,9% | no pobres incluidos |

- A cada lado de la línea el modelo falla en cerca de 4 de cada 10 hogares; al doble de la línea, casi nunca.
- Ni dar más peso a esos hogares ni un modelo "especialista" para los dudosos ayudó. Con estas preguntas, ese error es prácticamente irreducible.

---

## 4. Implicaciones de política pública

### 4.1 Del F1 a la focalización: leakage vs. undercoverage

En focalización, un **falso negativo** es un hogar pobre excluido (*undercoverage*, error de exclusión) y un **falso positivo** es un hogar no pobre incluido (*leakage*, error de inclusión). Con el umbral de 0,39 y las predicciones fuera de muestra de train (matriz de confusión en `metricas.dat`):

| | Predicho pobre | Predicho no pobre |
| :--- | :---: | :---: |
| **Pobre** (33.024) | 26.635 (80,7%) | 6.389 (19,3%) |
| **No pobre** (131.934) | 8.854 (6,7%) | 123.080 (93,3%) |

- **Cobertura (recall):** 80,7% de los hogares pobres se detecta; el *undercoverage* es de 19,3%.
- **Precisión:** 75,1%. Uno de cada cuatro beneficiarios no es pobre.

El F1 pesa igual los dos errores, y un programa social no tiene por qué hacerlo. **El umbral es una decisión de política:** bajarlo aumenta la cobertura a costa de más *leakage* (Kleinberg et al., 2015; Grosh y Baker, 1995).

### 4.2 Matices sociodemográficos

| Atributo | Grupo | % pobres | Sensibilidad | Falsos positivos | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Sexo del jefe | Hombre | 18,3 | 78,9 | 6,2 | 0,764 |
| | Mujer | 22,4 | 82,7 | 7,4 | 0,793 |
| Empleo del jefe | Formal (cotiza a pensión) | 5,6 | **65,1** | 1,7 | 0,674 |
| | Informal | 26,1 | 80,2 | **10,1** | 0,769 |
| | No ocupado | 24,7 | 84,8 | 7,8 | 0,813 |
| Zona | Cabecera | 19,0 | 80,8 | 6,2 | 0,780 |
| | Resto (rural) | 30,0 | 80,0 | **12,6** | 0,764 |
| Educación del jefe | Primaria o menos | 28,8 | 83,4 | 10,8 | 0,794 |
| | Superior | 7,3 | **65,4** | 2,3 | 0,674 |
| Tamaño del hogar | 1 o 2 personas | 10,7 | **66,5** | 3,8 | 0,672 |
| | 5 o más | 35,8 | 88,8 | **12,1** | 0,844 |
| Menores | Sin menores de 18 | 9,3 | 65,2 | 3,1 | 0,667 |
| | Con menores de 18 | 29,6 | 85,1 | 10,9 | 0,807 |

Fuente: [`equidad_mejor.tex`](../output/tables/equidad_mejor.tex).

- En los grupos con más precariedad (zona rural, jefe informal, hogares grandes) el modelo **sobreestima** la pobreza: tiene más errores de inclusión.
- Los pobres "que no lo parecen" (jefe formal o con educación superior, hogares pequeños o sin menores) quedan **excluidos** con más frecuencia: la sensibilidad es de 65–67%.
- **Jefatura femenina:** el modelo detecta un poco mejor a los hogares pobres con jefa (+3,8 puntos), con algo más de falsos positivos (+1,2 puntos).

Este patrón no viene de una regla discriminatoria, sino de **tasas base distintas**. Con una sola regla de decisión no se pueden igualar a la vez la sensibilidad y la tasa de falsos positivos entre grupos (Kleinberg, Mullainathan y Raghavan, 2017).

### 4.3 Pobreza monetaria vs. multidimensional

El modelo mide **estrictamente pobreza monetaria**. Educación, salud y subsidios aportan poco (caída del F1 de 0,021, 0,009 y 0,012) una vez se conoce el trabajo del hogar.

1. **Las dimensiones miden cosas distintas.** El IPM colombiano (Angulo et al., 2016) y la pobreza monetaria (DANE, 2019) se traslapan solo en parte.
2. **Riesgo de focalización.** Focalizar transferencias monetarias con indicadores de salud, educación o subsidios genera errores sistemáticos, y las variables observables y conocidas son manipulables (Camacho y Conover, 2011).

### 4.4 Viabilidad operativa: ¿una encuesta corta y barata?

**A favor**
- No hay que preguntar **montos de ingreso**, la parte más cara y subdeclarada de una encuesta. Bastan preguntas sí/no sobre fuentes de ingreso, el módulo laboral y la vivienda.
- La formalidad y el oficio se pueden verificar con registros administrativos, como la PILA.

**En contra**
- El modelo necesita el módulo laboral de **cada miembro**, unas 60 preguntas por persona: un costo parecido a una visita del SISBÉN IV (CONPES 3877, 2016).
- Los ingresos no laborales y la búsqueda de trabajo son difíciles de verificar en territorio.
- Hay que reentrenarlo cuando cambien la línea de pobreza y el mercado laboral (datos de 2018).

---

## 5. Cierre: qué nos hizo ganar

| Pilar | Qué hicimos |
| :--- | :--- |
| **Validación** | Folds por hogar, validación cruzada anidada sin fuga, la misma partición para comparar todo y otras particiones para confirmar. Decidimos con la validación cruzada. |
| **Uso de las herramientas** | Categóricas nativas, pérdida Tweedie, stacking con modelos por persona, parada temprana y pesos cerca de la línea. |
| **Intuición económica** | Modelar el hogar, predecir el ingreso y reconstruir la regla del DANE con el arriendo imputado, y buscar en el cuestionario los ingresos sin empleo. |

> Con la IA, cualquiera combina variables. La ventaja está en saber qué predecir y en cómo comprobarlo.

---

## 6. Cómo replicar

Desde la raíz del repositorio:

```sh
Rscript script/01_data_cleaning.r                      # bases limpias por persona
Rscript script/01b_bases_hogar.r                      # bases por hogar (data/train_hogar.rds, data/test_hogar.rds)
Rscript script/05b_xgboost_hogar.r                     # modelo final, permutaciones (secciones 14-16) y submission
Rscript script/06_comparacion_modelos.r                # tablas de comparación y equidad
Rscript script/11_figuras_presentacion.r               # figuras y datos de las slides (necesita ps2_predicciones_fuera_de_muestra/)
cd presentacion && latexmk -pdf best_equipo_04.tex     # presentación
```

Submission final: `output/submissions/XGBH_nativo_comb_..._peso_w07_threshold_039.csv`.

---

## Referencias

- Angulo, R., Díaz, Y. y Pardo, R. (2016). The Colombian Multidimensional Poverty Index: Measuring poverty in a public policy context. *Social Indicators Research, 127*(1), 1–38.
- Becker, G. S. (1964). *Human capital: A theoretical and empirical analysis, with special reference to education*. National Bureau of Economic Research y Columbia University Press.
- Camacho, A. y Conover, E. (2011). Manipulation of social program eligibility. *American Economic Journal: Economic Policy, 3*(2), 41–65.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD* (pp. 785–794).
- Collins, D., Morduch, J., Rutherford, S. y Ruthven, O. (2009). *Portfolios of the poor: How the world's poor live on $2 a day*. Princeton University Press.
- CONPES 3877 (2016). *Declaración de importancia estratégica del Sistema de Identificación de Potenciales Beneficiarios (SISBÉN IV)*. DNP.
- DANE (2019). *Pobreza monetaria y multidimensional en Colombia 2018* [Boletín técnico].
- DANE y DNP (2012). *Misión para el Empalme de las Series de Empleo, Pobreza y Desigualdad (MESEP): metodología de pobreza monetaria*.
- Deaton, A. (1997). *The analysis of household surveys: A microeconometric approach to development policy*. Johns Hopkins University Press y Banco Mundial.
- Fernández, C. y Villar, L. (2017). The impact of lowering the payroll tax on informality in Colombia. *Economía, 18*(1), 125–155.
- Grosh, M. y Baker, J. (1995). *Proxy means tests for targeting social programs* (LSMS Working Paper 118). Banco Mundial.
- Kleinberg, J., Ludwig, J., Mullainathan, S. y Obermeyer, Z. (2015). Prediction policy problems. *American Economic Review, 105*(5), 491–495.
- Kleinberg, J., Mullainathan, S. y Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. En *8th Innovations in Theoretical Computer Science Conference (ITCS)*.
- Lewis, W. A. (1954). Economic development with unlimited supplies of labour. *The Manchester School, 22*(2), 139–191.
- Ravallion, M. (2016). *The economics of poverty: History, measurement, and policy*. Oxford University Press.
