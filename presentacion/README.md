# ¿Quién es pobre en Colombia? Predicción de la pobreza monetaria con la GEIH 2018

**Equipo 04:** Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
Big Data y Machine Learning Aplicado a la Economía, Universidad de los Andes.

Este documento acompaña la presentación [`best_equipo_04.tex`](best_equipo_04.tex) y tiene la narrativa y el sustento técnico completos. Los números salen de [`latex/resultados.tex`](../latex/resultados.tex) y de las tablas de [`02_outputs/tables/`](../02_outputs/tables/).

---

## 1. Overview

**Modelo ganador:** un clasificador **XGBoost por hogar** (163 variables, categóricas nativas) alimentado por modelos auxiliares por persona: el ingreso de cada miembro, estimado con pérdida **Tweedie**, y su probabilidad de ser pobre. El clasificador se combina 70/30 con una regresión XGBoost del ingreso del hogar, y un hogar se clasifica como pobre si la probabilidad combinada supera **0,39**.

| Métrica | Valor |
| :--- | :---: |
| F1 de validación cruzada (5 folds por hogar) | **0,7775** |
| F1 con umbral honesto (escogido con los otros 4 folds) | 0,7775 |
| F1 con otras dos particiones en folds | 0,7784 / 0,7767 |
| **Kaggle (public leaderboard)** | **0,7775** |

**Takeaways**

1. **Con la IA, combinar variables ya no diferencia a nadie.** Cualquier equipo puede generar interacciones, logaritmos, índices y agregados en minutos, y por eso todos los grupos llegaban a F1 parecidos con los mismos datos públicos. En nuestro caso, las rondas de "más combinaciones" no sumaron nada (sección 3.2).
2. **Lo que nos hizo ganar fueron tres cosas:**
   - una **validación concienzuda** (CV anidada sin fuga, folds por hogar, F1 honesto, ruido medido);
   - un **uso inteligente de las herramientas** (categóricas nativas, pérdida Tweedie, stacking, parada temprana);
   - **intuición económica** (la pobreza es del hogar, el arriendo imputado reproduce la regla del DANE y había preguntas del cuestionario que nadie usaba).
3. La pobreza es **del hogar** y se explica sobre todo por el **trabajo**: formalidad, posición ocupacional, horas e ingresos no laborales.
4. Educación, salud y subsidios casi no predicen el ingreso una vez se conoce el trabajo. Medir pobreza monetaria **no es** medir pobreza multidimensional.

---

## 2. Datos

### 2.1 Muestra y variable objetivo

Usamos la muestra de entrenamiento de la GEIH 2018 (DANE): **543.103 personas en 164.958 hogares**. Un hogar es pobre cuando su ingreso per cápita de la unidad de gasto, **con arriendo imputado** (`Ingpcug`), es menor que la línea de pobreza de su dominio. Verificamos que esa regla se cumple en el 100% de los hogares. Son pobres el **20,0% de los hogares** y el 25,1% de las personas. La mediana de la línea de pobreza es $280.072.

Ningún monto de ingreso está en test. El modelo tiene que **reconstruir el ingreso** a partir de las preguntas que sí están: empleo, vivienda, demografía y fuentes de ingreso.

### 2.2 Justificación económica de las variables

| Bloque | Mecanismo económico | Sustento |
| :--- | :--- | :--- |
| Mercado laboral y formalidad | El trabajo es la principal fuente de ingreso del hogar. La informalidad (no cotizar a pensión, empresas pequeñas, cuenta propia) separa los ingresos estables de los precarios. | DANE–DNP (MESEP, 2012); Fernández y Villar (2017) |
| Ingresos no laborales | Pensiones, arriendos, remesas y ayudas: hogares con ingreso aunque nadie esté ocupado. | DANE (2019) |
| Vivienda y arriendo | El arriendo imputado **forma parte** de la definición de ingreso del DANE para propietarios y usufructuarios. El arriendo también aproxima la riqueza. | DANE–DNP (MESEP, 2012) |
| Demografía y dependencia | El mismo ingreso, dividido entre más miembros y más menores, deja al hogar por debajo de la línea. | Ravallion (2016) |
| Capital humano y protección social | Retornos a la educación; el régimen de salud (contributivo o subsidiado) aproxima la formalidad. | Angulo, Díaz y Pardo (2016) |

### 2.3 Qué revelaron los datos

- El ingreso per cápita sube con el **tamaño de la empresa** (un proxy de formalidad), con la educación superior y con estar ocupado. Es más bajo en las viviendas en posesión sin título ([`ingreso_dummies.png`](../02_outputs/figures/ingreso_dummies.png)).
- El sexo de la persona casi no cambia el ingreso per cápita promedio.
- En un CART exploratorio con todas las variables, la proporción de ocupados del hogar y el arriendo por persona sobre la línea concentran el 47% de la reducción de impureza.
- El ingreso es **muy asimétrico** y tiene muchos ceros por persona (inactivos, menores). Esto motivó la pérdida Tweedie (sección 3.4). El percentil 99 del ingreso per cápita es $4,4 millones.

### 2.4 El pivote: la pobreza es del hogar

| Etapa | F1 (CV) |
| :--- | :---: |
| **Fase 1 (fallida):** modelo por persona, probabilidad promediada en el hogar (log odds) | 0,7055 |
| **Fase 2 (pivote):** una fila por hogar (base de agregados `02`) | 0,7534 |
| + ingreso predicho por persona y stacking con la probabilidad por persona | 0,7598 |
| + categóricas nativas de XGBoost | 0,7612 |
| + 16 preguntas por persona que ningún modelo usaba | 0,7714 |
| + ingreso predicho afinado e hiperparámetros | 0,7733 |
| + pérdida Tweedie en el ingreso por persona | 0,7772 |
| + combinación con la regresión del ingreso (70/30) | **0,7775** |

En la fase 1, modelar a cada persona y promediar sus probabilidades dentro del hogar llegó a un techo de 0,706. En la fase 2 consolidamos la información en el hogar con sumas de dummies, proporciones, índices y promedios grupales: número de ocupados y de formales, educación máxima de los adultos, dependencia, hacinamiento y arriendo por persona. Ese cambio subió el F1 en **0,05**, el mayor salto del proyecto, porque captura las dinámicas del hogar. Por ejemplo, un ocupado formal sostiene a varios dependientes, y eso no se ve en una fila por persona.

**Contraste.** Generar más combinaciones de variables no sumó nada:

| Prueba | F1 |
| :--- | :---: |
| Interacciones de las variables más importantes | 0,7593 (vs 0,7603) |
| Logaritmos y cuadrados | 0,7588 |
| Índices sin supervisar y supervisados | 0,7605–0,7610 (vs 0,7612) |

Los árboles ya arman esas combinaciones, y una transformación monótona produce los mismos cortes.

---

## 3. Mejor modelo

### 3.1 Estructura (tres niveles)

Diagrama: [`02_outputs/figures/estructura_modelo_slide.png`](../02_outputs/figures/estructura_modelo_slide.png) (fuente: [`latex/estructura_modelo_slide.tex`](../latex/estructura_modelo_slide.tex)).

1. **Modelos auxiliares por persona** (XGBoost):
   - el ingreso de cada miembro, en cinco versiones (V4, V4t, V4e, V4f y **V4w con Tweedie**);
   - la probabilidad de ser pobre (P, Pe);
   - la probabilidad de que el ingreso del hogar esté por debajo de 0,8 y 1,25 líneas (C080, C125);
   - la tasa de pobreza del oficio.

   Como usan `pobre` o el ingreso, se estiman con **validación cruzada anidada**.
2. **Resúmenes por hogar:** suma, máximo, mínimo, el valor del jefe y el ingreso por ocupado. La variable clave es el **ingreso Tweedie predicho más el arriendo imputado, por persona y sobre la línea**, que reproduce la fórmula del DANE.
3. **Modelos por hogar:** un clasificador XGBoost y una regresión XGBoost de $z = \log(\text{ingreso}/\text{línea})$. Ambos usan esos resúmenes junto con la base por hogar, y sus probabilidades se combinan: $p = 0{,}7\,p_{\text{clasif}} + 0{,}3\,p_{\text{regr}}$; el hogar es pobre si $p > 0{,}39$.

### 3.2 Entrenamiento y validación (pilar: validación)

- **Folds por hogar** (5): todos los miembros de un hogar quedan en el mismo fold. Todos los modelos del equipo usan los mismos folds.
- **Validación cruzada anidada:** las variables que usan la etiqueta (ingreso predicho, probabilidad por persona, tasa del oficio) se estiman sin ver el fold que se evalúa, con 26 modelos por variable.
- **Parada temprana:** el entrenamiento para tras 100 rondas sin mejorar la log-loss del fold de validación.
- **Umbral** escogido por F1 en una grilla de 0,10 a 0,85.
- **F1 honesto:** con el umbral escogido en los otros 4 folds, el optimismo es de 0,0005 o menos.
- **Ruido:** con otras particiones en folds, el F1 tiene una desviación estándar de 0,0003. Las diferencias de menos de 0,001 entre modelos son ruido.
- **Leaderboard:** el puntaje público de Kaggle usa solo el 20% de test y tiene una desviación estándar de unos 0,006, así que no distingue mejoras de 0,001. **Escogimos los modelos por la validación cruzada, no por Kaggle.**
- **Sin fuga:** las 16 preguntas nuevas (pagos en especie, búsqueda de trabajo, ingresos de desocupados e inactivos) tienen la misma frecuencia y distribución en train y en test.

### 3.3 Hiperparámetros (pilar: herramientas)

| Parámetro (clasificador por hogar) | Valor |
| :--- | :---: |
| `objective` | `binary:logistic` |
| `eta` (learning rate) | 0,02 (bajo: aprendizaje iterativo y profundo) |
| `max_depth` | 8 |
| `min_child_weight` | 50 |
| `colsample_bytree` | 0,5 (50% de las variables por árbol) |
| `subsample` | 0,8 |
| Número de árboles | 613–869 (parada temprana, según el fold) |
| `gamma` (poda) / `lambda` (L2, ridge) / `alpha` (L1, lasso) | 0 / 1 / 0 |
| Variables categóricas | nativas de XGBoost (ni códigos enteros ni one-hot) |

La grilla final (profundidad 4, 6 u 8; `min_child_weight` 10 o 50; `colsample` 0,5 u 0,8) da una superficie plana, entre 0,7723 y 0,7733. Los parámetros de regularización `lambda` y `alpha` quedaron **fijos** desde las iteraciones anteriores, para aislar el efecto de las nuevas funciones de pérdida y variables. Las categóricas nativas superaron a los códigos enteros, sobre todo en los modelos por persona, donde el oficio tiene 84 niveles.

La regresión por hogar usa `eta` 0,05 y profundidad 6, con la misma regularización.

### 3.4 Funciones de pérdida y pesos (pilares: herramientas e intuición económica)

**Pérdida Tweedie en el ingreso por persona** (`reg:tweedie`, potencia $\rho = 1{,}2$; un modelo para el ingreso laboral y otro para el no laboral):

$$L(y,\mu) = -\frac{y\,\mu^{1-\rho}}{1-\rho} + \frac{\mu^{2-\rho}}{2-\rho}, \qquad 1 < \rho < 2$$

Con error cuadrático sobre $\log y$, la predicción de cada persona se acerca a la **mediana**. Al sumar a los miembros del hogar, el ingreso queda subestimado. La distribución Tweedie, que está entre la Poisson y la Gamma, admite masa en cero y valores positivos asimétricos, y predice la **media** del ingreso en pesos, que es la que importa al sumar. Con ella:

- el $R^2$ del ingreso del hogar pasa de 0,63 a 0,66;
- el F1 pasa de 0,7733 a **0,7772**, la mayor mejora después de las preguntas nuevas;
- con otras dos particiones, el modelo le gana al anterior a las preguntas nuevas por 0,015 en las tres.

**Pesos exponenciales en la regresión del ingreso**, con $z_i = \log(\text{ingreso}_i/\text{línea}_i)$:

$$w_i = 1 + 3\,e^{-|z_i|/0{,}25}$$

La función da hasta 4 veces más peso a los hogares **cerca de la línea, de los dos lados**. En una regresión, un error en un hogar rico no cambia su clasificación; uno cerca de la línea sí. Con estos pesos la regresión llega a un F1 de 0,7694, frente a 0,7684 sin ellos, y entra con 30% en la combinación final.

Los mismos pesos aplicados **al clasificador** no ayudaron (0,7689–0,7712, frente a 0,7715): los árboles ya separan lo que se puede separar. La función no castiga específicamente los falsos positivos, sino los errores cerca del umbral en las dos direcciones.

**Stacking:** el ingreso predicho y la probabilidad por persona entran como variables del clasificador. En cada ronda probamos una nueva versión del ingreso predicho (afinada, con variables del hogar, por componentes, Tweedie) como variable adicional. Así, XGBoost seleccionaba automáticamente las versiones útiles, y la validación cruzada indicaba cuándo una versión dejaba de aportar.

### 3.5 Importancia de las variables: pruebas de permutación

En cada fold se reestiman el clasificador y la regresión con los otros cuatro folds y se permutan al azar los valores de una variable (o de un grupo) entre los hogares del fold de validación. La importancia es la caída del F1 con el umbral de 0,39, promediada sobre 5 folds y 3 permutaciones. Como las variables correlacionadas se reemplazan entre sí, también se permutan **grupos** completos.

**Por grupo de variables (modelo final):**

| Grupo | Variables | Caída del F1 | D.E. | Ganancia (%) |
| :--- | :---: | :---: | :---: | :---: |
| Ingreso predicho por persona (todas las versiones) | 26 | **0,4232** | 0,0081 | 71,7 |
| Probabilidad de ingreso menor a 0,8 y 1,25 líneas | 8 | 0,0607 | 0,0154 | 6,5 |
| Probabilidad de ser pobre por persona (stacking) | 8 | 0,0437 | 0,0061 | 13,9 |
| Base por hogar de 02 (sin ubicación) | 89 | 0,0097 | 0,0024 | 4,0 |
| Otras construidas (expansión, mejor ocupado, productos, razones) | 24 | 0,0051 | 0,0014 | 2,8 |
| Ubicación (dominio, departamento, zona, línea) | 4 | 0,0022 | 0,0014 | 0,8 |
| Tasa de pobreza del oficio | 3 | 0,0005 | 0,0007 | 0,4 |

Sin el ingreso predicho, el F1 cae de 0,778 a 0,35. La variable individual más importante es `ing_arr_pc_lp_w`: el ingreso Tweedie predicho más el arriendo imputado, por persona y sobre la línea, que es **la fórmula del DANE** con el ingreso reconstruido. Permutarla resta 0,043 al F1 aunque las demás versiones del ingreso sigan en el modelo.

**Por concepto, de extremo a extremo.** Se permutan las variables del concepto en las personas y en el hogar, y la perturbación pasa por toda la cadena de modelos. Agrupamos los conceptos en categorías para leerlos mejor:

| Categoría | Concepto | Total | Directa (hogar) | Vía modelos auxiliares |
| :--- | :--- | :---: | :---: | :---: |
| **Mercado laboral** | Formalidad, posición y oficio | **0,226** | 0,002 | 0,226 |
| | Ocupación y horas de trabajo | 0,117 | 0,005 | 0,122 |
| | Búsqueda de trabajo e ingresos de desocupados | 0,070 | 0,000 | 0,070 |
| **Ingresos no laborales** | Pensiones, arriendos, remesas, ayudas | 0,143 | 0,001 | 0,137 |
| **Vivienda** | Vivienda y arriendo | 0,118 | 0,037 | 0,045 |
| **Demografía y territorio** | Edad, sexo y tamaño del hogar | 0,050 | 0,035 | 0,023 |
| | Ubicación y línea de pobreza | 0,024 | 0,009 | 0,017 |
| **Capital humano y protección social** | Educación | 0,021 | 0,006 | 0,024 |
| | Primas, subsidios y pagos en especie | 0,012 | 0,005 | 0,008 |
| | Salud | 0,009 | 0,001 | 0,010 |

Casi todo el efecto del trabajo pasa por los modelos auxiliares: el ingreso predicho resume el empleo de cada miembro. La vivienda y el tamaño del hogar tienen un efecto directo porque entran en la fórmula del ingreso per cápita con arriendo imputado. La educación y la salud importan poco porque su información ya está en la posición ocupacional, el oficio y la formalidad. Las importancias no se suman entre conceptos, porque los conceptos están correlacionados y sus efectos interactúan.

Fuentes: [`importancia_grupos.tex`](../02_outputs/tables/importancia_grupos.tex), [`importancia_conceptos.tex`](../02_outputs/tables/importancia_conceptos.tex), [`importancia_variables.tex`](../02_outputs/tables/importancia_variables.tex), [`importancia_directas.tex`](../02_outputs/tables/importancia_directas.tex) (secciones 14 y 15 de [`11_xgboost_hogar.R`](../01_code/11_xgboost_hogar.R)).

### 3.6 Diagnósticos: dónde se equivoca

- El 30% de los errores está en hogares con ingreso a menos de 10% de la línea, que son apenas el 7% de los hogares. El 82% de los errores está a menos de 50% de la línea (33% de los hogares).
- Justo al lado de la línea el modelo se equivoca en el 40–46% de los hogares; con ingresos de más del doble de la línea, en el 0,6%.
- Ni dar más peso a esos hogares ni un modelo "especialista" para los hogares dudosos ayudó. Con estas variables, ese error es prácticamente irreducible.
- Ningún grupo (ciudad, tamaño, tenencia, sexo del jefe) concentra errores de forma llamativa.

---

## 4. Implicaciones de política pública

### 4.1 Del F1 a la focalización: leakage vs. undercoverage

En focalización, un **falso negativo** es un hogar pobre excluido (*undercoverage*, error de exclusión) y un **falso positivo** es un hogar no pobre incluido (*leakage*, error de inclusión). Con el umbral de 0,39 y las predicciones fuera de muestra de train:

- **Cobertura (sensibilidad):** cerca del 81% de los hogares pobres se detecta, así que el *undercoverage* es de cerca del 19%.
- **Precisión:** cerca del 75%. El *leakage* es de 1 de cada 4 beneficiarios, que corresponde al 6,7% de los hogares no pobres.

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

Fuente: [`equidad_mejor.tex`](../02_outputs/tables/equidad_mejor.tex).

- En los grupos con más precariedad (zona rural, jefe informal, hogares grandes) el modelo **sobreestima** la pobreza: tiene más errores de inclusión. Asocia las características precarias con la pobreza, y en esos grupos también hay muchos hogares precarios que no son pobres monetariamente.
- Los pobres "que no lo parecen" (jefe formal o con educación superior, hogares pequeños o sin menores) quedan **excluidos** con más frecuencia: la sensibilidad es de 65–67%.
- **Jefatura femenina:** el modelo detecta un poco mejor a los hogares pobres con jefa (+3,8 puntos) con algo más de falsos positivos (+1,2 puntos). Es una diferencia pequeña y a favor de la inclusión.

Este patrón no viene de una regla discriminatoria, sino de **tasas base distintas**. Con una sola regla de decisión no se pueden igualar a la vez la sensibilidad y la tasa de falsos positivos entre grupos (Kleinberg, Mullainathan y Raghavan, 2017). En el contexto colombiano, donde la política social se focaliza con el SISBÉN, la brecha más relevante es la exclusión de hogares pobres formales o pequeños. Se podría corregir con umbrales por grupo o con verificación complementaria, a costa de más inclusión.

### 4.3 Pobreza monetaria vs. multidimensional

El modelo mide **estrictamente pobreza monetaria**. Educación, salud y subsidios aportan poco al poder predictivo (caída del F1 de 0,021, 0,009 y 0,012) una vez se conoce el trabajo del hogar. Esto tiene dos lecturas:

1. **Las dimensiones miden cosas distintas.** El IPM colombiano (Angulo et al., 2016) y la pobreza monetaria (DANE, 2019) se traslapan solo en parte. Un hogar puede tener privaciones en salud o educación sin estar bajo la línea de ingreso, y al revés. Más que independencia estadística, lo que encontramos es que esas dimensiones agregan poca información sobre el **ingreso** que no esté ya en el mercado laboral.
2. **Riesgo de focalización.** Si un programa de transferencias monetarias se focaliza con indicadores de salud, educación o acceso a subsidios, comete errores sistemáticos de inclusión y exclusión. Además, cuando las variables del instrumento son observables y conocidas, los hogares pueden ajustar sus respuestas (Camacho y Conover, 2011, sobre el SISBÉN).

### 4.4 Viabilidad operativa: ¿una encuesta corta y barata?

**A favor**
- No hay que preguntar **montos de ingreso**, la parte más cara, más sensible y más subdeclarada de una encuesta de hogares. Bastan preguntas sí/no sobre fuentes de ingreso, el módulo laboral y la vivienda.
- La formalidad y el oficio se pueden verificar con registros administrativos, como la PILA.

**En contra**
- El modelo necesita el módulo laboral de **cada miembro** del hogar, unas 60 preguntas por persona. No es un formulario de 10 preguntas: el costo es intermedio, parecido a una visita del SISBÉN IV (CONPES 3877, 2016).
- Los ingresos no laborales y la búsqueda de trabajo son difíciles de verificar en territorio.
- El modelo se entrenó con datos de 2018 y hay que reentrenarlo cuando cambien la línea de pobreza y el mercado laboral.

**Ojo con las versiones reducidas.** Un clasificador con solo 20 variables pierde poco F1 (0,7580 frente a 0,7612), pero esas 20 son sobre todo salidas de los modelos auxiliares, que siguen necesitando el módulo de cada persona. Reducir el número de variables del modelo no reduce, por sí solo, el costo de la encuesta.

---

## 5. Cierre: qué nos hizo ganar

| Pilar | Qué hicimos |
| :--- | :--- |
| **Validación concienzuda** | CV anidada sin fuga, folds por hogar, F1 honesto, ruido medido con otras particiones. Decidimos con la CV, no con el leaderboard. |
| **Herramientas** | Categóricas nativas, pérdida Tweedie, stacking con modelos por persona, parada temprana y paralelismo para correr cientos de pruebas que retoman si se interrumpen. |
| **Intuición económica** | La pobreza es del hogar; el arriendo imputado reproduce la regla del DANE; 16 preguntas del cuestionario (ingresos sin estar ocupado, pagos en especie) que ningún modelo usaba. |

> Con la IA, cualquiera combina variables. La ventaja está en saber qué preguntar y en cómo comprobarlo.

---

## 6. Cómo replicar

Desde la raíz del repositorio:

```sh
Rscript 01_code/01_data_cleaning.R                      # bases limpias por persona
Rscript 01_code/02_bases_hogar.R                       # bases por hogar (data/train_hogar.rds, data/test_hogar.rds)
Rscript 01_code/11_xgboost_hogar.R                     # modelo final, permutaciones (secciones 14-15) y submission
Rscript 01_code/15_comparacion_modelos.R                # tablas de equidad
cd presentacion && latexmk -pdf best_equipo_04.tex     # presentación
```

Submission final: `02_outputs/predictions/XGBH_nativo_comb_..._peso_w07_threshold_039.csv`.

---

## Referencias

- Angulo, R., Díaz, Y. y Pardo, R. (2016). The Colombian Multidimensional Poverty Index: Measuring poverty in a public policy context. *Social Indicators Research*, 127(1), 1–38.
- Camacho, A. y Conover, E. (2011). Manipulation of social program eligibility. *American Economic Journal: Economic Policy*, 3(2), 41–65.
- Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.
- CONPES 3877 (2016). Declaración de importancia estratégica del Sistema de Identificación de Potenciales Beneficiarios (SISBÉN IV). DNP.
- DANE (2019). Pobreza monetaria y multidimensional en Colombia 2018. Boletín técnico.
- DANE y DNP (2012). Misión para el Empalme de las Series de Empleo, Pobreza y Desigualdad (MESEP).
- Fernández, C. y Villar, L. (2017). The impact of lowering the payroll tax on informality in Colombia. *Economía*, 18(1), 125–155.
- Grosh, M. y Baker, J. (1995). Proxy means tests for targeting social programs. LSMS Working Paper 118, Banco Mundial.
- Kleinberg, J., Ludwig, J., Mullainathan, S. y Obermeyer, Z. (2015). Prediction policy problems. *American Economic Review*, 105(5), 491–495.
- Kleinberg, J., Mullainathan, S. y Raghavan, M. (2017). Inherent trade-offs in the fair determination of risk scores. *ITCS*.
- Ravallion, M. (2016). *The Economics of Poverty*. Oxford University Press.
