# Material guardado para el set de diapositivas de comparación de modelos

Este material quedó fuera de `best_equipo_04.tex` por indicación del equipo y se usará en otro set de diapositivas. Fuentes: [`output/tables/modelos_f1.tex`](../output/tables/modelos_f1.tex), [`output/tables/equidad_brechas.tex`](../output/tables/equidad_brechas.tex), [`output/figures/equidad_errores.png`](../output/figures/equidad_errores.png), [`latex/resultados.tex`](../latex/resultados.tex) y el historial de commits de `main`.

## Mejor modelo de cada categoría (misma evaluación: 5 folds por hogar, F1 por hogar)

| Categoría | Modelo | F1 en su script | F1 (5 folds) | F1 honesto | Umbral | Kaggle |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| CART | Por persona, x1, cp 2,5e-05, up-sampling | 0,670 | 0,6685 | 0,6679 | 0,62 | — |
| Naive Bayes | Por persona, 16 variables (x1), laplace 0 | 0,675 | 0,6755 | 0,6752 | 0,42 | — |
| XGBoost por persona | 24 variables (x3), 500 rondas | 0,706 | 0,7055 | 0,7055 | 0,32 | — |
| Logit lasso | Por hogar, 1.159 columnas con interacciones (`06_logit_interacciones.r`, main) | 0,745 | 0,7451 | 0,7449 | 0,37 | 0,74 (lasso de 04) |
| LightGBM | Por hogar (`07_boosting.r`, main) | 0,7537 | — | — | 0,37 | 0,75 |
| XGBoost por hogar | Solo la base por hogar de 01b | — | 0,7534 | 0,7532 | 0,36 | — |
| XGBoost por hogar | Mejor sin Tweedie, con la regresión del ingreso | — | 0,7738 | 0,7736 | 0,38 | — |
| **Random forest** | Por hogar, variables del mejor XGBoost (`ranger`, 500 árboles, mtry 30) | — | **0,7752** | 0,7750 | 0,40 | — |
| **XGBoost por hogar (final)** | Tweedie + regresión del ingreso | — | **0,7775** | 0,7775 | 0,39 | **0,7775** |

Los Kaggle del logit y de LightGBM vienen de los mensajes de commit de `main` ("Kaggle 0,74", "Kaggle 0,75"). Faltan los demás: [INSERT PUBLIC SCORE HERE].

## Random forest

- Con las mismas variables queda ligeramente por debajo de XGBoost: 0,7752 frente a 0,7775.
- Combinado con XGBoost: 0,7773 (honesto 0,7770). No aporta sobre XGBoost solo; la diferencia es ruido (menos de 0,001).

## Logit lasso (pesos, interpretabilidad)

- F1 de 0,7451 con 1.159 columnas (interacciones de a pares, `ns(df = 4)` en 8 continuas, `dominio ×` 10 variables), lambda 5,4e-05, umbral 0,37 y 794 coeficientes distintos de 0.
- Logit sin penalizar con las k primeras variables del ranking del lasso (`logit_parsimonia.tex` de main): 15 variables dan 0,697, 40 dan 0,727 y las 94 dan 0,736.
- Tiene una correlación de 0,92 con XGBoost en log odds y su peso óptimo en la combinación es 0: se equivoca en los mismos hogares.
- Argumentos a desarrollar: reimputación (coeficientes aplicables a otras encuestas), divulgación técnica, interpretabilidad (signos y magnitudes) a un costo de unos 0,03 de F1.

## Brechas de equidad por modelo (puntos porcentuales de sensibilidad entre el grupo mejor y el peor detectado)

| Modelo | Sexo jefe | Empleo jefe | Zona | Edad jefe | Educ. jefe | Tamaño | Menores | Vivienda |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CART | 0,5 | 15,9 | 6,0 | 13,0 | 13,7 | 31,3 | 24,6 | 11,8 |
| Naive Bayes | 2,0 | 33,4 | 12,9 | 9,3 | 30,6 | 38,8 | 31,4 | 19,9 |
| XGBoost por persona | 1,2 | 14,0 | 2,6 | 10,5 | 22,9 | 23,7 | 22,3 | 11,5 |
| Logit lasso | 4,2 | 22,2 | 0,0 | 7,0 | 19,3 | 25,7 | 23,8 | 7,2 |
| XGBoost hogar (solo 01b) | 3,8 | 21,0 | 0,2 | 6,3 | 18,9 | 22,1 | 20,4 | 7,0 |
| XGBoost hogar sin Tweedie | 3,7 | 20,0 | 1,3 | 6,6 | 18,2 | 22,0 | 19,4 | 6,5 |
| Random forest | 4,4 | 21,4 | 0,2 | 6,9 | 17,7 | 21,8 | 19,1 | 6,2 |
| XGBoost hogar final | 3,8 | 19,7 | 0,8 | 6,9 | 18,0 | 22,4 | 19,9 | 6,4 |

Los modelos más precisos reducen casi todas las brechas. Por ejemplo, la brecha de falsos positivos entre zonas pasa de 27,8 puntos con naive Bayes a 6,5 con el mejor modelo. La excepción es el sexo del jefe: la brecha es de 1–2 puntos en los modelos por persona y de unos 4 en los modelos por hogar, siempre a favor de los hogares con jefa.
