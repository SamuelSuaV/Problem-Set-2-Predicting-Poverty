# Copia sin cambios de script/01b_bases_hogar.r de origin/main (commit 63fde6d),
# para usar la base por hogar en las pruebas de esta carpeta.

###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   01b_bases_hogar.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Construir las bases a nivel hogar (una fila por hogar,
#                      con agregados de sus miembros) a partir de las bases
#                      limpias por persona.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Agregar personas al hogar
# 4. Guardar bases a nivel hogar

################################################################################

# Input:  Bases limpias creadas en 01_data_cleaning.r; train_clean.rds ,
#         test_clean.rds .
# Output: train_hogar.rds , test_hogar.rds .

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse   # Manipulación de datos.
)

################################################################################

## 2. Cargar datos

train_clean <- read_rds(here("data", "train_clean.rds"))
test_clean  <- read_rds(here("data", "test_clean.rds"))

################################################################################

## 3. Agregar personas al hogar

# Una fila por hogar con agregados de sus miembros, para modelos que predicen
# directamente el hogar (pobre y la submission son del hogar). Los conteos y
# proporciones usan solo la unidad de gasto (sin servicio doméstico,
# pensionistas ni trabajadores, parentesco 6-8), que es el denominador de
# ingreso = Ingpcug. Las preguntas que no aplican (p. ej. de trabajo para no
# ocupados) cuentan como 0 en los conteos, así que no quedan NA.
agregar_hogar <- function(df) {
  df <- df |>
    mutate(
      en_ug  = !parentesco_jefe_hogar %in% c(6, 7, 8),
      jefe   = parentesco_jefe_hogar == 1,
      adulto = anos >= 18,
      ocup   = ocupado == 1 & en_ug,
      formal = ocup & cotiza_pension %in% 1,  # ocupado que cotiza a pensión
      pensionado = cotiza_pension %in% 3 | recibio_pension_jubilacion %in% 1,
      # Número de primas, subsidios y bonificaciones que recibe (empleo formal).
      beneficios = rowSums(across(c(recibio_primas, prima_servicios, prima_navidad,
                                    prima_vacaciones, subsidio_alimentacion,
                                    subsidio_transporte, subsidio_familiar,
                                    subsidio_educativo, recibio_bonificaciones,
                                    bonificaciones_anuales), ~ .x %in% 1)),
      horas = coalesce(horas_trabajo_semana, 0) + coalesce(horas_segundo_trabajo, 0),
      gran_oficio = oficio %/% 10  # primer dígito de la CIUO-68
    )
  base <- df |>
    summarize(
      .by = id,
      # Ubicación y umbral
      dominio = first(dominio), depto = first(depto), cabecera = first(cabecera),
      log_lp = log(first(linea_pobreza)),
      # Demografía
      n_ug = first(personas_unidad_gasto), log_n_ug = log(n_ug),
      n_menor6  = sum(anos < 6 & en_ug), n_menor12 = sum(anos < 12 & en_ug),
      n_menor18 = sum(anos < 18 & en_ug), n_mayor65 = sum(anos >= 65 & en_ug),
      dependencia  = (n_menor18 + n_mayor65) / pmax(sum(anos >= 18 & anos < 65 & en_ug), 1),
      prop_menores = n_menor18 / n_ug,
      prop_mujer_ug = mean(mujer[en_ug]),
      # Cuántos generan ingreso
      n_ocup = sum(ocup), ocup_pc = n_ocup / n_ug,
      ocup_por_adulto = n_ocup / pmax(sum(adulto & en_ug), 1),
      n_desocup = sum(desocupado %in% 1 & en_ug), desocup_pc = n_desocup / n_ug,
      n_inactivo_adulto = sum(inactivo %in% 1 & anos >= 18 & anos < 65 & en_ug),
      horas_pc = sum(horas[en_ug]) / n_ug,
      horas_formal_pc   = sum(horas[formal]) / n_ug,
      horas_informal_pc = sum(horas[ocup & !formal]) / n_ug,
      n_segundo_trab = sum(segundo_trabajo %in% 1 & en_ug),
      n_quiere_mas_horas = sum(quiere_mas_horas %in% 1 & en_ug),
      n_oficios_hogar = sum(actividad_sem_pasada %in% 4 & adulto & en_ug),
      n_estudia = sum(actividad_sem_pasada %in% 3 & en_ug),
      n_pareja_ocup = sum(parentesco_jefe_hogar == 2 & ocup),
      n_hijos_ocup  = sum(parentesco_jefe_hogar == 3 & ocup),
      # Calidad del empleo
      n_formal = sum(formal), formal_pc = n_formal / n_ug,
      prop_formal_ocup = if_else(n_ocup > 0, n_formal / n_ocup, 0),
      n_asalariado    = sum(ocup & posicion_ocupacional %in% 1),
      n_gobierno      = sum(ocup & posicion_ocupacional %in% 2),
      n_domestico     = sum(ocup & posicion_ocupacional %in% 3),
      n_cuenta_propia = sum(ocup & posicion_ocupacional %in% 4),
      n_patron        = sum(ocup & posicion_ocupacional %in% 5),
      n_sin_pago      = sum(ocup & posicion_ocupacional %in% c(6, 7)),
      n_jornalero     = sum(ocup & posicion_ocupacional %in% 8),
      n_empresa_sola   = sum(ocup & tamano_empresa %in% 1),
      n_empresa_peq    = sum(ocup & tamano_empresa %in% 2:3),
      n_empresa_grande = sum(ocup & tamano_empresa %in% 8:9),
      beneficios_pc = sum(beneficios[en_ug]) / n_ug,
      n_con_beneficios = sum(beneficios > 0 & en_ug),
      n_horas_extra = sum(recibio_horas_extra %in% 1 & en_ug),
      max_log_meses = if (any(ocup)) max(log_meses_empresa[ocup]) else 0,
      n_of_profesional    = sum(ocup & gran_oficio %in% 0:1),
      n_of_directivo      = sum(ocup & gran_oficio %in% 2),
      n_of_administrativo = sum(ocup & gran_oficio %in% 3),
      n_of_comercio       = sum(ocup & gran_oficio %in% 4),
      n_of_servicios      = sum(ocup & gran_oficio %in% 5),
      n_of_agro           = sum(ocup & gran_oficio %in% 6),
      n_of_obrero         = sum(ocup & gran_oficio %in% 7:9),
      # Salud
      prop_contributivo = sum(salud_contributivo %in% 1 & en_ug) / n_ug,
      prop_subsidiado   = sum(salud_subsidiado %in% 1 & en_ug) / n_ug,
      n_especial    = sum(salud_especial %in% 1 & en_ug),
      n_no_afiliado = sum(afiliado_salud %in% 0 & en_ug),
      # Educación (anos_educ_imp; NA cuenta como 0 años)
      educ_max = max(coalesce(anos_educ_imp[en_ug], 0)),
      educ_media_adultos = coalesce(mean(anos_educ_imp[adulto & en_ug], na.rm = TRUE), 0),
      educ_ocup_media    = coalesce(mean(anos_educ_imp[ocup], na.rm = TRUE), 0),
      educ_ocup_suma_pc  = sum(anos_educ_imp[ocup], na.rm = TRUE) / n_ug,
      prop_superior = sum(educ_superior %in% 1 & adulto & en_ug) / pmax(sum(adulto & en_ug), 1),
      n_sin_educ = sum(max_educ %in% 1:2 & adulto & en_ug),
      n_ocup_superior = sum(ocup & educ_superior %in% 1),
      superior_formal_pc = sum(formal & educ_superior %in% 1) / n_ug,
      # Jefe del hogar (en train cada hogar tiene exactamente un jefe)
      jefe_edad = anos[jefe][1], jefe_edad2 = jefe_edad^2, jefe_mujer = mujer[jefe][1],
      jefe_mujer_sola = jefe_mujer * !any(parentesco_jefe_hogar == 2),
      jefe_educ = coalesce(anos_educ_imp[jefe][1], 0),
      jefe_ocup = ocupado[jefe][1], jefe_formal = as.integer(formal[jefe][1]),
      jefe_desocup = as.integer(desocupado[jefe][1] %in% 1),
      jefe_log_meses = coalesce(log_meses_empresa[jefe][1], 0),
      # Códigos DANE; 0 = el jefe no está ocupado (o no está afiliado a salud).
      jefe_posicion = coalesce(posicion_ocupacional[jefe][1], 0),
      jefe_regimen  = coalesce(regimen_salud[jefe][1], 0),
      jefe_tamano   = coalesce(tamano_empresa[jefe][1], 0),
      jefe_oficio   = coalesce(oficio[jefe][1], 0),
      # Ingresos no laborales: algún miembro recibe
      any_pension    = as.integer(any(pensionado)),
      any_arriendos  = as.integer(any(recibio_arriendos_pensiones %in% 1)),
      any_intereses  = as.integer(any(dinero_intereses_dividendos %in% 1)),
      any_exterior   = as.integer(any(dinero_hogares_exterior %in% 1)),
      any_hogares_pais = as.integer(any(dinero_hogares_pais %in% 1)),
      any_ayudas     = as.integer(any(ayudas_instituciones %in% 1)),
      any_cesantias  = as.integer(any(dinero_cesantias %in% 1)),
      any_pension_alim = as.integer(any(recibio_pension_alimenticia %in% 1)),
      any_otros_ing  = as.integer(any(recibio_otros_ingresos %in% 1)),
      n_fuentes_no_lab = any_pension + any_arriendos + any_intereses +
                         any_exterior + any_cesantias,
      # Vivienda
      tenencia = first(tenencia_vivienda),
      cuartos_pc = first(cuartos_hogar) / first(personas_hogar),
      pers_por_dormitorio = first(personas_hogar) / pmax(first(cuartos_dormir), 1),
      log_arriendo_pc_lp = first(arriendo_pc_sobre_lp),
      arriendo_na = as.integer(is.na(log_arriendo_pc_lp)),
      arriendo_es_pagado = as.integer(!is.na(first(arriendo_pagado))),
      cuota = first(cuota_amortizacion),
      log_cuota_pc_lp = if_else(!is.na(cuota) & cuota > 0,
                                log(cuota / first(personas_hogar) / first(linea_pobreza)), 0)
    ) |>
    select(-cuota)
  # pobre solo existe en train.
  if ("pobre" %in% names(df)) {
    base <- left_join(distinct(df, id, pobre), base, by = "id", relationship = "one-to-one")
  }
  base
}
train_hogar <- agregar_hogar(train_clean)
test_hogar  <- agregar_hogar(test_clean)
rm(train_clean, test_clean)  # liberar memoria

# Hogares sin arriendo pagado ni estimado (arriendo_na = 1): se imputa la
# mediana de train en ambas bases, para que test use el mismo valor.
mediana_arriendo <- median(train_hogar$log_arriendo_pc_lp, na.rm = TRUE)
train_hogar <- train_hogar |> mutate(log_arriendo_pc_lp = coalesce(log_arriendo_pc_lp, mediana_arriendo))
test_hogar  <- test_hogar  |> mutate(log_arriendo_pc_lp = coalesce(log_arriendo_pc_lp, mediana_arriendo))

################################################################################

## 4. Guardar bases a nivel hogar
write_rds(train_hogar, here("data", "train_hogar.rds"))
write_rds(test_hogar,  here("data", "test_hogar.rds"))
