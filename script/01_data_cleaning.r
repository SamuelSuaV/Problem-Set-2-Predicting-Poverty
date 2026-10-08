###############################################################################
# Nombre del proyecto: Predicting Poverty
# Nombre del script:   01_data_cleaning.r
# Autores:             Maria Jose Perez, Juan Manuel Lozano, Samuel Suárez Valle
# Propósito:           Unir las bases de personas y hogares (train y test) y
#                      renombrar variables.
###############################################################################

# Estructura:
# 1. Cargar librerías
# 2. Cargar datos
# 3. Merge de individuos con sus hogares
# 4. Renombrar columnas
# 5. Variables editadas
# 6. Guardar bases limpias

################################################################################

# Input:  Dataframes crudos de hogares e individuos; train_hogares.csv ,
#         train_personas.csv , test_hogares.csv , test_personas.csv .
# Output: train_clean.rds , test_clean.rds .

################################################################################

## 1. Cargar librerías

library(pacman)
p_load(
  here,       # Rutas relativas a la raíz del proyecto.
  tidyverse   # Manipulación de datos.
)

################################################################################

## 2. Cargar datos
raw_train_hogares  <- read_csv(here("data", "train_hogares.csv"), guess_max = Inf)
raw_train_personas <- read_csv(here("data", "train_personas.csv"), guess_max = Inf)
raw_test_hogares   <- read_csv(here("data", "test_hogares.csv"), guess_max = Inf)
raw_test_personas  <- read_csv(here("data", "test_personas.csv"), guess_max = Inf)

################################################################################

## 3. Merge de individuos con sus hogares
# Las columnas comunes son idénticas entre personas y hogares para cada id,
# así que entran como llaves y no se duplican (.x/.y).
llaves <- c("id", "Clase", "Dominio", "Fex_c", "Depto", "Fex_dpto")
raw_train <- left_join(raw_train_personas, raw_train_hogares,
                       by = llaves, relationship = "many-to-one")
raw_test  <- left_join(raw_test_personas,  raw_test_hogares,
                       by = llaves, relationship = "many-to-one")

################################################################################

## 4. Renombrar columnas
# Nombre nuevo = nombre original.
nombres <- c(
    orden               = "Orden",
    clase               = "Clase",
    dominio             = "Dominio",
    sexo                = "P6020",
    anos                = "P6040",
    parentesco_jefe_hogar = "P6050",
    afiliado_salud      = "P6090",
    regimen_salud       = "P6100",
    max_educ            = "P6210",
    max_grado_escolar   = "P6210s1",
    actividad_sem_pasada = "P6240",
    oficio              = "Oficio",
    meses_empresa       = "P6426",
    posicion_ocupacional = "P6430",
    recibio_horas_extra = "P6510",
    recibio_primas      = "P6545",
    recibio_bonificaciones = "P6580",
    subsidio_alimentacion = "P6585s1",
    subsidio_transporte = "P6585s2",
    subsidio_familiar   = "P6585s3",
    subsidio_educativo  = "P6585s4",
    pago_especie_alimentos = "P6590",
    pago_especie_vivienda = "P6600",
    transporte_empresa  = "P6610",
    pago_especie_otros  = "P6620",
    prima_servicios     = "P6630s1",
    prima_navidad       = "P6630s2",
    prima_vacaciones    = "P6630s3",
    viaticos_permanentes = "P6630s4",
    bonificaciones_anuales = "P6630s6",
    horas_trabajo_semana = "P6800",
    tamano_empresa      = "P6870",
    cotiza_pension      = "P6920",
    segundo_trabajo     = "P7040",
    horas_segundo_trabajo = "P7045",
    posicion_segundo_trabajo = "P7050",
    quiere_mas_horas    = "P7090",
    diligencias_mas_horas = "P7110",
    disponible_mas_horas = "P7120",
    diligencias_cambiar_trabajo = "P7150",
    disponible_nuevo_trabajo = "P7160",
    primera_vez_busca_trabajo = "P7310",
    posicion_ultimo_trabajo = "P7350",
    ingreso_trabajo_desocupado = "P7422",
    ingreso_trabajo_desocupado_2 = "P7472",
    recibio_arriendos_pensiones = "P7495",
    recibio_pension_jubilacion = "P7500s2",
    recibio_pension_alimenticia = "P7500s3",
    recibio_otros_ingresos = "P7505",
    dinero_hogares_pais = "P7510s1",
    dinero_hogares_exterior = "P7510s2",
    ayudas_instituciones = "P7510s3",
    dinero_intereses_dividendos = "P7510s5",
    dinero_cesantias    = "P7510s6",
    dinero_otras_fuentes = "P7510s7",
    edad_trabajar       = "Pet",
    ocupado             = "Oc",
    desocupado          = "Des",
    inactivo            = "Ina",
    fex_c               = "Fex_c",
    depto               = "Depto",
    fex_dpto            = "Fex_dpto",
    # Variables del hogar
    cuartos_hogar       = "P5000",
    cuartos_dormir      = "P5010",
    tenencia_vivienda   = "P5090",
    cuota_amortizacion  = "P5100",
    arriendo_estimado   = "P5130",
    arriendo_pagado     = "P5140",
    personas_hogar      = "Nper",
    personas_unidad_gasto = "Npersug",
    linea_indigencia    = "Li",
    linea_pobreza       = "Lp"
)

# Variables que solo existen en train: se deja ingreso (Ingpcug, ingreso
# per cápita de la unidad de gasto, que define pobre = ingreso < Lp) y se
# eliminan las demás preguntas (P...) y agregados de ingreso.
train_clean <- raw_train |>
  rename(all_of(nombres), ingreso = Ingpcug, pobre = Pobre) |>
  select(-matches("^P[0-9]", ignore.case = FALSE),
         -c(Impa:Ingtot, Ingtotug, Ingtotugarr))
test_clean  <- raw_test  |> rename(all_of(nombres))

################################################################################
## 5. Variables editadas

# Preguntas sí/no: 1 = sí, 2 = no, 9 = no sabe. Quedan 1 = sí, 0 = no,
# NA = no sabe.
si_no <- c(
  "afiliado_salud", "recibio_horas_extra", "recibio_primas",
  "recibio_bonificaciones", "subsidio_alimentacion", "subsidio_transporte",
  "subsidio_familiar", "subsidio_educativo", "pago_especie_alimentos",
  "pago_especie_vivienda", "transporte_empresa", "pago_especie_otros",
  "prima_servicios", "prima_navidad", "prima_vacaciones",
  "viaticos_permanentes", "bonificaciones_anuales", "segundo_trabajo",
  "quiere_mas_horas", "diligencias_mas_horas", "disponible_mas_horas",
  "diligencias_cambiar_trabajo", "disponible_nuevo_trabajo",
  "primera_vez_busca_trabajo", "ingreso_trabajo_desocupado",
  "ingreso_trabajo_desocupado_2", "recibio_arriendos_pensiones",
  "recibio_pension_jubilacion", "recibio_pension_alimenticia",
  "recibio_otros_ingresos", "dinero_hogares_pais", "dinero_hogares_exterior",
  "ayudas_instituciones", "dinero_intereses_dividendos", "dinero_cesantias",
  "dinero_otras_fuentes"
)

# Categóricas: una dummy por categoría (nombre = código DANE), incluida
# "no sabe". La variable original se conserva.
categorias <- list(
  max_educ = c(educ_ninguno = 1, educ_preescolar = 2, educ_primaria = 3,
               educ_secundaria = 4, educ_media = 5, educ_superior = 6,
               educ_no_sabe = 9),
  regimen_salud = c(salud_contributivo = 1, salud_especial = 2,
                    salud_subsidiado = 3, salud_no_sabe = 9),
  tenencia_vivienda = c(vivienda_propia_pagada = 1, vivienda_propia_pagando = 2,
                        vivienda_arriendo = 3, vivienda_usufructo = 4,
                        vivienda_sin_titulo = 5, vivienda_otra = 6)
)

# Misma edición para train y test, para que las variables coincidan.
editar <- function(df) {
  for (v in names(categorias)) {
    for (d in names(categorias[[v]])) {
      df[[d]] <- as.numeric(df[[v]] == categorias[[v]][[d]])
    }
  }
  df |>
    mutate(
      meses_empresa = replace_na(meses_empresa, 0),
      tamano_empresa = replace_na(tamano_empresa, 0),
      across(all_of(si_no), ~ case_match(.x, 1 ~ 1, 2 ~ 0)),
      mujer    = case_match(sexo,  2 ~ 1, 1 ~ 0),  # 1 hombre, 2 mujer
      cabecera = case_match(clase, 1 ~ 1, 2 ~ 0),  # 1 cabecera, 2 resto
      # Montos de vivienda: 98 = no sabe, 99 = no informa.
      across(c(arriendo_estimado, arriendo_pagado, cuota_amortizacion),
             ~ if_else(.x %in% c(98, 99), NA, .x))
    ) |>
    select(-sexo, -clase) |>
    # Proporción del hogar en edad de trabajar (edad_trabajar: 1 o NA).
    mutate(prop_edad_trabajar = sum(edad_trabajar == 1, na.rm = TRUE) / personas_hogar,
           .by = id) |>
    # Variables 0/1 como enteros: ocupan la mitad de memoria que numeric.
    mutate(across(where(~ is.numeric(.x) && all(.x %in% c(0, 1, NA))),
                  as.integer))
}
train_clean <- editar(train_clean) |>
  # Hogares con 98 y 43 cuartos (2 dormitorios, 3 personas): error de
  # digitación. Solo en train; test nunca se filtra (hay que predecirlo todo).
  filter(!cuartos_hogar %in% c(98, 43))
test_clean  <- editar(test_clean)


################################################################################

## 6. Guardar bases limpias
write_rds(train_clean, here("data", "train_clean.rds"))
write_rds(test_clean,  here("data", "test_clean.rds"))
