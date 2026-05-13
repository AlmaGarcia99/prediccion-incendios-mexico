import streamlit as st
import pandas as pd
import joblib
import os
import gdown

# =========================
# DESCARGAR MODELO DESDE DRIVE
# =========================

if not os.path.exists("modelo_incendios.pkl"):
   
    file_id = "1tH7dXUOvnfrp9e9taGn2zsroAyEDk9Il"

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
    url,
    "modelo_incendios.pkl",
    quiet=False,
    fuzzy=True
)

# =========================
# CARGAR MODELO
# =========================

modelo = joblib.load("modelo_incendios.pkl")
columnas = joblib.load("columnas_modelo.pkl")



# =========================
# TITULO
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Incendios analizados", "71,000+")
col2.metric("Accuracy del modelo", "61%")
col3.metric("Estados monitoreados", "32")

st.title("Predicción de Incendios Forestales en México")

st.write("""
Sistema basado en Machine Learning para predecir
la severidad esperada de incendios forestales.
""")

# =========================
# FORMULARIO
# =========================

st.header("Ingresa datos del incendio")

estado = st.selectbox(
    "Estado",
    [
        "Jalisco",
        "México",
        "Chiapas",
        "Chihuahua",
        "Durango",
        "Oaxaca",
        "Guerrero",
        "Michoacán"
    ]
)

causa = st.selectbox(
    "Causa",
    [
        "Intencional",
        "Actividades agrícolas",
        "Fogatas",
        "Fumadores",
        "Actividades pecuarias",
        "Desconocidas"
    ]
)

vegetacion = st.selectbox(
    "Tipo de vegetación",
    [
        "Bosque de Pino",
        "Bosque de Pino-Encino",
        "Bosque de Encino",
        "Selva Baja Caducifolia"
    ]
)

mes_inicio = st.slider("Mes", 1, 12, 4)

duracion = st.number_input(
    "Duración del incendio (segundos)",
    min_value=0,
    value=50000
)

deteccion = st.number_input(
    "Tiempo de detección (segundos)",
    min_value=0,
    value=3600
)

llegada = st.number_input(
    "Tiempo de llegada (segundos)",
    min_value=0,
    value=7200
)

# =========================
# CREAR INPUT
# =========================

input_data = pd.DataFrame(columns=columnas)
input_data.loc[0] = 0

# Variables numéricas
input_data['Duracion'] = duracion
input_data['mes_inicio'] = mes_inicio
input_data['Deteccion'] = deteccion
input_data['Llegada'] = llegada

# Variables categóricas
col_estado = f"Estado_{estado}"
col_causa = f"Causa_{causa}"
col_vegetacion = f"Tipo_Vegetacion_{vegetacion}"

if col_estado in input_data.columns:
    input_data[col_estado] = 1

if col_causa in input_data.columns:
    input_data[col_causa] = 1

if col_vegetacion in input_data.columns:
    input_data[col_vegetacion] = 1

# =========================
# PREDICCION
# =========================

if st.button("Predecir severidad"):

    prediccion = modelo.predict(input_data)[0]

    st.success(f"Predicción del modelo: {prediccion}")

    if prediccion == 'Mayor a 100 Hectáreas':
        st.error('🔴 Riesgo EXTREMO')

    elif prediccion == '51 a 100 Hectáreas':
        st.warning('🟠 Riesgo ALTO')

    else:
        st.success('🟢 Riesgo MODERADO')


st.info("""
El modelo considera variables ambientales,
temporales y operativas para estimar
la severidad potencial del incendio.
""")
