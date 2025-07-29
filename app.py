import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Título y descripción
st.set_page_config(page_title="Detector de Fraude", layout="wide")
st.title("🕵️‍♀️ Detección de Fraude en Transacciones")
st.markdown("Carga los datos de una transacción para predecir si es fraudulenta.")

# Rutas a los archivos (con los nombres subidos a GitHub)
MODEL_PATH = "modelo_fraude_mejorado_3 (1).pkl"
SCALER_PATH = "scaler_model (2).pkl"
IMPUTER_PATH = "imputer_model (2).pkl"
NUMERIC_COLS_PATH = "numeric_cols (2).json"


# Cargar modelo y preprocesadores
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    with open(NUMERIC_COLS_PATH, "r") as f:
        numeric_cols = json.load(f)
    return model, scaler, imputer, numeric_cols


model, scaler, imputer, numeric_cols = load_artifacts()

# Formulario de entrada
with st.form("form_fraude"):
    st.subheader("🔢 Variables de entrada")
    user_input = {}
    for col in numeric_cols:
        user_input[col] = st.number_input(col, value=0.0)
    submitted = st.form_submit_button("📊 Predecir")

# Predicción
if submitted:
    X_input = pd.DataFrame([user_input])[numeric_cols]
    X_input_imputed = imputer.transform(X_input)
    X_input_scaled = scaler.transform(X_input_imputed)
    y_pred = model.predict(X_input_scaled)[0]
    y_prob = model.predict_proba(X_input_scaled)[0, 1]

    st.subheader("📌 Resultado:")
    if y_pred == 1:
        st.error(
            f"⚠️ La transacción es sospechosa de fraude (probabilidad: {y_prob:.2%})"
        )
    else:
        st.success(f"✅ Transacción legítima (probabilidad de fraude: {y_prob:.2%})")

    st.info(
        "Esta predicción fue realizada por un modelo RandomForest con heurística ponderada."
    )
