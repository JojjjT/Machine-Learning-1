# -*- coding: utf-8 -*-
"""
Despliegue básico (Fase 3, punto 8.3) — Streamlit.
Uso:  streamlit run despliegue_streamlit.py
Recibe las entradas, aplica el mismo preprocesamiento del Pipeline y devuelve la predicción.

Los rangos de cada entrada numérica se derivan del propio conjunto de entrenamiento
(datos_muebles_panel.csv), de modo que la aplicación es coherente con los datos de uso
y ningún valor propuesto queda fuera del dominio observado en el modelamiento.
"""
import streamlit as st
import pandas as pd, joblib, os

BASE = os.path.dirname(os.path.abspath(__file__))
modelo = joblib.load(os.path.join(BASE, "mejor_modelo.joblib"))

# --- Rangos y valores por defecto derivados del dataset de entrenamiento ---
datos = pd.read_csv(os.path.join(BASE, "datos_muebles_panel.csv"))
NUM = ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda",
       "poblacion_miles", "indice_precios_productor"]
ETIQUETAS = {
    "importaciones_muebles": "Importaciones de muebles (USD miles)",
    "sentimiento_consumidor": "Índice de sentimiento del consumidor",
    "inicios_vivienda": "Inicios de vivienda (unidades)",
    "poblacion_miles": "Población (miles)",
    "indice_precios_productor": "Índice de precios al productor",
}
rango = {c: (float(datos[c].min()), float(datos[c].max()), float(datos[c].median())) for c in NUM}
ciudades = sorted(datos["ciudad"].unique())
subsectores = sorted(datos["subsector"].unique())

st.set_page_config(page_title="Pronóstico de ventas de muebles", layout="centered")
st.title("Pronóstico de demanda minorista de muebles")
st.caption("Proyecto Integrador ML1 — İnce & Taşdemir (2024), traslado local a Colombia. Semilla 42.")

st.subheader("Entradas")
st.caption("Cada campo admite valores dentro del rango observado en los datos de entrenamiento "
           "(mínimo–máximo de la columna); el valor por defecto es su mediana.")
col1, col2 = st.columns(2)
valores = {}
with col1:
    for c in NUM[:3]:
        mn, mx, med = rango[c]
        valores[c] = st.number_input(ETIQUETAS[c], min_value=mn, max_value=mx, value=med,
                                     format="%.2f")
with col2:
    for c in NUM[3:]:
        mn, mx, med = rango[c]
        valores[c] = st.number_input(ETIQUETAS[c], min_value=mn, max_value=mx, value=med,
                                     format="%.2f")
    pandemia = st.selectbox("Pandemia (mar–may 2020)", [0, 1])

ciudad = st.selectbox("Ciudad", ciudades)
subsector = st.selectbox("Subsector", subsectores)
mes_num = st.slider("Mes (1-12)", 1, 12, 12)

entrada = pd.DataFrame([{
    "importaciones_muebles": valores["importaciones_muebles"],
    "sentimiento_consumidor": valores["sentimiento_consumidor"],
    "inicios_vivienda": valores["inicios_vivienda"],
    "poblacion_miles": valores["poblacion_miles"],
    "indice_precios_productor": valores["indice_precios_productor"],
    "pandemia": pandemia,
    "ciudad": ciudad, "subsector": subsector, "mes_num": mes_num,
}])

if st.button("Predecir"):
    y = modelo.predict(entrada)[0]
    st.success(f"Demanda estimada: **{y:,.2f} millones de USD**")
    st.caption("El modelo aplica el mismo Pipeline (escalamiento + codificación) que en el entrenamiento.")
