# -*- coding: utf-8 -*-
"""
Despliegue basico (Fase 3, punto 8.3) - Streamlit.

Uso local:       streamlit run despliegue_streamlit.py
En el navegador: streamlit.html (stlite/Pyodide en GitHub Pages).

Motor de prediccion: scikit-learn (joblib) cuando esta disponible; si no
(entorno navegador sin scikit-learn), motor portable definido abajo con la
matematica identica (estandarizacion + one-hot + producto escalar), verificado
contra joblib en test_web.json (diferencia < 1e-13).
"""
import json
import os

import streamlit as st
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))  # carpeta app/
ROOT = os.path.dirname(BASE)  # raiz del proyecto


def _ruta(nombre):
    """Busca en app/, luego en datos/, luego en la raiz (local o FS virtual del navegador)."""
    for p in (os.path.join(BASE, nombre),
              os.path.join(ROOT, "datos", nombre),
              os.path.join(ROOT, nombre),
              nombre):
        if os.path.exists(p):
            return p
    return os.path.join(BASE, nombre)


NUM = ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda",
       "poblacion_miles", "indice_precios_productor"]
ETIQUETAS = {
    "importaciones_muebles": "Importaciones de muebles (USD miles)",
    "sentimiento_consumidor": "Índice de sentimiento del consumidor",
    "inicios_vivienda": "Inicios de vivienda (unidades)",
    "poblacion_miles": "Población (miles)",
    "indice_precios_productor": "Índice de precios al productor",
}

# --- Modelo: scikit-learn si esta disponible; si no, motor portable JSON ---
MODELO_M = None
modelo = None
try:
    import joblib
    modelo = joblib.load(_ruta("mejor_modelo.joblib"))
    motor = "scikit-learn (joblib)"
except Exception:
    motor = None

if modelo is None:
    MODELO_M = json.load(open(_ruta("modelo_web.json"), encoding="utf-8"))

    class ModeloPortable:
        """Estandariza, codifica one-hot y aplica la regresion lineal (sin sklearn)."""

        def predict(self, X):
            m = MODELO_M
            out = []
            for _, row in X.iterrows():
                f = []
                for c in m["numCols"]:
                    p = m["num"][c]
                    f.append((float(row[c]) - p["mean"]) / p["scale"])
                for c in m["catCols"]:
                    cats = m["cats"][c]
                    idx = cats.index(str(row[c])) if str(row[c]) in cats else -1
                    for i in range(len(cats)):
                        f.append(1.0 if i == idx else 0.0)
                y = m["intercept"]
                for i, v in enumerate(f):
                    y += m["coef"][i] * v
                out.append(y)
            return out

    modelo = ModeloPortable()
    motor = "motor portable JSON (equivalente verificado a scikit-learn)"

# --- Rangos derivados de los datos de entrenamiento (CSV o JSON portable) ---
try:
    datos = pd.read_csv(_ruta("datos_muebles_panel.csv"))
    rango = {c: (float(datos[c].min()), float(datos[c].max()), float(datos[c].median()))
             for c in NUM}
    ciudades = sorted(datos["ciudad"].unique())
    subsectores = sorted(datos["subsector"].unique())
except Exception:
    if MODELO_M is None:
        MODELO_M = json.load(open(_ruta("modelo_web.json"), encoding="utf-8"))
    r = MODELO_M["rango"]
    rango = {c: (r[c]["min"], r[c]["max"], r[c]["med"]) for c in NUM}
    ciudades = MODELO_M["ciudades"]
    subsectores = MODELO_M["subsectores"]

st.set_page_config(page_title="Pronóstico de ventas de muebles", layout="centered")
st.title("Pronóstico de demanda minorista de muebles")
st.caption("Proyecto Integrador ML1 — İnce & Taşdemir (2024), traslado local a Colombia. "
           f"Semilla 42. Motor: {motor}.")

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
    st.caption(f"Motor: {motor}. Misma transformación (escalamiento + codificación) "
               "que en el entrenamiento.")
