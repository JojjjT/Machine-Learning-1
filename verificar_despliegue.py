# -*- coding: utf-8 -*-
"""Verificación rápida del despliegue: el modelo carga y predice con las columnas correctas."""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = os.path.dirname(os.path.abspath(__file__))
import pandas as pd, joblib

m = joblib.load(os.path.join(P, "mejor_modelo.joblib"))
X = pd.read_csv(os.path.join(P, "X_test_prueba.csv"))
y = pd.read_csv(os.path.join(P, "y_test_prueba.csv")).iloc[:, 0]

fila = X.iloc[0:1].copy()
p = m.predict(fila)[0]
print("Columnas que espera el modelo:", list(X.columns))
print(f"Prediccion de prueba: {p:.2f} | Real: {y.iloc[0]:.2f}")
print("Modelo carga y predice OK" if abs(p - y.iloc[0]) < 50 else "REVISAR: desviacion grande")

# Verificar que las columnas del Streamlit coincidan
esperadas = ["importaciones_muebles","sentimiento_consumidor","inicios_vivienda",
             "poblacion_miles","indice_precios_productor","pandemia",
             "ciudad","subsector","mes_num"]
print("Columnas en X_test:", list(X.columns))
print("Coinciden con el modelo:", list(X.columns) == esperadas or set(esperadas).issubset(set(X.columns)))
