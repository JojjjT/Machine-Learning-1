# -*- coding: utf-8 -*-
"""
Figuras de evidencia de la Segunda Entrega (Fase 2), calculadas con el conjunto
de VALIDACIÓN. El conjunto de prueba sigue sellado hasta la Entrega Final.

Reproduce la misma partición 60/20/20 (semilla 42), el mismo Pipeline y los
mismos modelos de segunda_entrega_modelado.py.
Uso, desde la raíz del repositorio:  python codigo/generar_figuras_segunda.py
Salidas: figuras/segunda_fig1_predicho_vs_real.png, segunda_fig2_residuos.png,
         segunda_fig3_mape_validacion.png
Elaborado con apoyo de inteligencia artificial generativa (declarado).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
df = pd.read_csv(os.path.join(BASE, "datos", "datos_muebles_panel.csv"))
OBJ = "ventas_muebles_millones"
NUM_ALL = ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda",
           "poblacion_miles", "indice_precios_productor", "pandemia"]
CAT = ["ciudad", "subsector", "mes_num"]
X, y = df[NUM_ALL + CAT], df[OBJ]

# misma partición que segunda_entrega_modelado.py: la prueba no se usa aquí
X_tv, _, y_tv, _ = train_test_split(X, y, test_size=0.20, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.25, random_state=SEED)

def mape(a, b):
    return float(np.mean(np.abs((a - b) / a)) * 100)

def pipe(modelo):
    pre = ColumnTransformer([("num", StandardScaler(), NUM_ALL),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", modelo)])

modelos = {"RL múltiple": pipe(LinearRegression()), "Ridge (α=1)": pipe(Ridge(alpha=1.0)),
           "KNN (k=15)": pipe(KNeighborsRegressor(n_neighbors=15))}
res = {}
for n, m in modelos.items():
    m.fit(X_tr, y_tr)
    res[n] = mape(y_va.values, m.predict(X_va))
media = mape(y_va.values, np.full(len(y_va), y_tr.mean()))
mes_mean = X_tr.assign(y=y_tr).groupby("mes_num")["y"].mean()
estacional = mape(y_va.values, X_va["mes_num"].map(mes_mean).values)

yhat = modelos["RL múltiple"].predict(X_va)
resid = y_va.values - yhat
os.makedirs(os.path.join(BASE, "figuras"), exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# Figura 1: pronóstico frente a real (validación)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_va, yhat, s=8, alpha=0.5, edgecolors="none")
lims = [min(y_va.min(), yhat.min()), max(y_va.max(), yhat.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="perfecto (y = ŷ)")
ax.set_xlabel("Demanda real (millones USD)"); ax.set_ylabel("Demanda pronosticada")
ax.set_title("Predicho vs. real — validación (1.080 filas)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(BASE, "figuras", "segunda_fig1_predicho_vs_real.png"), dpi=150); plt.close(fig)

# Figura 2: residuos (validación)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(yhat, resid, s=8, alpha=0.5, edgecolors="none")
ax.axhline(0, color="r", lw=1.5)
ax.set_xlabel("Pronóstico (millones USD)"); ax.set_ylabel("Residuo (real − pronóstico)")
ax.set_title(f"Residuos en validación — media {resid.mean():.2f}")
fig.tight_layout(); fig.savefig(os.path.join(BASE, "figuras", "segunda_fig2_residuos.png"), dpi=150); plt.close(fig)

# Figura 3: MAPE en validación, líneas base y modelos
fig, ax = plt.subplots(figsize=(6, 4))
nombres = ["Media\nhistórica", "Naïve\nestacional", "RL\nmúltiple", "Ridge\n(α=1)", "KNN\n(k=15)"]
valores = [media, estacional, res["RL múltiple"], res["Ridge (α=1)"], res["KNN (k=15)"]]
colores = ["#999999", "#bbbbbb", "#2E5B9A", "#5B84C4", "#8FA9D6"]
barras = ax.bar(nombres, valores, color=colores)
ax.axhline(10, color="r", ls="--", lw=1, label="umbral 10 % («sobresaliente»)")
ax.bar_label(barras, fmt="%.2f%%", fontsize=9)
ax.set_ylabel("MAPE en validación (%)"); ax.set_title("MAPE en validación: líneas base frente a modelos")
ax.set_ylim(0, 14); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(BASE, "figuras", "segunda_fig3_mape_validacion.png"), dpi=150); plt.close(fig)

print(f"Validación: media {media:.2f} % | naïve estacional {estacional:.2f} % | "
      + " | ".join(f"{n} {v:.2f} %" for n, v in res.items()))
print(f"Residuos RL múltiple: media {resid.mean():.3f}; pronósticos por debajo de lo real: {np.mean(resid > 0):.1%}")
