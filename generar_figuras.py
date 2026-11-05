# -*- coding: utf-8 -*-
"""Genera las figuras de evidencia para la Segunda Entrega (rúbrica: figuras interpretadas)."""
import numpy as np, pandas as pd, joblib, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

BASE = os.path.dirname(os.path.abspath(__file__))
P = BASE

modelo = joblib.load(P + r"\mejor_modelo.joblib")
X = pd.read_csv(P + r"\X_test_prueba.csv")
y = pd.read_csv(P + r"\y_test_prueba.csv").iloc[:, 0]
yhat = modelo.predict(X)

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# Figura 1: predicho vs real (dispersión + diagonal)
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y, yhat, s=8, alpha=0.5, edgecolors="none")
lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="perfecto (y = ŷ)")
ax.set_xlabel("Demanda real (millones USD)"); ax.set_ylabel("Demanda pronosticada")
ax.set_title("Predicho vs. real — conjunto de prueba (1.080 filas)")
ax.legend()
fig.tight_layout(); fig.savefig(P + r"\fig1_predicho_vs_real.png", dpi=150); plt.close(fig)

# Figura 2: residuos
resid = y - yhat
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(yhat, resid, s=8, alpha=0.5, edgecolors="none")
ax.axhline(0, color="r", lw=1.5)
ax.set_xlabel("Pronóstico (millones USD)"); ax.set_ylabel("Residuo (real − pronóstico)")
ax.set_title(f"Residuos — media {resid.mean():.3f}, |media| {np.abs(resid).mean():.3f} (sesgo ligero a subpronosticar)")
fig.tight_layout(); fig.savefig(P + r"\fig2_residuos.png", dpi=150); plt.close(fig)

# Figura 3: comparación MAPE modelo vs líneas base (validación y prueba)
fig, ax = plt.subplots(figsize=(6, 4))
nombres = ["Media histórica\n(base)", "Naïve estacional\n(base)", "RL múltiple\n(validación)", "RL múltiple\n(prueba)"]
mape = [12.15, 11.03, 4.08, 4.18]
colores = ["#999999", "#bbbbbb", "#2E5B9A", "#1B6B3A"]
barras = ax.bar(nombres, mape, color=colores)
ax.axhline(10, color="r", ls="--", lw=1, label="umbral 10 % («outstanding»)")
ax.bar_label(barras, fmt="%.2f%%", fontsize=9)
ax.set_ylabel("MAPE (%)"); ax.set_title("MAPE: modelo vs líneas base — cuanto más bajo, mejor")
ax.set_ylim(0, 14); ax.legend()
fig.tight_layout(); fig.savefig(P + r"\fig3_mape_comparacion.png", dpi=150); plt.close(fig)

print("Figuras generadas:")
for f in ["fig1_predicho_vs_real", "fig2_residuos", "fig3_mape_comparacion"]:
    print(f"  {f}.png")
print(f"\nCifras: R2={r2_score(y,yhat):.4f}  MAE={mean_absolute_error(y,yhat):.3f}  "
      f"MAPE={np.mean(np.abs((y-yhat)/y))*100:.2f}%")
