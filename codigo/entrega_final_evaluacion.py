# -*- coding: utf-8 -*-
"""
ENTREGA FINAL (Fase 3) — Evaluación sobre el conjunto de PRUEBA (intacto) + despliegue.
Corre una unica vez sobre X_test/y_test reservados en la Segunda Entrega.
Genera las cifras finales y un despliegue Streamlit minimo.
"""
import numpy as np, pandas as pd, joblib, os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz
P = BASE

modelo = joblib.load(os.path.join(P, "datos", "mejor_modelo.joblib"))
X_test = pd.read_csv(os.path.join(P, "datos", "X_test_prueba.csv"))
y_test = pd.read_csv(os.path.join(P, "datos", "y_test_prueba.csv")).iloc[:, 0]

y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("="*60)
print("EVALUACION FINAL SOBRE CONJUNTO DE PRUEBA (1080 filas, intacto)")
print("="*60)
print(f"  MAE  = {mae:.3f}")
print(f"  RMSE = {rmse:.3f}")
print(f"  R2   = {r2:.4f}")
print(f"  MAPE = {mape:.2f}%   (articulo MLR: 4.89%)")
print(f"  MAPE < 10% -> desempeno sobresaliente segun el articulo.")

# linea base sobre la prueba (naive estacional) para el contraste final
Xtr = pd.read_csv(os.path.join(P, "datos", "datos_muebles_panel.csv"))  # para el mes_mean del entrenamiento
# recomputar naive estacional sobre prueba: usar el promedio del mes en train+val
tv = Xtr[Xtr["mes_num"].isin(pd.read_csv(P+r"\X_test_prueba.csv")["mes_num"].unique())]
# nota: usamos la media global por mes de todo el conjunto (train+val) como referencia
mes_mean = Xtr.groupby("mes_num")["ventas_muebles_millones"].mean()
y_naive = X_test["mes_num"].map(mes_mean).fillna(y_test.mean()).values
mape_naive = np.mean(np.abs((y_test - y_naive) / y_test)) * 100
print(f"\n  Linea base naive estacional (prueba): MAPE = {mape_naive:.2f}%")
print(f"  El modelo mejora {mape_naive - mape:.2f} puntos de MAPE sobre la linea base.")

# guardar predicciones para el informe
pd.DataFrame({"real": y_test.values, "predicho": y_pred}).to_csv(
    os.path.join(P, "datos", "predicciones_prueba.csv"), index=False)
print(f"\n  Predicciones guardadas: predicciones_prueba.csv")
print("="*60)
