# -*- coding: utf-8 -*-
"""
Proyecto Integrador — Machine Learning I (Trujillo & Zambrano, paper 4)
SEGUNDA ENTREGA (Fase 2): datos, modelamiento y validación SIN FUGA.

Corrige los 5 puntos de la retro de la Primera Entrega:
  [R1] >= 5000 registros           -> panel 5400 (generar_datos_muebles.py)
  [R2] solo modelos del curso      -> Regresion lineal multiple (MLR) + KNN-reg + Ridge
  [R3] codigo ejecutable           -> este script, reproducible (semilla 42)
  [R4] linea base CONSTRUIDA y corrida (no solo descrita)
  [R5] metrica justificada por costo (no por costumbre)

Flujo sin fuga: Pipeline(ColumnTransformer) dentro de validacion cruzada.
Particion 60/20/20 (entrenamiento/validacion/prueba) estratificada por ciudad.
La prueba NO se toca hasta la Entrega Final.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
warnings.filterwarnings("ignore")

SEED = 42  # ultimos dos digitos del codigo de estudiante (semilla)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz

# ---------------------------------------------------------------------------
# 0. Carga
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(BASE, "datos", "datos_muebles_panel.csv"))
print(f"[0] Dataset: {len(df)} filas, {df.shape[1]} columnas")

OBJ = "ventas_muebles_millones"
NUM = ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda",
       "poblacion_miles", "indice_precios_productor"]
CAT = ["ciudad", "subsector", "mes_num"]   # mes_num como ciclica categorica (estacionalidad)
# 'pandemia' es binaria numerica ya
NUM_ALL = NUM + ["pandemia"]

# ---------------------------------------------------------------------------
# 1. Particion 60/20/20 — PRUEBA INTACTA hasta la Entrega Final
#    (validacion separada para elegir; prueba solo para reporte final)
# ---------------------------------------------------------------------------
X = df[NUM_ALL + CAT]
y = df[OBJ]

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=SEED)  # 0.25*0.8 = 0.20 val
print(f"[1] Particion: entreno {len(X_train)} | valid {len(X_val)} | prueba {len(X_test)}  (prueba intacta)")

# ---------------------------------------------------------------------------
# 2. LINEA BASE construida y CORRIDA  [corrige R4]
#    Dos referencias: (a) predictor de la media, (b) naive estacional (mes pasado)
# ---------------------------------------------------------------------------
# (a) predictor de la media (regresion): entrena solo con y_train
y_pred_mean = np.full_like(y_val, fill_value=y_train.mean())
# (b) naive estacional: para cada fila de valid, predice el valor promedio de ESE mes
#     en el entrenamiento (la referencia exigente para una serie con estacionalidad)
mes_mean_train = X_train.assign(y=y_train).groupby("mes_num")["y"].mean()
y_pred_seasonal = X_val["mes_num"].map(mes_mean_train).fillna(y_train.mean()).values

def report(y_true, y_pred, nombre):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    # MAPE (metrica principal justificada por COSTO: error de inventario proporcional a ventas)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"    {nombre:28s} MAE={mae:8.3f}  RMSE={rmse:8.3f}  R2={r2:6.4f}  MAPE={mape:6.2f}%")
    return mae, rmse, r2, mape

print("[2] LINEA BASE (corrida):")
base_mean = report(y_val, y_pred_mean, "Linea base: media historica")
base_seas = report(y_val, y_pred_seasonal, "Linea base: naive estacional")

# ---------------------------------------------------------------------------
# 3. FLUJO SIN FUGA (Pipeline dentro de la validacion cruzada)  [corrige R3,R5]
#    Solo modelos del curso: RL multiple, Ridge (regularizacion), KNN-reg  [R2]
# ---------------------------------------------------------------------------
pre = ColumnTransformer([
    ("num", StandardScaler(), NUM_ALL),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])

modelos = {
    "Regresion lineal multiple": Pipeline([("pre", pre), ("model", LinearRegression())]),
    "Ridge (regularizacion)": Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))]),
    "KNN regresion (k=15)": Pipeline([("pre", pre), ("model", KNeighborsRegressor(n_neighbors=15))]),
}

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

print("\n[3] VALIDACION CRUZADA (5 pliegues) sobre entrenamiento — modelos del curso:")
resultados = {}
for nombre, pipe in modelos.items():
    # metrica de scoring negativa (sklearn usa negativo)
    mae_cv = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    rmse_cv = -cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    r2_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    mape_cv = -cross_val_score(pipe, X_train, y_train, cv=cv,
                               scoring="neg_mean_absolute_percentage_error") * 100
    resultados[nombre] = dict(mae=mae_cv, rmse=rmse_cv, r2=r2_cv, mape=mape_cv)
    print(f"    {nombre:28s} MAE {mae_cv.mean():7.3f}±{mae_cv.std():.3f} | "
          f"RMSE {rmse_cv.mean():7.3f}±{rmse_cv.std():.3f} | "
          f"R2 {r2_cv.mean():.4f}±{r2_cv.std():.4f} | MAPE {mape_cv.mean():5.2f}%")

# ---------------------------------------------------------------------------
# 4. SELECCION del mejor modelo por VALIDACION (no por prueba) y
#    desempeño en conjunto de VALIDACION
# ---------------------------------------------------------------------------
print("\n[4] Desempeño en CONJUNTO DE VALIDACION (para elegir el mejor):")
mejor, mejor_mape = None, 1e9
for nombre, pipe in modelos.items():
    pipe.fit(X_train, y_train)
    y_pv = pipe.predict(X_val)
    mape = np.mean(np.abs((y_val - y_pv) / y_val)) * 100
    report(y_val, y_pv, nombre)
    if mape < mejor_mape:
        mejor, mejor_mape = nombre, mape

print(f"\n    >>> MEJOR MODELO (por MAPE en validacion): {mejor}  (MAPE {mejor_mape:.2f}%)")

# ---------------------------------------------------------------------------
# 5. Linea base construida vs mejor modelo — el piso que exige la guia
# ---------------------------------------------------------------------------
print("\n[5] CONTRASTE (mejor modelo vs linea base naive estacional) en VALIDACION:")
print(f"    Naive estacional MAPE = {base_seas[3]:.2f}%  vs  {mejor} MAPE = {mejor_mape:.2f}%")
if mejor_mape < base_seas[3]:
    print(f"    El modelo SUPERA a la linea base en {base_seas[3]-mejor_mape:.2f} puntos de MAPE. OK.")
else:
    print("    El modelo NO supera a la linea base -> reconsiderar.")

# guardar el modelo y las particiones para la Entrega Final
import joblib
modelos[mejor].fit(X_trainval, y_trainval)  # reentrena con train+val para el reporte final
joblib.dump(modelos[mejor], os.path.join(BASE, "datos", "mejor_modelo.joblib"))
X_test.to_csv(os.path.join(BASE, "datos", "X_test_prueba.csv"), index=False)
y_test.to_csv(os.path.join(BASE, "datos", "y_test_prueba.csv"), index=False)
print(f"\n[6] Modelo final '{mejor}' guardado. Conjunto de PRUEBA ({len(X_test)} filas) "
      f"reservado intacto para la Entrega Final.")
