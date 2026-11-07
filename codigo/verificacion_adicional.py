# -*- coding: utf-8 -*-
"""
Verificación adicional (Segunda entrega) — Proyecto Integrador ML1.
Reproduce las cifras y figuras del video:
  - correlaciones (fila a fila, por los 60 meses y por los 12 meses del calendario)
  - índice estacional recuperado de los datos frente al del artículo (Tabla 3)
  - líneas base, validación cruzada y validación (misma partición, semilla 42)
  - ajuste de hiperparámetros por validación cruzada: alfa de Ridge y k de KNN
  - importancia por permutación del mejor modelo en validación
  - validación temporal: entrena con 2019-2022 y valida con 2023
Uso (desde la raíz del repositorio):  python codigo/verificacion_adicional.py
Salidas: carpeta resultados_verificacion/ (resultados.json y CSV de las figuras).
Elaborado con apoyo de inteligencia artificial generativa (declarado).
"""
import json, sys, pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "datos" / "datos_muebles_panel.csv"
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "resultados_verificacion"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
df = pd.read_csv(DATA)
R = {}

OBJ = "ventas_muebles_millones"
NUM = ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda",
       "poblacion_miles", "indice_precios_productor"]
CAT = ["ciudad", "subsector", "mes_num"]
NUM_ALL = NUM + ["pandemia"]
X = df[NUM_ALL + CAT]; y = df[OBJ]

# ---------------- datos ----------------
R["filas"] = len(df); R["faltantes"] = int(df.isna().sum().sum())
R["meses"] = df["mes"].nunique(); R["ciudades"] = df["ciudad"].nunique(); R["subsectores"] = df["subsector"].nunique()
R["y_desc"] = df[OBJ].describe().round(3).to_dict()
R["corr_fila"] = {c: round(df[OBJ].corr(df[c]), 3) for c in ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda"]}
m60 = df.groupby("mes").mean(numeric_only=True)
R["corr_60meses"] = {c: round(m60[OBJ].corr(m60[c]), 3) for c in ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda"]}
m12 = df.groupby("mes_num").mean(numeric_only=True)
R["corr_12meses_calendario"] = {c: round(m12[OBJ].corr(m12[c]), 3) for c in ["importaciones_muebles", "sentimiento_consumidor", "inicios_vivienda"]}

# serie nacional mensual (promedio de las 90 filas de cada mes)
serie = m60[[OBJ]].reset_index(); serie["t"] = np.arange(1, 61)
serie[["t", OBJ]].to_csv(OUT / "serie_mensual.csv", index=False)
# índice estacional recuperado: razón a la media móvil centrada 2x12, sin los meses de pandemia
s = m60[OBJ].values
cma = pd.Series(s).rolling(12, center=True).mean().rolling(2).mean().shift(-1).values
ratio = s / cma
pand = m60["pandemia"].values > 0
mn = m60["mes_num"].values.astype(int)
idx = np.array([np.nanmean(ratio[(mn == k) & ~pand]) for k in range(1, 13)])
idx = idx / idx.mean()
paper = np.array([0.930, 0.899, 1.043, 0.971, 1.001, 0.993, 1.005, 1.036, 0.997, 1.022, 1.054, 1.051])
pd.DataFrame({"mes": range(1, 13), "paper": paper, "sim": idx.round(4)}).to_csv(OUT / "estacional.csv", index=False)
R["estacional_max_dif"] = round(float(np.max(np.abs(idx - paper))), 4)
R["estacional_corr"] = round(float(np.corrcoef(idx, paper)[0, 1]), 4)
R["caida_covid_abril"] = round(float(m60.loc["2020-04", OBJ] / m60.loc["2019-04", OBJ] - 1), 3)

# ---------------- partición (idéntica al repositorio) ----------------
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.20, random_state=SEED)
Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.25, random_state=SEED)
R["particion"] = [len(Xtr), len(Xva), len(Xte)]
# cuántos meses del calendario aparecen a la vez en entrenamiento y validación
R["meses_compartidos_tr_va"] = int(len(set(df.loc[Xtr.index, "mes"]) & set(df.loc[Xva.index, "mes"])))

def met(yt, yp):
    return dict(mae=round(mean_absolute_error(yt, yp), 3), rmse=round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
                r2=round(r2_score(yt, yp), 4), mape=round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 2))

R["base_media"] = met(yva, np.full(len(yva), ytr.mean()))
mes_mean = Xtr.assign(y=ytr).groupby("mes_num")["y"].mean()
R["base_estacional"] = met(yva, Xva["mes_num"].map(mes_mean).values)

pre = ColumnTransformer([("num", StandardScaler(), NUM_ALL), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
def mk(m): return Pipeline([("pre", pre), ("model", m)])
modelos = {"MLR": mk(LinearRegression()), "Ridge": mk(Ridge(alpha=1.0)), "KNN": mk(KNeighborsRegressor(n_neighbors=15))}
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
R["cv"] = {}; R["val"] = {}
for n, p in modelos.items():
    mape = -cross_val_score(p, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_percentage_error") * 100
    r2 = cross_val_score(p, Xtr, ytr, cv=cv, scoring="r2")
    mae = -cross_val_score(p, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_error")
    R["cv"][n] = dict(mape=round(mape.mean(), 2), mape_sd=round(mape.std(), 2), r2=round(r2.mean(), 4), r2_sd=round(r2.std(), 4),
                      mae=round(mae.mean(), 3), mae_sd=round(mae.std(), 3), mape_pliegues=[round(v, 2) for v in mape])
    p.fit(Xtr, ytr); R["val"][n] = met(yva, p.predict(Xva))

# ajuste de hiperparámetros con validación cruzada (lo que pide 7.3)
gr = GridSearchCV(mk(Ridge()), {"model__alpha": [0.01, 0.1, 1, 10, 100]}, cv=cv, scoring="neg_mean_absolute_percentage_error").fit(Xtr, ytr)
gk = GridSearchCV(mk(KNeighborsRegressor()), {"model__n_neighbors": [5, 10, 15, 25, 50]}, cv=cv, scoring="neg_mean_absolute_percentage_error").fit(Xtr, ytr)
R["grid_ridge"] = {str(a): round(-s * 100, 3) for a, s in zip(gr.cv_results_["param_model__alpha"], gr.cv_results_["mean_test_score"])}
R["grid_knn"] = {str(a): round(-s * 100, 3) for a, s in zip(gk.cv_results_["param_model__n_neighbors"], gk.cv_results_["mean_test_score"])}

# lectura del mejor modelo en validación
mlr = modelos["MLR"]
yp = mlr.predict(Xva)
pd.DataFrame({"real": yva.values.round(3), "pred": yp.round(3), "res": (yva.values - yp).round(3)}).to_csv(OUT / "val_pred.csv", index=False)
R["resid_media"] = round(float(np.mean(yva.values - yp)), 3)
R["frac_subpronostico"] = round(float(np.mean(yva.values > yp)), 3)
nombres = list(mlr.named_steps["pre"].get_feature_names_out())
coef = mlr.named_steps["model"].coef_
R["coef_num"] = {n.replace("num__", ""): round(float(c), 3) for n, c in zip(nombres, coef) if n.startswith("num__")}
cm = {int(float(n.split("_")[-1])): float(c) for n, c in zip(nombres, coef) if n.startswith("cat__mes_num")}
base = np.mean(list(cm.values()))
R["coef_mes_rel"] = {k: round(v - base, 2) for k, v in sorted(cm.items())}
pi = permutation_importance(mlr, Xva, yva, scoring="neg_mean_absolute_percentage_error", n_repeats=10, random_state=SEED)
imp = {c: (round(float(m) * 100, 2), round(float(s) * 100, 2)) for c, m, s in zip(X.columns, pi.importances_mean, pi.importances_std)}
R["perm_imp_mape_pts"] = dict(sorted(imp.items(), key=lambda kv: -kv[1][0]))
pd.DataFrame([(c, m, s) for c, (m, s) in R["perm_imp_mape_pts"].items()], columns=["var", "imp", "sd"]).to_csv(OUT / "importancia.csv", index=False)

# prueba extra: validación temporal (entrena 2019-2022, valida 2023)
tr = df["anio"] <= 2022; va = df["anio"] == 2023
p = mk(LinearRegression()).fit(X[tr], y[tr])
R["temporal_mlr"] = met(y[va], p.predict(X[va]))
mm = df[tr].groupby("mes_num")[OBJ].mean()
R["temporal_base_estacional"] = met(y[va], X[va]["mes_num"].map(mm).values)
pk = mk(KNeighborsRegressor(n_neighbors=15)).fit(X[tr], y[tr])
R["temporal_knn"] = met(y[va], pk.predict(X[va]))

(OUT / "resultados.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1, ensure_ascii=False))
