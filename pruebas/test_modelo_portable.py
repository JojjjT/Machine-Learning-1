# -*- coding: utf-8 -*-
"""Verifica que ModeloPortable (JSON) predice igual que scikit-learn (joblib)."""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd, joblib

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz del proyecto (carpeta pruebas/)
sys.path.insert(0, os.path.join(P, "app"))  # modelo_portable vive en app/
from modelo_portable import ModeloPortable, cargar_modelo

port = ModeloPortable(cargar_modelo(os.path.join(P, "app", "modelo_web.json")))
skl = joblib.load(os.path.join(P, "datos", "mejor_modelo.joblib"))
tests = json.load(open(os.path.join(P, "pruebas", "test_web.json"), encoding="utf-8"))

maxd = 0.0
for t in tests:
    df = pd.DataFrame([t["entrada"]])
    # dtypes correctos: mes_num debe ser entero (como en X_test) para el OneHot de sklearn
    df["mes_num"] = df["mes_num"].astype(int)
    df["pandemia"] = df["pandemia"].astype(float)
    yp = float(port.predict(df)[0])
    ys = float(skl.predict(df)[0])
    maxd = max(maxd, abs(yp - ys), abs(ys - t["y_py"]))
    assert abs(yp - t["y_py"]) < 1e-9, f"registro mismatch: {yp} vs {t['y_py']}"

print(f"portable vs scikit-learn vs registro: max diff = {maxd:.2e} -> {'EQUIVALENTE' if maxd < 1e-9 else 'FALLA'}")
