# -*- coding: utf-8 -*-
"""
Prueba funcional de despliegue_streamlit.py con AppTest (streamlit.testing.v1).
Escenario A: entorno local (joblib + CSV disponibles).
Escenario B: simulacion del navegador (joblib bloqueado + CSV ausente) -> motor portable JSON.
Ambos escenarios deben producir la prediccion sin excepcion.
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # raiz del proyecto (carpeta pruebas/)
os.chdir(P)
from streamlit.testing.v1 import AppTest

def run_app():
    at = AppTest.from_file(os.path.join(P, "app", "despliegue_streamlit.py"), default_timeout=60)
    at.run()
    assert not at.exception, f"excepcion al arrancar: {at.exception}"
    at.button[0].click().run()
    assert not at.exception, f"excepcion al predecir: {at.exception}"
    assert len(at.success) >= 1, "no hubo bloque de exito con la prediccion"
    return at

# --- Escenario A: local ---
atA = run_app()
print(f"[A local]       {atA.success[0].value}")

# --- Escenario B: navegador (joblib bloqueado, CSV ausente) ---
import builtins, pandas as pd, importlib
_real_import = builtins.__import__
def _sin_joblib(name, *a, **k):
    if name == "joblib":
        raise ImportError("joblib no disponible (simulacion navegador)")
    return _real_import(name, *a, **k)
_real_read_csv = pd.read_csv
def _sin_csv(path, *a, **k):
    if "datos_muebles" in str(path):
        raise FileNotFoundError(path)
    return _real_read_csv(path, *a, **k)

builtins.__import__ = _sin_joblib
pd.read_csv = _sin_csv
sys.modules.pop("despliegue_streamlit", None)
try:
    atB = run_app()
    print(f"[B navegador]   {atB.success[0].value}")
    caps = [c.value for c in atB.caption]
    motor_browser = next((c for c in caps if "Motor" in c), "")
    assert "portable" in motor_browser, f"no uso el motor portable: {motor_browser}"
    print(f"[B motor]       {motor_browser[:110]}")
finally:
    builtins.__import__ = _real_import
    pd.read_csv = _real_read_csv

print("TEST APP: OK (local + navegador)")
