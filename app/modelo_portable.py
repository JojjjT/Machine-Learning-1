# -*- coding: utf-8 -*-
"""
Motor portable del modelo: carga el modelo exportado a JSON (modelo_web.json) y predice
con la misma matemática que el Pipeline de scikit-learn:
    estandarizacion (media/desv) -> one-hot -> producto escalar + intercepto.
Se usa como alternativa a joblib/scikit-learn cuando no estan disponibles
(p. ej. ejecucion en el navegador con stlite/Pyodide en GitHub Pages).
Equivalencia verificada contra joblib en test_web.json (diferencia < 1e-13).
"""
import json, os


def cargar_modelo(ruta_json):
    """Carga modelo_web.json y devuelve el dict con num/coef/intercept/rango/cats."""
    with open(ruta_json, encoding="utf-8") as f:
        return json.load(f)


class ModeloPortable:
    """Interfaz minima compatible con sklearn: predict(DataFrame) -> lista de flotantes."""

    def __init__(self, m):
        self.m = m

    def predict(self, X):
        m = self.m
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
