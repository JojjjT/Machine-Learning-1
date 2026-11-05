# -*- coding: utf-8 -*-
"""
Genera index.html: versión estática de la aplicación de pronóstico (GitHub Pages).
El modelo (Pipeline: StandardScaler + OneHotEncoder + LinearRegression) se exporta a JSON
y se ejecuta en JavaScript en el navegador — misma predicción, sin servidor.
También emite test_web.json (filas de prueba con predicción en Python) para verificar
la equivalencia Python ↔ JavaScript.
"""
import sys, json, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, joblib

P = os.path.dirname(os.path.abspath(__file__))
datos = pd.read_csv(os.path.join(P, "datos_muebles_panel.csv"))

# ---- Extraer el Pipeline entrenado ----
pipe = joblib.load(os.path.join(P, "mejor_modelo.joblib"))
pre = pipe.named_steps["pre"]
lin = pipe.named_steps["model"]

# numéricas
_, scaler, num_cols = pre.transformers_[0]
num_cols = list(num_cols)
num_par = {c: {"mean": float(scaler.mean_[i]), "scale": float(scaler.scale_[i])}
           for i, c in enumerate(num_cols)}

# categóricas
_, ohe, cat_cols = pre.transformers_[1]
cat_cols = list(cat_cols)
cats = {c: [str(v) for v in ohe.categories_[i]] for i, c in enumerate(cat_cols)}

coef = [float(x) for x in lin.coef_]
intercept = float(lin.intercept_)
esperado = len(num_cols) + sum(len(v) for v in cats.values())
assert len(coef) == esperado, f"coef {len(coef)} != features {esperado}"

NUM = num_cols
ETIQUETAS = {
    "importaciones_muebles": "Importaciones de muebles (USD miles)",
    "sentimiento_consumidor": "Índice de sentimiento del consumidor",
    "inicios_vivienda": "Inicios de vivienda (unidades)",
    "poblacion_miles": "Población (miles)",
    "indice_precios_productor": "Índice de precios al productor",
    "pandemia": "Pandemia (mar-may 2020)",
}
rango = {c: {"min": float(datos[c].min()), "max": float(datos[c].max()),
             "med": float(datos[c].median())} for c in NUM}
ciudades = sorted(datos["ciudad"].unique().tolist())
subsectores = sorted(datos["subsector"].unique().tolist())

MODELO = {
    "numCols": NUM, "num": num_par, "catCols": cat_cols, "cats": cats,
    "coef": coef, "intercept": intercept, "rango": rango,
    "ciudades": ciudades, "subsectores": subsectores,
}

# ---- Filas de prueba Python ↔ JS ----
X = pd.read_csv(os.path.join(P, "X_test_prueba.csv"))
y = pipe.predict(X)
muestra = []
for i in [0, 7, 42, 100, 500, 999]:
    fila = {k: (float(v) if k in NUM else str(v)) for k, v in X.iloc[i].to_dict().items()}
    muestra.append({"entrada": fila, "y_py": float(y[i])})
json.dump(muestra, open(os.path.join(P, "test_web.json"), "w"), ensure_ascii=False, indent=1)

# ---- HTML ----
modelo_json = json.dumps(MODELO, ensure_ascii=False)
opt_ciudad = "\n".join(f'          <option value="{c}">{c}</option>' for c in ciudades)
opt_sub = "\n".join(f'          <option value="{s}">{s}</option>' for s in subsectores)
campos_num = ""
for etq, c in [("Importaciones de muebles (USD miles)","importaciones_muebles"),
               ("Índice de sentimiento del consumidor","sentimiento_consumidor"),
               ("Inicios de vivienda (unidades)","inicios_vivienda"),
               ("Población (miles)","poblacion_miles"),
               ("Índice de precios al productor","indice_precios_productor")]:
    r = rango[c]
    campos_num += f'''
        <div class="campo">
          <label for="{c}">{etq}</label>
          <input type="number" id="{c}" step="0.01"
                 min="{r['min']:.2f}" max="{r['max']:.2f}" value="{r['med']:.2f}">
          <small>Rango de entrenamiento: {r['min']:.1f} – {r['max']:.1f}</small>
        </div>'''

HTML = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronóstico de demanda minorista de muebles — Proyecto Integrador ML1</title>
<style>
  :root {{ --tinta:#1a1a1a; --azul:#1f3051; --acento:#2e5b9a; --linea:#d0d7e2; --fondo:#ffffff; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--fondo); color:var(--tinta);
         font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         line-height:1.55; font-size:16px; }}
  header {{ background:var(--azul); color:#fff; padding:28px 20px; }}
  header .cont {{ max-width:880px; margin:0 auto; }}
  header h1 {{ margin:0 0 6px; font-size:1.6rem; line-height:1.3; }}
  header p {{ margin:2px 0; opacity:.92; font-size:.95rem; }}
  main {{ max-width:880px; margin:0 auto; padding:24px 20px 48px; }}
  section {{ margin-bottom:34px; }}
  h2 {{ font-size:1.2rem; color:var(--azul); border-bottom:2px solid var(--linea);
        padding-bottom:6px; margin:0 0 14px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:14px; }}
  .campo label {{ display:block; font-weight:600; font-size:.9rem; margin-bottom:4px; }}
  .campo input, .campo select {{ width:100%; padding:9px 10px; font-size:1rem;
      border:1px solid var(--linea); border-radius:5px; background:#fff; }}
  .campo input:focus, .campo select:focus {{ outline:2px solid var(--acento); outline-offset:1px; }}
  .campo small {{ color:#555; font-size:.78rem; }}
  button {{ margin-top:16px; background:var(--acento); color:#fff; border:none;
      padding:12px 26px; font-size:1.02rem; border-radius:5px; cursor:pointer; min-height:44px; }}
  button:hover {{ background:#1f3f73; }}
  #resultado {{ margin-top:14px; padding:12px 14px; border:1px solid var(--linea);
      border-radius:5px; background:#f4f7fb; font-size:1.05rem; display:none; }}
  #resultado.error {{ background:#fdf0f0; border-color:#c98a8a; }}
  table {{ border-collapse:collapse; width:100%; font-size:.93rem; }}
  th, td {{ border:1px solid var(--linea); padding:8px 10px; text-align:left; }}
  th {{ background:#eef1f6; color:var(--azul); }}
  fig {{ display:block; margin:14px 0; }}
  fig img {{ max-width:100%; height:auto; border:1px solid var(--linea); border-radius:4px; }}
  fig figcaption {{ font-size:.85rem; color:#555; margin-top:5px; }}
  footer {{ border-top:1px solid var(--linea); padding:18px 20px; font-size:.87rem;
      color:#555; text-align:center; }}
  footer a {{ color:var(--acento); }}
</style>
</head>
<body>
<header>
  <div class="cont">
    <h1>Pronóstico de demanda minorista de muebles</h1>
    <p>Proyecto Integrador — Machine Learning I (2026-2) · Universidad Externado de Colombia</p>
    <p>Joshua Samad Trujillo Lagos y Juan Andrés Zambrano Mora</p>
    <p>Artículo base: İnce &amp; Taşdemir (2024), <em>Systems</em>, 12(6), 219 — regresión lineal múltiple aplicada a un panel mes × ciudad × subsector en Colombia.</p>
  </div>
</header>
<main>
  <section>
    <h2>Resultados del modelo</h2>
    <table>
      <tr><th>Métrica</th><th>Validación</th><th>Prueba (1.080 filas)</th></tr>
      <tr><td>MAPE (métrica principal)</td><td>4,08 %</td><td>4,18 %</td></tr>
      <tr><td>R²</td><td>0,8833</td><td>0,8729</td></tr>
      <tr><td>Línea base naïve estacional (MAPE)</td><td>11,03 %</td><td>10,67 %</td></tr>
    </table>
    <p style="font-size:.9rem;color:#555;">El modelo supera a la línea base en 6,50 puntos de MAPE y se
    mantiene bajo el umbral del 10 % que el artículo califica como desempeño sobresaliente.</p>
  </section>

  <section>
    <h2>Predictor interactivo</h2>
    <p style="font-size:.93rem;">Todos los campos admiten valores dentro del rango observado en los datos
    de entrenamiento; el valor por defecto es la mediana de cada columna. La predicción se calcula en el
    navegador con el mismo pipeline del entrenamiento (estandarización + codificación one-hot + regresión lineal).</p>
    <div class="grid">{campos_num}
      <div class="campo">
        <label for="pandemia">Pandemia (mar–may 2020)</label>
        <select id="pandemia"><option value="0">0 — sin pandemia</option><option value="1">1 — pandemia</option></select>
      </div>
      <div class="campo">
        <label for="ciudad">Ciudad</label>
        <select id="ciudad">
{opt_ciudad}
        </select>
      </div>
      <div class="campo">
        <label for="subsector">Subsector</label>
        <select id="subsector">
{opt_sub}
        </select>
      </div>
      <div class="campo">
        <label for="mes_num">Mes (1–12)</label>
        <input type="range" id="mes_num" min="1" max="12" step="1" value="12"
               oninput="document.getElementById('mes_txt').textContent=this.value">
        <small>Mes seleccionado: <strong id="mes_txt">12</strong> (diciembre, pico estacional)</small>
      </div>
    </div>
    <button id="btn" type="button">Predecir</button>
    <div id="resultado" role="status" aria-live="polite"></div>
  </section>

  <section>
    <h2>Evidencia gráfica</h2>
    <figure><img src="fig1_predicho_vs_real.png" alt="Dispersión de predicho contra real en la prueba">
      <figcaption>Figura 1. Pronóstico frente a valor real (prueba): puntos alrededor de la diagonal, sin sesgo apreciable.</figcaption></figure>
    <figure><img src="fig2_residuos.png" alt="Residuos contra pronóstico">
      <figcaption>Figura 2. Residuos centrados en cero, sin patrón estructurado.</figcaption></figure>
    <figure><img src="fig3_mape_comparacion.png" alt="Comparación del MAPE: modelo contra líneas base">
      <figcaption>Figura 3. MAPE: modelo (4,08 % / 4,18 %) frente a las líneas base (11–12 %).</figcaption></figure>
  </section>

  <section>
    <h2>Informes</h2>
    <p><a href="Segunda_Entrega_APA.pdf">Segunda Entrega (Fase 2) — PDF</a> ·
       <a href="Entrega_Final_APA.pdf">Entrega Final (Fase 3) — PDF</a></p>
  </section>
</main>
<footer>
  <p>Trabajo académico de Machine Learning I. Referencia: İnce, M. N., &amp; Taşdemir, Ç. (2024).
  <a href="https://doi.org/10.3390/systems12060219">doi.org/10.3390/systems12060219</a></p>
</footer>

<script id="modelo-datos" type="application/json">{modelo_json}</script>
<script>
// --MODELO CORE START--
const MODELO = JSON.parse(document.getElementById("modelo-datos").textContent);
function predecir(v) {{
  const f = [];
  for (const c of MODELO.numCols) {{
    const p = MODELO.num[c];
    f.push((Number(v[c]) - p.mean) / p.scale);
  }}
  for (const c of MODELO.catCols) {{
    const cats = MODELO.cats[c];
    const idx = cats.indexOf(String(v[c]));
    for (let i = 0; i < cats.length; i++) f.push(idx === -1 ? 0 : (i === idx ? 1 : 0));
  }}
  let y = MODELO.intercept;
  for (let i = 0; i < f.length; i++) y += MODELO.coef[i] * f[i];
  return y;
}}
// --MODELO CORE END--
document.getElementById("btn").addEventListener("click", () => {{
  try {{
    const v = {{}};
    for (const c of MODELO.numCols) v[c] = Number(document.getElementById(c).value);
    v.pandemia = Number(document.getElementById("pandemia").value);
    v.ciudad = document.getElementById("ciudad").value;
    v.subsector = document.getElementById("subsector").value;
    v.mes_num = Number(document.getElementById("mes_num").value);
    const y = predecir(v);
    const el = document.getElementById("resultado");
    el.className = "";
    el.style.display = "block";
    el.innerHTML = "<strong>Demanda estimada: " + y.toFixed(2) +
      " millones de USD</strong><br><small>Mes " + v.mes_num + " · " + v.ciudad +
      " · " + v.subsector + " · misma transformación que el entrenamiento (escala + one-hot).</small>";
  }} catch (e) {{
    const el = document.getElementById("resultado");
    el.className = "error"; el.style.display = "block";
    el.textContent = "Error: " + e.message;
  }}
}});
</script>
</body>
</html>
'''

open(os.path.join(P, "index.html"), "w", encoding="utf-8", newline="\n").write(HTML)
print(f"index.html generado ({os.path.getsize(os.path.join(P,'index.html'))/1024:.0f} KB)")
print(f"Modelo: {len(num_cols)} num + {sum(len(v) for v in cats.values())} one-hot = {esperado} features, intercept={intercept:.4f}")
print(f"test_web.json: {len(muestra)} filas de verificacion")
