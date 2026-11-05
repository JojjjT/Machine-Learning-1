# Proyecto Integrador — Machine Learning I (2026-2)

**Equipo:** Joshua Samad Trujillo Lagos & Juan Andrés Zambrano Mora
**Artículo base:** İnce, M. N. & Taşdemir, Ç. (2024). *Forecasting Retail Sales for Furniture and
Furnishing Items through the Employment of Multiple Linear Regression and Holt–Winters Models.*
Systems, 12(6), 219. https://doi.org/10.3390/systems12060219
**Traslado local:** Pronóstico de la demanda minorista de muebles en Colombia.

## 🌐 Sitio desplegado (GitHub Pages)

| Página | URL |
|---|---|
| **Inicio** — resumen, resultados, figuras e informes | **https://jojjjt.github.io/Machine-Learning-1/** |
| **App Streamlit completa** — interactiva, corriendo en el navegador (stlite) | **https://jojjjt.github.io/Machine-Learning-1/streamlit.html** |
| Predictor estático (versión ligera en JavaScript) | https://jojjjt.github.io/Machine-Learning-1/ |

Publicado por el workflow `.github/workflows/deploy.yml`: en cada push a `main` se ejecuta
Streamlit en CI (prueba funcional + health check) y, solo si pasa, se despliega el sitio.
Activación única: **Settings → Pages → Source: GitHub Actions**.

## Requisitos
- Python 3.12
- scikit-learn 1.9.1 · pandas · numpy · joblib · matplotlib · streamlit 1.66+
- Semilla fija: **42** (reproducibilidad total)

## Estructura del repositorio

```
├── index.html                → página de inicio (predictor en JavaScript)
├── streamlit.html            → app Streamlit desplegada en Pages (stlite/Pyodide)
├── app/                      → despliegue Streamlit (motor dual + modelo portable)
├── codigo/                   → scripts ejecutables (datos → modelo → figuras → evaluación)
├── datos/                    → dataset, modelo entrenado y particiones de prueba
├── figuras/                  → 3 figuras de evidencia
├── informes/                 → entregas APA 7 (PDF + DOCX)
├── pruebas/                  → tests (equivalencia de motores, AppTest, rutas web)
└── .github/workflows/        → deploy.yml (CI + Pages)
```

## Archivos y orden de ejecución

| # | Archivo | Qué hace |
|---|---|---|
| 1 | `codigo/generar_datos_muebles.py` | Genera el dataset simulado fundamentado (5.400 filas) → `datos/datos_muebles_panel.csv` |
| 2 | `codigo/segunda_entrega_modelado.py` | Partición 60/20/20, línea base **corrida**, pipeline sin fuga, CV 5-pliegues, selección del mejor modelo. Guarda `datos/mejor_modelo.joblib` + `datos/X/y_test_prueba.csv` |
| 3 | `codigo/generar_figuras.py` | Genera las 3 figuras de evidencia → `figuras/` |
| 4 | `codigo/entrega_final_evaluacion.py` | Evalúa el modelo final una vez sobre la **prueba intacta** → `datos/predicciones_prueba.csv` |
| 5 | `app/despliegue_streamlit.py` | Despliegue local: `streamlit run app/despliegue_streamlit.py` |
| 6 | `codigo/generar_index_web.py` | Regenera `index.html` y `app/modelo_web.json` tras cambios en el modelo |
| ✓ | `codigo/verificar_despliegue.py` | Verificación: el modelo carga, predice y sus columnas coinciden con las del despliegue |

## Informes (APA 7 — formato + citas + referencias) — carpeta `informes/`
- `informes/Segunda_Entrega_APA` — Fase 2: datos, modelamiento, validación (PDF + DOCX)
- `informes/Entrega_Final_APA` — Fase 3: modelo final, resultados, despliegue, límites (PDF + DOCX)

## Entregas del proyecto (20 % de la nota)
| Fase | Peso | Nota / Estado |
|---|---|---|
| Primera entrega (comprensión y planteamiento) | 20 % | 2,2 / 5,0 (retroalimentación recibida) |
| Segunda entrega (datos, modelamiento, validación) | 40 % | **Preparada** — informe APA + código + figuras |
| Entrega final (consolidación, despliegue, comunicación) | 40 % | **Preparada** — informe APA + despliegues |

## Correcciones aplicadas respecto a la Primera Entrega
1. **≥ 5.000 registros** → panel mes × ciudad × subsector (5.400 ≥ 5.000)
2. **Solo modelos del curso** → RL múltiple, Ridge, KNN (HW como referencia externa)
3. **Código ejecutable** → 6 scripts reproducibles con semilla 42
4. **Línea base construida y corrida** (no solo descrita) → MAPE 11,03 % de piso
5. **MAPE justificado por costo** (no por costumbre) + declaración de su asimetría

## Resultados clave
- **Prueba (1.080 filas):** MAPE 4,18 % · R² 0,8729 · MAE 4,286
- **Validación:** MAPE 4,08 % · R² 0,8833 (brecha con prueba: 0,10 pts → sin sobreajuste)
- **Línea base naïve estacional:** MAPE 10,67 % en prueba → el modelo mejora 6,50 puntos
- **Artículo (MLR):** MAPE 4,89 % · R² 0,7673 — resultado comparable

## Despliegue local (Streamlit)
```bash
streamlit run app/despliegue_streamlit.py
```
Recibe 9 entradas (5 numéricas, pandemia, ciudad, subsector, mes), aplica el mismo Pipeline
de preprocesamiento que el entrenamiento, y devuelve la demanda estimada en millones de USD.
Los rangos de cada campo se derivan del dataset de entrenamiento (coherencia entre datos de
uso y de entrenamiento).

## Despliegue web (GitHub Pages)
- **`index.html`** — página de inicio: resumen del proyecto, resultados, figuras de evidencia
  e informes enlazados. El predictor corre en JavaScript con el mismo Pipeline (estandarización
  + one-hot + regresión lineal); la equivalencia con Python está verificada en
  `pruebas/test_web.json` (6 filas, diferencia < 1e-13).
- **`streamlit.html`** — la **app Streamlit completa** corriendo en el navegador con
  stlite/Pyodide (sin servidor): mismos campos, mismos rangos y el mismo botón de predicción.
  Usa el motor portable (`app/modelo_web.json` + `app/modelo_portable.py`), verificado
  equivalente a scikit-learn (`pruebas/test_modelo_portable.py`, diferencia < 1e-13).

El workflow `.github/workflows/deploy.yml` **ejecuta Streamlit en CI antes de publicar**
(`streamlit-check`: prueba funcional con AppTest + health check del servidor real) y solo
despliega si pasa.

## Pruebas locales
```bash
python pruebas/test_modelo_portable.py      # motor portable == scikit-learn
python pruebas/test_despliegue_app.py       # app en modo local y modo navegador
python pruebas/verificar_rutas_web.py       # ningún enlace roto en index.html/streamlit.html
```
