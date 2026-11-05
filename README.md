# Proyecto Integrador — Machine Learning I (2026-2)

**Equipo:** Joshua Samad Trujillo Lagos & Juan Andrés Zambrano Mora (Cancelación)
**Artículo base:** İnce, M. N. & Taşdemir, Ç. (2024). *Forecasting Retail Sales for Furniture and
Furnishing Items through the Employment of Multiple Linear Regression and Holt–Winters Models.*
Systems, 12(6), 219. https://doi.org/10.3390/systems12060219
**Traslado local:** Pronóstico de la demanda minorista de muebles en Colombia.

## Requisitos
- Python 3.12
- scikit-learn 1.9.1 · pandas · numpy · joblib · matplotlib · streamlit 1.66+
- Semilla fija: **42** (reproducibilidad total)

## Archivos y orden de ejecución

| # | Archivo | Qué hace |
|---|---|---|
| 1 | `generar_datos_muebles.py` | Genera el dataset simulado fundamentado (5.400 filas) → `datos_muebles_panel.csv` |
| 2 | `segunda_entrega_modelado.py` | Partición 60/20/20, línea base **corrida**, pipeline sin fuga, CV 5-pliegues, selección del mejor modelo. Guarda `mejor_modelo.joblib` + `X/y_test_prueba.csv` |
| 3 | `generar_figuras.py` | Genera las 3 figuras de evidencia (predicho vs. real, residuos, MAPE vs línea base) |
| 4 | `entrega_final_evaluacion.py` | Evalúa el modelo final una vez sobre la **prueba intacta** → `predicciones_prueba.csv` |
| 5 | `despliegue_streamlit.py` | Despliegue básico: `streamlit run despliegue_streamlit.py` |
| ✓ | `verificar_despliegue.py` | Verificación: el modelo carga, predice y sus columnas coinciden con las del despliegue |

## Informes (APA 7 — formato + citas + referencias)
- `Segunda_Entrega_APA` — Fase 2: datos, modelamiento, validación (PDF + DOCX)
- `Entrega_Final_APA` — Fase 3: modelo final, resultados, despliegue, límites (PDF + DOCX)

## Entregas del proyecto (20 % de la nota)
| Fase | Peso | Nota / Estado |
|---|---|---|
| Primera entrega (comprensión y planteamiento) | 20 % | 2,2 / 5,0 (retroalimentación recibida) |
| Segunda entrega (datos, modelamiento, validación) | 40 % | **Preparada** — informe APA + código + figuras |
| Entrega final (consolidación, despliegue, comunicación) | 40 % | **Preparada** — informe APA + Streamlit + evaluación en prueba |

## Correcciones aplicadas respecto a la Primera Entrega
1. **≥ 5.000 registros** → panel mes × ciudad × subsector (5.400 ≥ 5.000)
2. **Solo modelos del curso** → RL múltiple, Ridge, KNN (HW como referencia externa)
3. **Código ejecutable** → 5 scripts reproducibles con semilla 42
4. **Línea base construida y corrida** (no solo descrita) → MAPE 11,03 % de piso
5. **MAPE justificado por costo** (no por costumbre) + declaración de su asimetría

## Resultados clave
- **Prueba (1.080 filas):** MAPE 4,18 % · R² 0,8729 · MAE 4,286
- **Validación:** MAPE 4,08 % · R² 0,8833 (brecha con prueba: 0,10 pts → sin sobreajuste)
- **Línea base naïve estacional:** MAPE 10,67 % en prueba → el modelo mejora 6,50 puntos
- **Artículo (MLR):** MAPE 4,89 % · R² 0,7673 — resultado comparable

## Despliegue (Streamlit, local)
```bash
streamlit run despliegue_streamlit.py
```
Recibe 9 entradas (5 numéricas, pandemia, ciudad, subsector, mes), aplica el mismo Pipeline
de preprocesamiento que el entrenamiento, y devuelve la demanda estimada en millones de USD.

## Despliegue web (GitHub Pages)
El repositorio incluye `index.html`: versión estática del predictor donde el mismo Pipeline
(estandarización + one-hot + regresión lineal) se ejecuta en JavaScript. La equivalencia con
Python está verificada en `test_web.json` (6 filas de prueba, diferencia < 1e-13).

El workflow `.github/workflows/deploy.yml` publica el sitio en Pages. Activación:
**Settings → Pages → Source: GitHub Actions**. Cada push a `main` despliega automáticamente
en `https://<usuario>.github.io/<repo>/`.

Para regenerar `index.html` tras cambios en el modelo:
```bash
python generar_index_web.py
```
