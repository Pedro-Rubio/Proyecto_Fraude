# Detección de Fraude en Transacciones — MVP de Ciencia de Datos

> **Objetivo:** construir un sistema reproducible de detección de fraude que va desde la exploración y limpieza de datos, pasando por entrenamiento y validación sin fuga de información, hasta el **despliegue operativo** en una app de **Streamlit** con una política de **triage** (ALTO_RIESGO / REVISAR / OK) y tableros analíticos en **Tableau/Power BI**.

---

## 🧭 Tabla de contenidos

- [Contexto del negocio](#contexto-del-negocio)
- [Arquitectura general](#arquitectura-general)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Entorno](#entorno)
- [Datos](#datos)
- [Pipeline de features y entrenamiento](#pipeline-de-features-y-entrenamiento)
- [Prevención de fuga de información](#prevención-de-fuga-de-información)
- [Métricas y resultados](#métricas-y-resultados)
- [Política de triage (MVP operativo)](#política-de-triage-mvp-operativo)
- [App Streamlit (scoring por lote)](#app-streamlit-scoring-por-lote)
- [Dashboards BI (Tableau/Power BI)](#dashboards-bi-tableaupower-bi)
- [Pruebas unitarias](#pruebas-unitarias)
- [Cómo replicar](#cómo-replicar)
- [Hoja de ruta / Mejoras futuras](#hoja-de-ruta--mejoras-futuras)
- [Responsabilidad y limitaciones](#responsabilidad-y-limitaciones)
- [Autor](#autor)

---

## 💼 Contexto del negocio

Los fraudes con tarjeta impactan directamente en pérdidas financieras y experiencia del cliente. Este MVP prioriza:

1. **Detectar casos de alto riesgo** con **precisión mínima garantizada** (para evitar bloqueos injustificados).
2. **Maximizar recall** bajo capacidad humana limitada mediante una bandeja **REVISAR** ordenada por **pérdida esperada** (*expected_loss = prob*fraude × amount*).
3. Entregar **insights ejecutables** (ciudades/giros de mayor incidencia, bandas de monto, horarios) para prevención y monitoreo.

---

## 🏗 Arquitectura general

- **Ingesta/Limpieza:** `cards_data.csv`, `transactions_data.csv`, `users_data.csv` → `dataset_final_limpio.csv`.
- **Features & Split:** ingeniería de variables con **GroupKFold** por cliente y corte temporal (evita fuga).
- **Modelos:** *Logistic Regression* + *RandomForest* (baselines) y opción *LightGBM* (si está disponible).
- **Selección & Umbral:** métrica objetivo **PR AUC** y ajuste de umbral por **coste de errores** o **precisión mínima**.
- **Despliegue:** app **Streamlit** para scoring por lote (CSV), con **triage** y descarga de resultados.
- **BI:** dashboard en **Tableau/Power BI/Looker Studio** con KPIs, serie temporal, top ciudades/giros y mapa.

---

## 📁 Estructura del repositorio

```text
proyecto_fraude/
├─ app.py                                    # App Streamlit (MVP operativo)
├─ requirements.txt                          # Dependencias (pinneadas a Py 3.11 / sklearn 1.4)
├─ runtime.txt                               # python-3.11 (Streamlit Cloud)
├─ sample_data/
│  └─ sample_10_transactions.csv             # CSV de prueba para la app
├─ artifacts_noleak_v1/                      # Artefactos para producción
│  ├─ best_so_far.pkl                        # Pipeline (prep + modelo)
│  ├─ triage_thresholds.json                 # Umbrales/parametría por defecto del MVP
├─ notebooks/
│  ├─ Proyecto_Data_Fraude_Depurado.ipynb    # Exploración + limpieza + carga + Desarrollo de ML
├─ Data Visualizacion/
│  ├─ Proyecto_Power_ BI                     # Visualizaciones desarrolladas en Microsoft Power BI
│  ├─ Proyecto_Looker_Studio                 # Visualizaciones desarrolladas en Looker Studio
└─ README.md
```

---

## 🧰 Entorno

Recomendado: **Python 3.11** (coherente con el pickle de sklearn).

`runtime.txt`
```
python-3.11
```

`requirements.txt`
```
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
joblib==1.4.2
matplotlib==3.8.4
```

> ⚠️ Usar otra versión de `scikit-learn` puede romper la carga del `best_so_far.pkl`.

---

## 🗂 Datos

- **transactions_data.csv**: transacciones (amount, mcc, merchant_city/state, zip, hour/day, use_chip, etc.)
- **cards_data.csv**: atributos de tarjeta (card_brand/tipo, fechas de emisión/expiración, chip, dark web flag, etc.)
- **users_data.csv**: atributos del cliente (edad, género, ingresos aproximados o per cápita del zip, etc.)

🧽 **Limpieza (resumen):**

- Tipos coherentes (`date` a datetime, `amount` a float, ids a int/string).
- Tratamiento de nulos (imputación por mediana/moda según tipo).
- Eliminación de columnas con >50% nulos y alta cardinalidad textual sin valor.
- Features de **tiempo** (año, mes, hora, día semana), **riesgo de monto**, **frecuencias por cliente**, etc.

Salida: **`dataset_final_limpio.csv`**.

---

## 🔧 Pipeline de features y entrenamiento

- **Preprocesamiento** vía `ColumnTransformer`:
  - Numéricas: `SimpleImputer(median)` + `StandardScaler`.
  - Categóricas: `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')`.
- **Split sin fuga**:
  - **GroupKFold** por `client_id` (evita que el mismo cliente esté en train y test).
  - **Corte temporal** (train: fechas anteriores, test: posteriores) para simular producción.
- **Baselines**:
  - `LogisticRegression(class_weight='balanced')`
  - `RandomForestClassifier(class_weight='balanced')`
- **Selección** por **PR AUC** (promedio en CV).
- **Umbrales**:
  - Por coste (`C_FP`, `C_FN`) **o** por precisión mínima (ej. 0.80) maximizando recall.

Artefactos clave: **`artifacts_noleak_v1/best_so_far.pkl`** y **`triage_thresholds.json`**.

---

## 🧪 Prevención de fuga de información

- No se usan variables derivadas de la etiqueta.
- Split por **cliente** y **tiempo**.
- Imputación/escala/OHE ajustados **solo** en train (vía pipeline).
- Evaluación en **holdout** verdaderamente out-of-sample.

---

## 📊 Métricas y resultados

- **PR AUC** (objetivo) y **ROC AUC**.
- Reporte de clasificación a umbral óptimo (coste o precisión mínima).
- Ejemplo reciente (dataset de demostración muy desbalanceado):
  - PR AUC (holdout) ≈ **0.99**
  - ROC AUC (holdout) ≈ **1.00**
  - Umbral seleccionado para **Precision ≥ 0.60** con **Recall ≈ 1.00** en **ALTO_RIESGO**.

> ⚠️ En datasets sintéticos o con reglas “fáciles”, las métricas pueden ser extremadamente altas. En producción, se espera descenso y se ajusta umbral/capacidad de revisión.

---

## 🧭 Política de triage (MVP operativo)

1. **ALTO_RIESGO:** probabilidad ≥ `thr_high` (por defecto 0.99 o se re-estima con `is_fraud` del CSV).
2. **REVISAR:** Top-N casos (configurable) por **pérdida esperada** `prob * amount`.
3. **OK:** resto.

Archivos de configuración (editables):
- `artifacts_noleak_v1/triage_thresholds.json`:
  ```json
  {
    "thr_high": 0.99,
    "min_precision_high": 0.80,
    "review_capacity": 200
  }
  ```

---

## 🌐 App Streamlit (scoring por lote)

**Funcionalidad:**
- Subir CSV con columnas esperadas por el pipeline.
- Scoring por lote y asignación de triage.
- Descarga del CSV con `score`, `expected_loss` y `triage`.
- Si el CSV trae `is_fraud`, la app muestra métricas (PR AUC, ROC AUC, reporte y matriz).
- Panel de info del modelo: columnas num/cat y (si aplica) importancias.

**Ejecución local:**
```bash
# 1) Crear venv (opcional)
python -m venv .venv && source .venv/bin/activate  # (Win: .venv\Scripts\activate)

# 2) Instalar deps
pip install -r requirements.txt

# 3) Ejecutar la app
streamlit run app.py
```

**Despliegue en Streamlit Cloud:**
- Asegura `runtime.txt` con **python-3.11**.
- `requirements.txt` con versiones pinneadas.
- Sube `artifacts_noleak_v1/best_so_far.pkl` y `triage_thresholds.json` al repo.

---

## 📈 Dashboards BI (Tableau/Power BI/ Looker Studio)

**Tablero Resumen Ejecutivo:**
- KPIs: **Total Transacciones**, **Total Fraudes**, **Tasa de Fraude**, **Monto Total**.
- Serie temporal por **mes** (fraudes y/o transacciones).
- **Top 10** ciudades/giros por fraudes (y monto defraudado).
- **Mapa** (burbuja por ciudad o coordenadas).
- Filtros globales: **rango de fecha**, **estado/país**, **tipo de tarjeta**.

> En Tableau/Power BI, `amount` → numérico; crear **banda de monto**; derivar **Fecha Mes** con `DATETRUNC('month', date)` (Tableau) o `EOMONTH` (DAX).

---

## ✅ Pruebas unitarias

`tests/test_model.py`
- **Forma de salida:** para input `n×p` devuelve `n` scores/clases.
- **Rango de predict_proba:** `[0,1]`.
- **Dominio de etiquetas:** `{0,1}`.

Ejemplo:
```python
import joblib, numpy as np

def test_proba_output_range():
    pipe = joblib.load("artifacts_noleak_v1/best_so_far.pkl")
    X = np.random.rand(5, 27)  # adapta p a tus columnas numéricas
    y_prob = pipe.predict_proba(X)[:, 1]
    assert np.all((0 <= y_prob) & (y_prob <= 1))
```
Ejecución:
```bash
pytest -q
```

---

## 🔁 Cómo replicar

1. **Preparar datos**
   - Coloca tus CSV en `data/` o ajusta paths en los notebooks.
   - Ejecuta `notebooks/01_eda_y_limpieza.ipynb` ⇒ `dataset_final_limpio.csv`.

2. **Features + entrenamiento**
   - Ejecuta `notebooks/03_training_noleak.ipynb` para entrenar con **GroupKFold + corte temporal**.
   - El notebook guarda `best_so_far.pkl` y métricas en `artifacts_noleak_v1/`.

3. **App local**
   - `pip install -r requirements.txt`
   - `streamlit run app.py`

4. **App nube**
   - Sube repo a GitHub (incluyendo `artifacts_noleak_v1/`).
   - Deploy en Streamlit Cloud.

---

## 🗺 Hoja de ruta / Mejoras futuras

- **Más modelos:** calibración, LightGBM/XGBoost con *early stopping*.
- **Detección secuencial:** features basadas en ventanas por cliente/merchant.
- **Monitorización en producción:** drift, performance y umbrales adaptativos.
- **Explainability:** SHAP/Permutación para explicar decisiones.
- **Integración tiempo real:** API para scoring online + colas (Kafka/Redis).
- **Riesgo/Gobernanza:** auditoría, bitácoras y *model cards* ampliadas.

---

## ⚖️ Responsabilidad y limitaciones

- Datos **sintéticos o anonimizados** para portafolio.  
- Métricas altas pueden deberse a distribuciones favorables; en producción se espera ruido, *concept drift* y costo de etiquetado.  
- Evitar sesgos: revisar disparidad por segmentos (género/zipcode/edad) antes de liberar decisiones automatizadas.

---

## 👤 Autor

**Ezequiel Gonzalez** — Data Science & Analytics 
- *ezequiel.gonzalez08a@gmail.com*
**Pedro Rubio**  

- App Streamlit *https://proyectofraude-hyd2ycaphdnqqbeo87scer.streamlit.app/*, BI con Tableau/Power BI.
- Notebook/Google Colab: *https://colab.research.google.com/drive/1_Ed65bITdC714VqEDTFk9ouYxYGiwDoL?usp=sharing*
- Drive/Datasets/Notebook & Artefactos *https://drive.google.com/drive/folders/1NkZ6kv_qt_HE2uL2GLREaLmqH1jvzBQY?usp=sharing*
- Contacto: *srdelosdatos@gmail.com* — *www.linkedin.com/in/srdelosdatos* — 
