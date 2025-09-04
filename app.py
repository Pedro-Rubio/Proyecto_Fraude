# app.py

import os, io, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ─────────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────────
st.set_page_config(page_title="Fraude • MVP", layout="wide")
ART_DIR = "artifacts_noleak_v1"
PIPE_PATH = os.path.join(ART_DIR, "best_so_far.pkl")
THR_PATH  = os.path.join(ART_DIR, "triage_mvp_thresholds.json")

# ─────────────────────────────────────────────
# Utilidades de carga / esquema
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline(path: str):
    with open(path, "rb") as f:
        return joblib.load(f)

def _find_column_transformer(pipeline):
    """Devuelve el ColumnTransformer del pipeline (busca el step 'prep' o el primero que sea ColumnTransformer)."""
    ct = None
    if hasattr(pipeline, "named_steps") and "prep" in pipeline.named_steps:
        ct = pipeline.named_steps["prep"]
    else:
        # Búsqueda por tipo
        from sklearn.compose import ColumnTransformer
        for _, step in getattr(pipeline, "steps", []):
            if isinstance(step, ColumnTransformer):
                ct = step
                break
    return ct

def get_training_schema(pipeline) -> Tuple[List[str], List[str], List[str]]:
    """
    Obtiene columnas numéricas y categóricas (lista de nombres) desde el ColumnTransformer.
    Retorna (num_cols, cat_cols, required_cols).
    """
    ct = _find_column_transformer(pipeline)
    if ct is None:
        return [], [], []

    # Intenta recuperar por nombre de transformador
    num_cols, cat_cols = [], []
    try:
        if "num" in ct.named_transformers_:
            num_cols = list(ct.named_transformers_["num"][2] if isinstance(ct.named_transformers_["num"], tuple) else ct.transformers_[0][2])
        if "cat" in ct.named_transformers_:
            cat_cols = list(ct.named_transformers_["cat"][2] if isinstance(ct.named_transformers_["cat"], tuple) else ct.transformers_[1][2])
    except Exception:
        # Fallback genérico a ct.transformers_
        try:
            num_cols = list(ct.transformers_[0][2])
            cat_cols = list(ct.transformers_[1][2])
        except Exception:
            pass

    required_cols = list(num_cols) + list(cat_cols)
    return num_cols, cat_cols, required_cols

def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def align_to_schema(df_in: pd.DataFrame, num_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """
    Crea las columnas faltantes, ordena y castea tipos
    para que coincidan con lo que espera el ColumnTransformer.
    """
    df = df_in.copy()

    # Crear columnas faltantes
    for c in num_cols + cat_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Casteo por tipo
    for c in num_cols:
        df[c] = _safe_to_numeric(df[c])
    for c in cat_cols:
        # Mantener string; los NaN los maneja el imputer del pipeline
        df[c] = df[c].astype("string")

    # Orden final (extras quedan fuera; el CT con remainder='drop' los ignoraría igual)
    return df[num_cols + cat_cols]

def threshold_for_min_precision(y_true, y_prob, target_precision=0.8) -> Optional[float]:
    """
    Devuelve el umbral más alto que cumpla la precisión objetivo.
    Si no encuentra ninguno, retorna None.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    mask = prec[:-1] >= target_precision
    if not mask.any():
        return None
    idxs = np.where(mask)[0]
    best_idx = idxs[np.argmax(rec[:-1][mask])]
    return float(thr[best_idx])

def build_download(df: pd.DataFrame, filename: str) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

# ─────────────────────────────────────────────
# Cargar artefactos (pipeline + thresholds)
# ─────────────────────────────────────────────
if not os.path.exists(PIPE_PATH):
    st.error(f"No se encontró el pipeline en `{PIPE_PATH}`. Sube tus artefactos al repo.")
    st.stop()

pipeline = load_pipeline(PIPE_PATH)
num_cols, cat_cols, required_cols = get_training_schema(pipeline)

cfg_default = {"thr_high": 0.99, "min_precision_high": 0.80, "review_capacity": 200}
if os.path.exists(THR_PATH):
    try:
        cfg_default.update(json.load(open(THR_PATH)))
    except Exception:
        pass

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
st.sidebar.write("Artefacto:", f"`{PIPE_PATH}`")
with st.sidebar.expander("Parámetros MVP", expanded=True):
    min_prec_high = st.slider("Precisión mínima (ALTO_RIESGO)", 0.50, 0.99, float(cfg_default["min_precision_high"]), 0.01)
    review_capacity = st.number_input("Capacidad de revisión (REVISAR)", min_value=0, step=10, value=int(cfg_default["review_capacity"]))
    allow_reestimate_thr = st.checkbox("Reestimar umbral si el CSV trae `is_fraud`", value=True)
st.sidebar.caption("Política: bloqueo por precisión mínima + revisión Top-N por pérdida esperada (prob * amount).")

# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────
st.title("🔎 Detección de Fraude — MVP Operativo")
tab1, tab2, tab3 = st.tabs(["📦 Scoring por lote (CSV)", "📈 Métricas (si hay is_fraud)", "ℹ️ Info del modelo"])

# ============ TAB 1: Scoring por lote ======================
with tab1:
    st.subheader("📦 Cargar CSV para scoring")
    st.write("El CSV debe contener como mínimo las columnas que el modelo espera.")
    st.code(", ".join(required_cols) if required_cols else "El pipeline no expone ColumnTransformer.", language="markdown")

    file = st.file_uploader("Sube un CSV de transacciones", type=["csv"])
    if file is None:
        st.info("Opcional: prueba con un ejemplo pequeño en `sample_data/sample_10_transactions.csv` si existe.")
        demo_path = os.path.join("sample_data", "sample_10_transactions.csv")
        if os.path.exists(demo_path):
            df_demo = pd.read_csv(demo_path)
            st.dataframe(df_demo.head(10), use_container_width=True)
    else:
        try:
            df_raw = pd.read_csv(file, low_memory=False)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()

        st.success(f"CSV cargado: {len(df_raw):,} filas")
        st.dataframe(df_raw.head(50), use_container_width=True)

        # Alinear al esquema de entrenamiento
        if required_cols:
            df_infer = align_to_schema(df_raw, num_cols, cat_cols)
        else:
            df_infer = df_raw.copy()

        # Scoring robusto (algunos estimadores no implementan predict_proba)
        try:
            y_prob = pipeline.predict_proba(df_infer)[:, 1]
        except Exception:
            # fallback: decision_function escalado a [0,1]
            score = pipeline.decision_function(df_infer)
            y_prob = (score - score.min()) / (score.max() - score.min() + 1e-12)

        scored = df_raw.copy()
        scored["score"] = y_prob

        # Pérdida esperada
        if "amount" in scored.columns:
            scored["expected_loss"] = scored["score"] * scored["amount"].fillna(0)
        else:
            scored["expected_loss"] = scored["score"] * 0.0

        # Umbral ALTO_RIESGO
        thr_high = cfg_default.get("thr_high", 0.99)
        if allow_reestimate_thr and "is_fraud" in scored.columns:
            try:
                thr_est = threshold_for_min_precision(
                    scored["is_fraud"].astype(int).values,
                    scored["score"].values,
                    target_precision=min_prec_high
                )
                if thr_est is not None:
                    thr_high = thr_est
            except Exception:
                pass

        # Triage: ALTO_RIESGO + REVISAR (Top-N por expected_loss) + OK
        high_mask = scored["score"].values >= thr_high
        triage = np.array(["OK"] * len(scored), dtype=object)
        triage[np.where(high_mask)[0]] = "ALTO_RIESGO"

        rem_mask = ~high_mask
        order_low = np.argsort(scored.loc[rem_mask, "expected_loss"].values)[::-1]
        review_take = min(review_capacity, order_low.shape[0])
        if review_take > 0:
            low_idx = np.where(rem_mask)[0]
            review_idx = low_idx[order_low[:review_take]]
            triage[review_idx] = "REVISAR"
        scored["triage"] = triage

        # Guardar en sesión para pestaña de métricas
        st.session_state["scored_df"] = scored
        st.session_state["thr_high"] = float(thr_high)

        # Resumen por banda
        def band_report(df_band):
            n = len(df_band)
            if "is_fraud" in df_band.columns:
                tp = int(df_band["is_fraud"].sum())
                fp = n - tp
                prec = tp/n if n > 0 else 0.0
                return n, tp, fp, prec
            return n, None, None, None

        col1, col2, col3 = st.columns(3)
        for band, col in zip(["ALTO_RIESGO", "REVISAR", "OK"], [col1, col2, col3]):
            subset = scored[scored["triage"] == band]
            n, tp, fp, prec = band_report(subset)
            with col:
                st.metric(band, f"{n:,} casos", help="TP/FP solo si el CSV trae is_fraud")
                if tp is not None:
                    st.caption(f"TP={tp} | FP={fp} | Prec={prec:.3f}")

        st.download_button(
            "⬇️ Descargar CSV con scores y triage",
            data=build_download(scored, "scored_with_triage.csv"),
            file_name="scored_with_triage.csv",
            mime="text/csv"
        )

        # Vista rápida: top por score
        st.write("🔎 Top 50 por score")
        st.dataframe(scored.sort_values("score", ascending=False).head(50), use_container_width=True)

# ============ TAB 2: Métricas (si hay is_fraud) ============
with tab2:
    st.subheader("📈 Métricas")
    scored = st.session_state.get("scored_df")
    thr_high = st.session_state.get("thr_high", cfg_default.get("thr_high", 0.99))
    if scored is None:
        st.info("Primero sube un CSV en la pestaña anterior.")
    elif "is_fraud" not in scored.columns:
        st.warning("El CSV no trae `is_fraud`, no se pueden calcular métricas.")
    else:
        y_true = scored["is_fraud"].astype(int).values
        y_prob = scored["score"].values
        pr_auc = average_precision_score(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        st.write(f"**PR AUC:** {pr_auc:.4f} | **ROC AUC:** {roc_auc:.4f}")

        # PR Curve
        prec, rec, thr = precision_recall_curve(y_true, y_prob)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec, prec, lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Curva Precision-Recall")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # Reporte @thr_high
        y_pred = (y_prob >= float(thr_high)).astype(int)
        st.text("Reporte @thr_high")
        st.text(classification_report(y_true, y_pred, digits=4))
        st.text("Matriz de confusión @thr_high")
        st.write(confusion_matrix(y_true, y_pred))

# ============ TAB 3: Info del modelo =======================
with tab3:
    st.subheader("ℹ️ Información del modelo")
    st.write("**Columnas numéricas (expectativa de entrenamiento)**:", ", ".join(num_cols))
    st.write("**Columnas categóricas (expectativa de entrenamiento)**:", ", ".join(cat_cols))

    # Intento de introspección de importancias
    try:
        clf = pipeline.named_steps.get("clf")
        ct  = _find_column_transformer(pipeline)

        if hasattr(clf, "coef_"):
            # Nombres OHE (si existen) para columnas categóricas
            ohe_names = []
            try:
                ohe = ct.named_transformers_["cat"].named_steps["ohe"]  # type: ignore
                try:
                    ohe_names = list(ohe.get_feature_names_out(cat_cols))
                except Exception:
                    ohe_names = [f"cat_{i}" for i in range(len(cat_cols))]
            except Exception:
                pass

            feature_names = list(num_cols) + list(ohe_names)
            coefs = pd.DataFrame({"feature": feature_names, "coef": clf.coef_.ravel()})
            top = coefs.assign(abscoef=lambda d: d["coef"].abs()).sort_values("abscoef", ascending=False).head(20)
            st.caption("Coeficientes absolutos (Top 20)")
            st.dataframe(top, use_container_width=True)

        elif hasattr(clf, "feature_importances_"):
            # Nota: para árboles, el mapeo a nombres tras OHE es más complejo; dejamos indicativo.
            fi = pd.DataFrame({
                "feature": list(num_cols) + ["OHE_*"],
                "importance": clf.feature_importances_
            }).sort_values("importance", ascending=False).head(20)
            st.caption("Feature importances (modelo de árboles)")
            st.dataframe(fi, use_container_width=True)
        else:
            st.info("El estimador no expone coeficientes ni importancias.")
    except Exception as e:
        st.warning(f"No se pudo introspeccionar importancias: {e}")

    st.caption(f"scikit-learn runtime: importado con la versión que hay en requirements.txt")
