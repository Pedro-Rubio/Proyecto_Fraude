import os, json, joblib, io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score, classification_report, confusion_matrix

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────
st.set_page_config(page_title="Fraude • MVP", layout="wide")
ART_DIR = "artifacts_noleak_v1"

# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────
@st.cache_resource
def load_pipeline(path: str):
    with open(path, "rb") as f:
        return joblib.load(f)

def extract_required_columns(pipeline):
    """Obtiene las columnas esperadas por el ColumnTransformer del pipeline."""
    prep = pipeline.named_steps.get("prep")
    if prep is None:
        return None, None, []
    # num
    num_cols = prep.transformers_[0][2] if len(prep.transformers_) > 0 else []
    # cat
    cat_cols = prep.transformers_[1][2] if len(prep.transformers_) > 1 else []
    req = list(num_cols) + list(cat_cols)
    return num_cols, cat_cols, req

def align_columns(df_in: pd.DataFrame, required_cols):
    """Crea columnas faltantes como NaN y reordena para matchear lo esperado."""
    df = df_in.copy()
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
    # El ColumnTransformer con remainder='drop' ignora extras, pero ordenamos por prolijidad
    return df[required_cols]

def threshold_for_min_precision(y_true, y_prob, target_precision=0.8):
    """Devuelve el umbral más alto que cumpla la precisión objetivo (elige mayor recall)."""
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
# Cargar artefactos
# ─────────────────────────────────────────────
pipe_path = os.path.join(ART_DIR, "best_so_far.pkl")
if not os.path.exists(pipe_path):
    st.error(f"No se encontró el pipeline en {pipe_path}. Sube tus artefactos a la carpeta del repo.")
    st.stop()

pipeline = load_pipeline(pipe_path)
num_cols, cat_cols, required_cols = extract_required_columns(pipeline)

# Umbrales por defecto (opcional)
cfg_default = {"thr_high": 0.99, "min_precision_high": 0.80, "review_capacity": 200}
cfg_path = os.path.join(ART_DIR, "triage_mvp_thresholds.json")
if os.path.exists(cfg_path):
    try:
        cfg_default.update(json.load(open(cfg_path)))
    except Exception:
        pass

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Configuración")
st.sidebar.write("Artefactos:", f"`{pipe_path}`")
with st.sidebar.expander("Parámetros MVP", expanded=True):
    min_prec_high = st.slider("Precisión mínima (ALTO_RIESGO)", 0.5, 0.99, float(cfg_default["min_precision_high"]), 0.01)
    review_capacity = st.number_input("Capacidad de revisión (REVISAR)", min_value=0, step=10, value=int(cfg_default["review_capacity"]))
    allow_reestimate_thr = st.checkbox("Reestimar umbral con etiqueta (si el CSV tiene is_fraud)", value=True)
st.sidebar.caption("La política MVP: bloqueo por precisión mínima + revisión Top-N por pérdida esperada (prob * amount).")

st.title("🔎 Detección de Fraude — MVP Operativo")

# ─────────────────────────────────────────────
# Tabs principales
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📦 Scoring por lote (CSV)", "📈 Métricas (si hay is_fraud)", "ℹ️ Info del modelo"])

# ============ TAB 1: Scoring lote ==========================
with tab1:
    st.subheader("📦 Cargar CSV para scoring")
    st.write("El CSV debe contener **al menos** las columnas que el modelo espera. Columnas requeridas detectadas:")
    st.code(", ".join(required_cols) if required_cols else "El pipeline no expone ColumnTransformer.", language="markdown")

    file = st.file_uploader("Sube un CSV de transacciones", type=["csv"])
    if file is None:
        st.info("Opcional: prueba con un ejemplo pequeño.")
        demo_path = os.path.join("sample_data", "sample_10_transactions.csv")
        if os.path.exists(demo_path):
            df_raw = pd.read_csv(demo_path)
            st.dataframe(df_raw.head(10))
        else:
            st.warning("No se encontró sample_10_transactions.csv en sample_data/")
    else:
        df_raw = pd.read_csv(file)
        st.success(f"CSV cargado. Filas: {len(df_raw):,}")
        st.dataframe(df_raw.head(50), use_container_width=True)

        # Alinear columnas
        if required_cols:
            df_infer = align_columns(df_raw, required_cols)
        else:
            # Si no hay ColumnTransformer, pasar todo y confiar en el pipeline
            df_infer = df_raw.copy()

        # Scoring
        y_prob = pipeline.predict_proba(df_infer)[:, 1]
        scored = df_raw.copy()
        scored["score"] = y_prob

        # Pérdida esperada
        if "amount" in scored.columns:
            scored["expected_loss"] = scored["score"] * scored["amount"].fillna(0)
        else:
            scored["expected_loss"] = scored["score"] * 0.0  # fallback

        # Umbral ALTO_RIESGO
        thr_high = cfg_default.get("thr_high", 0.99)
        if allow_reestimate_thr and "is_fraud" in scored.columns:
            try:
                thr_est = threshold_for_min_precision(scored["is_fraud"].values.astype(int), scored["score"].values, target_precision=min_prec_high)
                if thr_est is not None:
                    thr_high = thr_est
            except Exception:
                pass

        # Triage: ALTO_RIESGO + REVISAR (TopN por expected_loss) + OK
        high_mask = scored["score"].values >= thr_high
        low_mask  = ~high_mask
        order_low = np.argsort(scored.loc[low_mask, "expected_loss"].values)[::-1]
        review_take = min(review_capacity, order_low.shape[0])
        triage = np.array(["OK"] * len(scored), dtype=object)
        triage[np.where(high_mask)[0]] = "ALTO_RIESGO"
        if review_take > 0:
            low_idx = np.where(low_mask)[0]
            review_idx = low_idx[order_low[:review_take]]
            triage[review_idx] = "REVISAR"
        scored["triage"] = triage

        # Resumen
        def band_report(df_band):
            n = len(df_band)
            if "is_fraud" in df_band.columns:
                tp = int(df_band["is_fraud"].sum())
                fp = n - tp
                prec = tp/n if n>0 else 0.0
                return n, tp, fp, prec
            else:
                return n, None, None, None

        col1, col2, col3 = st.columns(3)
        for band, col in zip(["ALTO_RIESGO","REVISAR","OK"], [col1,col2,col3]):
            subset = scored[scored["triage"]==band]
            n, tp, fp, prec = band_report(subset)
            with col:
                st.metric(band, f"{n:,} casos", help="TP/FP visibles solo si el CSV trae is_fraud")
                if tp is not None:
                    st.caption(f"TP={tp} | FP={fp} | Prec={prec:.3f}")

        # Descargas
        st.download_button(
            "⬇️ Descargar CSV con scores y triage",
            data=build_download(scored, "scored.csv"),
            file_name="scored_with_triage.csv",
            mime="text/csv"
        )

# ============ TAB 2: Métricas ==============================
with tab2:
    st.subheader("📈 Métricas (requiere columna is_fraud en el CSV)")
    uploaded = st.session_state.get("last_uploaded_df", None)
    if file is None:
        st.info("Vuelve a la pestaña anterior y sube un CSV con columna `is_fraud` para ver métricas.")
    else:
        df_eval = scored  # ya contiene score
        if "is_fraud" not in df_eval.columns:
            st.warning("El CSV no trae `is_fraud`, no se pueden calcular métricas.")
        else:
            y_true = df_eval["is_fraud"].astype(int).values
            y_prob = df_eval["score"].values
            pr_auc = average_precision_score(y_true, y_prob)
            roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
            st.write(f"**PR AUC:** {pr_auc:.4f} | **ROC AUC:** {roc_auc:.4f}")

            # PR Curve
            prec, rec, thr = precision_recall_curve(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(rec, prec, lw=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Curva Precision-Recall")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Reporte @thr_high
            y_pred = (y_prob >= thr_high).astype(int)
            st.text("Reporte @thr_high")
            st.text(classification_report(y_true, y_pred, digits=4))
            st.text("Matriz de confusión @thr_high")
            st.write(confusion_matrix(y_true, y_pred))

# ============ TAB 3: Info del modelo =======================
with tab3:
    st.subheader("ℹ️ Información del modelo")
    st.write("**Columnas numéricas**:", ", ".join(num_cols))
    st.write("**Columnas categóricas**:", ", ".join(cat_cols))

    # Intento de introspección de coeficientes/feature importances
    st.write("**Importancia de features (si aplica):**")
    try:
        clf = pipeline.named_steps['clf']
        if hasattr(clf, "coef_"):
            # Para models lineales
            st.caption("Coeficientes absolutos (top 20)")
            prep = pipeline.named_steps['prep']
            # nombres OHE (si existen)
            try:
                ohe = prep.named_transformers_['cat'].named_steps['ohe']
                try:
                    ohe_names = ohe.get_feature_names_out(cat_cols)
                except Exception:
                    ohe_names = [f"cat_{i}" for i in range(len(cat_cols))]
            except Exception:
                ohe_names = []

            feature_names = list(num_cols) + list(ohe_names)
            coefs = pd.DataFrame({"feature": feature_names, "coef": clf.coef_.ravel()})
            top = coefs.assign(abscoef=lambda d: d["coef"].abs()).sort_values("abscoef", ascending=False).head(20)
            st.dataframe(top)
        elif hasattr(clf, "feature_importances_"):
            st.caption("Feature importances (modelo de árboles)")
            fi = pd.DataFrame({
                "feature": list(num_cols) + ["OHE_..."],  # simplificado
                "importance": clf.feature_importances_
            }).sort_values("importance", ascending=False).head(20)
            st.dataframe(fi)
        else:
            st.info("El estimador no expone coeficientes ni importancias.")
    except Exception as e:
        st.warning(f"No se pudo introspeccionar importancias: {e}")
