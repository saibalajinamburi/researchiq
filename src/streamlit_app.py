import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from src.inference import ResearchIQInference


ROOT_DIR = Path(__file__).resolve().parent.parent
API_URL = os.getenv("RESEARCHIQ_API_URL", "").rstrip("/")
LOCAL_INFERENCE = ResearchIQInference()


st.set_page_config(page_title="ResearchIQ", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.5rem; max-width: 1180px; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    div[data-testid="stStatusWidget"] { visibility: hidden; height: 0; }
    .stButton>button { width: 100%; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, timeout: int = 5):
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict, timeout: int = 60):
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def running_with_api() -> bool:
    return bool(API_URL)


def get_health():
    if running_with_api():
        return api_get("/health")
    info = LOCAL_INFERENCE.info()
    return {
        "status": "ok",
        "model_exists": Path(info["classifier_path"]).exists(),
        "metadata_exists": True,
        "version": "0.1.0",
    }


def get_model_info():
    if running_with_api():
        return api_get("/model/info", timeout=30)
    return LOCAL_INFERENCE.info()


def predict_single(text: str):
    if running_with_api():
        return api_post("/predict", {"text": text})
    return LOCAL_INFERENCE.predict(text)


with st.sidebar:
    st.title("ResearchIQ")
    mode_label = API_URL if running_with_api() else "Direct inference mode"
    st.caption(mode_label)
    if running_with_api():
        st.link_button("API Docs", f"{API_URL}/docs")
        st.link_button("Metrics", f"{API_URL}/metrics")
    st.link_button("MLflow", os.getenv("RESEARCHIQ_MLFLOW_URL", "http://127.0.0.1:5000"))

    try:
        health = get_health()
        st.success(health["status"])
    except Exception as exc:
        st.error(f"API offline: {exc}")


predict_tab, monitor_tab, insights_tab = st.tabs(["Predict", "Monitor", "Insights"])

with predict_tab:
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.subheader("Scientific Abstract")
        default_text = (
            "We propose a transformer-based representation learning method for robust "
            "scientific document classification across machine learning and statistical domains."
        )
        text = st.text_area("Abstract", value=default_text, height=220, label_visibility="collapsed")
        submitted = st.button("Classify", type="primary")

    with right:
        st.subheader("Prediction")
        if submitted:
            try:
                result = predict_single(text)
                st.metric("Category", result["category"])
                st.metric("Confidence", f"{result['confidence']:.2%}")
                chart_data = pd.DataFrame(result["top_categories"])
                fig = px.bar(
                    chart_data,
                    x="probability",
                    y="category",
                    orientation="h",
                    range_x=[0, 1],
                    text=chart_data["probability"].map(lambda value: f"{value:.1%}"),
                    color="category",
                    color_discrete_sequence=["#2f6f73", "#8f5f2d", "#4c5f88"],
                )
                fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0))
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))
        else:
            st.info("Ready")


with monitor_tab:
    col_a, col_b, col_c = st.columns(3)
    try:
        health = get_health()
        info = get_model_info()
        col_a.metric("Service", health["status"])
        col_b.metric("Classes", len(info["classes"]))
        col_c.metric("Cache", info["cache"]["size"])

        metadata = info["metadata"]
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": "Model", "Value": metadata["model_name"]},
                    {"Field": "Macro F1", "Value": f"{metadata['macro_f1']:.4f}"},
                    {"Field": "Rows", "Value": metadata["n_samples"]},
                    {"Field": "Features", "Value": metadata["n_features"]},
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.error(str(exc))

    if running_with_api():
        metrics_url = f"{API_URL}/metrics"
        st.link_button("Open Prometheus Metrics", metrics_url)


with insights_tab:
    comparison_path = ROOT_DIR / "reports" / "phase3_final_model_comparison.csv"
    if comparison_path.exists():
        st.subheader("Final Model Run")
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    class_dist = ROOT_DIR / "reports" / "figures" / "class_distribution.png"
    abstract_len = ROOT_DIR / "reports" / "figures" / "abstract_length_dist.png"
    if class_dist.exists():
        col_a.image(str(class_dist), use_column_width=True)
    if abstract_len.exists():
        col_b.image(str(abstract_len), use_column_width=True)

    phase4_path = ROOT_DIR / "models" / "phase4_onnx" / "phase4_metadata.json"
    if phase4_path.exists():
        st.subheader("ONNX Export")
        st.json(phase4_path.read_text(encoding="utf-8"))
