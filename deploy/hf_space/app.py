from pathlib import Path
import time

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import ResearchIQInference


ROOT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

SUPPORTED_DOMAINS = {
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "cs.AI": "Artificial Intelligence",
    "cs.CL": "Computation and Language",
    "cs.CV": "Computer Vision",
    "cs.LG": "Machine Learning",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.RO": "Robotics",
    "math.PR": "Probability",
    "math.ST": "Statistics Theory",
    "physics.comp-ph": "Computational Physics",
    "q-bio.NC": "Neurons and Cognition",
    "q-bio.QM": "Quantitative Methods",
    "stat.ML": "Machine Learning (Statistics)"
}


@st.cache_resource
def get_inference_engine() -> ResearchIQInference:
    engine = ResearchIQInference()
    engine.load()
    return engine


st.set_page_config(page_title="ResearchIQ", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 1.25rem; max-width: 1160px; }
    .stButton > button { width: 100%; border-radius: 6px; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    .domain-desc { font-size: 0.85em; color: #a0a0a0; margin-bottom: 8px; display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = get_inference_engine()
info = engine.info()
metadata = info["metadata"]

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=64)
    st.title("Supported Domains")
    st.markdown("ResearchIQ can classify abstracts into the following **15** arXiv categories:")
    
    # Scrollable container for domains so sidebar doesn't get too long
    with st.container(height=500, border=True):
        for code, desc in SUPPORTED_DOMAINS.items():
            st.markdown(f"**{code}**<br><span class='domain-desc'>{desc}</span>", unsafe_allow_html=True)
            
    st.divider()
    st.caption("Engine: `ONNX Runtime`")
    st.caption("Embeddings: `MiniLM INT8`")
    st.caption(f"Latency: `< 10ms`")

# Main Content
st.title("🔬 ResearchIQ")
st.caption("High-speed ONNX-powered scientific abstract classifier")

st.link_button("View Model Card", "https://huggingface.co/spaces/saibalajiomg/researchiq")

top_left, top_mid, top_right = st.columns(3)
top_left.metric("Total Classes", len(info["classes"]))
top_mid.metric("Macro F1 Score", f"{metadata['macro_f1']:.4f}")
top_right.metric("Training Samples", f"{metadata['n_samples']:,}")

predict_tab, insights_tab, about_tab = st.tabs(["Predict", "Insights", "About"])

with predict_tab:
    col_a, col_b = st.columns([1.2, 0.8], gap="large")
    with col_a:
        example_abstracts = {
            "Machine Learning (cs.LG)": "We introduce a transformer-based machine learning method for scientific abstract classification using dense embeddings and calibrated probabilities.",
            "Computer Vision (cs.CV)": "We present a novel deep learning architecture for image segmentation using convolutional neural networks and self-attention mechanisms, achieving state-of-the-art results on the COCO dataset.",
            "Astrophysics (astro-ph.GA)": "Observations of the galactic center using the James Webb Space Telescope reveal new insights into the formation of supermassive black holes and their accretion disks.",
            "Probability (math.PR)": "We prove a new limit theorem for random matrices under weak moment conditions, extending the Wigner semicircle law.",
            "Custom": ""
        }
        
        selected_example = st.selectbox("Try an example abstract", list(example_abstracts.keys()))
        
        abstract = st.text_area(
            "Abstract Text", 
            value=example_abstracts[selected_example], 
            height=220,
            placeholder="Paste a scientific abstract here..."
        )
        run_prediction = st.button("Classify Paper", type="primary")

    with col_b:
        st.subheader("Prediction Results")
        if run_prediction and abstract.strip():
            with st.spinner("Analyzing semantic structure..."):
                start_time = time.perf_counter()
                result = engine.predict(abstract)
                inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            st.success("Analysis Complete!")
            
            col_c1, col_c2 = st.columns(2)
            col_c1.metric("Predicted Category", result["category"])
            col_c2.metric("Confidence Score", f"{result['confidence']:.2%}")
            
            st.progress(result['confidence'])
            st.caption(f"⚡ ONNX Inference Time: **{inference_time_ms:.2f} ms**")
            
            st.markdown("#### Probability Distribution")
            chart_df = pd.DataFrame(result["top_categories"])
            fig = px.bar(
                chart_df,
                x="probability",
                y="category",
                orientation="h",
                text=chart_df["probability"].map(lambda value: f"{value:.1%}"),
                color="category",
                color_discrete_sequence=["#2ca02c", "#ff7f0e", "#1f77b4"],
                range_x=[0, 1],
            )
            fig.update_layout(height=240, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Awaiting input... Paste an abstract and click Classify.")

with insights_tab:
    comparison_path = ROOT_DIR / "reports" / "phase3_final_model_comparison.csv"
    if comparison_path.exists():
        st.subheader("Final Model Run Metrics")
        st.dataframe(pd.read_csv(comparison_path), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    class_fig = FIGURES_DIR / "class_distribution.png"
    length_fig = FIGURES_DIR / "abstract_length_dist.png"
    if class_fig.exists():
        col_a.image(str(class_fig), use_column_width=True, caption="Class Distribution")
    if length_fig.exists():
        col_b.image(str(length_fig), use_column_width=True, caption="Abstract Lengths")

with about_tab:
    st.subheader("System Architecture Overview")
    st.write(
        "ResearchIQ uses a quantized ONNX MiniLM embedding model and an ONNX-exported Scikit-Learn classifier "
        "to categorize scientific abstracts instantly without heavy PyTorch dependencies."
    )
    st.json(
        {
            "final_model": metadata["model_name"],
            "macro_f1": metadata["macro_f1"],
            "n_samples": metadata["n_samples"],
            "n_features": metadata["n_features"],
            "classes": info["classes"],
            "providers": info["providers"],
            "cache": info["cache"],
        }
    )
