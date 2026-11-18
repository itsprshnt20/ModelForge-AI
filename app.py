"""ModelForge AI Streamlit entry point."""
from pathlib import Path

import pandas as pd
import streamlit as st

from workbench import MODELS, ValidationError, evaluate, fit_models, model_card, parse_csv, target_labels

SAMPLE = Path(__file__).resolve().parent / "samples" / "synthetic_churn.csv"
METRICS = ("Balanced accuracy", "Precision", "Recall", "F1", "Average precision")
STYLE = """
<style>
.stApp, [data-testid="stHeader"] { background: #000; }
.stMainBlockContainer { max-width: 75rem; padding-top: 3.4rem; padding-bottom: 4rem; }
.st-key-hero, .st-key-dataset_panel, .st-key-config_panel, .st-key-results_panel {
    border: 1px solid #273b32; border-radius: 1.15rem;
    background: #090e0c; padding: clamp(1rem, 2.5vw, 1.55rem);
}
.st-key-hero {
    padding: clamp(1.3rem, 4vw, 2.65rem);
    background: radial-gradient(ellipse at 95% 0%, #1a362b 0%, #0a1610 45%, #050806 80%);
}
.st-key-hero h1 { font-size: clamp(2.5rem, 7vw, 4.8rem); line-height: 1; letter-spacing: -.07em; }
[data-testid="stBaseButton-primary"]:not(:disabled) {
    color: #072519; background: #a0f4cb; border-color: #a0f4cb; font-weight: 700;
}
[data-testid="stBaseButton-primary"]:not(:disabled) p { color: #072519; }
[data-testid="stMetric"] { background: #101914; border: 1px solid #263c31; border-radius: .8rem; padding: .8rem; }
button:focus-visible, input:focus-visible { outline: 2px solid #a0f4cb; outline-offset: 2px; }
</style>
"""


def load(raw, source):
    st.session_state.data_epoch += 1
    st.session_state.run = None
    try:
        st.session_state.frame = parse_csv(raw)
    except ValidationError as exc:
        st.session_state.frame = None
        st.session_state.input_error = str(exc)
        return
    st.session_state.source = source
    st.session_state.input_error = None


def render_results(run):
    with st.container(key="results_panel"):
        st.subheader("Model comparison")
        st.text(f"Positive class: {run.positive} · {run.train_size} training rows · "
                f"{len(run.observed)} validation rows")
        st.caption("Same validation rows for each model. Classification metrics use a threshold of 0.50.")
        rows = []
        for name in MODELS:
            score = evaluate(run.observed, run.probabilities[name], 0.50)
            rows.append({"Model": name, **{label: score[label] for label in METRICS}})
        comparison = pd.DataFrame(rows)
        st.dataframe(comparison, hide_index=True, use_container_width=True,
                     column_config={name: st.column_config.NumberColumn(format="%.3f") for name in METRICS})
        with st.expander("Average precision comparison"):
            st.bar_chart(comparison, x="Model", y="Average precision", color="#A0F4CB")

        st.subheader("Threshold explorer")
        st.caption("Validation-set exploration. Threshold selection here needs an independent test set.")
        model = st.selectbox("Explore model", MODELS, index=1)
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)
        metrics = evaluate(run.observed, run.probabilities[model], threshold)
        scores = st.columns(3)
        for column, name in zip(scores, ("Precision", "Recall", "F1")):
            column.metric(name, f"{metrics[name]:.3f}")
        st.markdown("**Confusion matrix**")
        st.caption("Compare actual labels with predictions at the selected threshold.")
        matrix = st.columns(2)
        matrix[0].metric("True negatives (TN)", metrics["TN"])
        matrix[1].metric("False positives (FP)", metrics["FP"])
        matrix = st.columns(2)
        matrix[0].metric("False negatives (FN)", metrics["FN"])
        matrix[1].metric("True positives (TP)", metrics["TP"])
        st.download_button("Download model card", model_card(run, model, threshold),
                           "modelforge-model-card.md", "text/markdown", on_click="ignore")


st.set_page_config(page_title="ModelForge AI", layout="wide")
st.html(STYLE)
st.session_state.setdefault("data_epoch", 0)
with st.container(key="hero"):
    st.title("ModelForge AI")
    st.write("Compare binary classifiers, explore thresholds, and export the evidence behind a model.")
    st.caption("Local machine learning workbench · Maintained by Prashant Chauhan")

with st.container(key="dataset_panel"):
    st.subheader("Dataset")
    if st.button("Load demo", type="primary"):
        load(SAMPLE.read_bytes(), "Synthetic customer churn")
    st.file_uploader("Upload a CSV (up to 3 MiB)", type=["csv"], key="upload")
    if st.button("Load uploaded CSV"):
        upload = st.session_state.upload
        if upload:
            load(upload.getvalue(), f"User-uploaded CSV ({upload.name})")
        else:
            st.warning("Choose a CSV file before loading it.")
    if st.session_state.get("input_error"):
        st.error(st.session_state.input_error)
    frame = st.session_state.get("frame")
    if frame is None:
        st.info("Load the synthetic demo or upload a CSV to inspect its data quality.")
    else:
        st.text(st.session_state.source)
        row, col, missing = st.columns(3)
        row.metric("Rows", len(frame))
        col.metric("Columns", len(frame.columns))
        missing.metric("Missing values", int(frame.isna().sum().sum()))
        with st.expander("Data preview", expanded=True):
            st.dataframe(frame.head(8), hide_index=True, use_container_width=True)
        with st.expander("Missing values by column"):
            st.dataframe(frame.isna().sum().rename("Missing values").rename_axis("Column").reset_index(),
                         hide_index=True, use_container_width=True)

active_key = None
if frame is not None:
    with st.container(key="config_panel"):
        st.subheader("Training setup")
        is_demo = st.session_state.source == "Synthetic customer churn"
        epoch = st.session_state.data_epoch
        columns = list(frame.columns)
        target = st.selectbox("Binary target", columns,
                              index=columns.index("churn") if is_demo else None, key=f"target_{epoch}")
        if target is None:
            st.info("Select the target column to see its class balance and choose a positive class.")
        else:
            try:
                labels = target_labels(frame, target)
            except ValidationError as exc:
                st.error(str(exc))
            else:
                st.dataframe(frame[target].value_counts().rename("Rows").rename_axis("Class").reset_index(),
                             hide_index=True, use_container_width=True)
                positive = st.selectbox("Positive class", labels, index=None, key=f"positive_{epoch}_{target}")
                candidates = [name for name in columns if name != target]
                default = [name for name in ("customer_id", "cancellation_ticket") if name in candidates]
                excluded = st.multiselect("Exclude fields", candidates, default=default if is_demo else [],
                                          key=f"exclude_{epoch}_{target}",
                                          help="Exclude identifiers and any fields unavailable at prediction time.")
                active_key = (epoch, target, positive, tuple(excluded))
                if st.button("Train models", type="primary", disabled=positive is None):
                    st.session_state.run = None
                    try:
                        with st.spinner("Training three models on the same stratified split…"):
                            st.session_state.run = fit_models(frame, target, positive, excluded,
                                                              st.session_state.source)
                        st.session_state.run_key = active_key
                    except ValidationError as exc:
                        st.error(str(exc))
                if positive is None:
                    st.info("Choose the positive class before training.")
    run = st.session_state.get("run")
    if run is not None and st.session_state.get("run_key") == active_key:
        render_results(run)
