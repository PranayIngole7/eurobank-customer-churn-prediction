"""Model insights and evaluation page."""

import streamlit as st

from dashboard.components import render_header


def render_model_insights():
    render_header(
        "Model Insights",
        "Explore model evaluation, explainability, and fairness.",
    )

    st.info("Model insights implementation is coming next.")