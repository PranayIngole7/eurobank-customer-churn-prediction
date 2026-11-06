"""Batch customer prediction page."""

import streamlit as st

from dashboard.components import render_header


def render_batch_prediction():
    render_header(
        "Batch Customer Prediction",
        "Upload customer records and generate churn predictions.",
    )

    st.info("Batch prediction implementation is coming next.")