"""Single customer prediction page."""

import streamlit as st

from dashboard.components import render_header


def render_single_prediction():
    render_header(
        "Single Customer Prediction",
        "Estimate churn risk for an individual customer.",
    )

    st.info("Single customer prediction implementation is coming next.")