"""Eurobank Customer Churn Prediction Dashboard."""

import streamlit as st

from dashboard.overview import render_overview
from dashboard.single_prediction import render_single_prediction
from dashboard.batch_prediction import render_batch_prediction
from dashboard.model_insights import render_model_insights


st.set_page_config(
    page_title="Eurobank Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Run the dashboard application."""

    st.sidebar.title("Eurobank")
    st.sidebar.caption("Customer Churn Prediction")

    st.sidebar.divider()

    selected_page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Single Customer Prediction",
            "Batch Prediction",
            "Model Insights",
        ],
    )

    if selected_page == "Overview":
        render_overview()

    elif selected_page == "Single Customer Prediction":
        render_single_prediction()

    elif selected_page == "Batch Prediction":
        render_batch_prediction()

    elif selected_page == "Model Insights":
        render_model_insights()

    st.sidebar.divider()

    st.sidebar.caption(
        "Eurobank Customer Churn Prediction | ML Project"
    )


if __name__ == "__main__":
    main()