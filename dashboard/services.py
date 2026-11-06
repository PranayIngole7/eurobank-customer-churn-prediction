"""Shared dashboard services."""

import streamlit as st

from src.prediction_service import PredictionService


@st.cache_resource
def get_prediction_service():
    """Load and reuse the persisted prediction service."""
    return PredictionService()