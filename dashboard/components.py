"""Reusable Streamlit dashboard components."""

import streamlit as st


def render_header(title, description):
    """Render a consistent page header."""

    st.title(title)
    st.markdown(description)
    st.divider()


def render_section_header(title, description=None):
    """Render a section heading."""

    st.subheader(title)

    if description:
        st.caption(description)


def render_metric_card(label, value, help_text=None):
    """Render a dashboard metric."""

    st.metric(
        label=label,
        value=value,
        help=help_text,
    )


def render_info(message):
    """Display an informational message."""

    st.info(message)


def render_warning(message):
    """Display a warning message."""

    st.warning(message)