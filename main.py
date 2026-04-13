"""
Iris-Shield entry point.
Run: uv run streamlit run main.py --server.port 8501
"""
import streamlit as st

st.set_page_config(
    page_title="Iris-Shield",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui.styles import inject_css
inject_css()

from ui.dashboard import run_dashboard
run_dashboard()
