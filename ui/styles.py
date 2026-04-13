"""Custom CSS theming."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.iris-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem; border-radius: 12px; margin-bottom: 1.5rem; color: white;
    text-align: center;
}
.iris-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
.iris-header p { opacity: 0.8; margin: 0.5rem 0 0; font-size: 1rem; }
.caption-box {
    padding: 1rem 1.2rem; border-radius: 8px; margin: 0.5rem 0;
    font-size: 1rem; line-height: 1.5;
}
.caption-before { background: rgba(255,82,82,0.15); border-left: 4px solid #ff5252; }
.caption-after { background: rgba(0,230,118,0.15); border-left: 4px solid #00e676; }
</style>
"""

def inject_css():
    import streamlit as st
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
