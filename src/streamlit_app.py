"""
Interactive Optical Flow Registration App
Using scikit-image for registration and Streamlit for UI
"""

import streamlit as st

import tab_app, tab_edu, tab_slides

st.set_page_config(page_title="Optical Flow Registration", layout="wide")

st.title("Image Registration with Optical Flow")
st.write("An interactive Streamlit demo for estimation of elastic movement between frames, using optical-flow-based non-linear image registration.")
st.image("https://github.com/sherylposting/data/blob/main/opticalflow/opticalflow_fig.png?raw=true", width=400)

tab1_ui, tab2_ui, tab3_ui = st.tabs(["Registration Tool", "What is Optical Flow Registration?", "Slide Deck"])

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=SUSE+Mono&display=swap');

    h1, h2, h3, h4, h5, h6 {
        font-family: 'SUSE Mono' !important;
    }

    h1 {
        font-size: 26px !important;
    }

    h2 {
        font-size: 20px !important;
    }

    h3 {
        font-size: 16px !important;
    }

    .katex-display {
    text-align: left !important;
    margin-left: 0 !important;
    }

    .katex-display > .katex {
        display: flex !important;
        justify-content: flex-start !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

with tab1_ui:
    tab_app.render()
with tab2_ui:
    tab_edu.render()
with tab3_ui:
    tab_slides.render()