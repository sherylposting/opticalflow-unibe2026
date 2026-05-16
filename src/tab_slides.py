import streamlit as st
import streamlit_book as stb

# ── Slides tab ────────────────────────────────────────────────────────────
def render():
    slides_url = "https://docs.google.com/presentation/d/e/2PACX-1vQkvhuAnkkI5hBgYxh7nYyx8hoWmi7D_ZnYZO6NFe2Nz2usJRiF_3SxBdU2J4OtMOV_yxwNmxnZA1PD/pubembed?start=false&loop=false"

    st.markdown(
        f"""
        <div style="
        width:100%;
        max-width:1000px;
        aspect-ratio:16/9;
        margin:auto;
        ">
            <iframe src={slides_url} 
                frameborder="0" 
                width="100%" 
                height="100%" 
                allowfullscreen="true" 
                mozallowfullscreen="true" 
                webkitallowfullscreen="true">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True
    )