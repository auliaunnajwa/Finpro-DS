import streamlit as st

# ⬇️ WAJIB PALING ATAS
st.set_page_config(
    page_title="Clustering Perilaku Sosial Siswa",
    page_icon="👩‍💻",
    layout="wide"
)

# CSS padding
st.markdown("""
<style>
.block-container { 
    padding-top: 2rem; 
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER TENGAH
# =========================
st.markdown("""
<div style="text-align:center; margin-top:20px; margin-bottom:10px;">
    <h1>👩‍💻 Clustering Perilaku Sosial Siswa</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-bottom:30px;">
    <p style="font-size:16px;"><b>Final Project</b> – Data Analyst & Data Science</p>
    <p style="font-size:14px; color:gray;">Jakarta, 12 Januari 2026</p>
</div>
""", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "About Dataset",
    "Dashboards",
    "Machine Learning",
    "Prediction App",
    "Contact Me"
])

with tab1:
    import about
    about.about_dataset()

with tab2:
    import visualisasi
    visualisasi.chart()

with tab3:
    import machine_learning
    machine_learning.ml_model()

with tab4:
    import prediction
    prediction.prediction_app()

with tab5:
    import kontak
    kontak.contact_me()
