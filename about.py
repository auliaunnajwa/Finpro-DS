import streamlit as st

def about_dataset():
    st.markdown("### Tentang Dataset")

    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        link = "https://upload.wikimedia.org/wikipedia/commons/c/cb/3D_Social_Networking.jpg"
        st.image(
            link,
            use_container_width=True,
            caption="Clustering Perilaku Sosial Siswa"
        )

    with col2:
        st.markdown("""
        <div style="
            background-color:#f8fbff;
            padding:24px;
            border-radius:14px;
            height:100%;
            box-shadow:0 4px 12px rgba(0,0,0,0.04);
        ">
            <h4>Dataset Student Social Network Profile</h4>
            <p style="font-size:16px; line-height:1.7;">
                Dataset ini berisi data profil sosial siswa sebanyak <b>10.000 baris</b>.
                Dataset mencakup informasi <b>tahun kelulusan</b>, <b>jumlah teman di media sosial</b>,
                serta <b>ketertarikan siswa</b> terhadap berbagai aktivitas seperti olahraga,
                musik, seni, dan gaya hidup.
            </p>
            <p style="font-size:16px; line-height:1.7;">
                Data ini digunakan untuk menganalisis <b>pola perilaku sosial siswa</b>
                dan mengelompokkan siswa berdasarkan karakteristik yang dimiliki
                menggunakan pendekatan <b>clustering</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)