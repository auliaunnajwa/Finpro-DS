import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def chart():
    df = pd.read_excel("students_clustered.xlsx")

    # =========================
    # Warna biru pastel (konsisten)
    # =========================
    PASTEL_BLUES = ["#1f77b4", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
    PASTEL_BLUE_MAIN = "#6baed6"   # dipakai bar/scatter
    KPI_BG = "#eef6ff"            # biru pastel sangat muda

    # =========================
    # Feature Engineering (untuk KPI & chart)
    # =========================
    interest_cols = [
        'dance', 'band', 'marching', 'music', 'rock', 'hair', 'dress', 'blonde',
        'basketball', 'football', 'soccer', 'softball', 'volleyball',
        'swimming', 'cheerleading', 'baseball', 'tennis', 'sports'
    ]
    interest_cols = [c for c in interest_cols if c in df.columns]

    arts_cols = [c for c in ['dance','band','marching','music','rock','hair','dress','blonde'] if c in df.columns]
    sports_cols = [c for c in ['basketball','football','soccer','softball','volleyball','swimming','cheerleading','baseball','tennis','sports'] if c in df.columns]

    df["arts_interest"] = df[arts_cols].sum(axis=1) if len(arts_cols) > 0 else 0
    df["sports_interest"] = df[sports_cols].sum(axis=1) if len(sports_cols) > 0 else 0
    df["active_interest_count"] = (df[interest_cols] > 0).sum(axis=1) if len(interest_cols) > 0 else 0
    df["total_interest"] = df[interest_cols].sum(axis=1) if len(interest_cols) > 0 else 0

    # =========================
    # KPI Cards (semua biru pastel)
    # =========================
    total_siswa = int(len(df))
    avg_friends = df["NumberOffriends"].mean() if "NumberOffriends" in df.columns else np.nan
    avg_active_interest = df["active_interest_count"].mean() if len(interest_cols) > 0 else np.nan
    dominant = "Arts" if df["arts_interest"].sum() > df["sports_interest"].sum() else "Sports"

    st.markdown("""
    <style>
    .kpi-card{
        padding:16px 18px;
        border-radius:14px;
        border: 1px solid rgba(31,119,180,0.18);
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        height: 110px;
        display:flex;
        flex-direction:column;
        justify-content:center;
        background: #eef6ff;
    }
    .kpi-title{font-size:14px; margin:0; color:#0b2540; opacity:0.9;}
    .kpi-value{font-size:40px; font-weight:800; margin:4px 0 0 0; line-height:1; color:#0b2540;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### KPI")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Total Siswa</p>
            <p class="kpi-value">{total_siswa:,}</p>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        val = f"{avg_friends:,.1f}" if not np.isnan(avg_friends) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Rata-rata Jumlah Teman</p>
            <p class="kpi-value">{val}</p>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        val = f"{avg_active_interest:.2f}" if not np.isnan(avg_active_interest) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Rata-rata Minat Aktif</p>
            <p class="kpi-value">{val}</p>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-title">Minat Dominan</p>
            <p class="kpi-value">{dominant}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # Pie Charts (judul kecil + proporsional + biru pastel)
    # =========================
    col5, col6 = st.columns(2)

    def style_pie(fig):
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=55),
            legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center", title_text=""),
            font=dict(size=13),
        )
        fig.update_traces(textposition="inside", textinfo="percent")
        return fig

    with col5:
        st.markdown("#### Distribusi Tahun Kelulusan Siswa")
        if "gradyear" in df.columns:
            graduation_year_count = df["gradyear"].value_counts().reset_index()
            graduation_year_count.columns = ["gradyear", "count"]

            fig1 = px.pie(
                graduation_year_count,
                names="gradyear",
                values="count",
                hole=0.35,
                color_discrete_sequence=PASTEL_BLUES
            )
            fig1 = style_pie(fig1)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Kolom 'gradyear' tidak ditemukan di dataset.")

    with col6:
        st.markdown("#### Proporsi Siswa Berdasarkan Jumlah Minat")
        if len(interest_cols) > 0:
            bins = [-1, 3, 6, len(interest_cols)]
            labels = ["Low", "Medium", "High"]
            df["interest_level"] = pd.cut(df["active_interest_count"], bins=bins, labels=labels)

            interest_level_count = df["interest_level"].value_counts().reset_index()
            interest_level_count.columns = ["interest_level", "count"]

            fig2 = px.pie(
                interest_level_count,
                names="interest_level",
                values="count",
                hole=0.35,
                color_discrete_sequence=PASTEL_BLUES
            )
            fig2 = style_pie(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Tidak ada kolom minat yang ditemukan untuk visualisasi proporsi jumlah minat.")

    st.markdown("---")

    # =========================
    # 3. Visualisasi rata-rata minat (bar chart -> biru pastel)
    # =========================
    st.subheader("Rata-rata Minat Siswa")
    st.write("Menampilkan rata-rata minat siswa pada berbagai aktivitas.")

    interest_columns = [c for c in [
        'basketball', 'football', 'soccer', 'softball', 'volleyball',
        'swimming', 'cheerleading', 'baseball','tennis', 'sports',
        'dance', 'band', 'marching', 'music', 'rock',
        'hair', 'dress', 'blonde',
    ] if c in df.columns]

    if len(interest_columns) > 0:
        avg_interests = df[interest_columns].mean().reset_index()
        avg_interests.columns = ["Interest", "Average Score"]

        fig_avg = px.bar(
            avg_interests,
            x="Interest",
            y="Average Score",
            labels={"Interest": "Minat", "Average Score": "Rata-rata Nilai"},
            color_discrete_sequence=[PASTEL_BLUE_MAIN]
        )
        fig_avg.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_avg, use_container_width=True)
    else:
        st.info("Tidak ada kolom minat yang ditemukan untuk visualisasi rata-rata.")

    st.markdown("---")

    # =========================
    # 4. Distribusi jumlah minat aktif (bar chart -> biru pastel)
    # =========================
    st.subheader("Distribusi Jumlah Minat Aktif Siswa")
    st.write("Jumlah minat aktif menunjukkan seberapa beragam minat yang dimiliki seorang siswa.")

    if len(interest_columns) > 0:
        df["active_interest_count"] = df[interest_columns].gt(0).sum(axis=1)
        active_interest_count = df["active_interest_count"].value_counts().reset_index()
        active_interest_count.columns = ["active_interest_count", "count"]

        fig5 = px.bar(
            active_interest_count,
            x="active_interest_count",
            y="count",
            labels={"active_interest_count": "Jumlah Minat Aktif", "count": "Jumlah Siswa"},
            color_discrete_sequence=[PASTEL_BLUE_MAIN]
        )
        fig5.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Tidak ada kolom minat yang ditemukan untuk visualisasi distribusi jumlah minat aktif.")

    st.markdown("---")

    # =========================
    # 5. Arts vs Sports Scatter (marker biru pastel)
    # =========================
    st.subheader("Pola Minat: Arts vs Sports (Scatter)")
    fig_scatter = px.scatter(
        df,
        x="arts_interest",
        y="sports_interest",
        labels={"arts_interest": "Minat Arts", "sports_interest": "Minat Sports"}
    )
    fig_scatter.update_traces(marker=dict(size=7, opacity=0.6, color=PASTEL_BLUE_MAIN))
    fig_scatter.update_layout(height=500, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_scatter, use_container_width=True)