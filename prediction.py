import streamlit as st
import pandas as pd
import joblib

def build_engineered_features(row: dict) -> dict:
    interest_cols = [
        "basketball","football","soccer","softball","volleyball",
        "swimming","cheerleading","baseball","tennis","sports",
        "dance","band","marching","music","rock","hair","dress","blonde"
    ]
    existing = [c for c in interest_cols if c in row]

    row["total_interest"] = sum(float(row[c]) for c in existing)
    row["active_interest_count"] = sum(1 for c in existing if float(row[c]) > 0)

    arts_cols = [c for c in ["dance","band","marching","music","rock","hair","dress","blonde"] if c in row]
    row["arts_interest"] = sum(float(row[c]) for c in arts_cols)

    sports_cols = [c for c in ["basketball","football","soccer","softball","volleyball",
                               "swimming","cheerleading","baseball","tennis","sports"] if c in row]
    row["sports_interest"] = sum(float(row[c]) for c in sports_cols)

    return row


def prediction_app():
    st.markdown("## 🧙‍♂️ Prediction App")
    st.caption("Masukkan data siswa untuk memprediksi cluster.")

    # Load model, scaler, dan fitur yang dipakai KMeans
    model = joblib.load("Finpro_model.pkl")
    scaler = joblib.load("Finpro_scaler.pkl")
    used_cols = joblib.load("Finpro_features.pkl")  # ✅ fitur FIX sesuai training

    # ========= INPUT =========
    st.markdown("### Data Dasar")
    c1, c2 = st.columns(2)
    with c1:
        gradyear = st.number_input("Tahun Kelulusan", 2000, 2035, 2015)
    with c2:
        num_friends = st.number_input("Jumlah Teman di Media Sosial", 0, 10000, 50)

    st.markdown("### Minat (0-100)")
    colA, colB, colC = st.columns(3)

    with colA:
        basketball = st.slider("Basketball", 0, 100, 0)
        football = st.slider("Football", 0, 100, 0)
        soccer = st.slider("Soccer", 0, 100, 0)
        softball = st.slider("Softball", 0, 100, 0)
        volleyball = st.slider("Volleyball", 0, 100, 0)

    with colB:
        swimming = st.slider("Swimming", 0, 100, 0)
        cheerleading = st.slider("Cheerleading", 0, 100, 0)
        baseball = st.slider("Baseball", 0, 100, 0)
        tennis = st.slider("Tennis", 0, 100, 0)
        sports = st.slider("Sports (umum)", 0, 100, 0)

    with colC:
        dance = st.slider("Dance", 0, 100, 0)
        band = st.slider("Band", 0, 100, 0)
        marching = st.slider("Marching", 0, 100, 0)
        music = st.slider("Music", 0, 100, 0)
        rock = st.slider("Rock", 0, 100, 0)
        hair = st.slider("Hair", 0, 100, 0)
        dress = st.slider("Dress", 0, 100, 0)
        blonde = st.slider("Blonde", 0, 100, 0)

    st.markdown("---")

    if st.button("Prediksi Cluster", use_container_width=True):
        # 1) bikin row input
        row = {
            "gradyear": gradyear,
            "NumberOffriends": num_friends,
            "basketball": basketball,
            "football": football,
            "soccer": soccer,
            "softball": softball,
            "volleyball": volleyball,
            "swimming": swimming,
            "cheerleading": cheerleading,
            "baseball": baseball,
            "tennis": tennis,
            "sports": sports,
            "dance": dance,
            "band": band,
            "marching": marching,
            "music": music,
            "rock": rock,
            "hair": hair,
            "dress": dress,
            "blonde": blonde,
        }

        # 2) tambahin fitur turunan (kalau dipakai training)
        row = build_engineered_features(row)

        input_df = pd.DataFrame([row])

        # 3) samakan kolom untuk scaler (kalau scaler simpan feature_names_in_)
        if hasattr(scaler, "feature_names_in_"):
            expected_scaler = list(scaler.feature_names_in_)
            for col in expected_scaler:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_scaler]

        # 4) scaling
        input_scaled = scaler.transform(input_df)

        # 5) IMPORTANT: ambil hanya kolom yang dipakai KMeans saat training
        # Caranya: cari index kolom used_cols dari expected_scaler
        if hasattr(scaler, "feature_names_in_"):
            expected_scaler = list(scaler.feature_names_in_)
            idx = [expected_scaler.index(c) for c in used_cols]
            input_scaled_used = input_scaled[:, idx]
        else:
            st.error("Scaler tidak punya feature_names_in_. Simpan scaler dari DataFrame saat fit.")
            return

        # 6) predict
        cluster_pred = model.predict(input_scaled_used)[0]
        st.success(f"✅ Prediksi cluster: **{cluster_pred}**")