import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def ml_model():
    # =========================
    # 0) Load Dataset
    # =========================
    df = pd.read_excel("students_clustered.xlsx")

    st.write("### Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

    # =========================
    # 1) Ambil Kolom Numerik
    # =========================
    numbers = df.select_dtypes(include=[np.number]).columns.tolist()
    numbers = [c for c in numbers if c not in ["cluster", "clusters"]]  # buang kolom cluster kalau ada

    if len(numbers) == 0:
        st.error("Tidak ada kolom numerik. Proses machine learning tidak bisa dilanjutkan.")
        st.stop()

    df_ml = df[numbers].copy()

    # =========================================================
    # 2) Keep OUTLIER (tidak dihapus)
    # =========================================================
    st.write("### 1. Outlier Handling")
    st.info("Pada dataset ini, data **tidak dihapus outlier**. Semua baris numerik tetap digunakan.")
    df_clean = df_ml.copy()
    st.markdown("---")

    # =========================
    # 3) Normalisasi (StandardScaler)
    # =========================
    st.write("### 2. Normalisasi menggunakan StandardScaler")

    df_clean_safe = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_clean_safe),
        columns=df_clean_safe.columns,
        index=df_clean_safe.index
    )

    st.write("**Preview data setelah normalisasi:**")
    st.dataframe(df_scaled.head(10), use_container_width=True)
    st.markdown("---")

    # =========================
    # 4) Correlation Heatmap + Drop corr > 0.8
    # =========================
    st.write("### 3. Correlation Heatmap")

    corr_df = df_clean_safe.copy()

    # buang kolom varians 0
    corr_df = corr_df.loc[:, corr_df.var() > 0]

    # buang kolom yang hampir selalu 0 (biar heatmap nggak banyak kosong)
    zero_ratio = (corr_df == 0).mean()
    corr_df = corr_df.loc[:, zero_ratio < 0.99]

    if corr_df.shape[1] < 2:
        st.warning("Heatmap tidak dapat ditampilkan karena kolom yang tersisa kurang dari 2.")
        st.stop()

    threshold = 0.80
    corr_abs = corr_df.corr().abs()

    # segitiga atas agar tidak double
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

    # kolom yang harus di-drop (punya korelasi > threshold dengan kolom lain)
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]

    # daftar pasangan korelasi tinggi untuk ditampilkan
    high_pairs = []
    for col in upper.columns:
        high = upper[col][upper[col] > threshold]
        for row_name, val in high.items():
            high_pairs.append((row_name, col, float(val)))

    # dataframe final untuk heatmap
    corr_df_filtered = corr_df.drop(columns=to_drop, errors="ignore")

    corr_filtered = corr_df_filtered.corr().round(2)

    fig_heat = px.imshow(
        corr_filtered,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )
    fig_heat.update_layout(height=650, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # =========================
    # 5) Elbow (mulai k=1) & Silhouette (mulai k=2)
    # =========================
    st.write("### 4. Menentukan Jumlah Cluster (Elbow & Silhouette)")

    # IMPORTANT:
    # - Clustering pakai fitur hasil drop korelasi (lebih aman)
    # - Tetap pakai versi scaled agar skala setara
    used_cols = corr_df_filtered.columns.tolist()

    if len(used_cols) < 2:
        st.warning("Kolom yang tersisa untuk clustering kurang dari 2. Turunkan threshold atau cek data.")
        st.stop()

    X = df_scaled[used_cols].values

    # Elbow boleh dari k=1
    k_elbow = list(range(1, 11))
    inertias = []
    for k in k_elbow:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    # Silhouette harus dari k=2
    k_sil = list(range(2, 11))
    sil_scores = []
    for k in k_sil:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))

    colA, colB = st.columns(2)

    with colA:
        fig_elbow = px.line(
            x=k_elbow, y=inertias, markers=True,
            labels={"x": "Jumlah Cluster (k)", "y": "Inertia"},
            title="Elbow Method"
        )
        fig_elbow.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with colB:
        fig_sil = px.line(
            x=k_sil, y=sil_scores, markers=True,
            labels={"x": "Jumlah Cluster (k)", "y": "Silhouette Score"},
            title="Silhouette Score"
        )
        fig_sil.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_sil, use_container_width=True)

    best_k = k_sil[int(np.argmax(sil_scores))]
    best_sil = float(np.max(sil_scores))

    st.info(
        f"- Elbow: cari titik “siku” saat penurunan inertia mulai melambat.\n"
        f"- Silhouette: semakin tinggi semakin baik.\n"
        f"- Kandidat terbaik (silhouette tertinggi): **k = {best_k}** (score **{best_sil:.3f}**)."
    )
    st.markdown("---")

    # =========================================================
    # 6) TRAIN KMEANS FINAL (FIXED k=2)
    # =========================================================
    st.write("### 5. Training Model KMeans Final")

    FINAL_K = 2  # fixed
    kmeans_final = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
    final_labels = kmeans_final.fit_predict(X)

    df_result = df_clean_safe.copy()
    df_result["cluster"] = final_labels

    st.success(f"✅ Training KMeans dengan k = {FINAL_K}")
    st.markdown("---")

    # =========================================================
    # 7) DISTRIBUSI CLUSTER
    # =========================================================
    st.write("### 6. Distribusi Cluster")

    cluster_count = df_result["cluster"].value_counts().sort_index().reset_index()
    cluster_count.columns = ["cluster", "jumlah_siswa"]

    fig_cluster = px.pie(
        cluster_count,
        names="cluster",
        values="jumlah_siswa",
        hole=0.45,
        title="Proporsi Jumlah Siswa per Cluster"
    )
    fig_cluster.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_cluster, use_container_width=True)
    st.dataframe(cluster_count, use_container_width=True)
    st.markdown("---")

    # =========================================================
    # 8) PROFILING CLUSTER (MEAN)
    # =========================================================
    st.write("### 7. Profiling Cluster (Rata-rata Fitur per Cluster)")

    profile_mean = df_result.groupby("cluster").mean(numeric_only=True).round(2)
    st.dataframe(profile_mean, use_container_width=True)

    st.info(
    "🔹 **Low Interest – Passive Students**\n\n"
    "Cluster ini merepresentasikan siswa yang:\n"
    "1. Tidak terlalu aktif dalam kegiatan olahraga maupun seni\n"
    "2. Memiliki ketertarikan yang terbatas pada aktivitas ekstrakurikuler\n"
    "3. Cenderung pasif dan lebih fokus pada aktivitas sosial standar tanpa minat dominan tertentu\n\n"
    "📌 **Karakter umum:** siswa dengan tingkat partisipasi dan eksplorasi minat yang rendah."
)

    st.info(
    "🔹 **High Interest – Active Students**\n\n"
    "Cluster ini merepresentasikan siswa yang:\n"
    "1. Aktif mengikuti berbagai kegiatan seni dan/atau olahraga\n"
    "2. Memiliki minat yang beragam dan tingkat keterlibatan tinggi\n"
    "3. Lebih ekspresif serta aktif dalam aktivitas sosial dan ekstrakurikuler\n\n"
    "📌 **Karakter umum:** siswa dengan minat tinggi dan kecenderungan aktif dalam pengembangan diri."
)

    st.markdown("---")

    # =========================================================
    # 9) TOP MINAT PER CLUSTER
    # =========================================================
    st.write("### 8. Top Minat per Cluster")

    interest_cols = [
        'dance', 'band', 'marching', 'music', 'rock', 'hair', 'dress', 'blonde',
        'basketball', 'football', 'soccer', 'softball', 'volleyball',
        'swimming', 'cheerleading', 'baseball', 'tennis', 'sports'
    ]
    interest_cols = [c for c in interest_cols if c in df_result.columns]

    if len(interest_cols) > 0:
        top_rows = []
        for c in sorted(df_result["cluster"].unique()):
            means = df_result[df_result["cluster"] == c][interest_cols].mean().sort_values(ascending=False)
            top3 = means.head(3)

            top_rows.append({
                "cluster": int(c),
                "Top 1": f"{top3.index[0]} ({top3.iloc[0]:.2f})",
                "Top 2": f"{top3.index[1]} ({top3.iloc[1]:.2f})",
                "Top 3": f"{top3.index[2]} ({top3.iloc[2]:.2f})",
            })

        top_df = pd.DataFrame(top_rows)
        st.dataframe(top_df, use_container_width=True)
    else:
        st.warning("Kolom minat tidak ditemukan untuk profiling top minat.")
    st.markdown("---")

    # =========================================================
    # 10) PCA VISUALIZATION (2D) - pakai X yg sudah filtered
    # =========================================================
    st.write("### 9. Visualisasi Cluster (PCA 2D)")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"], index=df_scaled.index)
    pca_df["cluster"] = final_labels

    fig_pca = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="cluster",
        title="Visualisasi Cluster dengan PCA (2D)",
        opacity=0.75
    )
    fig_pca.update_layout(height=520, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig_pca, use_container_width=True)

    st.caption(
        f"Explained Variance Ratio: PC1={pca.explained_variance_ratio_[0]:.2f}, "
        f"PC2={pca.explained_variance_ratio_[1]:.2f}"
    )
    st.markdown("---")

    # =========================================================
    # 11) Evaluasi dengan Silhouette Score (FINAL_K)
    # =========================================================
    st.write("### 10. Evaluasi Cluster")

    silhouette_avg = silhouette_score(X, final_labels)
    st.metric("Silhouette Score (k=2)", f"{silhouette_avg:.3f}")

    # Save PKL 
    import joblib
    joblib.dump(kmeans_final, "Finpro_model.pkl")
    joblib.dump(scaler, "Finpro_scaler.pkl")
    joblib.dump(used_cols, "Finpro_used_cols.pkl")
    joblib.dump(used_cols, "Finpro_features.pkl")
