import pandas as pd
import streamlit as st
from utils import model_final, pca
import charts
import model as M

r, info = model_final(), pca()
st.title("Reduksi Dimensi dengan PCA")
st.caption("Pada proyek ini PCA dipakai untuk visualisasi dan diagnosis, bukan sebagai langkah wajib sebelum K-Means.")

c1, c2, c3 = st.columns(3)
c1.metric("Fitur sebelum PCA", 3)
c2.metric("Varians PC1 + PC2", f"{info['cumulative'][1] * 100:.1f}%")
c3.metric("Varians PC1 + PC2 + PC3", f"{info['cumulative'][2] * 100:.1f}%")

t1, t2, t3 = st.tabs(["Scree Plot", "Interpretasi Komponen", "Segmen pada ruang PCA"])
with t1:
    charts.tampil(charts.scree(info))
    st.caption("Karena hanya ada tiga fitur, mereduksi dimensi tidak menghemat banyak; dua komponen sudah menangkap sebagian besar varians.")
with t2:
    load = pd.DataFrame(info["components"][:2].T, index=[M.SINGKAT[c] for c in M.FITUR], columns=["PC1", "PC2"]).round(3)
    st.dataframe(load)
    st.write("Semakin besar nilai mutlak *loading*, semakin kuat fitur itu membentuk komponen. "
             "Nilai dihitung pada data yang sudah di-Yeo-Johnson dan distandardisasi.")
with t3:
    charts.tampil(charts.pca_scatter(info["proj"], r["labels"], r["nama"], info["cumulative"][1]))
    st.info("Segmen tampak terpisah pada proyeksi 2D, yang menguatkan hasil klasterisasi. Komponen utama sulit dijelaskan ke pihak bisnis, "
            "sehingga model dan profil segmen tetap memakai fitur asli.")
