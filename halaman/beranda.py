import streamlit as st
from utils import DOSEN, KAMPUS, PEMBUAT, load_data, model_final

df, r = load_data(), model_final()
m = r["metrik"]

st.title("Segmentasi Pelanggan Mall")
st.caption("Implementasi Machine Learning (unsupervised) berbasis web dengan Streamlit")

st.markdown(
    "Dashboard ini memaparkan seluruh tahapan **pengelompokan pelanggan mall tanpa label** pada dataset *Mall Customers*: "
    "pengenalan data, EDA, preprocessing dengan **Yeo-Johnson**, reduksi dimensi (PCA), pemilihan jumlah klaster dan algoritma "
    "**K-Means**, hingga **menentukan segmen pelanggan baru** secara langsung."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Jumlah pelanggan", f"{len(df):,}")
c2.metric("Jumlah segmen (K)", len(r["nama"]))
c3.metric("Silhouette", f"{m['Silhouette']:.3f}")
c4.metric("Davies-Bouldin", f"{m['Davies-Bouldin']:.3f}", help="Semakin kecil semakin baik")

st.divider()
st.subheader("Alur pengerjaan")
a, b, c, d = st.columns(4)
for kol, (jd, isi) in zip([a, b, c, d], [
    ("1. Dataset & EDA", "200 pelanggan, 3 fitur numerik, tanpa data hilang dan duplikat. Tidak ada kolom target."),
    ("2. Preprocessing", "Yeo-Johnson + standardisasi agar fitur sebanding dan distribusi lebih simetris."),
    ("3. Modelling", "Memilih K lewat Silhouette, Davies-Bouldin, Calinski-Harabasz; membandingkan K-Means, Ward, GMM, DBSCAN."),
    ("4. Penerapan", "Profil dan strategi tiap segmen, serta penentuan segmen pelanggan baru satu per satu atau lewat CSV."),
]):
    with kol.container(border=True):
        st.markdown(f"**{jd}**")
        st.write(isi)

st.divider()
st.subheader("Anggota Kelompok 3")
cols = st.columns(3)
for kol, (nama, nim) in zip(cols, PEMBUAT):
    with kol.container(border=True):
        st.markdown(f"**{nama}**")
        st.caption(f"NIM {nim}")
if DOSEN:
    st.write(f"Dosen pengampu: **{DOSEN}**")
st.caption(KAMPUS)

with st.expander("Mengapa segmentasi pelanggan penting?"):
    st.markdown(
        "- Promosi yang seragam sering kurang tepat sasaran; segmen memungkinkan pesan yang berbeda untuk kelompok yang berbeda.\n"
        "- Pelanggan berpenghasilan tinggi tetapi hemat butuh pendekatan berbeda dari pelanggan muda yang royal tetapi berpenghasilan rendah.\n"
        "- Karena tidak ada label, kualitas klaster dinilai dengan **indeks internal** dan kegunaannya bagi bisnis, bukan dengan akurasi."
    )
with st.expander("Apa keunggulan website ini?"):
    st.markdown(
        "Website ini tidak hanya menampilkan data, tetapi menjalankan **pipeline Yeo-Johnson + K-Means** secara langsung: "
        "pengguna dapat mengubah K, membandingkan penskalaan dan algoritma, melihat uji stabilitas, serta memasukkan pelanggan baru."
    )
