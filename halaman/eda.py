import streamlit as st
from utils import NAMA_PANJANG, load_data
import charts
import model as M

df = load_data()
st.title("Eksplorasi Data (EDA)")
st.caption("Pola apa yang tampak pada pelanggan sebelum dikelompokkan?")

with st.sidebar:
    st.header("Filter EDA")
    gender = st.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
    umin, umax = st.slider("Rentang usia", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
dff = df[df["Gender"].isin(gender) & df["Age"].between(umin, umax)]
if len(dff) < 5:
    st.warning("Data terlalu sedikit untuk filter ini. Ubah filter di sidebar.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Pelanggan terfilter", f"{len(dff):,}")
c2.metric("Rata-rata Income (k$)", f"{dff[M.INC].mean():.1f}", f"{dff[M.INC].mean() - df[M.INC].mean():+.1f} vs semua", delta_color="off")
c3.metric("Rata-rata Spending", f"{dff[M.SPD].mean():.1f}", f"{dff[M.SPD].mean() - df[M.SPD].mean():+.1f} vs semua", delta_color="off")

t1, t2, t3, t4, t5, t6 = st.tabs(["Distribusi", "Gender", "Usia", "Income vs Spending", "Korelasi", "Outlier"])
with t1:
    pilih = st.radio("Pilih fitur", M.FITUR, horizontal=True, format_func=lambda c: NAMA_PANJANG[c])
    per_gender = st.toggle("Pisahkan menurut gender")
    charts.tampil(charts.hist_numerik(dff, pilih, per_gender))
    st.caption("Ketiga fitur sudah hampir simetris (|skewness| < 0,5); Age sedikit menceng ke kanan.")
with t2:
    charts.tampil(charts.distribusi_gender(dff))
    tb = dff.groupby("Gender")[M.FITUR].mean().round(2)
    st.dataframe(tb)
    st.caption("Rata-rata antar-gender mirip; gender bukan pembeda utama pada data ini.")
with t3:
    charts.tampil(charts.spending_per_usia(dff))
    st.caption("Kelompok usia 26-35 paling tinggi skor belanjanya, 46-55 paling rendah.")
with t4:
    charts.tampil(charts.scatter_gender(dff))
    st.caption("Titik membentuk beberapa kelompok yang terpisah jelas; inilah dasar dilakukannya klasterisasi.")
with t5:
    kiri, kanan = st.columns(2)
    with kiri:
        charts.tampil(charts.korelasi(dff))
    with kanan:
        st.markdown("**Cara membaca:** mendekati 1 atau -1 berarti hubungan linear kuat.\n\n"
                    "- `Age` dan `Spending Score` berkorelasi negatif sedang: makin tua, skor belanja cenderung turun.\n"
                    "- `Income` hampir tidak berkorelasi dengan fitur lain, sehingga tidak ada multikolinearitas. "
                    "Korelasi rendah bukan berarti tanpa struktur: pola kelompok justru muncul pada kombinasi Income dan Spending.")
with t6:
    st.dataframe(M.tabel_outlier_iqr(dff))
    st.caption("Aturan IQR 1,5x. Outlier yang ditemukan adalah pelanggan berpendapatan tinggi yang valid, jadi dipertahankan.")

with st.expander("Ringkasan temuan utama"):
    kor = dff[M.FITUR].corr()
    st.markdown(
        f"- Korelasi Age-Spending: **{kor.loc[M.AGE, M.SPD]:.2f}**; Income-Spending: **{kor.loc[M.INC, M.SPD]:.2f}**.\n"
        f"- Rata-rata Spending perempuan {dff.loc[dff['Gender'] == 'Female', M.SPD].mean():.1f} dan laki-laki "
        f"{dff.loc[dff['Gender'] == 'Male', M.SPD].mean():.1f} (data terfilter).\n"
        "- Pola kelompok terlihat jelas pada diagram Income vs Spending, bukan pada fitur tunggal.")
