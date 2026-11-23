import pandas as pd
import streamlit as st
from utils import fitur, load_data, model_final
import charts
import model as M

df, r = load_data(), model_final()
X = fitur()

st.title("Tentukan Segmen Satu Pelanggan")
st.caption("Isi profil pelanggan, lalu lihat segmen, tingkat kepastian, dan strategi yang disarankan.")

CONTOH = {
    "Isi manual": (35, 60, 50),
    "Contoh: premium muda": (30, 90, 85),
    "Contoh: kaya tetapi hemat": (45, 95, 15),
    "Contoh: muda, income rendah, royal": (23, 25, 80),
}
contoh = st.selectbox("Profil awal", list(CONTOH))
u0, i0, s0 = CONTOH[contoh]
k = contoh.replace(" ", "_")

with st.form("form_segmen"):
    c1, c2, c3 = st.columns(3)
    usia = c1.slider("Usia (tahun)", int(X[M.AGE].min()), int(X[M.AGE].max()), u0, key=f"{k}_u")
    income = c2.slider("Pendapatan tahunan (k$)", int(X[M.INC].min()), int(X[M.INC].max()), i0, key=f"{k}_i")
    belanja = c3.slider("Spending Score (1-100)", 1, 100, s0, key=f"{k}_s")
    kirim = st.form_submit_button("Tentukan segmen", type="primary")

if kirim:
    baru = pd.DataFrame([{M.AGE: usia, M.INC: income, M.SPD: belanja}])
    hsl = M.prediksi(r["pipe"], baru).iloc[0]
    kl = int(hsl["Klaster"])
    nama = r["nama"][kl]
    row = r["profil"].loc[kl]

    st.divider()
    st.subheader(f"Hasil: {nama}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Segmen", nama)
    m2.metric("Tingkat kepastian", f"{hsl['Kepastian'] * 100:.0f}%",
              help="1 - (jarak ke centroid terdekat / jarak ke centroid terdekat kedua). Mendekati 0% = di perbatasan dua segmen.")
    m3.metric("Anggota segmen", f"{int(row['Jumlah'])} pelanggan")
    st.progress(float(min(max(hsl["Kepastian"], 0), 1)), text="Seberapa jelas pelanggan ini masuk segmen tersebut")
    if hsl["Kepastian"] < 0.25:
        st.warning("Pelanggan ini berada di dekat perbatasan dua segmen; strategi segmen lain mungkin juga relevan.")

    kiri, kanan = st.columns([3, 2])
    with kiri:
        charts.tampil(charts.scatter_segmen(df, r["labels"], r["nama"], r["pusat"], titik_baru=(income, belanja)))
    with kanan:
        st.markdown("**Strategi yang disarankan**")
        st.write(row["Strategi"])
        charts.tampil(charts.bar_bandingkan(
            {"Usia": usia, "Income": income, "Spending": belanja},
            {"Usia": row["Usia"], "Income": row["Pendapatan"], "Spending": row["Belanja"]}))
    st.caption("Gender tidak dipakai model. Hasil adalah penempatan ke segmen terdekat, bukan prediksi perilaku di masa depan.")
