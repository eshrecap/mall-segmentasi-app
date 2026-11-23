import pandas as pd
import streamlit as st
from utils import load_data, model_final
import charts
import model as M

r = model_final()
st.title("Segmentasi Massal (CSV)")
st.caption("Unggah CSV berisi banyak pelanggan, lalu unduh hasil segmennya.")

with st.expander("Kolom yang dibutuhkan"):
    st.write(", ".join(f"`{c}`" for c in M.FITUR))
    st.caption("Kolom lain (mis. CustomerID, Gender) boleh ada dan akan dibawa ke hasil. Format sama dengan Mall_Customers.csv.")

file = st.file_uploader("Unggah file CSV", type=["csv"])
pakai_contoh = st.checkbox("Tidak punya file? Pakai 50 baris contoh dari dataset")

data = None
if file is not None:
    try:
        data = pd.read_csv(file)
    except Exception as e:  # file rusak tidak boleh menjatuhkan aplikasi
        st.error(f"Gagal membaca file: {e}")
        st.stop()
elif pakai_contoh:
    data = load_data().sample(50, random_state=7).reset_index(drop=True)

if data is None:
    st.info("Unggah file CSV atau centang contoh data untuk memulai.")
    st.stop()

hilang = [c for c in M.FITUR if c not in data.columns]
if hilang:
    st.error("Kolom berikut tidak ada di file: " + ", ".join(hilang))
    st.stop()

num = data[M.FITUR].apply(pd.to_numeric, errors="coerce")
valid = num.notna().all(axis=1)
if (~valid).any():
    st.warning(f"{int((~valid).sum())} baris dilewati karena nilai kosong atau bukan angka.")
if valid.sum() == 0:
    st.error("Tidak ada baris valid untuk diproses.")
    st.stop()

hsl = M.prediksi(r["pipe"], num[valid])
out = data.loc[valid].copy()
out["Segmen"] = hsl["Klaster"].map(r["nama"]).values
out["Kepastian (%)"] = (hsl["Kepastian"] * 100).round(1).values

c1, c2, c3 = st.columns(3)
c1.metric("Pelanggan diproses", f"{len(out):,}")
c2.metric("Jumlah segmen terisi", out["Segmen"].nunique())
c3.metric("Rata-rata kepastian", f"{out['Kepastian (%)'].mean():.0f}%")

t1, t2 = st.tabs(["Tabel hasil", "Sebaran segmen"])
with t1:
    awal = [c for c in ["CustomerID"] if c in out.columns]
    st.dataframe(out[awal + ["Segmen", "Kepastian (%)"] + M.FITUR].sort_values("Kepastian (%)"), hide_index=True)
    st.download_button("Download hasil segmentasi (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="hasil_segmentasi.csv", mime="text/csv")
    st.caption("Diurutkan dari kepastian terendah: pelanggan di perbatasan segmen ada di bagian atas.")
with t2:
    charts.tampil(charts.distribusi_segmen(out["Segmen"].value_counts()))
