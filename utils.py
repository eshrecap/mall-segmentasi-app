"""Fungsi bersama untuk semua halaman: pemuatan data, model (di-cache), tema, dan data referensi."""
from pathlib import Path

import pandas as pd
import streamlit as st

import model as M

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- identitas
PEMBUAT = [
    ("Dina Aura Nur", "241011401694"),
    ("Nur Hidayat", "241011401641"),
    ("Nur Rizka Sasi Kirana", "241011401355"),
]
DOSEN = ""   # isi nama dosen pengampu bila ingin ditampilkan di Beranda
KAMPUS = "Program Studi Teknik Informatika S-1, Fakultas Ilmu Komputer, Universitas Pamulang"

KAMUS = {
    "CustomerID": ("Identitas", "Nomor unik pelanggan (tidak dipakai sebagai fitur)"),
    "Gender": ("Demografi", "Jenis kelamin (Female / Male); tidak dimasukkan ke K-Means, dipakai untuk profil segmen"),
    "Age": ("Fitur klaster", "Usia pelanggan (tahun)"),
    "Annual Income (k$)": ("Fitur klaster", "Pendapatan tahunan, dalam ribuan dolar"),
    "Spending Score (1-100)": ("Fitur klaster", "Skor belanja yang diberikan mall (1 = rendah, 100 = tinggi)"),
}
NAMA_PANJANG = {M.AGE: "Usia", M.INC: "Pendapatan Tahunan (k$)", M.SPD: "Skor Belanja (1-100)"}

INDIGO = "#4B5FD6"

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1200px;}
h1, h2, h3 {color: #2B3A9E;}
div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #D5DAF5; border-radius: 14px;
                             padding: 12px 16px; box-shadow: 0 1px 4px rgba(75,95,214,.10);}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {background: #E7EAFB; border-radius: 10px 10px 0 0; padding: 8px 16px;}
.stTabs [aria-selected="true"] {background: #FFFFFF;}
</style>
"""


def pasang_tema():
    st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data & model
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "Mall_Customers.csv")


@st.cache_data
def fitur() -> pd.DataFrame:
    return load_data()[M.FITUR].copy()


@st.cache_resource
def model_final() -> dict:
    """Pipeline akhir dilatih saat aplikasi start (200 baris, < 1 detik), jadi tidak perlu file .joblib
    yang bisa gagal dimuat bila versi scikit-learn berbeda."""
    df = load_data()
    r = M.latih_final(df[M.FITUR])
    r["profil"] = M.profil_segmen(df, r["labels"])
    r["nama"] = r["profil"]["Segmen"].to_dict()
    r["pusat"] = {k: (row["Pendapatan"], row["Belanja"]) for k, row in r["profil"].iterrows()}
    return r


@st.cache_data
def sweep_semua() -> pd.DataFrame:
    X = fitur()
    return pd.concat([M.sweep_k(X, s) for s in M.SCALERS], ignore_index=True)


@st.cache_data
def algoritma(k: int) -> pd.DataFrame:
    return M.bandingkan_algoritma(M.buat_scaler("Yeo-Johnson").fit_transform(fitur()), k)


@st.cache_data
def bic() -> pd.DataFrame:
    return M.bic_gmm(M.buat_scaler("StandardScaler").fit_transform(fitur()))


@st.cache_data
def dbscan() -> pd.DataFrame:
    return M.dbscan_grid(M.buat_scaler("StandardScaler").fit_transform(fitur()))


@st.cache_data
def stabil(n_init: int) -> pd.DataFrame:
    return M.stabilitas(fitur(), n_init=n_init)


@st.cache_data
def paradoks() -> pd.DataFrame:
    y, _, _ = M.label_prioritas(load_data())
    return M.paradoks_akurasi(fitur(), y)


@st.cache_data
def pca() -> dict:
    return M.pca_info(model_final()["Z"])


def nama_segmen(klaster: int) -> str:
    return model_final()["nama"][klaster]
