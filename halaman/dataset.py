import pandas as pd
import streamlit as st
from utils import KAMUS, load_data
import charts
import model as M

df = load_data()
st.title("Tentang Dataset")
st.caption("Sumber: Mall Customers (Kaggle), format CSV")

tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan", "Kamus Kolom", "Lihat Data", "Kualitas Data"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Baris", f"{df.shape[0]:,}")
    c2.metric("Kolom", df.shape[1])
    c3.metric("Perempuan", f"{(df['Gender'] == 'Female').sum()}")
    c4.metric("Laki-laki", f"{(df['Gender'] == 'Male').sum()}")
    st.markdown(
        "- **Jenis masalah:** klasterisasi (unsupervised learning)\n"
        "- **Target:** tidak ada; kelompok ditemukan dari kemiripan fitur\n"
        "- **Fitur klaster:** `Age`, `Annual Income (k$)`, `Spending Score (1-100)`\n"
        "- **Tujuan:** membagi pelanggan menjadi segmen yang bermakna bagi strategi pemasaran")
    charts.tampil(charts.distribusi_gender(df))
    desk = df[M.FITUR].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(2)
    desk.columns = ["Rata-rata", "Std", "Min", "Q1", "Median", "Q3", "Maks"]
    st.dataframe(desk)

with tab2:
    kamus = pd.DataFrame([(k, v[0], v[1], str(df[k].dtype)) for k, v in KAMUS.items()],
                         columns=["Kolom", "Peran", "Keterangan", "Tipe data"])
    st.dataframe(kamus, hide_index=True)

with tab3:
    kiri, kanan = st.columns(2)
    gender = kiri.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
    umin, umax = kanan.slider("Rentang usia", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
    sub = df[df["Gender"].isin(gender) & df["Age"].between(umin, umax)]
    st.write(f"Menampilkan {len(sub):,} baris")
    st.dataframe(sub, hide_index=True)
    st.download_button("Download data terfilter (CSV)", sub.to_csv(index=False).encode("utf-8"),
                       file_name="mall_terfilter.csv", mime="text/csv")

with tab4:
    c1, c2, c3 = st.columns(3)
    c1.metric("Nilai hilang", int(df.isna().sum().sum()))
    c2.metric("Baris duplikat", int(df.duplicated().sum()))
    c3.metric("CustomerID unik", f"{df['CustomerID'].nunique()} / {len(df)}")
    st.subheader("Rentang nilai dan skala")
    rentang = pd.DataFrame({"Min": df[M.FITUR].min(), "Maks": df[M.FITUR].max()})
    rentang["Rentang"] = rentang["Maks"] - rentang["Min"]
    st.dataframe(rentang)
    st.info("Rentang tiap fitur berbeda (Usia 52, Income 122, Spending 98), sehingga **standardisasi wajib** sebelum K-Means "
            "yang memakai jarak Euclidean.")
    st.subheader("Kolom yang tidak dipakai sebagai fitur")
    st.markdown("- `CustomerID`: hanya penanda, tidak membawa informasi pola (dibuang).\n"
                "- `Gender`: biner dan bukan ukuran jarak yang wajar bagi K-Means; sebarannya diperiksa setelah klasterisasi.")
