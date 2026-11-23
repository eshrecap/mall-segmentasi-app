import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import PowerTransformer
from utils import NAMA_PANJANG, fitur
import charts
import model as M

X = fitur()
st.title("Preprocessing: Yeo-Johnson")
st.caption("Menyamakan skala dan merapikan distribusi fitur sebelum K-Means, tanpa membuang data.")

st.progress(1.0, text="Alur: Buang CustomerID -> Pilih 3 fitur numerik -> Yeo-Johnson -> Standardisasi -> K-Means")

with st.expander("1. Mengapa perlu penskalaan?", expanded=True):
    rentang = (X.max() - X.min()).rename("Rentang nilai")
    st.dataframe(rentang)
    st.write("K-Means memakai jarak Euclidean. Tanpa penskalaan, fitur dengan rentang besar (Income) mendominasi jarak. "
             "Karena itu setiap fitur diubah ke rata-rata 0 dan simpangan baku 1.")

with st.expander("2. Yeo-Johnson: skewness sebelum dan sesudah", expanded=True):
    t = M.tabel_skew(X)
    lam = M.lambda_yj(X)
    tabel = t.copy()
    tabel["Lambda (λ)"] = pd.Series(lam)
    st.dataframe(tabel)
    charts.tampil(charts.skew_bar(t))
    st.write("`PowerTransformer(method='yeo-johnson', standardize=True)` mengestimasi λ tiap fitur lewat kemungkinan maksimum, "
             "lalu otomatis menstandardisasi hasilnya.")
    st.warning("Jujur soal hasilnya: pada tiga fitur asli yang sudah hampir simetris, manfaatnya kecil. Age dan Income menjadi lebih simetris, "
               "tetapi Spending Score justru sedikit lebih menceng (-0,047 menjadi -0,222). Keunggulan Yeo-Johnson paling nyata pada fitur turunan yang sangat miring.")

with st.expander("3. Lihat bentuk distribusinya"):
    kol = st.selectbox("Pilih fitur", M.FITUR, format_func=lambda c: NAMA_PANJANG[c])
    Z = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(X)
    charts.tampil(charts.hist_sebelum_sesudah(X[kol], Z[:, M.FITUR.index(kol)], NAMA_PANJANG[kol]))

with st.expander("4. Mengapa Yeo-Johnson, bukan log atau Box-Cox?"):
    # fitur turunan: perkalian Income x Spending (sangat miring ke kanan)
    from scipy.stats import skew
    est = (X[M.INC] * X[M.SPD]).to_frame("Estimasi_Belanja")
    yj = PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(est)[:, 0]
    lg = np.log1p(est["Estimasi_Belanja"])
    tb = pd.DataFrame({"Skewness": [skew(est["Estimasi_Belanja"]), skew(lg), skew(yj)]},
                      index=["Asli", "log1p", "Yeo-Johnson"]).round(3)
    st.dataframe(tb)
    st.markdown(
        "- **Log** tidak aman untuk nol/negatif dan di sini mengoreksi berlebihan (miring ke sisi sebaliknya).\n"
        "- **Box-Cox** hanya untuk nilai positif.\n"
        "- **Yeo-Johnson** menerima nol dan negatif, parameter dipelajari dari data, dan monoton sehingga outlier valid tidak dibuang.")
    st.caption("Fitur turunan `Estimasi_Belanja = Income x Spending` dibuat khusus untuk demo ini (tidak dipakai pada model akhir).")

with st.expander("5. Kode pipeline"):
    st.code('''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans

pipeline = Pipeline([
    ("scaler", PowerTransformer(method="yeo-johnson", standardize=True)),
    ("kmeans", KMeans(n_clusters=6, n_init=10, random_state=42)),
])
pipeline.fit(df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]])''', language="python")
    st.caption("Untuk produksi, `fit` hanya pada data latih lalu simpan pipeline; pelanggan baru cukup `pipeline.predict(...)`.")
