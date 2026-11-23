import pandas as pd
import streamlit as st
from utils import algoritma, load_data, model_final, sweep_semua
import model as M

df, r = load_data(), model_final()
m, prof = r["metrik"], r["profil"]
sw = sweep_semua()
s6 = sw[sw["K"] == M.K_FINAL].set_index("Scaler")["Silhouette"]
alg = algoritma(M.K_FINAL)

st.title("Kesimpulan & Saran")

st.subheader("Kesimpulan")
prio = prof.sort_values("Prioritas", ascending=False).iloc[0]
st.markdown(
    f"1. Dataset **{len(df)} pelanggan** dengan tiga fitur numerik, tanpa data hilang dan duplikat, layak dikelompokkan tanpa label.\n"
    f"2. **Yeo-Johnson** dipilih karena aman untuk nilai nol/negatif, parameternya dipelajari dari data, dan menaikkan Silhouette K = {M.K_FINAL} "
    f"dari {s6['StandardScaler']:.4f} menjadi {s6['Yeo-Johnson']:.4f}.\n"
    f"3. **K-Means** dipilih karena unggul atas Ward ({alg.loc['Ward (aglomeratif)', 'Silhouette']:.4f}) dan Gaussian Mixture "
    f"({alg.loc['Gaussian Mixture', 'Silhouette']:.4f}) pada ketiga indeks, stabil terhadap inisialisasi, dan mudah ditafsirkan.\n"
    f"4. Model akhir menghasilkan **{len(prof)} segmen** (Silhouette {m['Silhouette']:.3f}, Davies-Bouldin {m['Davies-Bouldin']:.3f}, "
    f"Calinski-Harabasz {m['Calinski-Harabasz']:.1f}).\n"
    f"5. Segmen **{prio['Segmen']}** ({int(prio['Jumlah'])} pelanggan) memuat {int(prio['Prioritas'])} dari {int(prof['Prioritas'].sum())} Pelanggan Prioritas.")

st.subheader("Ringkasan segmen")
tb = prof[["Segmen", "Jumlah", "Usia", "Pendapatan", "Belanja"]].set_index("Segmen").sort_values("Belanja", ascending=False)
tb.columns = ["Jumlah", "Usia", "Income (k$)", "Spending"]
st.dataframe(tb)

st.subheader("Saran bisnis")
st.markdown(
    "- **Premium Muda**: jaga dengan program VIP dan akses awal koleksi baru.\n"
    "- **Kaya tetapi Hemat**: segmen dengan potensi terbesar; uji kampanye personal untuk menaikkan skor belanja.\n"
    "- **Royal Berpenghasilan Rendah**: gunakan promo dan paket hemat, dengan batas risiko kredit.\n"
    "- Evaluasi segmen secara berkala dan latih ulang pipeline bila sebaran pelanggan berubah.")

st.subheader("Keterbatasan")
st.markdown(
    "- N = 200 dan tidak ada label sebenarnya, sehingga evaluasi bersifat internal.\n"
    "- Silhouette sekitar 0,46 menunjukkan struktur sedang, bukan pemisahan tegas; K = 5 juga cukup dekat dan BIC Gaussian Mixture memilih 5 komponen.\n"
    "- Yeo-Johnson sedikit memperburuk simetri Spending Score; manfaatnya pada tiga fitur asli tergolong sedang.\n"
    "- Label Pelanggan Prioritas dibuat dari kuantil fitur yang sama sehingga bukan validasi independen.\n"
    "- Pada dashboard ini transformasi dipelajari dari seluruh data; untuk produksi, latih hanya pada data latih lalu simpan pipeline.")
