import pandas as pd
import streamlit as st
from utils import algoritma, bic, dbscan, model_final, paradoks, stabil, sweep_semua
import charts
import model as M

r = model_final()
sw = sweep_semua()
st.title("Modelling & Evaluasi")
st.caption("Tanpa label, kualitas klaster dinilai dengan indeks internal. Akurasi tidak berlaku di sini.")

m = r["metrik"]
st.success(f"Model akhir: **Yeo-Johnson + K-Means (K = {M.K_FINAL})**. Silhouette {m['Silhouette']:.4f}, "
           f"Davies-Bouldin {m['Davies-Bouldin']:.4f}, Calinski-Harabasz {m['Calinski-Harabasz']:.2f}, inertia {r['pipe'][-1].inertia_:.3f}.")

t1, t2, t3, t4, t5 = st.tabs(["Pemilihan K", "Perbandingan Algoritma", "Silhouette per Segmen", "Stabilitas", "Paradoks Akurasi"])

with t1:
    metrik_pilih = st.radio("Metrik", ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Inertia"], horizontal=True)
    rentang = st.slider("Rentang K", 2, 10, (2, 10))
    s = sw[sw["K"].between(*rentang)]
    charts.tampil(charts.kurva_k(s, metrik_pilih, M.K_FINAL if rentang[0] <= M.K_FINAL <= rentang[1] else None))
    pv = s.pivot(index="K", columns="Scaler", values=metrik_pilih).round(4)
    st.dataframe(pv.style.highlight_max(axis=0, color="#D5DAF5") if metrik_pilih in ("Silhouette", "Calinski-Harabasz")
                 else pv.style.highlight_min(axis=0, color="#D5DAF5") if metrik_pilih == "Davies-Bouldin" else pv)
    st.caption("Silhouette tertinggi pada K = 6 untuk kedua penskalaan; Yeo-Johnson menaikkannya dari 0,4284 menjadi 0,4565. "
               "Inertia selalu turun saat K naik, sehingga tidak bisa dipakai sendirian.")

with t2:
    k = st.slider("Jumlah klaster untuk perbandingan", 3, 8, M.K_FINAL)
    tb = algoritma(k)
    charts.tampil(charts.bar_algoritma(tb))
    st.dataframe(tb)
    st.caption("Dihitung pada data terskala Yeo-Johnson. Pada K = 6, K-Means unggul di ketiga indeks.")
    kiri, kanan = st.columns(2)
    with kiri:
        st.markdown("**Gaussian Mixture: BIC**")
        charts.tampil(charts.bic_plot(bic()))
        st.caption("BIC (data StandardScaler) memilih K = 5, tetapi selisihnya dengan K = 3 dan 4 kecil; Silhouette tetap mendukung K = 6.")
    with kanan:
        st.markdown("**DBSCAN (StandardScaler)**")
        db = dbscan()
        st.dataframe(db[db["Jumlah klaster"] >= 2].sort_values("Noise (%)").head(12), hide_index=True)
        st.caption("Konfigurasi yang menghasilkan 2+ klaster membuang banyak titik sebagai noise karena kepadatan antar-segmen tidak seragam.")

with t3:
    sil = M.silhouette_per_klaster(r["Z"], r["labels"])
    charts.tampil(charts.silhouette_plot(sil, r["nama"]))
    ring = sil.groupby("Klaster")["silhouette"].agg(["mean", "min", lambda s: (s < 0).sum()])
    ring.columns = ["Rata-rata", "Minimum", "Titik silhouette < 0"]
    ring.index = [r["nama"][i] for i in ring.index]
    st.dataframe(ring.round(3))
    st.caption("Silhouette rata-rata sekitar 0,46: struktur klaster sedang, ada sebagian pelanggan di perbatasan antar-segmen.")

with t4:
    n_init = st.radio("Nilai n_init K-Means", [10, 1], horizontal=True, help="n_init = banyaknya inisialisasi centroid yang dicoba")
    s = stabil(n_init)
    a, b, c = st.columns(3)
    a.metric("Silhouette min - maks", f"{s['Silhouette'].min():.4f} - {s['Silhouette'].max():.4f}")
    b.metric("ARI minimum", f"{s['ARI vs seed 0'].min():.3f}")
    c.metric("ARI rata-rata", f"{s['ARI vs seed 0'].mean():.3f}")
    charts.tampil(charts.stabilitas_plot(s))
    st.caption("30 inisialisasi acak. Dengan n_init = 10 hasilnya stabil; n_init = 1 lebih mudah terjebak optimum lokal.")

with t5:
    st.write("**Paradoks akurasi pada klasterisasi.** Metrik yang tampak bagus bisa menyesatkan. Di sini dibuat label bantu "
             "*Pelanggan Prioritas* (Income dan Spending sama-sama di kuantil 75 ke atas, 22 pelanggan = 11%). "
             "Setiap klaster 'memilih' kelas mayoritasnya, lalu dihitung purity (mirip akurasi) dan recall kelas Prioritas.")
    p = paradoks()
    charts.tampil(charts.paradoks_plot(p))
    st.dataframe(p.round(3).set_index("K"))
    st.markdown("- **K = 1** sudah mencapai purity 0,89 tetapi recall 0: tebakan 'bukan Prioritas' untuk semua orang.\n"
                "- **K = 200** mencapai purity 1,0: tiap pelanggan satu klaster, tidak ada makna segmentasi.\n"
                "- Karena itu K dipilih dengan Silhouette, Davies-Bouldin, dan Calinski-Harabasz, bukan purity.")
    st.caption("Dihitung pada seluruh data tanpa validasi silang. Label bantu dibuat dari fitur yang sama dengan klasterisasi, "
               "jadi bukan validasi independen.")
