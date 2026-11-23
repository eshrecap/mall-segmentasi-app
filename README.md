# Dashboard Segmentasi Pelanggan Mall

Tugas Machine Learning, Kelompok 3, Teknik Informatika S-1, Universitas Pamulang.
Dashboard Streamlit untuk segmentasi pelanggan (unsupervised) pada dataset Mall Customers:
**Yeo-Johnson + K-Means (K = 6)**.

## Menjalankan
```bash
python -m venv env
env\Scripts\activate          # Windows  (Linux/Mac: source env/bin/activate)
pip install -r requirements.txt
python -m streamlit run app.py
```
Buka http://localhost:8501. Model dilatih otomatis saat aplikasi mulai (200 baris, kurang dari 1 detik),
jadi tidak ada file model yang perlu dibuat atau dimuat.

## Halaman
| Kelompok | Halaman | Isi |
| --- | --- | --- |
| Pengantar | Beranda, Tentang Dataset, EDA | ringkasan, kamus kolom, kualitas data, grafik interaktif dengan filter |
| Pengolahan & Model | Preprocessing, PCA, Modelling & Evaluasi | Yeo-Johnson, pemilihan K, K-Means vs Ward/GMM/DBSCAN, stabilitas, paradoks akurasi |
| Penerapan | Profil Segmen, Tentukan Segmen, Segmentasi Massal, Kesimpulan | persona dan strategi, form pelanggan baru, unggah CSV |

## Struktur
```
app.py          navigasi (st.navigation)
model.py        logika inti tanpa Streamlit (pipeline, metrik, penamaan segmen)
utils.py        data, cache, tema, identitas
charts.py       semua grafik
halaman/        satu file per halaman
tests/          uji logika inti:  python -m pytest -q
data/           Mall_Customers.csv
```
