"""Logika inti segmentasi (murni Python, tanpa Streamlit) sehingga mudah diuji.

Pipeline akhir: PowerTransformer (Yeo-Johnson, standardize=True) -> KMeans (K=6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

RS = 42
K_FINAL = 6
AGE, INC, SPD = "Age", "Annual Income (k$)", "Spending Score (1-100)"
FITUR = [AGE, INC, SPD]
SINGKAT = {AGE: "Usia", INC: "Pendapatan", SPD: "Skor Belanja"}
SCALERS = ["Yeo-Johnson", "StandardScaler"]


# ----------------------------------------------------------------------------- pipeline
def buat_scaler(nama: str):
    if nama == "Yeo-Johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)
    return StandardScaler()


def bangun_pipeline(k: int = K_FINAL, scaler: str = "Yeo-Johnson", n_init: int = 10, seed: int = RS) -> Pipeline:
    return Pipeline([("scaler", buat_scaler(scaler)),
                     ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=seed))])


def latih_final(X: pd.DataFrame, k: int = K_FINAL, scaler: str = "Yeo-Johnson") -> dict:
    pipe = bangun_pipeline(k, scaler).fit(X)
    Z = pipe[:-1].transform(X)
    labels = pipe.predict(X)
    return {"pipe": pipe, "Z": Z, "labels": labels, "metrik": metrik(Z, labels)}


def metrik(Z, labels) -> dict:
    if len(set(labels)) < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    return {"Silhouette": float(silhouette_score(Z, labels)),
            "Davies-Bouldin": float(davies_bouldin_score(Z, labels)),
            "Calinski-Harabasz": float(calinski_harabasz_score(Z, labels))}


# ----------------------------------------------------------------------------- data & statistik
def lambda_yj(X: pd.DataFrame) -> dict:
    pt = PowerTransformer(method="yeo-johnson", standardize=True).fit(X)
    return dict(zip(X.columns, np.round(pt.lambdas_, 3)))


def tabel_skew(X: pd.DataFrame) -> pd.DataFrame:
    Z = pd.DataFrame(PowerTransformer(method="yeo-johnson", standardize=True).fit_transform(X), columns=X.columns)
    return pd.DataFrame({"Skewness asli": [skew(X[c]) for c in X.columns],
                         "Skewness Yeo-Johnson": [skew(Z[c]) for c in X.columns]}, index=X.columns).round(3)


def tabel_outlier_iqr(df: pd.DataFrame, cols=FITUR) -> pd.DataFrame:
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({"Fitur": c, "Batas bawah": round(lo, 2), "Batas atas": round(hi, 2),
                     "Jumlah outlier": int(((df[c] < lo) | (df[c] > hi)).sum())})
    return pd.DataFrame(rows).set_index("Fitur")


def label_prioritas(df: pd.DataFrame):
    """Pelanggan Prioritas = Income dan Spending sama-sama >= kuantil 75 (label bantu untuk evaluasi)."""
    q_inc, q_spd = df[INC].quantile(.75), df[SPD].quantile(.75)
    return ((df[INC] >= q_inc) & (df[SPD] >= q_spd)).astype(int), float(q_inc), float(q_spd)


# ----------------------------------------------------------------------------- penamaan segmen
STRATEGI = {
    "Premium Muda": "Pelanggan prioritas: berpenghasilan tinggi dan royal. Program loyalitas VIP, akses awal koleksi baru, layanan personal.",
    "Kaya tetapi Hemat": "Potensi terbesar yang belum tergarap. Penawaran personal, produk bernilai tinggi, undangan acara eksklusif.",
    "Royal Berpenghasilan Rendah": "Peka promo. Diskon, paket hemat, cicilan ringan; waspadai risiko kredit.",
    "Hemat Berpenghasilan Rendah": "Prioritas rendah. Komunikasi berbiaya rendah (notifikasi, kupon digital).",
    "Paruh Baya Menengah": "Segmen stabil dan terbesar. Pertahankan lewat produk kebutuhan rutin dan poin loyalitas.",
    "Muda Menengah": "Belum terikat. Kampanye pengenalan, keanggotaan, dan promo musiman.",
}
URUTAN_NAMA = list(STRATEGI)


def _level(v, rendah, tinggi):
    return "L" if v < rendah else ("H" if v > tinggi else "M")


def beri_nama(usia: float, income: float, belanja: float) -> str:
    """Nama segmen dari rata-rata centroid (bukan dari nomor klaster, yang bisa berubah antar-versi)."""
    li, ls = _level(income, 40, 70), _level(belanja, 35, 65)
    peta = {("H", "H"): "Premium Muda", ("H", "L"): "Kaya tetapi Hemat",
            ("L", "H"): "Royal Berpenghasilan Rendah", ("L", "L"): "Hemat Berpenghasilan Rendah"}
    if (li, ls) in peta:
        return peta[(li, ls)]
    if (li, ls) == ("M", "M"):
        return "Muda Menengah" if usia < 35 else "Paruh Baya Menengah"
    kata = {"L": "rendah", "M": "menengah", "H": "tinggi"}
    return f"Pendapatan {kata[li]}, belanja {kata[ls]}"


def profil_segmen(df: pd.DataFrame, labels) -> pd.DataFrame:
    d = df.copy()
    d["Klaster"] = labels
    d["Prioritas"], _, _ = label_prioritas(d)
    g = d.groupby("Klaster").agg(Jumlah=("Klaster", "size"), Usia=(AGE, "mean"), Pendapatan=(INC, "mean"),
                                 Belanja=(SPD, "mean"), Perempuan=("Gender", lambda s: (s == "Female").mean() * 100),
                                 Prioritas=("Prioritas", "sum"))
    nama = [beri_nama(r.Usia, r.Pendapatan, r.Belanja) for r in g.itertuples()]
    seen = {}
    for i, n in enumerate(nama):                      # jaga agar nama unik
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            nama[i] = f"{n} ({seen[n]})"
    g["Segmen"] = nama
    g["Strategi"] = [STRATEGI.get(n, "Tinjau profil segmen ini bersama tim pemasaran.") for n in nama]
    g["Persen"] = g["Jumlah"] / g["Jumlah"].sum() * 100
    return g.round(2)


# ----------------------------------------------------------------------------- evaluasi
def sweep_k(X: pd.DataFrame, scaler: str, ks=range(2, 11)) -> pd.DataFrame:
    rows = []
    for k in ks:
        pipe = bangun_pipeline(k, scaler).fit(X)
        Z, lab = pipe[:-1].transform(X), pipe.predict(X)
        rows.append({"K": k, "Scaler": scaler, **metrik(Z, lab), "Inertia": float(pipe[-1].inertia_)})
    return pd.DataFrame(rows)


def bandingkan_algoritma(Z, k: int) -> pd.DataFrame:
    algo = {"K-Means": KMeans(k, n_init=10, random_state=RS).fit_predict(Z),
            "Ward (aglomeratif)": AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z),
            "Gaussian Mixture": GaussianMixture(k, n_init=5, random_state=RS).fit(Z).predict(Z)}
    return pd.DataFrame({n: metrik(Z, l) for n, l in algo.items()}).T.round(4)


def bic_gmm(Z, ks=range(2, 11)) -> pd.DataFrame:
    return pd.DataFrame({"K": list(ks), "BIC": [GaussianMixture(k, n_init=5, random_state=RS).fit(Z).bic(Z) for k in ks]})


def dbscan_grid(Z) -> pd.DataFrame:
    rows = []
    for eps in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for ms in [3, 5, 8]:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z)
            n_kl = len(set(lab)) - (1 if -1 in lab else 0)
            noise = float((lab == -1).mean() * 100)
            sil = np.nan
            mask = lab != -1
            if n_kl >= 2 and mask.sum() > n_kl:
                sil = silhouette_score(Z[mask], lab[mask])
            rows.append({"eps": eps, "min_samples": ms, "Jumlah klaster": n_kl, "Noise (%)": round(noise, 1),
                         "Silhouette (tanpa noise)": round(sil, 3) if sil == sil else np.nan})
    return pd.DataFrame(rows)


def silhouette_per_klaster(Z, labels) -> pd.DataFrame:
    s = silhouette_samples(Z, labels)
    return pd.DataFrame({"Klaster": labels, "silhouette": s})


def stabilitas(X: pd.DataFrame, scaler: str = "Yeo-Johnson", k: int = K_FINAL, n: int = 30, n_init: int = 10) -> pd.DataFrame:
    Z = buat_scaler(scaler).fit_transform(X)
    ref = None
    rows = []
    for seed in range(n):
        lab = KMeans(k, n_init=n_init, random_state=seed).fit_predict(Z)
        if ref is None:
            ref = lab
        rows.append({"seed": seed, "Silhouette": silhouette_score(Z, lab), "ARI vs seed 0": adjusted_rand_score(ref, lab)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- paradoks akurasi
def purity(labels, y) -> float:
    d = pd.DataFrame({"k": labels, "y": np.asarray(y)})
    return float(d.groupby("k")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(d))


def paradoks_akurasi(X: pd.DataFrame, y, ks=(1, 2, 3, 6, 10, 25, 50, 100, 200)) -> pd.DataFrame:
    Z = buat_scaler("Yeo-Johnson").fit_transform(X)
    y = np.asarray(y)
    rows = []
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=RS).fit_predict(Z)
        d = pd.DataFrame({"k": lab, "y": y})
        mayoritas = d.groupby("k")["y"].transform(lambda s: int(s.mean() >= 0.5))   # prediksi = suara mayoritas klaster
        tp = int(((mayoritas == 1) & (d["y"] == 1)).sum())
        rows.append({"K": k, "Purity (mirip akurasi)": purity(lab, y), "Recall kelas Prioritas": tp / max(int(y.sum()), 1),
                     "Silhouette": float(silhouette_score(Z, lab)) if 1 < len(set(lab)) < len(Z) else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- PCA
def pca_info(Z) -> dict:
    p = PCA(random_state=RS).fit(Z)
    return {"explained": p.explained_variance_ratio_, "cumulative": np.cumsum(p.explained_variance_ratio_),
            "components": p.components_, "proj": PCA(n_components=2, random_state=RS).fit_transform(Z)}


# ----------------------------------------------------------------------------- prediksi
def prediksi(pipe: Pipeline, Xbaru: pd.DataFrame) -> pd.DataFrame:
    Z = pipe[:-1].transform(Xbaru[FITUR])
    d = pipe[-1].transform(Z)
    urut = np.sort(d, axis=1)
    return pd.DataFrame({"Klaster": d.argmin(axis=1), "Jarak ke centroid": urut[:, 0],
                         "Kepastian": 1 - urut[:, 0] / urut[:, 1]}, index=Xbaru.index)
