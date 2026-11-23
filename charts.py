"""Semua grafik (matplotlib) bertema indigo. Setiap fungsi mengembalikan Figure; tampilkan dengan tampil()."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import AGE, FITUR, INC, SINGKAT, SPD, URUTAN_NAMA

INDIGO, INDIGO_MUDA, INDIGO_TUA, KORAL, ABU, HIJAU = "#4B5FD6", "#B7C0F2", "#2B3A9E", "#E5736E", "#8B93B5", "#35A89A"
BG = "#F6F7FD"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": "#FFFFFF", "axes.edgecolor": "#CDD3F0",
    "axes.grid": True, "grid.color": "#E7EAFB", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlecolor": INDIGO_TUA, "axes.labelcolor": "#33395C", "xtick.color": "#33395C",
    "ytick.color": "#33395C", "font.size": 10, "legend.frameon": False,
})
WARNA_SEGMEN = dict(zip(URUTAN_NAMA, [INDIGO, "#F2A65A", KORAL, ABU, HIJAU, "#9C7BD9"]))
GENDER = {"Female": KORAL, "Male": INDIGO}
_CADANGAN = ["#6C757D", "#B08968", "#2A9D8F", "#E9C46A", "#264653", "#A8DADC"]


def warna_segmen(nama: str) -> str:
    if nama in WARNA_SEGMEN:
        return WARNA_SEGMEN[nama]
    return _CADANGAN[sum(map(ord, nama)) % len(_CADANGAN)]


def tampil(fig):
    import streamlit as st
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _fig(w=6.5, h=3.6, n=1):
    return plt.subplots(1, n, figsize=(w, h))


# ----------------------------------------------------------------------------- dataset & EDA
def distribusi_gender(df):
    fig, (a, b) = _fig(8, 3.2, 2)
    c = df["Gender"].value_counts().reindex(["Female", "Male"]).fillna(0)
    a.bar(["Perempuan", "Laki-laki"], c.values, color=[KORAL, INDIGO])
    for i, v in enumerate(c.values):
        a.text(i, v + 1.5, f"{int(v)}", ha="center")
    a.set_title("Jumlah pelanggan"); a.set_ylim(0, max(c.max() * 1.18, 1))
    b.pie(c.values, labels=["Perempuan", "Laki-laki"], colors=[KORAL, INDIGO], autopct="%1.1f%%", startangle=90,
          wedgeprops={"width": 0.45, "edgecolor": BG})
    b.set_title("Proporsi"); b.grid(False)
    return fig


def hist_numerik(df, col, by_gender=False):
    fig, (a, b) = _fig(9, 3.6, 2)
    if by_gender:
        for g, c in GENDER.items():
            a.hist(df.loc[df["Gender"] == g, col], bins=20, alpha=.65, color=c, label=g)
        a.legend()
        data = [df.loc[df["Gender"] == g, col] for g in GENDER]
        bp = b.boxplot(data, patch_artist=True, medianprops={"color": "#1E2540"})
        b.set_xticks([1, 2]); b.set_xticklabels(list(GENDER))
        for patch, c in zip(bp["boxes"], [GENDER["Female"], GENDER["Male"]]):
            patch.set_facecolor(c); patch.set_alpha(.55)
    else:
        a.hist(df[col], bins=20, color=INDIGO, alpha=.85)
        a.axvline(df[col].mean(), color=KORAL, ls="--", label=f"Rata-rata {df[col].mean():.1f}")
        a.axvline(df[col].median(), color=HIJAU, ls=":", label=f"Median {df[col].median():.1f}"); a.legend()
        bp = b.boxplot([df[col]], patch_artist=True, medianprops={"color": "#1E2540"}, vert=False)
        bp["boxes"][0].set_facecolor(INDIGO_MUDA); b.set_yticks([])
    a.set_title(f"Distribusi {col}"); a.set_xlabel(col); a.set_ylabel("Jumlah")
    b.set_title("Boxplot")
    return fig


def spending_per_usia(df):
    d = df.copy()
    d["Kelompok Usia"] = pd.cut(d[AGE], [17, 25, 35, 45, 55, 70], labels=["18-25", "26-35", "36-45", "46-55", "56-70"])
    g = d.groupby("Kelompok Usia", observed=True)[SPD].agg(["mean", "count"])
    fig, ax = _fig(7, 3.6)
    bars = ax.bar(g.index.astype(str), g["mean"], color=INDIGO_MUDA)
    bars[int(np.argmax(g["mean"].values))].set_color(INDIGO)
    for r, (m_, n) in zip(bars, g.values):
        ax.text(r.get_x() + r.get_width() / 2, m_ + 1, f"{m_:.1f}\n(n={int(n)})", ha="center", fontsize=8)
    ax.axhline(df[SPD].mean(), color=KORAL, ls="--", lw=1, label=f"Rata-rata {df[SPD].mean():.1f}")
    ax.set(title="Rata-rata Spending Score per kelompok usia", xlabel="Kelompok usia", ylabel="Spending Score")
    ax.set_ylim(0, g["mean"].max() * 1.3); ax.legend()
    return fig


def scatter_gender(df, x=INC, y=SPD):
    fig, ax = _fig(7, 4.4)
    for g, c in GENDER.items():
        s = df[df["Gender"] == g]
        ax.scatter(s[x], s[y], s=28, alpha=.7, color=c, label=g)
    ax.set(title=f"{SINGKAT.get(x, x)} vs {SINGKAT.get(y, y)}", xlabel=x, ylabel=y); ax.legend()
    return fig


def korelasi(df):
    c = df[FITUR].corr()
    lab = [SINGKAT[k] for k in FITUR]
    fig, ax = _fig(5, 4)
    im = ax.imshow(c, cmap="Purples", vmin=-1, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(lab, rotation=15, ha="right"); ax.set_yticks(range(3)); ax.set_yticklabels(lab)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", color="white" if abs(c.iloc[i, j]) > .55 else "#1E2540")
    ax.grid(False); ax.set_title("Korelasi Pearson"); fig.colorbar(im, ax=ax, shrink=.8)
    return fig


# ----------------------------------------------------------------------------- preprocessing
def skew_bar(t: pd.DataFrame):
    fig, ax = _fig(7, 3.4)
    x = np.arange(len(t)); w = .38
    ax.bar(x - w / 2, t["Skewness asli"], w, color=INDIGO_MUDA, label="Asli")
    ax.bar(x + w / 2, t["Skewness Yeo-Johnson"], w, color=INDIGO, label="Yeo-Johnson")
    ax.axhline(0, color="#1E2540", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([SINGKAT.get(i, i) for i in t.index]); ax.set_ylabel("Skewness"); ax.legend()
    ax.set_title("Kemencengan sebelum dan sesudah Yeo-Johnson")
    return fig


def hist_sebelum_sesudah(asli, transformasi, nama):
    fig, (a, b) = _fig(9, 3.4, 2)
    a.hist(asli, bins=20, color=INDIGO_MUDA); a.set_title(f"{nama} (asli)")
    b.hist(transformasi, bins=20, color=INDIGO); b.set_title(f"{nama} (Yeo-Johnson, standar)")
    return fig


# ----------------------------------------------------------------------------- PCA
def scree(info):
    ind, cum = info["explained"], info["cumulative"]
    idx = np.arange(1, len(ind) + 1)
    fig, ax = _fig(6.5, 3.6)
    ax.bar(idx, ind, color=INDIGO_MUDA, label="Individual")
    ax.step(idx, cum, where="mid", color=INDIGO_TUA, lw=2, label="Kumulatif")
    for i, v in enumerate(cum):
        ax.text(idx[i], v + .02, f"{v:.0%}", ha="center", fontsize=8)
    ax.set(title="Scree plot", xlabel="Komponen utama", ylabel="Proporsi varians", ylim=(0, 1.15)); ax.set_xticks(idx); ax.legend()
    return fig


def pca_scatter(proj, labels, nama_map, var2):
    fig, ax = _fig(7, 4.6)
    for k in sorted(set(labels)):
        m_ = labels == k
        ax.scatter(proj[m_, 0], proj[m_, 1], s=28, alpha=.8, color=warna_segmen(nama_map[k]), label=nama_map[k])
    ax.set(title=f"Proyeksi PCA 2D (varians {var2:.1%})", xlabel="PC1", ylabel="PC2"); ax.legend(fontsize=8, loc="best")
    return fig


# ----------------------------------------------------------------------------- evaluasi
def kurva_k(sw: pd.DataFrame, metrik_kol, ks_tandai=None):
    fig, ax = _fig(7, 3.6)
    for s, c in zip(sw["Scaler"].unique(), [INDIGO, ABU]):
        d = sw[sw["Scaler"] == s]
        ax.plot(d["K"], d[metrik_kol], marker="o", color=c, lw=2.2, label=s)
        if metrik_kol in ("Silhouette", "Calinski-Harabasz"):
            b = d.loc[d[metrik_kol].idxmax()]
            ax.annotate(f"K={int(b['K'])}", (b["K"], b[metrik_kol]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=c)
    if ks_tandai:
        ax.axvline(ks_tandai, color=KORAL, ls="--", lw=1, label=f"K akhir = {ks_tandai}")
    ax.set(title=f"{metrik_kol} terhadap jumlah klaster K", xlabel="K", ylabel=metrik_kol); ax.legend()
    return fig


def bar_algoritma(t: pd.DataFrame):
    fig, ax = _fig(10, 3.2, 3)
    for a, (c, bagus) in zip(ax, [("Silhouette", "besar"), ("Davies-Bouldin", "kecil"), ("Calinski-Harabasz", "besar")]):
        v = t[c]
        terbaik = v.idxmax() if bagus == "besar" else v.idxmin()
        a.bar(range(len(v)), v.values, color=[INDIGO if i == terbaik else INDIGO_MUDA for i in v.index])
        a.set_xticks(range(len(v))); a.set_xticklabels([i.split(" ")[0] for i in v.index], fontsize=8)
        a.set_title(f"{c}\n(lebih {bagus} lebih baik)", fontsize=9)
    return fig


def silhouette_plot(sil: pd.DataFrame, nama_map):
    fig, ax = _fig(7, 4.4)
    y0 = 0
    rata = sil["silhouette"].mean()
    for k in sorted(sil["Klaster"].unique()):
        v = np.sort(sil.loc[sil["Klaster"] == k, "silhouette"].values)
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=warna_segmen(nama_map[k]), alpha=.85)
        ax.text(-0.02, y0 + len(v) / 2, nama_map[k], ha="right", va="center", fontsize=7)
        y0 += len(v) + 6
    ax.axvline(rata, color=KORAL, ls="--", label=f"Rata-rata {rata:.3f}")
    ax.set(title="Silhouette per pelanggan, dikelompokkan per segmen", xlabel="Koefisien silhouette"); ax.set_yticks([]); ax.legend(loc="lower right")
    ax.grid(False)
    return fig


def bic_plot(b: pd.DataFrame):
    fig, ax = _fig(6, 3.2)
    ax.plot(b["K"], b["BIC"], marker="o", color=INDIGO)
    k = int(b.loc[b["BIC"].idxmin(), "K"])
    ax.axvline(k, color=KORAL, ls="--", label=f"BIC terendah: K = {k}")
    ax.set(title="BIC Gaussian Mixture", xlabel="Jumlah komponen", ylabel="BIC (lebih kecil lebih baik)"); ax.legend()
    return fig


def stabilitas_plot(st_df: pd.DataFrame):
    fig, (a, b) = _fig(9, 3.2, 2)
    a.hist(st_df["Silhouette"], bins=12, color=INDIGO); a.set_title("Silhouette pada 30 inisialisasi acak")
    a.set_xlabel("Silhouette")
    b.hist(st_df["ARI vs seed 0"], bins=12, color=HIJAU); b.set_title("ARI terhadap inisialisasi pertama")
    b.set_xlabel("ARI")
    return fig


def paradoks_plot(t: pd.DataFrame):
    fig, ax = _fig(7.5, 3.8)
    x = np.arange(len(t))
    ax.plot(x, t["Purity (mirip akurasi)"], marker="o", color=KORAL, lw=2, label="Purity (mirip akurasi)")
    ax.plot(x, t["Recall kelas Prioritas"], marker="s", color=INDIGO, lw=2, label="Recall kelas Prioritas")
    ax.plot(x, t["Silhouette"], marker="^", color=HIJAU, lw=2, label="Silhouette")
    ax.set_xticks(x); ax.set_xticklabels(t["K"].astype(str)); ax.set_ylim(-.05, 1.08)
    ax.set(title="Purity naik terus ketika K membesar, makna segmentasi tidak", xlabel="Jumlah klaster K", ylabel="Skor"); ax.legend(fontsize=8)
    return fig


# ----------------------------------------------------------------------------- profil & prediksi
def scatter_segmen(df, labels, nama_map, pusat=None, titik_baru=None):
    fig, ax = _fig(7.5, 4.8)
    for k in sorted(set(labels)):
        m_ = labels == k
        ax.scatter(df.loc[m_, INC], df.loc[m_, SPD], s=30, alpha=.8, color=warna_segmen(nama_map[k]), label=nama_map[k])
    if pusat is not None:
        for k, (x, y) in pusat.items():
            ax.scatter(x, y, marker="X", s=140, color="#1E2540", edgecolor="white", zorder=5)
    if titik_baru is not None:
        ax.scatter(*titik_baru, marker="*", s=420, color="gold", edgecolor="#1E2540", linewidth=1.3, zorder=6, label="Pelanggan baru")
    ax.set(title="Segmen pelanggan pada Income vs Spending", xlabel=INC, ylabel=SPD); ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def bar_profil(prof: pd.DataFrame):
    fig, ax = _fig(10, 3.4, 3)
    for a, (c, jd) in zip(ax, [("Usia", "Usia rata-rata"), ("Pendapatan", "Income rata-rata (k$)"), ("Belanja", "Spending Score rata-rata")]):
        p = prof.sort_values(c)
        a.barh([n for n in p["Segmen"]], p[c], color=[warna_segmen(n) for n in p["Segmen"]])
        a.set_title(jd, fontsize=9)
        a.tick_params(axis="y", labelsize=7)
        for i, v in enumerate(p[c]):
            a.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=8)
    return fig


def radar(prof: pd.DataFrame, X: pd.DataFrame, pilih):
    cols = [("Usia", AGE), ("Pendapatan", INC), ("Belanja", SPD)]
    ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, polar=True)
    for _, r in prof[prof["Segmen"].isin(pilih)].iterrows():
        v = [(r[a] - X[b].min()) / (X[b].max() - X[b].min()) for a, b in cols]
        v += v[:1]
        ax.plot(ang, v, color=warna_segmen(r["Segmen"]), lw=2, label=r["Segmen"])
        ax.fill(ang, v, color=warna_segmen(r["Segmen"]), alpha=.12)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels([c[0] for c in cols]); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.08), ncol=1)
    return fig


def bar_bandingkan(nilai: dict, rata: dict):
    fig, ax = _fig(6, 3)
    x = np.arange(len(nilai)); w = .38
    ax.bar(x - w / 2, list(nilai.values()), w, color="#F2C94C", label="Pelanggan ini")
    ax.bar(x + w / 2, list(rata.values()), w, color=INDIGO, label="Rata-rata segmen")
    ax.set_xticks(x); ax.set_xticklabels(list(nilai)); ax.legend(); ax.set_title("Pelanggan vs rata-rata segmen")
    return fig


def distribusi_segmen(jumlah: pd.Series):
    fig, ax = _fig(6.5, 3.4)
    jumlah = jumlah.sort_values()
    ax.barh(jumlah.index, jumlah.values, color=[warna_segmen(n) for n in jumlah.index])
    for i, v in enumerate(jumlah.values):
        ax.text(v + .3, i, f"{int(v)}", va="center", fontsize=8)
    ax.set_title("Jumlah pelanggan per segmen"); ax.tick_params(axis="y", labelsize=8)
    return fig
