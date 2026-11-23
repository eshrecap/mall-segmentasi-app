"""Uji logika inti. Jalankan:  python -m pytest -q   (atau  python tests/test_model.py)"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import model as M  # noqa: E402

DF = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "Mall_Customers.csv")
X = DF[M.FITUR]


def test_data_bersih():
    assert DF.shape == (200, 5) and DF.isna().sum().sum() == 0 and DF.duplicated().sum() == 0


def test_metrik_akhir():
    r = M.latih_final(X)
    assert abs(r["metrik"]["Silhouette"] - 0.4565) < 5e-4
    assert abs(r["metrik"]["Davies-Bouldin"] - 0.8096) < 5e-4
    assert abs(r["metrik"]["Calinski-Harabasz"] - 143.84) < 0.05


def test_nama_segmen_unik_dan_lengkap():
    r = M.latih_final(X)
    p = M.profil_segmen(DF, r["labels"])
    assert len(p) == 6 and p["Segmen"].is_unique and set(p["Segmen"]) == set(M.URUTAN_NAMA)
    assert p["Jumlah"].sum() == 200


def test_prioritas_terkumpul_di_satu_segmen():
    r = M.latih_final(X)
    p = M.profil_segmen(DF, r["labels"])
    assert p["Prioritas"].sum() == 22 and p["Prioritas"].max() == 22


def test_prediksi_sama_dengan_label_latih():
    r = M.latih_final(X)
    hsl = M.prediksi(r["pipe"], X)
    assert (hsl["Klaster"].values == r["labels"]).all()
    assert hsl["Kepastian"].between(0, 1).all()


def test_paradoks_k1_recall_nol():
    y, _, _ = M.label_prioritas(DF)
    t = M.paradoks_akurasi(X, y, ks=(1, 6))
    assert abs(t.loc[0, "Purity (mirip akurasi)"] - 0.89) < 1e-9 and t.loc[0, "Recall kelas Prioritas"] == 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK", n)
