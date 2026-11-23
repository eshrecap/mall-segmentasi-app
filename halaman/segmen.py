import streamlit as st
from utils import fitur, load_data, model_final
import charts
import model as M

df, r = load_data(), model_final()
prof = r["profil"].sort_values("Belanja", ascending=False)
st.title("Profil Segmen")
st.caption("Siapa saja pelanggan di tiap segmen, dan apa strategi yang cocok?")

c1, c2, c3 = st.columns(3)
c1.metric("Jumlah segmen", len(prof))
terbesar = prof.sort_values("Jumlah", ascending=False).iloc[0]
c2.metric("Segmen terbesar", terbesar["Segmen"], f"{int(terbesar['Jumlah'])} pelanggan ({terbesar['Persen']:.1f}%)", delta_color="off")
prio = prof.sort_values("Prioritas", ascending=False).iloc[0]
c3.metric("Segmen Pelanggan Prioritas", prio["Segmen"], f"{int(prio['Prioritas'])} dari {int(prof['Prioritas'].sum())}", delta_color="off")

t1, t2, t3 = st.tabs(["Peta Segmen", "Tabel Profil", "Kartu Strategi"])
with t1:
    tampil_pusat = st.toggle("Tandai pusat klaster (centroid)", value=True)
    charts.tampil(charts.scatter_segmen(df, r["labels"], r["nama"], r["pusat"] if tampil_pusat else None))
    st.caption("Segmen Paruh Baya Menengah dan Muda Menengah tampak bertumpuk di bidang ini karena keduanya dibedakan oleh usia, bukan oleh Income atau Spending.")
    charts.tampil(charts.bar_profil(prof))
with t2:
    tb = prof[["Segmen", "Jumlah", "Persen", "Usia", "Pendapatan", "Belanja", "Perempuan", "Prioritas"]].set_index("Segmen")
    tb.columns = ["Jumlah", "Persen (%)", "Usia", "Income (k$)", "Spending", "Perempuan (%)", "Pelanggan Prioritas"]
    st.dataframe(tb)
    st.download_button("Download profil segmen (CSV)", tb.to_csv().encode("utf-8"), file_name="profil_segmen.csv", mime="text/csv")
    st.caption("Persentase perempuan berkisar sempit antar segmen dengan N kecil per segmen, sehingga gender bukan dasar segmentasi yang kuat.")
    pilih = st.multiselect("Bandingkan dengan radar", list(prof["Segmen"]), default=list(prof["Segmen"])[:2])
    if pilih:
        charts.tampil(charts.radar(prof.reset_index(), fitur(), pilih))
with t3:
    for _, row in prof.iterrows():
        with st.container(border=True):
            a, b = st.columns([1, 2])
            a.markdown(f"**{row['Segmen']}**")
            a.caption(f"{int(row['Jumlah'])} pelanggan ({row['Persen']:.1f}%)")
            b.write(f"Usia rata-rata {row['Usia']:.0f} tahun, income {row['Pendapatan']:.0f}k$, spending {row['Belanja']:.0f}.")
            b.write(f"**Strategi:** {row['Strategi']}")
st.caption("Nama dan strategi segmen dibuat otomatis dari rata-rata tiap klaster; tinjau bersama tim pemasaran sebelum dipakai.")
