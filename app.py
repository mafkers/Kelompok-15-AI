import streamlit as st
import requests
import pandas as pd
import time
import os
import pickle
from src.a_star_scheduler import AStarScheduler, Pasien
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="Penjadwalan UGD (A* & NLP)", layout="wide")

st.title("Sistem Penjadwalan UGD")

st.header("1. Cek Prioritas Penanganan")
keluhan_input = st.text_area("Isi Keluhan Pasien (Contoh: 'pasien muntah darah dan pingsan')")
if st.button("Cek Prioritas"):
    if keluhan_input.strip():
        with st.spinner("Menganalisis keluhan..."):
            kegawatan_hasil = None
            try:
                res = requests.post("http://localhost:8000/predict_kegawatan", json={"keluhan": keluhan_input}, timeout=5)
                if res.status_code == 200 and res.json().get("status") == 200:
                    kegawatan_hasil = res.json().get("kegawatan")
            except:
                pass
                
            if kegawatan_hasil is None:
                # Handle fallback lokal jika API mati
                try:
                    base_dir = os.path.dirname(__file__)
                    vec_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
                    mod_path = os.path.join(base_dir, 'models', 'kegawatan_model.pkl')
                    if os.path.exists(vec_path) and os.path.exists(mod_path):
                        with open(vec_path, 'rb') as f:
                            vec = pickle.load(f)
                        with open(mod_path, 'rb') as f:
                            model = pickle.load(f)
                            
                        factory = StemmerFactory()
                        stemmer = factory.create_stemmer()
                        teks_bersih = stemmer.stem(keluhan_input)
                        vec_transform = vec.transform([teks_bersih])
                        if vec_transform.nnz == 0:
                            kegawatan_hasil = 3
                        else:
                            kegawatan_hasil = int(model.predict(vec_transform)[0])
                except Exception as e:
                    st.error(f"Gagal memprediksi secara lokal: {e}")

            if kegawatan_hasil:
                label = "Level 1 (Merah/Gawat Darurat)" if kegawatan_hasil == 1 else "Level 2 (Kuning/Darurat)" if kegawatan_hasil == 2 else "Level 3 (Hijau/Tidak Gawat)"
                st.success(f"**Tingkat Kegawatan Pasien:** {label}")
    else:
        st.warning("Mohon masukkan keluhan pasien.")

st.divider()

st.header("2. Jadwal Antrean Dokter")

if st.button("Tracking Prioritas Antrean"):
    with st.spinner("Menyusun jadwal..."):
        data = None
        
        try:
            res = requests.post("http://localhost:8000/schedule", timeout=5)
            if res.status_code == 200:
                hasil = res.json()
                if hasil.get("status") == 200:
                    data = hasil.get("data")
        except:
                pass
            
        if data is None:
            try:
                base_dir = os.path.dirname(__file__)
                dataset_path = os.path.join(base_dir, 'data', 'dataset_ugd.csv')
                df = pd.read_csv(dataset_path)
                
                daftar_pasien = []
                for _, row in df.iterrows():
                    p = Pasien(row['id_pasien'], row['waktu_kedatangan'], row['tingkat_kegawatan'], row['estimasi_penanganan'])
                    daftar_pasien.append(p)
                    
                scheduler = AStarScheduler(daftar_pasien)
                data = scheduler.jalankan()
            except Exception as e:
                st.error(f"Gagal memproses algoritma: {e}")
                data = None

        if data:
            df_hasil = pd.DataFrame(data)
            df_hasil.columns = ["ID Pasien", "Tingkat Kegawatan", "Waktu Datang", "Waktu Mulai", "Waktu Selesai", "Waktu Tunggu"]
            
            # Set index DataFrame
            df_hasil.set_index("ID Pasien", inplace=True)
            
            # Force CSS align center untuk Streamlit table
            st.markdown("""
                <style>
                table td, table th {
                    text-align: center !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            def highlight_kegawatan(val):
                if val == 1:
                    return 'background-color: #ffcccc; color: black;'
                elif val == 2:
                    return 'background-color: #ffffcc; color: black;'
                elif val == 3:
                    return 'background-color: #ccffcc; color: black;'
                return ''
            
            styled_df = df_hasil.style.map(highlight_kegawatan, subset=["Tingkat Kegawatan"])
            st.table(styled_df)
