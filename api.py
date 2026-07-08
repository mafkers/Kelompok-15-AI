import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.a_star_scheduler import AStarScheduler, Pasien
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = FastAPI()

base_dir = os.path.dirname(__file__)
dataset_path = os.path.join(base_dir, 'data', 'dataset_ugd.csv')

# Load NLP Models
models_dir = os.path.join(base_dir, 'models')
vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')
model_path = os.path.join(models_dir, 'kegawatan_model.pkl')

if os.path.exists(vectorizer_path) and os.path.exists(model_path):
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(model_path, 'rb') as f:
        nlp_model = pickle.load(f)
else:
    vectorizer, nlp_model = None, None

factory = StemmerFactory()
stemmer = factory.create_stemmer()

class KegawatanRequest(BaseModel):
    keluhan: str

@app.get("/")
def home():
    return {"msg": "API Penjadwalan UGD (A* Search & NLP) Aktif"}

@app.post("/predict_kegawatan")
def predict_kegawatan(req: KegawatanRequest):
    if not vectorizer or not nlp_model:
        return {"status": 500, "message": "Model NLP belum dilatih."}
    
    try:
        # Preprocessing text
        keluhan_bersih = stemmer.stem(req.keluhan)
        
        # Inference
        vec = vectorizer.transform([keluhan_bersih])
        
        # Jika kata tidak ada di vocabulary (misal: tulisan ngawur/kosong), default ke Level 3
        if vec.nnz == 0:
            prediksi = 3
        else:
            prediksi = nlp_model.predict(vec)[0]
        
        return {
            "status": 200,
            "message": "Berhasil",
            "kegawatan": int(prediksi)
        }
    except Exception as e:
        return {"status": 500, "message": f"Terjadi kesalahan: {e}"}

@app.post("/schedule")
def schedule_patients():
    try:
        if not os.path.exists(dataset_path):
            return {"status": 404, "message": "Dataset UGD tidak ditemukan."}

        df = pd.read_csv(dataset_path)
        daftar_pasien = []
        for _, row in df.iterrows():
            p = Pasien(row['id_pasien'], row['waktu_kedatangan'], row['tingkat_kegawatan'], row['estimasi_penanganan'])
            daftar_pasien.append(p)
            
        scheduler = AStarScheduler(daftar_pasien)
        jadwal_optimal = scheduler.jalankan()
        
        if not jadwal_optimal:
            return {"status": 500, "message": "Gagal menemukan jadwal."}

        return {
            "status": 200,
            "message": "Berhasil",
            "data": jadwal_optimal
        }

    except Exception as e:
        print("Error:", e)
        return {"status": 500, "message": "Terjadi kesalahan di server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
