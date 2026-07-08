# Sistem Penjadwalan Otomatis UGD

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)

## Anggota Kelompok 15

| Nama Lengkap | NIM | Pembagian Tugas |
| :--- | :--- | :--- |
| **Alexander Tristan** | 241103000063 | Arsitektur REST API (FastAPI), Integrasi Frontend UI (Streamlit), dan Penulisan Dokumentasi. |
| **Muhammad Naufal** | 24110300010 | Pengembangan Model NLP (Naive Bayes & Sastrawi), Evaluasi Model, dan Pipeline Training. |
| **Ikmal Luthfi** | 24110300023 | Implementasi Algoritma A* Search, Perancangan Fungsi Heuristik, dan Eksplorasi Data (EDA). |

Repositori ini adalah implementasi sistem penjadwalan otomatis untuk Unit Gawat Darurat (UGD) yang menggabungkan **Natural Language Processing (NLP)** dan algoritma pencarian heuristik **A* Search**.

## Arsitektur Sistem

1. **Input Layer (Streamlit):** Menerima teks keluhan pasien.
2. **Preprocessing (Sastrawi):** Membuang imbuhan (stemming) agar model NLP lebih fokus pada kata dasar.
3. **Inference (Naive Bayes):** Memprediksi tingkat kegawatan (Prioritas 1/2/3).
4. **Scheduling (A* Search):** Menyusun jadwal dokter yang mengutamakan prioritas tinggi (1), dengan heuristik waktu tunggu.

## Setup & Instalasi

1. Clone repositori ini
```bash
git clone https://github.com/mafkers/Kelompok-15-AI.git
```
2. acceses file lewat cmd (command prompt)
```
cd Kelompok-15-AI
```
3. Buat Virtual Environment (opsional tapi disarankan)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

4. Install dependencies
```bash
pip install -r requirements.txt
```

5. Jalankan Backend (API)
```bash
uvicorn api:app --reload
```

6. Jalankan Frontend (Streamlit)
```bash
streamlit run app.py
```

## Dokumentasi API (REST API)

Sistem ini mengekspos REST API yang bisa dikonsumsi oleh client mana pun.

### 1. Prediksi Kegawatan (`POST /predict_kegawatan`)
Menerima teks keluhan pasien dan mengembalikan prediksi Kegawatan.

**Request Body (JSON):**
```json
{
  "keluhan": "pasien muntah darah dan pingsan"
}
```

**Response (200 OK):**
```json
{
  "status": 200,
  "message": "Berhasil",
  "kegawatan": 1
}
```

**Response (500 Error - Model belum dilatih):**
```json
{
  "status": 500,
  "message": "Model NLP belum dilatih."
}
```

### 2. Generate Jadwal (`POST /schedule`)
Membaca dataset internal dan menyusun jadwal optimal dengan A* Search.

**Request:** (Tidak butuh parameter)

**Response (200 OK):**
```json
{
  "status": 200,
  "message": "Berhasil",
  "data": [
    {
      "id": "P-001",
      "kegawatan": 3,
      "waktu_datang": "08:00",
      "waktu_mulai": "08:00",
      "waktu_selesai": "08:15",
      "waktu_tunggu": 0
    }
  ]
}
```

**Response (500 Error):**
```json
{
  "status": 500,
  "message": "Terjadi kesalahan di server"
}
```
