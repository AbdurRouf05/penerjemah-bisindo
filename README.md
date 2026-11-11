# Penerjemah BISINDO ke Teks dan Suara (Real-Time)

MediaPipe Holistic + MLP (huruf) + LSTM (kata) + edge-tts.
Panduan lengkap: [docs/PANDUAN_PROYEK.md](docs/PANDUAN_PROYEK.md)

## 1. Install (sekali)
```powershell
cd "D:\kuliah\masin larning\tugas_akhir_bisindo"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Versi yang sudah teruji: Python 3.11.9, mediapipe 0.10.21, tensorflow 2.19.1, numpy 1.26.4.

## 2. Urutan menjalankan (semua dari folder root proyek)
| Langkah | Perintah | Waktu |
|---|---|---|
| Cek webcam | `.\.venv\Scripts\python.exe -m src.cek_webcam` | - |
| Ekstraksi huruf | `.\.venv\Scripts\python.exe -m src.ekstrak_huruf` | ±40–60 menit |
| Ekstraksi kata | `.\.venv\Scripts\python.exe -m src.ekstrak_kata` | ±1–2 jam |
| Training huruf | `.\.venv\Scripts\python.exe -m src.latih_huruf` | ±5–15 menit |
| Training kata | `.\.venv\Scripts\python.exe -m src.latih_kata` | ±30–60 menit |
| **Aplikasi** | `.\.venv\Scripts\python.exe -m src.app` | - |

**Terputus (laptop mati / Ctrl+C)?** Jalankan ulang perintah yang sama — ekstraksi lanjut dari
checkpoint (`data/landmarks/_checkpoint/`), training lanjut dari epoch terakhir (`models/_backup/`).
Yang sudah selesai tidak diulang.

## 3. Tombol aplikasi
`h` mode huruf · `k` mode kata · `SPASI` spasi · `BACKSPACE` hapus · `c` bersihkan ·
`ENTER` ucapkan · `q` keluar

## 4. Rekam data sendiri (anggota kelompok)
Panduan lengkap untuk perekam (instal ringan, tanpa TensorFlow):
**[docs/PANDUAN_REKAM.md](docs/PANDUAN_REKAM.md)**
```powershell
.\.venv\Scripts\python.exe -m src.rekam_data --peraga budi --mode huruf --label semua
.\.venv\Scripts\python.exe -m src.cek_rekaman
```
Latih ulang model huruf dengan semua rekaman:
`.\.venv\Scripts\python.exe -m src.latih_huruf --pakai-rekaman`

## Isi repo
Model `.keras` dan hasil ekstraksi landmark (`data/landmarks/`) ikut di-commit, jadi
aplikasi bisa langsung dijalankan tanpa training. Data mentah (`data/raw/`) tidak ikut
karena terlalu besar.

## 5. Struktur
- `src/features.py` — fungsi ekstraksi bersama (standar 126 / 201 fitur)
- `data/raw/` — data asli (jangan diubah) · `data/landmarks/` — hasil ekstraksi `.npz`
- `models/` — model terlatih · `reports/` — grafik, confusion matrix, tabel hasil
