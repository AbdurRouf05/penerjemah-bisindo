# PANDUAN PROYEK UAS — Penerjemah BISINDO ke Teks dan Suara

> **Judul proyek:** Penerjemah Bahasa Isyarat Indonesia (BISINDO) ke Teks dan Suara secara Real-Time Menggunakan MediaPipe dan LSTM
>
> **Jenis:** Tugas UAS Machine Learning, kelompok 6 orang.
>
> Dokumen ini adalah **satu-satunya sumber kebenaran** proyek. Kalau ada yang bertentangan dengan dokumen ini, ikuti dokumen ini lalu diskusikan di grup.

---

## DAFTAR ISI

0. [Cara memakai dokumen ini (WAJIB DIBACA, termasuk oleh AI)](#0-cara-memakai-dokumen-ini)
1. [Ringkasan proyek dalam 1 menit](#1-ringkasan-proyek)
2. [Keputusan yang sudah DIKUNCI](#2-keputusan-yang-sudah-dikunci)
3. [Lingkungan kerja (laptop, Python, library)](#3-lingkungan-kerja)
4. [Struktur folder](#4-struktur-folder)
5. [Inventaris data yang sudah ada](#5-inventaris-data)
6. [Standar data / kontrak antar-anggota (PENTING)](#6-standar-data--kontrak-antar-anggota)
7. [Pembagian tugas 6 anggota](#7-pembagian-tugas-6-anggota)
8. [Kartu tugas rinci (T0–T9)](#8-kartu-tugas-rinci)
9. [Jadwal dan urutan pengerjaan](#9-jadwal-dan-urutan-pengerjaan)
10. [Laporan dan presentasi](#10-laporan-dan-presentasi)
11. [Kesalahan umum dan solusinya](#11-kesalahan-umum-dan-solusinya)
12. [Glosarium (istilah sulit)](#12-glosarium)

---

## 0. CARA MEMAKAI DOKUMEN INI

### Untuk anggota kelompok (manusia)
1. Baca bagian 1, 2, 6, dan 7 dulu.
2. Cari namamu/kode anggotamu di bagian 7, lalu buka kartu tugasmu di bagian 8.
3. Kerjakan langkah demi langkah. Tugas dianggap selesai **hanya jika semua poin "SELESAI JIKA" terpenuhi**.
4. Kalau minta bantuan AI (ChatGPT, Gemini, Claude, Copilot, dll.), **tempelkan ke AI: bagian 0, 2, 3, 6, dan kartu tugasmu di bagian 8.** Jangan cuma bilang "buatkan program bahasa isyarat" — hasilnya tidak akan cocok dengan pekerjaan anggota lain.

### ATURAN UNTUK MODEL AI yang membantu mengerjakan
Jika kamu adalah model AI yang diberi dokumen ini, patuhi aturan berikut:

1. **Jangan mengubah keputusan di bagian 2.** Jangan menyarankan ganti ke ASL, ganti ke YOLO/CNN gambar, ganti judul, dsb.
2. **Selalu pakai standar data di bagian 6** (jumlah fitur, urutan fitur, normalisasi, nama label, format file). Ini yang membuat pekerjaan 6 orang bisa digabung.
3. **Selalu impor fungsi dari `src/features.py`** untuk ekstraksi fitur. Jangan menulis ulang fungsi ekstraksi versi sendiri.
4. **Jangan menyentuh folder `00_konsep_dasar`, `01_tfidf`, `02_word2vec`, `03_naive_bayes`** di `D:\kuliah\masin larning\`. Itu materi kuliah lain.
5. **Jangan mendownload data besar (> 100 MB) tanpa izin pengguna.** Sebutkan ukurannya dulu.
6. **Jangan menghapus data di `data/raw/`** tanpa izin pengguna.
7. Sistem operasi utama: **Windows 11, PowerShell**. Pakai path Windows. Python ada di `.venv\Scripts\python.exe`.
8. Laptop **tidak punya GPU NVIDIA**. Semua training harus bisa berjalan di CPU (model kecil, data landmark).
9. **JANGAN membalik (flip) frame kamera sebelum diproses MediaPipe.** Flip hanya boleh untuk tampilan layar.
10. Setelah mengerjakan, **cek semua poin "SELESAI JIKA"** di kartu tugas dan laporkan hasilnya dengan jujur (termasuk jika gagal).
11. Tulis kode sederhana, beri komentar singkat dalam Bahasa Indonesia, dan jangan menambah fitur di luar kartu tugas.

---

## 1. RINGKASAN PROYEK

**Apa yang dibuat:** aplikasi di laptop yang membaca **video webcam**, mengenali **isyarat BISINDO** (huruf A–Z dan kata), menyusunnya jadi **kalimat teks**, lalu **membacakannya dengan suara** Bahasa Indonesia, secara real-time.

**Alur sistem:**

```
🎥 Webcam (video real-time)
      ↓
MediaPipe Holistic  → mengambil titik koordinat tangan + badan tiap frame
      ↓
┌─────────────┴─────────────┐
MODE HURUF                  MODE KATA
1 frame → 126 angka         30 frame → (30 × 201) angka
Model MLP                   Model LSTM
└─────────────┬─────────────┘
      ↓
Penyusun kalimat  → "saya mau makan"
      ↓
Teks di layar + 🔊 Suara (edge-tts, Bahasa Indonesia)
```

**Kenapa metodenya begini (singkat, untuk dijelaskan saat presentasi):**

| Komponen | Fungsi | Alasan dipilih |
|---|---|---|
| MediaPipe Holistic | Ubah gambar → titik koordinat tangan/badan | Ringan (jalan di CPU), tidak terpengaruh background/warna kulit/cahaya seberat CNN gambar |
| MLP | Kenali **huruf** (pose diam, 1 frame) | Huruf = pose statis, cukup model sederhana & cepat |
| LSTM | Kenali **kata** (gerakan, 30 frame) | Kata = gerakan berurutan; LSTM punya "ingatan" urutan waktu |
| Pembanding: Random Forest, SVM (huruf); GRU, 1D-CNN (kata) | Bukti ilmiah bahwa pilihan model tepat | Penguji suka perbandingan |
| Penyusun kalimat | Rangkai huruf/kata jadi kalimat, cegah kata dobel | Agar output berupa kalimat, bukan kata lepas |
| edge-tts | Teks → suara Indonesia | Gratis, suara natural, tidak perlu melatih model suara |

**Data yang dibutuhkan:**
- Huruf → dataset **gambar** (untuk training). Saat aplikasi berjalan, input tetap video webcam.
- Kata → dataset **video**.
- Suara → **tidak butuh dataset** (edge-tts sudah jadi).

---

## 2. KEPUTUSAN YANG SUDAH DIKUNCI

Jangan diubah tanpa kesepakatan seluruh kelompok.

| Hal | Keputusan |
|---|---|
| Bahasa isyarat | **BISINDO** (bukan ASL, bukan SIBI) |
| Cakupan | Huruf A–Z **dan** kata → kalimat → suara |
| Ekstraksi fitur | **MediaPipe Holistic** (bukan CNN dari piksel gambar, bukan YOLO) |
| Model huruf | **MLP** (pembanding: Random Forest, SVM) |
| Model kata | **LSTM** (pembanding: GRU, 1D-CNN) |
| Panjang sekuens kata | **30 frame** |
| Suara | **edge-tts**, suara `id-ID-GadisNeural` (cadangan offline: `pyttsx3`) |
| Tampilan aplikasi | **OpenCV window** (wajib) — Streamlit hanya opsional |
| Bahasa pemrograman | **Python 3.11** |
| Library ML | **TensorFlow/Keras** (MLP/LSTM/GRU/CNN), **scikit-learn** (RF, SVM, metrik) |
| Variasi BISINDO | Dataset kata utama berasal dari **variasi Banten** (WL-BISINDO) → ditulis di batasan masalah |
| Fitur bonus (hanya jika waktu cukup) | Suara → teks (Vosk/Whisper) agar dua arah |

---

## 3. LINGKUNGAN KERJA

### 3.1 Laptop referensi (milik koordinator)
- HP 14s-fq2xxx, **AMD Ryzen 7 5825U** (8 core/16 thread), **RAM 16 GB**
- GPU: AMD Radeon terintegrasi → **tidak bisa CUDA**, training pakai **CPU** (cukup, karena data berupa angka koordinat)
- Webcam: HP TrueVision HD
- Windows 11, Python **3.11.9**
- Disk D: masih ±70 GB kosong

Laptop anggota lain: minimal RAM 8 GB, Python 3.10 atau 3.11. **Hindari Python 3.12/3.13** (MediaPipe/TensorFlow bisa bermasalah).

### 3.2 Lokasi proyek
```
D:\kuliah\masin larning\tugas_akhir_bisindo\
```
(Nama folder `tugas_akhir_bisindo` biarkan saja walau ini UAS — mengganti nama akan merusak path.)

### 3.3 Setup Python (sekali per laptop)
Buka **PowerShell**, lalu:

```powershell
cd "D:\kuliah\masin larning\tugas_akhir_bisindo"

# 1. Buat virtual environment (di laptop koordinator SUDAH ADA, lewati langkah ini)
python -m venv .venv

# 2. Aktifkan (lakukan setiap membuka terminal baru)
.\.venv\Scripts\Activate.ps1
#    Jika muncul error "running scripts is disabled", jalankan sekali:
#    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install library
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3.4 Isi `requirements.txt`
File ini sudah tersedia di root proyek:
```
numpy
pandas
opencv-python
mediapipe
tensorflow
scikit-learn
matplotlib
seaborn
tqdm
edge-tts
pygame
pyttsx3
kaggle
```

### 3.5 ⚠️ Cek penting MediaPipe (WAJIB setelah install)
Proyek ini memakai API **`mp.solutions.holistic`** (API "legacy"). Beberapa versi MediaPipe terbaru bisa saja menghapus API ini. Cek dengan:

```powershell
python -c "import mediapipe as mp; print(mp.__version__); print(mp.solutions.holistic.Holistic)"
```

- Jika tercetak versi dan `<class ...Holistic>` → **aman**.
- Jika error `AttributeError: module 'mediapipe' has no attribute 'solutions'` → turunkan versi:
  ```powershell
  python -m pip install "mediapipe==0.10.21"
  ```
  lalu cek ulang. Jika masih gagal, coba `0.10.14`. **Semua anggota harus memakai versi MediaPipe yang sama** — catat versinya di grup.

### 3.6 Cek TensorFlow
```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.reduce_sum(tf.ones([2,2])).numpy())"
```
Harus mencetak versi dan `4.0`. Pesan peringatan (warning) tentang CPU/oneDNN boleh diabaikan.

### 3.7 Berbagi file antar anggota
- **Kode** (`src/`, `notebooks/`, `docs/`): GitHub (repo private) atau Google Drive.
- **Data mentah** (`data/raw/`, ±3,5 GB): **JANGAN di-upload ke GitHub.** Bagikan lewat Google Drive/flashdisk, atau setiap anggota download sendiri (lihat 5.3).
- **Data hasil ekstraksi** (`data/landmarks/`, kecil, puluhan MB) dan **model** (`models/`): bagikan lewat Google Drive.

---

## 4. STRUKTUR FOLDER

```
tugas_akhir_bisindo/
├── .venv/                      ← virtual environment Python (jangan diedit/di-upload)
├── requirements.txt            ← daftar library
├── README.md                   ← penunjuk ke dokumen ini
├── docs/
│   └── PANDUAN_PROYEK.md       ← DOKUMEN INI
├── data/
│   ├── raw/                    ← data asli hasil download (JANGAN DIUBAH)
│   │   ├── huruf/              ← dataset gambar huruf A–Z
│   │   ├── kata/               ← dataset video/landmark kata
│   │   └── campuran/           ← dataset video campuran (Mendeley)
│   ├── landmarks/              ← hasil ekstraksi MediaPipe (.npz) — dibuat oleh T2, T3, T4
│   │   ├── huruf/
│   │   ├── kata/
│   │   └── rekaman_sendiri/
│   └── labels/                 ← daftar label (.json) — dibuat oleh T1
├── models/                     ← model terlatih (.keras / .joblib) — dibuat oleh T5, T6
├── reports/                    ← grafik, confusion matrix, tabel hasil — dibuat oleh T5, T6, T8
├── notebooks/                  ← eksplorasi (opsional)
└── src/
    ├── download_mendeley.py    ← SUDAH ADA: download dataset Mendeley
    ├── features.py             ← T1: fungsi ekstraksi fitur BERSAMA (kontrak bagian 6)
    ├── cek_webcam.py           ← T1: uji MediaPipe + webcam
    ├── ekstrak_huruf.py        ← T2
    ├── ekstrak_kata.py         ← T3
    ├── rekam_data.py           ← T4
    ├── latih_huruf.py          ← T5
    ├── latih_kata.py           ← T6
    ├── penyusun_kalimat.py     ← T7
    ├── suara.py                ← T7
    └── app.py                  ← T8: aplikasi real-time utama
```

---

## 5. INVENTARIS DATA

Semua data berada di `D:\kuliah\masin larning\tugas_akhir_bisindo\data\raw\`.

### 5.1 Data HURUF (gambar, untuk model huruf)

| Folder | Isi | Jumlah file | Struktur | Lisensi | Sumber |
|---|---|---|---|---|---|
| `huruf/bisindo_7peraga/Mendeley BISINDO/<A..Z>/` | Gambar huruf, 7 peraga, acuan resmi Kemendikbud | 8.040 jpg | Subfolder = label huruf | CC BY 4.0 | https://data.mendeley.com/datasets/4xnkvr88tk/1 |
| `huruf/multimodal_corpus/Train/<A..Z>/` dan `.../Test/<A..Z>/` | Gambar huruf, 1 peraga, indoor/outdoor, terang/gelap | ±14.000 jpg (+ file .xml bounding box, **abaikan .xml**) | Subfolder = label | CC BY 4.0 | https://data.mendeley.com/datasets/235c78xbmk/2 |
| `huruf/alphabet_image_data/02. Resized Images/<A..Z>/` | Berbagai HP, latar, orang | 4.466 jpg | Subfolder = label (cek ulang) | CC BY 4.0 | https://data.mendeley.com/datasets/ywnjpbcz8m/1 |
| `huruf/bisindo_recognition/BISINDO.zip` | Gambar abjad BISINDO | 1 zip (21 MB) | **Belum diekstrak** — T2 harus ekstrak & cek | CC BY 4.0 | https://data.mendeley.com/datasets/xh6bmx73n6/1 |
| `huruf/dataset_of_bisindo/` | Training data / Testing data | 156 png | Label kemungkinan di nama file — T2 cek | CC BY 4.0 | https://data.mendeley.com/datasets/m8y229yvrm |
| `huruf/bisindo_70_30/` | Training Data / Test Data | 260 png | Label kemungkinan di nama file — T2 cek | CC BY 4.0 | https://data.mendeley.com/datasets/4gtxgjy6xv |

### 5.2 Data KATA

**(a) WL-BISINDO — dataset kata UTAMA**
- Folder: `kata/wl_bisindo/` — **1.600 video .mp4** (langsung, tanpa subfolder)
- Nama file: `signer{S}_label{L}_sample{N}.mp4`, contoh `signer0_label10_sample3.mp4`
  - `S` = nomor peraga (0–4, total 5 peraga) — **verifikasi** dengan melihat semua nama file
  - `L` = nomor label kata (0–31)
  - `N` = nomor ulangan (1–10)
- 32 kata × 5 peraga × 10 ulangan = 1.600 video. Variasi **Banten**.
- Lisensi **CC BY-NC 4.0** (boleh untuk tugas kuliah, wajib cantumkan sumber, tidak boleh dijual).
- Sumber: https://www.kaggle.com/datasets/glennleonali/wl-bisindo , https://github.com/AceKinnn/WL-BISINDO
- Peta label (nomor → kata):

| L | Kata | L | Kata | L | Kata | L | Kata |
|---|---|---|---|---|---|---|---|
| 0 | air | 8 | motor | 16 | mengapa | 24 | datang |
| 1 | belajar | 9 | saya | 17 | bagaimana | 25 | teman |
| 2 | cari | 10 | terima kasih | 18 | merah | 26 | keluarga |
| 3 | hari | 11 | tuli | 19 | kuning | 27 | rumah |
| 4 | ingat | 12 | apa | 20 | hijau | 28 | pagi |
| 5 | lagi | 13 | siapa | 21 | hitam | 29 | siang |
| 6 | maaf | 14 | kapan | 22 | dengar | 30 | sore |
| 7 | makan | 15 | di mana | 23 | berangkat | 31 | malam |

**(b) BISINDO 40 kata (landmark) — dataset kata TAMBAHAN**
- Folder: `kata/bisindo_40kata_landmark/BISINDO Dataset/<Kategori>/<Kata>/<Kata>_001.npy`
- 2.400 file `.npy`, **setiap file berbentuk (30, 447)** = 30 frame × 447 fitur, sudah dinormalisasi oleh pembuatnya.
- 40 kata:
  - Kata Ganti: Dia, Kalian, Kami, Kamu, Kita, Mereka, Saya
  - Kata Kerja: Belajar, Berdiri, Duduk, Makan, Mandi, Melihat, Membaca, Menulis, Minum, Tidur
  - Kata Sifat: Baik, Bingung, Marah, Pendek, Ramah, Sabar, Sedih, Senang, Tinggi
  - Kata Tanya: Apa, Bagaimana, Berapa, Dimana, Kapan, Kemana, Siapa
  - Sapaan: Apa Kabar, Halo, Selamat Malam, Selamat Pagi, Selamat Siang, Selamat Sore, Terima Kasih
- ⚠️ **Format 447 fitur BERBEDA dengan standar kita (201 fitur)** dan **tidak ada video aslinya**. Lihat tugas T3 bagian B untuk cara menanganinya.
- Lisensi CC BY 4.0. Sumber: https://data.mendeley.com/datasets/yzhzkzbnmw/3 (ada README.txt dan file kamus .xlsx di folder tersebut).

**(c) BISINDO Video Dataset (Mendeley) — opsional, untuk demo/uji**
- Folder: `campuran/bisindo_video_dataset/` — 6 video panjang (huruf, angka, hari, keluarga, perkenalan, cerita) + 1.000 gambar huruf.
- Hanya 1 video per kategori → **tidak cukup untuk training**, gunakan untuk uji coba/demo.
- Sumber: https://data.mendeley.com/datasets/f33k9w86wr/3

### 5.3 Cara mendapatkan data (untuk anggota yang belum punya)
- **Cara termudah:** salin folder `data/raw/` dari laptop koordinator (flashdisk/Google Drive).
- **Download sendiri:**
  ```powershell
  .\.venv\Scripts\python.exe src\download_mendeley.py
  ```
  (±1,9 GB, aman diulang — file yang sudah lengkap dilewati).
  WL-BISINDO harus download manual dari Kaggle (login), lalu ekstrak semua `.mp4` ke `data/raw/kata/wl_bisindo/`.

### 5.4 Ringkasan kosakata
- WL-BISINDO: 32 kata. 40-kata landmark: 40 kata. Kata yang sama di keduanya (9): saya, makan, belajar, apa, siapa, kapan, di mana/dimana, bagaimana, terima kasih. Total unik ±63 kata.
- **Kosakata final aplikasi ditetapkan di T3** (lihat bagian 6.5).

---

## 6. STANDAR DATA / KONTRAK ANTAR-ANGGOTA

> Bagian ini adalah "perjanjian" agar pekerjaan semua anggota bisa digabung. **Semua anggota & semua AI wajib mengikutinya.** Implementasinya ada di `src/features.py` (dibuat di T1, kode lengkap ada di bawah).

### 6.1 Ekstraktor
- Selalu pakai **`mediapipe.solutions.holistic.Holistic`**.
  - Gambar diam (dataset huruf): `static_image_mode=True`
  - Video & webcam: `static_image_mode=False`
  - `model_complexity=1`, `min_detection_confidence=0.5`, `min_tracking_confidence=0.5`
- Input MediaPipe harus **RGB** (OpenCV membaca **BGR** → wajib `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`).
- **Jangan flip frame** sebelum `holistic.process(...)`.

### 6.2 Fitur HURUF = vektor 126 angka (1 frame)
Urutan:
1. Tangan kiri: 21 titik × (x, y, z) = 63 angka
2. Tangan kanan: 21 titik × (x, y, z) = 63 angka

Normalisasi **per tangan**:
- Kurangi semua titik dengan titik 0 (pergelangan/wrist) → pergelangan jadi (0,0,0).
- Bagi dengan jarak terjauh (pada sumbu x,y) dari pergelangan ke titik lain → ukuran tangan seragam.
- Jika tangan tidak terdeteksi → 63 angka nol.

Jika **kedua tangan tidak terdeteksi** → sampel **dibuang** (saat ekstraksi dataset) atau **tidak diprediksi** (saat aplikasi).

### 6.3 Fitur KATA = matriks (30, 201) (30 frame)
Per frame 201 angka, urutan:
1. Fitur tangan 126 angka (persis sama dengan 6.2)
2. Pose badan atas: titik pose **0–24** (25 titik × x,y,z = 75 angka), dinormalisasi:
   - Kurangi titik tengah kedua bahu (pose 11 dan 12)
   - Bagi dengan lebar bahu (jarak x,y antara pose 11 dan 12)
   - Jika pose tidak terdeteksi → 75 angka nol

Panjang sekuens:
- Video diambil **semua frame**, lalu **di-resample menjadi tepat 30 frame** dengan indeks merata (`np.linspace`).
- Frame yang kedua tangannya tidak terdeteksi **tetap disimpan** (fitur tangan nol) — posisi pose tetap informatif.

### 6.4 Format file hasil ekstraksi
Simpan sebagai **satu file `.npz` per dataset** (bukan ribuan file kecil):

| File | Isi |
|---|---|
| `data/landmarks/huruf/<nama_dataset>.npz` | `X`: float32 (N, 126); `y`: string label (N,) contoh `"A"`; `sumber`: string nama file asal (N,); `split`: string `"train"`/`"test"`/`""` (N,) jika dataset sudah punya pembagian |
| `data/landmarks/kata/<nama_dataset>.npz` | `X`: float32 (N, 30, 201); `y`: string label (N,); `peraga`: string id peraga (N,) contoh `"wl_s0"`; `sumber`: nama file asal (N,) |

Cara simpan & baca:
```python
np.savez_compressed(path, X=X, y=y, sumber=sumber, split=split)
d = np.load(path, allow_pickle=True); X, y = d["X"], d["y"]
```

### 6.5 Aturan nama label
- Huruf: **kapital satu karakter** `"A"`..`"Z"`.
- Kata: **huruf kecil, spasi diganti garis bawah**, tanpa tanda baca. Contoh: `"terima_kasih"`, `"selamat_pagi"`, `"apa_kabar"`.
- Penyamaan ejaan: `"Di mana"` dan `"Dimana"` → **`"di_mana"`**; `"Kemana"` → **`"ke_mana"`**.
- Daftar label final disimpan di `data/labels/label_huruf.json` dan `data/labels/label_kata.json` (list string, urutan = indeks kelas model). Dibuat oleh T5/T6 dari data training, **jangan diketik manual berbeda-beda**.

### 6.6 Pembagian data (split)
- `random_state=42` di mana pun ada pengacakan.
- **Huruf:** gabungkan semua dataset huruf → stratified split **70% train / 15% val / 15% test**. Jika dataset punya folder Train/Test sendiri (multimodal_corpus), boleh dipakai apa adanya, tetapi tuliskan di laporan.
- **Kata (WL-BISINDO):** dua skenario, laporkan **keduanya**:
  1. *Signer-dependent* (acak): stratified 70/15/15.
  2. *Signer-independent* (orang baru): train = peraga 0,1,2; val = peraga 3; test = peraga 4.
- **Rekaman sendiri (T4):** dipakai **hanya untuk test** "orang baru" (dan untuk kata tambahan jika diputuskan di T3).

### 6.7 Kode lengkap `src/features.py` (dibuat di T1 — salin persis)

```python
"""Fungsi ekstraksi fitur BERSAMA. Semua script WAJIB memakai file ini.

Standar (lihat docs/PANDUAN_PROYEK.md bagian 6):
- Fitur huruf  : 126 angka  (tangan kiri 63 + tangan kanan 63)
- Fitur kata   : 201 angka per frame (tangan 126 + pose 0-24 75), 30 frame
"""
import cv2
import numpy as np
import mediapipe as mp

N_HAND = 63          # 21 titik x 3
N_HANDS = 126        # 2 tangan
POSE_IDX = list(range(25))   # pose badan atas 0..24
N_POSE = 75          # 25 titik x 3
N_FRAME = N_HANDS + N_POSE   # 201
SEQ_LEN = 30

mp_holistic = mp.solutions.holistic

# Pasangan titik pose kiri-kanan (untuk augmentasi cermin)
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def buat_holistic(gambar_diam=False):
    """Buat objek MediaPipe Holistic dengan pengaturan standar proyek."""
    return mp_holistic.Holistic(
        static_image_mode=gambar_diam,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def proses_bgr(holistic, frame_bgr):
    """Jalankan MediaPipe pada frame BGR dari OpenCV. Frame TIDAK di-flip."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic.process(rgb)


def _fitur_tangan(landmarks):
    """21 titik tangan -> 63 angka, relatif pergelangan & diskalakan."""
    if landmarks is None:
        return np.zeros(N_HAND, dtype=np.float32)
    a = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
    a = a - a[0]
    skala = np.max(np.linalg.norm(a[:, :2], axis=1))
    if skala > 0:
        a = a / skala
    return a.flatten()


def fitur_huruf(results):
    """Hasil MediaPipe -> vektor 126 angka. Kembalikan None jika tidak ada tangan."""
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        return None
    return np.concatenate([
        _fitur_tangan(results.left_hand_landmarks),
        _fitur_tangan(results.right_hand_landmarks),
    ])


def _fitur_pose(pose_landmarks):
    """Pose 0..24 -> 75 angka, relatif titik tengah bahu & dibagi lebar bahu."""
    if pose_landmarks is None:
        return np.zeros(N_POSE, dtype=np.float32)
    lm = pose_landmarks.landmark
    p = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in POSE_IDX], dtype=np.float32)
    tengah = (p[11] + p[12]) / 2
    lebar = np.linalg.norm(p[11, :2] - p[12, :2])
    p = p - tengah
    if lebar > 0:
        p = p / lebar
    return p.flatten()


def fitur_frame(results):
    """Hasil MediaPipe -> vektor 201 angka untuk 1 frame (selalu ada nilainya)."""
    tangan = np.concatenate([
        _fitur_tangan(results.left_hand_landmarks),
        _fitur_tangan(results.right_hand_landmarks),
    ])
    return np.concatenate([tangan, _fitur_pose(results.pose_landmarks)])


def resample_sekuens(frames, panjang=SEQ_LEN):
    """List/array (T, 201) -> array (30, 201) dengan indeks merata."""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        return np.zeros((panjang, N_FRAME), dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, panjang).round().astype(int)
    return frames[idx]


def fitur_video(path_video, holistic=None):
    """File video -> array (30, 201). Kembalikan None jika video tak terbaca
    atau tidak ada tangan sama sekali di seluruh video."""
    tutup = holistic is None
    if tutup:
        holistic = buat_holistic(gambar_diam=False)
    cap = cv2.VideoCapture(str(path_video))
    frames, ada_tangan = [], False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = proses_bgr(holistic, frame)
        if res.left_hand_landmarks is not None or res.right_hand_landmarks is not None:
            ada_tangan = True
        frames.append(fitur_frame(res))
    cap.release()
    if tutup:
        holistic.close()
    if not frames or not ada_tangan:
        return None
    return resample_sekuens(frames)


def cermin_huruf(x):
    """Augmentasi cermin untuk fitur huruf (126): tukar kiri-kanan & balik x."""
    x = x.reshape(2, 21, 3).copy()
    x = x[::-1]          # tukar tangan kiri <-> kanan
    x[..., 0] *= -1      # balik sumbu x
    return x.reshape(-1)


def cermin_kata(seq):
    """Augmentasi cermin untuk sekuens kata (30, 201)."""
    seq = seq.copy()
    out = np.empty_like(seq)
    for t in range(seq.shape[0]):
        tangan = cermin_huruf(seq[t, :N_HANDS])
        pose = seq[t, N_HANDS:].reshape(25, 3).copy()
        for a, b in POSE_SWAP:
            pose[[a, b]] = pose[[b, a]]
        pose[:, 0] *= -1
        out[t] = np.concatenate([tangan, pose.reshape(-1)])
    return out
```

---

## 7. PEMBAGIAN TUGAS 6 ANGGOTA

Isi nama anggota pada kolom "Nama".

| Kode | Nama | Peran | Kartu tugas | Bagian laporan |
|---|---|---|---|---|
| **A1** | ________ | Koordinator, setup lingkungan, `features.py`, integrasi akhir | T0, T1, T9 | BAB 1 Pendahuluan + menggabungkan laporan |
| **A2** | ________ | Ekstraksi data huruf | T2 | BAB 3.1–3.2 (data & preprocessing huruf) |
| **A3** | ________ | Ekstraksi data kata + investigasi dataset 40 kata | T3 | BAB 3.3 (data & preprocessing kata) |
| **A4** | ________ | Alat perekam data + koordinasi rekaman 6 anggota | T4 | BAB 2 Tinjauan Pustaka |
| **A5** | ________ | Training & evaluasi model huruf DAN model kata | T5, T6 | BAB 4.1–4.2 (hasil model) |
| **A6** | ________ | Penyusun kalimat, suara, aplikasi real-time, pengujian sistem | T7, T8 | BAB 4.3 (pengujian sistem) + BAB 5 |

**Semua anggota:** ikut direkam sebagai peraga (T4), ikut uji aplikasi (T8), menyiapkan slide bagiannya masing-masing.

> Jika beban A5 terlalu berat, A2 membantu T5 (model huruf) setelah T2 selesai, dan A3 membantu T6 (model kata) setelah T3 selesai.

### Peta ketergantungan (siapa menunggu siapa)

```
T0 Setup ─► T1 features.py ─┬─► T2 Ekstrak huruf ─► T5 Model huruf ─┐
                            ├─► T3 Ekstrak kata  ─► T6 Model kata  ─┼─► T8 Aplikasi ─► T9 Integrasi & laporan
                            └─► T4 Rekam data ──────────────────────┤
                                T7 Kalimat + suara (bisa mulai kapan saja) ─┘
```

- **T1 harus selesai paling awal** (hari 1–2) karena semua bergantung padanya.
- **T7 tidak bergantung** pada siapa pun — A6 bisa mulai hari pertama.
- T4 bisa mulai segera setelah T1 selesai.

---

## 8. KARTU TUGAS RINCI

Format setiap kartu: **Tujuan → Input → Output → Langkah → SELESAI JIKA → Jebakan**.

---

### T0 — Setup lingkungan (A1, lalu setiap anggota di laptopnya)

**Tujuan:** semua anggota bisa menjalankan Python + MediaPipe + TensorFlow.

**Langkah:**
1. Ikuti bagian 3.3 sampai 3.6.
2. Catat versi: `python --version`, `pip show mediapipe tensorflow` → kirim ke grup.
3. Pastikan semua anggota memakai **versi MediaPipe yang sama**.

**SELESAI JIKA:**
- [ ] Perintah cek di 3.5 dan 3.6 berhasil di laptop masing-masing.
- [ ] Versi MediaPipe semua anggota sama (dicatat di grup).

**Jebakan:** Python 3.12/3.13 → pakai 3.11. Error ExecutionPolicy → lihat 3.3.

---

### T1 — `features.py` dan uji webcam (A1)

**Tujuan:** menyediakan fungsi ekstraksi bersama dan membuktikan MediaPipe berjalan dengan webcam.

**Output:**
- `src/features.py` (salin persis dari bagian 6.7)
- `src/cek_webcam.py`

**Langkah:**
1. Buat `src/features.py` dengan isi **persis** bagian 6.7.
2. Buat `src/cek_webcam.py` yang:
   - Membuka webcam `cv2.VideoCapture(0)`.
   - Setiap frame: `res = proses_bgr(holistic, frame)`.
   - Menggambar titik dengan `mp.solutions.drawing_utils.draw_landmarks(frame, res.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)` (sama untuk tangan kanan dan pose dengan `POSE_CONNECTIONS`).
   - Menampilkan teks: FPS, "Kiri: ada/tidak", "Kanan: ada/tidak", dan panjang vektor `fitur_frame(res)` (harus 201).
   - Untuk tampilan boleh `cv2.flip(frame, 1)` **setelah** diproses dan digambar.
   - Tombol `q` untuk keluar.
3. Jalankan: `.\.venv\Scripts\python.exe src\cek_webcam.py`
4. Uji fungsi lain dengan 1 video WL-BISINDO:
   ```python
   from src.features import fitur_video
   x = fitur_video(r"data\raw\kata\wl_bisindo\signer0_label7_sample1.mp4")
   print(x.shape)   # harus (30, 201)
   ```
   (jalankan dari folder root proyek; jika `from src.features` error, tambahkan file kosong `src/__init__.py`).

**SELESAI JIKA:**
- [ ] Jendela webcam muncul, titik tangan & badan tergambar, FPS ≥ 10.
- [ ] Panjang `fitur_frame` = 201.
- [ ] `fitur_video(...)` menghasilkan shape `(30, 201)`.
- [ ] `features.py` dibagikan ke semua anggota.

**Jebakan:** webcam dipakai aplikasi lain (Zoom/Meet) → tutup dulu. Warna aneh → lupa konversi BGR→RGB.

---

### T2 — Ekstraksi landmark HURUF (A2)

**Tujuan:** mengubah ±27.000 gambar huruf menjadi vektor 126 angka.

**Input:** semua folder di `data/raw/huruf/` (lihat 5.1).
**Output:** `data/landmarks/huruf/<nama_dataset>.npz` (format 6.4) + `reports/ekstraksi_huruf.csv`.

**Langkah:**
1. **Inspeksi dulu setiap dataset** (jangan langsung ekstrak):
   - Ekstrak `huruf/bisindo_recognition/BISINDO.zip` ke folder yang sama.
   - Untuk setiap dataset, cetak: daftar subfolder, 5 contoh nama file, jumlah file per label.
   - Tentukan bagaimana label diambil (nama folder atau nama file). Tulis temuannya sebagai komentar di atas `ekstrak_huruf.py`.
2. Buat `src/ekstrak_huruf.py`:
   - Satu objek `buat_holistic(gambar_diam=True)` dipakai ulang untuk semua gambar (jangan dibuat per gambar — lambat).
   - Untuk setiap gambar: `cv2.imread` → jika `None` lewati → `proses_bgr` → `fitur_huruf(res)` → jika `None` (tangan tak terdeteksi) catat sebagai gagal.
   - Gambar sangat besar boleh diperkecil dulu agar sisi terpanjang 640 piksel (`cv2.resize`).
   - Hanya proses ekstensi `.jpg .jpeg .png` (abaikan `.xml`, `.txt`, `.heic`).
   - Label → kapital 1 huruf (`"A"`..`"Z"`). Buang label di luar A–Z.
   - Simpan per dataset ke `.npz` dengan kunci `X, y, sumber, split`.
   - Tampilkan progress dengan `tqdm`.
3. Buat ringkasan `reports/ekstraksi_huruf.csv` kolom: `dataset, label, jumlah_gambar, berhasil, gagal, persen_berhasil`.
4. Ambil 5 contoh gambar per dataset, gambar landmark-nya, simpan ke `reports/contoh_huruf_<dataset>.png` (untuk laporan).

**SELESAI JIKA:**
- [ ] Ada 1 file `.npz` per dataset di `data/landmarks/huruf/`.
- [ ] `X.shape[1] == 126`, `X.dtype == float32`, tidak ada `NaN` (`np.isnan(X).any() == False`).
- [ ] Ringkasan CSV dibuat; total persentase berhasil dilaporkan.
- [ ] Setiap huruf A–Z punya minimal 100 sampel gabungan dari semua dataset (jika kurang, laporkan huruf mana).

**Jebakan:**
- Persentase gagal tinggi pada gambar yang hanya berisi tangan terpotong (crop) → normal; MediaPipe butuh tangan utuh. Laporkan saja.
- Jangan pakai `mp.solutions.hands` — harus `holistic` (konsistensi dengan aplikasi).
- Estimasi waktu: ±5–20 ms per gambar → 27.000 gambar ≈ 5–15 menit di Ryzen 7.

---

### T3 — Ekstraksi landmark KATA + investigasi dataset 40 kata (A3)

**Bagian A — WL-BISINDO (wajib)**

**Input:** `data/raw/kata/wl_bisindo/*.mp4`
**Output:** `data/landmarks/kata/wl_bisindo.npz` (X: (1600, 30, 201)) + `reports/ekstraksi_kata.csv`

**Langkah:**
1. Parse nama file dengan regex `signer(\d+)_label(\d+)_sample(\d+)\.mp4`. Pastikan ada 5 nilai `signer` berbeda dan 32 `label` berbeda; cetak jumlah video per (signer, label).
2. Ubah nomor label → nama kata pakai tabel di 5.2(a), lalu terapkan aturan nama 6.5 (contoh: 10 → `"terima_kasih"`, 15 → `"di_mana"`).
3. Untuk tiap video: `fitur_video(path, holistic)` (buat 1 objek holistic `gambar_diam=False`, **tapi buat ulang per video** agar tracking tidak terbawa dari video sebelumnya — atau panggil `fitur_video(path)` tanpa argumen holistic).
4. Simpan `X, y, peraga ("wl_s0".."wl_s4"), sumber`.
5. Catat video gagal (None) di `reports/ekstraksi_kata.csv`.
6. Buat 1 gambar plot: lintasan titik pergelangan tangan kanan (fitur indeks tertentu dari pose 16) sepanjang 30 frame untuk 3 kata berbeda → `reports/contoh_lintasan_kata.png`.

Estimasi waktu: 1.600 video × ±2–4 detik ≈ 1–2 jam. Jalankan saat laptop tidak dipakai; buat script **bisa dilanjutkan** (simpan hasil sementara setiap 100 video).

**Bagian B — Investigasi dataset 40 kata landmark (wajib dicoba, hasil boleh "tidak bisa")**

Masalah: dataset ini berbentuk (30, 447), berbeda dari standar kita (30, 201), dan tidak ada videonya.

Langkah:
1. Baca `README.txt`, file `.xlsx`, dan gambar struktur folder di `data/raw/kata/bisindo_40kata_landmark/`.
2. Cari paper-nya (judul: *"Indonesian Sign Language (BISINDO) Dataset: Landmark Extraction for capturing Non-Manual Markers"*, penulis Ramadhana, Ho, Luman, Sutoyo — BINUS). Cari tahu **titik mana saja** yang membentuk 447 fitur dan **normalisasinya**.
3. Keputusan (tulis di `docs/keputusan_dataset_40kata.md`):
   - **Opsi 1 (ideal):** jika cara ekstraksinya jelas, buat fungsi `fitur_frame_447(results)` di file terpisah `src/features_447.py` yang meniru persis cara mereka, sehingga aplikasi bisa memakai model yang dilatih pada data ini.
   - **Opsi 2 (aman):** jika tidak jelas, pakai dataset ini **hanya sebagai eksperimen pembanding** di laporan (latih LSTM pada data 447 fitur dan laporkan akurasinya), dan kata-kata pentingnya (halo, kamu, minum, tidur, apa_kabar, selamat_pagi, dll.) **direkam sendiri** di T4.
4. Konversi ke format kita: `data/landmarks/kata/bisindo40_447.npz` (X: (2400, 30, 447), label sesuai aturan 6.5, `peraga` diisi `"b40"` karena info peraga tidak ada — cek README).

**Bagian C — Menetapkan KOSAKATA FINAL (bersama A1, A4, A6)**
- Kosakata aplikasi = 32 kata WL-BISINDO + kata tambahan yang direkam sendiri di T4 (maksimal 10–15 kata, pilih yang membuat kalimat lebih bermakna, misalnya: `kamu, dia, halo, minum, tidur, mau, tidak, sakit, tolong, apa_kabar`).
- Tulis daftarnya di `docs/kosakata_final.md` beserta 10 **contoh kalimat uji** yang bisa dibentuk, contoh: "saya makan pagi", "teman saya tuli", "kamu di mana", "terima kasih teman".

**SELESAI JIKA:**
- [ ] `wl_bisindo.npz` ada, `X.shape == (N, 30, 201)` dengan N mendekati 1.600, tanpa NaN.
- [ ] Laporan video gagal dibuat.
- [ ] `docs/keputusan_dataset_40kata.md` berisi opsi yang dipilih + alasannya.
- [ ] `docs/kosakata_final.md` berisi daftar kata final + 10 kalimat uji.

**Jebakan:** jangan menggabungkan array 447 fitur dengan 201 fitur dalam satu model — ukurannya beda dan artinya beda.

---

### T4 — Alat perekam data + rekaman sendiri (A4, semua anggota jadi peraga)

**Tujuan:** (1) data uji "orang baru", (2) data kata tambahan dari kosakata final, (3) data kalimat uji.

**Output:**
- `src/rekam_data.py`
- `data/landmarks/rekaman_sendiri/kata_<peraga>.npz` dan `huruf_<peraga>.npz`
- (opsional) video mentah `.mp4` di `data/raw/rekaman_sendiri/<peraga>/<label>/`

**Spesifikasi `rekam_data.py`:**
1. Argumen command line: `--peraga A1` `--mode kata|huruf` `--label makan` `--jumlah 10`.
   Contoh: `.\.venv\Scripts\python.exe src\rekam_data.py --peraga A3 --mode kata --label makan --jumlah 10`
2. Tampilan webcam (di-flip untuk tampilan saja) dengan teks: label, peraga, nomor ulangan ke-berapa.
3. **Mode kata:** tekan `SPASI` → hitung mundur 3-2-1 → rekam **60 frame (±2 detik)** → `fitur_frame` per frame → `resample_sekuens` ke (30, 201) → simpan. Ulangi sampai `--jumlah`.
4. **Mode huruf:** tekan `SPASI` → ambil 1 frame setiap 0,2 detik selama 2 detik (10 sampel) → `fitur_huruf` → simpan yang tidak `None`. Geser/putar tangan sedikit selama perekaman agar bervariasi.
5. Tombol `r` = hapus rekaman terakhir (jika salah gerak), `q` = keluar & simpan.
6. Simpan dengan format 6.4 (`peraga` = kode anggota, contoh `"A3"`). Jika file sudah ada, **tambahkan** (append), jangan menimpa.
7. (Opsional) simpan juga video mentah dengan `cv2.VideoWriter`.

**Rencana rekaman (koordinasi A4):**
- Setiap anggota (6 orang) merekam:
  - 32 kata WL-BISINDO × **5 ulangan** → untuk **test orang baru** (WL-BISINDO hanya 5 peraga dari Banten).
  - Kata tambahan (kosakata final T3) × **20 ulangan** → untuk **training + test** kata tambahan.
  - Huruf A–Z × 1 sesi (10 sampel) → test orang baru untuk huruf.
- Pelajari gerakan dulu dari video WL-BISINDO (untuk 32 kata) dan dari kanal YouTube/materi **Pusbisindo** (untuk kata tambahan). Gerakan harus konsisten antar anggota.
- Variasikan: baju berbeda, latar berbeda, pencahayaan berbeda (siang/malam).
- Estimasi waktu per anggota: ±45–60 menit.

**SELESAI JIKA:**
- [ ] `rekam_data.py` berjalan untuk mode kata & huruf, bisa hapus rekaman terakhir.
- [ ] Ada file `.npz` rekaman untuk minimal 5 dari 6 anggota.
- [ ] Bentuk data: kata `(N, 30, 201)`, huruf `(N, 126)`.
- [ ] Tabel jumlah rekaman per anggota per label di `reports/rekaman_sendiri.csv`.

---

### T5 — Training & evaluasi model HURUF (A5)

**Input:** semua `data/landmarks/huruf/*.npz` (+ `rekaman_sendiri/huruf_*.npz` untuk test orang baru)
**Output:** `models/huruf_mlp.keras`, `models/huruf_rf.joblib`, `models/huruf_svm.joblib`, `data/labels/label_huruf.json`, file di `reports/huruf_*`

**Langkah:**
1. Gabungkan semua dataset huruf → `X (N,126)`, `y (N,)`.
2. `label_huruf = sorted(set(y))` → simpan ke `data/labels/label_huruf.json`. Ubah `y` ke indeks.
3. Split 70/15/15 stratified, `random_state=42`.
4. **Augmentasi hanya pada data train:**
   - Cermin: `cermin_huruf(x)` (meniru pengguna kidal) — tambahkan sebagai sampel baru.
   - Noise: `x + np.random.normal(0, 0.01, x.shape)`.
5. **Model utama MLP (Keras):**
   ```
   Input(126) → Dense(256, relu) → Dropout(0.3) → Dense(128, relu) → Dropout(0.3) → Dense(n_kelas, softmax)
   optimizer=Adam(1e-3), loss=sparse_categorical_crossentropy, metrics=[accuracy]
   epochs=100, batch_size=64, EarlyStopping(monitor=val_loss, patience=10, restore_best_weights=True)
   ```
6. **Pembanding (scikit-learn):** `RandomForestClassifier(n_estimators=300, random_state=42)` dan `SVC(kernel="rbf", C=10, probability=True)`.
7. **Evaluasi pada test set** (dan pada rekaman sendiri jika ada):
   - Accuracy, Precision, Recall, F1 (macro) → `classification_report`
   - Confusion matrix 26×26 → `reports/huruf_cm_mlp.png`
   - Grafik loss & accuracy train/val per epoch → `reports/huruf_kurva_mlp.png`
   - Waktu prediksi rata-rata per sampel (ms)
8. Simpan tabel ringkasan → `reports/huruf_perbandingan.csv` kolom: `model, accuracy, precision, recall, f1, ms_per_sampel, akurasi_orang_baru`.

**SELESAI JIKA:**
- [ ] 3 model tersimpan, `label_huruf.json` tersimpan.
- [ ] Tabel perbandingan + confusion matrix + kurva tersedia.
- [ ] Akurasi test MLP dilaporkan (target ≥ 90%; jika di bawah, laporkan dan analisis huruf yang tertukar).

---

### T6 — Training & evaluasi model KATA (A5, dibantu A3)

**Input:** `data/landmarks/kata/wl_bisindo.npz` + `data/landmarks/rekaman_sendiri/kata_*.npz`
**Output:** `models/kata_lstm.keras`, `models/kata_gru.keras`, `models/kata_cnn1d.keras`, `data/labels/label_kata.json`, file di `reports/kata_*`

**Langkah:**
1. Muat data, gabungkan kata tambahan dari rekaman sendiri (sesuai kosakata final T3).
2. `label_kata = sorted(set(y))` → `data/labels/label_kata.json`.
3. Dua skenario split (bagian 6.6): **signer-dependent** dan **signer-independent**. Rekaman sendiri untuk 32 kata WL-BISINDO → **hanya test**.
4. Augmentasi train: `cermin_kata(seq)`, noise kecil (σ=0.01), dan skala acak (`seq * uniform(0.9, 1.1)`).
5. **Model utama LSTM:**
   ```
   Input(30, 201) → LSTM(128, return_sequences=True) → Dropout(0.3)
                 → LSTM(64) → Dropout(0.3)
                 → Dense(64, relu) → Dense(n_kelas, softmax)
   Adam(1e-3), sparse_categorical_crossentropy, epochs=150, batch_size=32,
   EarlyStopping(val_loss, patience=15, restore_best_weights=True)
   ```
6. **Pembanding:**
   - GRU: sama persis, ganti `LSTM` → `GRU`.
   - 1D-CNN: `Conv1D(64,3,relu) → Conv1D(128,3,relu) → GlobalAveragePooling1D → Dropout(0.3) → Dense(64,relu) → Dense(n,softmax)`.
7. Evaluasi sama seperti T5 (accuracy, precision, recall, F1, confusion matrix, kurva, ms/prediksi) untuk **kedua skenario**.
8. Tabel → `reports/kata_perbandingan.csv` kolom: `model, skenario, accuracy, precision, recall, f1, ms_per_sampel`.
9. (Jika T3-B memilih Opsi 2) latih LSTM juga pada `bisindo40_447.npz` dan laporkan sebagai eksperimen tambahan.

**SELESAI JIKA:**
- [ ] 3 model tersimpan + `label_kata.json`.
- [ ] Tabel perbandingan untuk 2 skenario.
- [ ] Confusion matrix LSTM untuk kedua skenario.
- [ ] Analisis singkat: kata mana yang paling sering tertukar dan kenapa (contoh: gerakan mirip).

**Jebakan:** akurasi signer-independent **pasti lebih rendah** dari signer-dependent. Itu normal dan justru menarik untuk dibahas — jangan disembunyikan.

---

### T7 — Penyusun kalimat + suara (A6) — bisa dimulai hari pertama

**Output:** `src/penyusun_kalimat.py`, `src/suara.py`

**Spesifikasi `penyusun_kalimat.py`** — class `PenyusunKalimat`:

| Method | Perilaku |
|---|---|
| `tambah_prediksi(label, keyakinan, mode)` | Dipanggil setiap ada prediksi. Hanya **diterima** jika `keyakinan ≥ 0.80` **dan** label sama muncul **berturut-turut** (huruf: 8 kali, kata: 3 kali) **dan** sudah lewat **jeda 1 detik** sejak token terakhir diterima **dan** label berbeda dari token terakhir (kecuali sudah lewat 2 detik — agar huruf dobel seperti "AA" tetap bisa). |
| `spasi()` | Akhiri kata yang sedang dieja huruf demi huruf. |
| `hapus_terakhir()` | Hapus token/huruf terakhir. |
| `bersihkan()` | Kosongkan kalimat. |
| `teks()` | Kembalikan kalimat siap tampil: huruf yang dieja digabung jadi satu kata (`S`,`I`,`T`,`I` → `siti`), label kata diubah `_` → spasi, huruf pertama kalimat kapital. Contoh: `["saya","makan","pagi"]` → `"Saya makan pagi"`. |

Gunakan `time.monotonic()` untuk waktu. Tulis **unit test sederhana** di bagian `if __name__ == "__main__":` yang mensimulasikan urutan prediksi dan mencetak hasil `teks()`.

**Spesifikasi `suara.py`:**
```python
def ucapkan(teks: str) -> None
```
- Utama: `edge-tts` suara `id-ID-GadisNeural` → simpan ke file mp3 sementara → putar dengan `pygame.mixer`.
  ```python
  import asyncio, edge_tts
  asyncio.run(edge_tts.Communicate(teks, "id-ID-GadisNeural").save(path_mp3))
  ```
- Jika gagal (tidak ada internet) → **cadangan** `pyttsx3` (offline).
- Jalankan di **thread terpisah** (`threading.Thread(daemon=True)`) supaya video webcam tidak macet saat suara diputar.
- Jangan putar suara yang sama berulang-ulang.

**SELESAI JIKA:**
- [ ] Simulasi di `__main__` penyusun kalimat menghasilkan kalimat benar (tidak ada kata dobel).
- [ ] `python src\suara.py` mengucapkan "Halo, apa kabar?" dengan suara Indonesia.
- [ ] Saat internet dimatikan, suara cadangan tetap keluar.

---

### T8 — Aplikasi real-time + pengujian sistem (A6)

**Input:** `features.py`, model & label dari T5/T6, modul T7.
**Output:** `src/app.py`, `reports/pengujian_sistem.csv`, video demo.

**Spesifikasi `app.py`:**
1. Muat `models/huruf_mlp.keras`, `models/kata_lstm.keras`, `data/labels/label_huruf.json`, `data/labels/label_kata.json`.
2. Loop webcam: `res = proses_bgr(holistic, frame)` → simpan `fitur_frame(res)` ke `collections.deque(maxlen=30)`.
3. **Mode HURUF** (tombol `h`): setiap frame, jika `fitur_huruf(res)` tidak `None` → prediksi MLP → `penyusun.tambah_prediksi(label, prob, "huruf")`.
4. **Mode KATA** (tombol `k`): setiap **5 frame**, jika deque penuh (30) dan ada tangan terdeteksi di ≥ 10 dari 30 frame → prediksi LSTM pada `np.array(deque)[None]` → `penyusun.tambah_prediksi(label, prob, "kata")`.
5. Tombol: `SPASI` = spasi, `BACKSPACE` = hapus terakhir, `c` = bersihkan, `ENTER` = ucapkan kalimat (`suara.ucapkan`), `q` = keluar.
6. Tampilan (frame di-flip **hanya untuk tampilan**):
   - Landmark tangan & pose tergambar.
   - Kiri atas: mode aktif, prediksi saat ini + persen keyakinan, FPS.
   - Bawah: kotak hitam berisi kalimat (`penyusun.teks()`).
7. Pakai `model(x, training=False)` atau `model.predict(x, verbose=0)` — pilih yang lebih cepat dan ukur.

**Pengujian sistem (semua anggota):**
- Uji 10 kalimat uji dari `docs/kosakata_final.md`, masing-masing oleh minimal 3 anggota.
- Catat di `reports/pengujian_sistem.csv`: `penguji, kalimat_target, kalimat_hasil, benar_semua(ya/tidak), jumlah_kata_salah, waktu_detik`.
- Hitung: persentase kalimat benar, **Word Error Rate** sederhana (kata salah / total kata), rata-rata FPS, rata-rata waktu per kalimat.
- Uji kondisi: terang vs redup, background polos vs ramai → catat pengaruhnya.
- Rekam **video demo 1–2 menit** (pakai OBS / Xbox Game Bar `Win+G`).

**SELESAI JIKA:**
- [ ] Aplikasi berjalan ≥ 10 FPS di laptop koordinator.
- [ ] Bisa mengeja nama (mode huruf), menyusun kalimat (mode kata), dan mengucapkannya.
- [ ] `pengujian_sistem.csv` terisi + ringkasan angka.
- [ ] Video demo tersimpan.

---

### T9 — Integrasi akhir, README, laporan (A1)

**Langkah:**
1. Jalankan ulang seluruh alur di laptop koordinator dari nol: ekstraksi → training → app. Pastikan tidak ada path yang hanya ada di laptop anggota lain.
2. Lengkapi `README.md` di root: cara install, cara menjalankan tiap script (urutan), struktur folder.
3. Pastikan semua sumber dataset & lisensinya dicantumkan di laporan (bagian 5).
4. Gabungkan laporan (bagian 10) & slide presentasi.
5. Simpan backup akhir (kode + models + landmarks + laporan + video demo) di Google Drive kelompok.

**SELESAI JIKA:**
- [ ] Orang yang baru meng-clone bisa menjalankan aplikasi hanya dengan mengikuti README.
- [ ] Laporan & slide lengkap, semua bagian anggota masuk.

---

## 9. JADWAL DAN URUTAN PENGERJAAN

Sesuaikan dengan tanggal pengumpulan UAS. Usulan 4 minggu:

| Minggu | A1 | A2 | A3 | A4 | A5 | A6 |
|---|---|---|---|---|---|---|
| **1** | T0, T1 (hari 1–2), bantu setup anggota | T0, inspeksi dataset huruf, mulai T2 | T0, mulai T3-A, T3-B | T0, buat `rekam_data.py`, belajar gerakan | T0, siapkan kerangka script T5/T6, belajar Keras | T0, T7 selesai |
| **2** | Monitor progres, mulai BAB 1 | T2 selesai | T3 selesai + kosakata final | **Sesi rekaman semua anggota** | T5 (huruf) | Kerangka `app.py` dengan model dummy |
| **3** | Integrasi awal | Bantu T5 / BAB 3 | Bantu T6 / BAB 3 | BAB 2 | T6 (kata) | T8 dengan model asli |
| **4** | T9, gabung laporan & slide | Revisi laporan | Revisi laporan | Revisi laporan | BAB 4.1–4.2 | Pengujian sistem, video demo, BAB 4.3 & 5 |

**Rapat singkat** (15 menit, online) minimal 2× seminggu: tiap anggota lapor "sudah apa, kendala apa, berikutnya apa".

---

## 10. LAPORAN DAN PRESENTASI

### Kerangka laporan
- **BAB 1 Pendahuluan** (A1): latar belakang (komunitas Tuli Indonesia, kesenjangan komunikasi, BISINDO vs SIBI), rumusan masalah, tujuan, batasan masalah, manfaat.
- **BAB 2 Tinjauan Pustaka** (A4): BISINDO, MediaPipe Holistic, MLP, LSTM, GRU, 1D-CNN, Random Forest, SVM, Text-to-Speech, metrik evaluasi, minimal 5 penelitian terdahulu (tabel: peneliti, metode, data, hasil).
- **BAB 3 Metodologi** (A2, A3): alur penelitian, dataset (tabel bagian 5 + lisensi), ekstraksi fitur (standar 126/201 fitur), augmentasi, arsitektur model, skenario split, metrik.
- **BAB 4 Hasil dan Pembahasan** (A5, A6): tabel perbandingan model huruf & kata, confusion matrix, kurva training, signer-dependent vs independent, pengujian sistem real-time.
- **BAB 5 Kesimpulan dan Saran** (A6).
- **Daftar Pustaka** + sitasi semua dataset.

### Rumusan masalah (siap pakai)
1. Bagaimana merancang sistem yang dapat menerjemahkan isyarat BISINDO (huruf dan kata) menjadi teks dan suara secara real-time menggunakan MediaPipe dan LSTM?
2. Berapa tingkat akurasi model MLP dalam mengenali huruf dan model LSTM dalam mengenali kata BISINDO dibandingkan model pembanding?
3. Bagaimana kinerja sistem saat digunakan secara real-time oleh pengguna yang datanya tidak digunakan dalam pelatihan?

### Batasan masalah (siap pakai)
- Bahasa isyarat yang digunakan adalah BISINDO; dataset kata utama berasal dari variasi Banten.
- Huruf yang dikenali A–Z; kata yang dikenali terbatas pada kosakata final (±32–45 kata).
- Kalimat dibentuk dari kata yang diperagakan satu per satu dengan jeda, tanpa memproses tata bahasa BISINDO secara penuh.
- Satu orang di depan webcam, pencahayaan cukup.
- Ekspresi wajah (non-manual markers) tidak dimodelkan.

### Pertanyaan yang mungkin ditanyakan dosen
| Pertanyaan | Inti jawaban |
|---|---|
| Kenapa MediaPipe, bukan CNN gambar langsung? | Lebih ringan, real-time di CPU, tahan perbedaan background & warna kulit |
| Kenapa LSTM untuk kata? | Kata = gerakan berurutan, LSTM mengingat urutan waktu |
| Kenapa huruf pakai MLP? | Huruf = pose diam, cukup 1 frame, lebih cepat |
| Bagaimana kalau orang lain yang memakai? | Sudah diuji skenario signer-independent & rekaman anggota yang tidak ikut training |
| BISINDO beda per daerah? | Disebut di batasan masalah; dataset utama variasi Banten |
| Kenapa akurasi orang baru lebih rendah? | Variasi gaya isyarat antar orang; solusi: tambah peraga & augmentasi |

### Presentasi (±10–15 menit)
1. Masalah & tujuan (A1) → 2. Data (A2/A3) → 3. Metode (A4) → 4. Hasil model (A5) → 5. **Demo live** + video cadangan (A6) → 6. Kesimpulan (A1).
**Selalu siapkan video demo cadangan** jika demo live gagal (webcam/cahaya ruangan).

---

## 11. KESALAHAN UMUM DAN SOLUSINYA

| Gejala | Penyebab | Solusi |
|---|---|---|
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | Versi MediaPipe terlalu baru | Bagian 3.5: `pip install "mediapipe==0.10.21"` |
| `ModuleNotFoundError: src` | Script dijalankan dari folder yang salah | Jalankan dari root proyek; tambahkan `src/__init__.py` |
| Webcam hitam / tidak terbuka | Dipakai aplikasi lain / indeks salah | Tutup Zoom/Meet; coba `VideoCapture(1)` |
| Tangan sering tidak terdeteksi | Cahaya redup, tangan terlalu dekat/terpotong | Tambah cahaya, mundur ±1 meter |
| Akurasi test tinggi tapi aplikasi sering salah | Data training beda kondisi dengan webcam; frame di-flip sebelum MediaPipe; normalisasi beda | Pastikan selalu pakai `features.py`, jangan flip sebelum proses, tambah rekaman sendiri |
| Tangan kiri & kanan tertukar | Efek cermin | Augmentasi `cermin_huruf`/`cermin_kata` sudah menangani; jangan flip sebelum MediaPipe |
| Shape error `(30, 201)` vs `(30, 447)` | Mencampur dataset 40 kata dengan standar | Jangan dicampur, lihat T3-B |
| Training sangat lambat | Batch kecil / model terlalu besar | Pakai arsitektur di T5/T6, batch 32–64; Colab jika perlu |
| Aplikasi patah-patah saat suara diputar | TTS di thread utama | Jalankan `ucapkan` di thread terpisah (T7) |
| Kata dobel di kalimat | Tidak ada jeda/syarat stabil | Ikuti aturan `tambah_prediksi` di T7 |
| `edge-tts` error | Tidak ada internet | Cadangan `pyttsx3` |
| Disk penuh | Data mentah besar | Data mentah cukup di 1 laptop; anggota lain cukup pakai `data/landmarks/` |

---

## 12. GLOSARIUM

| Istilah | Arti sederhana |
|---|---|
| BISINDO | Bahasa Isyarat Indonesia, dipakai sehari-hari oleh komunitas Tuli |
| SIBI | Sistem Isyarat Bahasa Indonesia, versi resmi di sekolah (bukan yang kita pakai) |
| Landmark / keypoint | Titik koordinat (x, y, z) pada tangan/badan yang dideteksi MediaPipe |
| MediaPipe Holistic | Alat dari Google yang mendeteksi titik tangan, badan, dan wajah dari gambar |
| Frame | Satu gambar diam dari video (webcam ±30 frame per detik) |
| Sekuens | Urutan beberapa frame (di proyek ini 30 frame) |
| Normalisasi | Menyamakan skala/posisi data agar tidak terpengaruh jarak & posisi orang |
| Augmentasi | Memperbanyak data dengan memodifikasi sedikit (cermin, noise) |
| MLP | Jaringan saraf sederhana berlapis, untuk data 1 frame |
| LSTM / GRU | Jaringan saraf dengan "ingatan" untuk data berurutan |
| 1D-CNN | Jaringan konvolusi pada data berurutan |
| Random Forest / SVM | Algoritma machine learning klasik sebagai pembanding |
| Softmax | Lapisan akhir yang mengubah output jadi peluang tiap kelas |
| Epoch | Satu kali model melihat seluruh data training |
| Early stopping | Berhenti training otomatis jika model tidak membaik |
| Overfitting | Model hafal data training tapi buruk di data baru |
| Signer-dependent | Uji dengan orang yang sama seperti saat training |
| Signer-independent | Uji dengan orang yang **tidak** ada di data training (lebih sulit, lebih realistis) |
| Confusion matrix | Tabel yang menunjukkan kelas mana yang sering tertukar |
| Precision / Recall / F1 | Ukuran ketepatan model selain akurasi |
| WER (Word Error Rate) | Persentase kata salah dalam kalimat hasil terjemahan |
| TTS (Text-to-Speech) | Mengubah teks jadi suara |
| FPS | Frame per detik — ukuran kelancaran aplikasi real-time |
| venv | Lingkungan Python khusus proyek agar library tidak bentrok |
