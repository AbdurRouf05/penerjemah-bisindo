# Panduan Merekam Data Huruf (untuk anggota kelompok)

Tugas kamu: merekam isyarat huruf BISINDO dengan webcam. Rekamanmu membuat model
lebih pintar mengenali tangan orang yang berbeda-beda.
**Tidak perlu laptop kuat**: tidak ada training, hanya kamera + MediaPipe.

## 1. Persiapan (sekali saja, ±10 menit)

Butuh: **Python 3.11** ([python.org](https://www.python.org/downloads/release/python-3119/),
centang "Add python.exe to PATH" saat instal) dan **Git** ([git-scm.com](https://git-scm.com/)).

Buka PowerShell, lalu:
```powershell
git clone https://github.com/AbdurRouf05/penerjemah-bisindo.git
cd penerjemah-bisindo
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-rekam.txt
```
Unduhannya sekitar 250 MB.

## 2. Merekam

Ganti `budi` dengan **nama panggilanmu** (huruf kecil, tanpa spasi). Pakai nama yang
sama setiap kali merekam.

Rekam semua huruf A-Z (3 kali per huruf, ±15 menit):
```powershell
.\.venv\Scripts\python.exe -m src.rekam_data --peraga budi --mode huruf --label semua
```
Atau rekam huruf tertentu saja, misalnya yang sering salah:
```powershell
.\.venv\Scripts\python.exe -m src.rekam_data --peraga budi --mode huruf --label STMN --jumlah 5
```

Di jendela kamera:
| Tombol | Fungsi |
|---|---|
| `SPASI` | mulai rekam (hitung mundur 3-2-1, lalu rekam 2 detik) |
| `r` | hapus rekaman terakhir (kalau salah isyarat) |
| `n` | lewati huruf ini |
| `q` | berhenti (yang sudah direkam tetap tersimpan) |

**Tips agar datanya bagus:**
- Cahaya terang, wajah dan kedua tangan terlihat jelas, duduk ±60 cm dari kamera.
- Saat merekam 2 detik, **gerakkan tangan sedikit** (maju-mundur, agak miring).
  Tujuannya supaya model belajar banyak variasi.
- Huruf dua tangan: jangan sampai satu tangan menutupi tangan lainnya sepenuhnya.
- Pastikan garis kerangka tangan muncul di layar. Kalau tidak muncul, tangan tidak terbaca.

## 3. Cek hasil rekaman

```powershell
.\.venv\Scripts\python.exe -m src.cek_rekaman
```
Muncul jumlah sampel per huruf dan peringatan kalau ada yang kurang bagus.
Huruf yang diberi peringatan sebaiknya direkam ulang (`--label` huruf itu saja).
Rekaman baru otomatis **ditambahkan** ke file lama, tidak menimpa.

## 4. Mengirim data

Hasil rekamanmu ada di satu file: `data/landmarks/rekaman_sendiri/huruf_<nama>.npz`.

**Cara A (lewat Git):**
```powershell
git pull
git add data/landmarks/rekaman_sendiri/huruf_budi.npz
git commit -m "data: rekaman huruf budi"
git push
```
Syaratnya, kamu sudah diundang sebagai *collaborator* di repo GitHub.

**Cara B (paling mudah):** kirim file `huruf_<nama>.npz` itu ke penanggung jawab training
lewat WA atau Drive. Dia yang memasukkan file itu ke repo.

Setiap orang punya file sendiri, jadi data tidak akan saling bertabrakan.

## 5. Untuk penanggung jawab training (laptop kuat)

Setelah file rekaman teman masuk:
```powershell
git pull
.\.venv\Scripts\python.exe -m src.cek_rekaman
.\.venv\Scripts\python.exe -m src.latih_huruf --pakai-rekaman
```
Lalu commit dan push model baru (`models/huruf_mlp.keras`, `data/labels/`, `reports/`).
