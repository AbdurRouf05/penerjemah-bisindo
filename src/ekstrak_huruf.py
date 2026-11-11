"""T2 - Ekstraksi landmark HURUF dari semua dataset gambar.

Hasil inspeksi dataset (cara mengambil label):
- bisindo_7peraga/Mendeley BISINDO/<A..Z>/*.jpg        -> label = nama folder
- multimodal_corpus/{Train,Test}/<A..Z>/*.jpg           -> label = nama folder, split dari folder
                                                           (.xml bounding box diabaikan)
- alphabet_image_data/02. Resized Images/<A..Z>/*.jpg   -> label = nama folder
- bisindo_recognition/BISINDO/<A..Z>/*                  -> label = nama folder (hasil ekstrak zip)
- dataset_of_bisindo/{Training,Testing} data/A-01.png   -> label = huruf pertama nama file
- bisindo_70_30/{Training,Test} Data/A2.png             -> label = huruf pertama nama file

Jalankan (dari folder root proyek):
    .venv\\Scripts\\python.exe -m src.ekstrak_huruf
Bisa dihentikan kapan saja dan dijalankan ulang -> otomatis lanjut.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src import lanjut

ROOT = lanjut.ROOT
RAW = ROOT / "data" / "raw" / "huruf"
OUT = ROOT / "data" / "landmarks" / "huruf"
REPORTS = ROOT / "reports"
EKSTENSI = {".jpg", ".jpeg", ".png"}
HURUF = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _split_dari_path(p):
    s = str(p).lower()
    if "train" in s:
        return "train"
    if "test" in s:
        return "test"
    return ""


def daftar_dataset():
    """Kembalikan {nama_dataset: [(path, label, split), ...]}."""
    def per_folder(root):
        out = []
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in EKSTENSI and p.parent.name.upper() in HURUF:
                out.append((p, p.parent.name.upper(), _split_dari_path(p.relative_to(root))))
        return out

    def per_nama_file(root):
        out = []
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() in EKSTENSI and p.name[0].upper() in HURUF:
                out.append((p, p.name[0].upper(), _split_dari_path(p.relative_to(root))))
        return out

    ds = {
        "bisindo_7peraga": per_folder(RAW / "bisindo_7peraga"),
        "multimodal_corpus": per_folder(RAW / "multimodal_corpus"),
        "alphabet_image_data": per_folder(RAW / "alphabet_image_data"),
        "bisindo_recognition": per_folder(RAW / "bisindo_recognition" / "BISINDO"),
        "dataset_of_bisindo": per_nama_file(RAW / "dataset_of_bisindo"),
        "bisindo_70_30": per_nama_file(RAW / "bisindo_70_30"),
    }
    # split hanya bermakna untuk dataset yang punya folder train/test sendiri
    for nama in ["bisindo_7peraga", "alphabet_image_data", "bisindo_recognition"]:
        ds[nama] = [(p, l, "") for p, l, _ in ds[nama]]
    return ds


# ---------- bagian yang berjalan di proses worker ----------
_holistic = None


def _init_worker():
    global _holistic
    from src.features import buat_holistic
    _holistic = buat_holistic(gambar_diam=True)


def _kerja(path_str):
    from src.features import proses_bgr, fitur_huruf
    try:
        img = cv2.imread(path_str)
        if img is None:
            return path_str, None
        h, w = img.shape[:2]
        if max(h, w) > 640:  # perkecil gambar besar agar cepat
            s = 640 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
        return path_str, fitur_huruf(proses_bgr(_holistic, img))
    except Exception:
        return path_str, None
# -----------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    ringkasan = []

    for nama, items in daftar_dataset().items():
        path_npz = OUT / f"{nama}.npz"
        if path_npz.exists():
            print(f"[{nama}] sudah selesai, dilewati ({path_npz.name})")
            d = np.load(path_npz, allow_pickle=True)
            berhasil = set(d["sumber"])
        else:
            kunci = [str(p) for p, _, _ in items]
            hasil = lanjut.jalankan(f"huruf_{nama}", kunci, _kerja, _init_worker,
                                    workers=args.workers, simpan_tiap=500)
            X, y, sumber, split = [], [], [], []
            for p, label, sp in items:
                f = hasil.get(str(p))
                if f is not None:
                    X.append(f)
                    y.append(label)
                    sumber.append(str(p.relative_to(ROOT)))
                    split.append(sp)
            X = np.asarray(X, dtype=np.float32).reshape(-1, 126)
            np.savez_compressed(path_npz, X=X, y=np.array(y), sumber=np.array(sumber),
                                split=np.array(split))
            lanjut.hapus_checkpoint(f"huruf_{nama}")
            berhasil = set(sumber)
            print(f"[{nama}] tersimpan: X={X.shape}")

        # ringkasan per label
        stat = defaultdict(lambda: [0, 0])
        for p, label, _ in items:
            stat[label][0] += 1
            stat[label][1] += str(p.relative_to(ROOT)) in berhasil
        for label in sorted(stat):
            n, ok = stat[label]
            ringkasan.append([nama, label, n, ok, n - ok, round(100 * ok / max(n, 1), 1)])

    with open(REPORTS / "ekstraksi_huruf.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "label", "jumlah_gambar", "berhasil", "gagal", "persen_berhasil"])
        w.writerows(ringkasan)

    total = sum(r[2] for r in ringkasan)
    ok = sum(r[3] for r in ringkasan)
    print(f"\nTOTAL: {ok}/{total} berhasil ({100 * ok / max(total, 1):.1f}%)")
    per_huruf = defaultdict(int)
    for r in ringkasan:
        per_huruf[r[1]] += r[3]
    kurang = {h: n for h, n in per_huruf.items() if n < 100}
    print("Huruf < 100 sampel:", kurang if kurang else "tidak ada")


if __name__ == "__main__":
    main()
