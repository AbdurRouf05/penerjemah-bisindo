"""T4 - Alat perekam data sendiri (huruf / kata) dari webcam.

Contoh:
    .venv\\Scripts\\python.exe -m src.rekam_data --peraga A3 --mode kata --label makan --jumlah 10
    .venv\\Scripts\\python.exe -m src.rekam_data --peraga A3 --mode huruf --label B --jumlah 3

Tombol: SPASI = mulai rekam, r = hapus rekaman terakhir, q = keluar & simpan.
Hasil ditambahkan (append) ke data/landmarks/rekaman_sendiri/<mode>_<peraga>.npz
"""
import argparse
import string
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.features import (buat_holistic, fitur_frame, fitur_huruf, proses_bgr,
                          resample_sekuens)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "landmarks" / "rekaman_sendiri"
gambar = mp.solutions.drawing_utils
H = mp.solutions.holistic
FRAME_KATA = 60


def baca_frame(cap, holistic):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Webcam tidak terbaca.")
    res = proses_bgr(holistic, frame)
    gambar.draw_landmarks(frame, res.left_hand_landmarks, H.HAND_CONNECTIONS)
    gambar.draw_landmarks(frame, res.right_hand_landmarks, H.HAND_CONNECTIONS)
    gambar.draw_landmarks(frame, res.pose_landmarks, H.POSE_CONNECTIONS)
    return res, cv2.flip(frame, 1)


def tampil(img, baris, warna=(0, 255, 0)):
    for i, t in enumerate(baris):
        cv2.putText(img, t, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5)
        cv2.putText(img, t, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, warna, 2)
    cv2.imshow("Rekam data", img)
    return cv2.waitKey(1) & 0xFF


def simpan(path, sampel, label, peraga, mode):
    if not sampel:
        return
    X = np.stack(sampel).astype(np.float32)
    y = np.array([label] * len(X))
    pr = np.array([peraga] * len(X))
    sumber = np.array([f"webcam_{int(time.time())}"] * len(X))
    if path.exists():  # append, jangan menimpa
        d = np.load(path, allow_pickle=True)
        X = np.concatenate([d["X"], X])
        y = np.concatenate([d["y"], y])
        pr = np.concatenate([d["peraga"], pr])
        sumber = np.concatenate([d["sumber"], sumber])
    extra = {"split": np.array([""] * len(X))} if mode == "huruf" else {}
    np.savez_compressed(path, X=X, y=y, peraga=pr, sumber=sumber, **extra)
    print(f"Tersimpan {path.name}: total {len(X)} sampel")


def rekam_satu_label(cap, holistic, label, peraga, mode, jumlah, info):
    """Rekam `jumlah` kali untuk 1 label. Kembalikan (sampel, keluar?)."""
    rekaman = []  # tiap elemen = list sampel dari 1 kali rekam
    while len(rekaman) < jumlah:
        res, img = baca_frame(cap, holistic)
        k = tampil(img, [f"Label: {label}   ({info})   Peraga: {peraga}",
                         f"Rekaman {len(rekaman)}/{jumlah}",
                         "SPASI=rekam  r=hapus terakhir  n=lewati  q=keluar"])
        if k == ord("q"):
            return [s for r in rekaman for s in r], True
        if k == ord("n"):
            break
        if k == ord("r") and rekaman:
            rekaman.pop()
            print("Rekaman terakhir dihapus")
        if k != 32:
            continue
        # hitung mundur 3-2-1
        for n in (3, 2, 1):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1:
                _, img = baca_frame(cap, holistic)
                tampil(img, [f"{label}: siap... {n}"], (0, 200, 255))
        if mode == "kata":
            frames = []
            while len(frames) < FRAME_KATA:
                res, img = baca_frame(cap, holistic)
                frames.append(fitur_frame(res))
                tampil(img, [f"MEREKAM {len(frames)}/{FRAME_KATA}"], (0, 0, 255))
            rekaman.append([resample_sekuens(frames)])
        else:
            sampel, t0, t_ambil = [], time.monotonic(), 0.0
            while time.monotonic() - t0 < 2:
                res, img = baca_frame(cap, holistic)
                if time.monotonic() - t_ambil >= 0.2:
                    t_ambil = time.monotonic()
                    f = fitur_huruf(res)
                    if f is not None:
                        sampel.append(f)
                tampil(img, [f"MEREKAM {label}... {len(sampel)} sampel",
                             "gerakkan tangan sedikit (maju/mundur, miring)"], (0, 0, 255))
            if len(sampel) < 3:
                print(f"  {label}: tangan hampir tidak terdeteksi, rekaman diulang")
                continue
            rekaman.append(sampel)
        print(f"  {label}: rekaman {len(rekaman)} selesai ({len(rekaman[-1])} sampel)")
    return [s for r in rekaman for s in r], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peraga", required=True, help="nama perekam, mis. budi (tanpa spasi)")
    ap.add_argument("--mode", choices=["kata", "huruf"], required=True)
    ap.add_argument("--label", required=True,
                    help='huruf: "A", "STDX", atau "semua" | kata: mis. "makan"')
    ap.add_argument("--jumlah", type=int, default=None,
                    help="berapa kali rekam per label (bawaan: huruf 3, kata 10)")
    a = ap.parse_args()
    peraga = a.peraga.strip().lower().replace(" ", "_")
    if a.mode == "huruf":
        daftar = list(string.ascii_uppercase) if a.label.lower() == "semua"             else [h for h in a.label.upper() if h.isalpha()]
    else:
        daftar = [a.label.lower().replace(" ", "_")]
    jumlah = a.jumlah or (3 if a.mode == "huruf" else 10)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.mode}_{peraga}.npz"
    cap = cv2.VideoCapture(0)
    holistic = buat_holistic(gambar_diam=False)
    for i, label in enumerate(daftar, 1):
        print(f"[{i}/{len(daftar)}] {label}")
        sampel, keluar = rekam_satu_label(cap, holistic, label, peraga, a.mode, jumlah,
                                          f"{i}/{len(daftar)}")
        simpan(path, sampel, label, peraga, a.mode)   # simpan tiap label -> aman jika berhenti
        if keluar:
            break
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
