"""Cek hasil rekaman sendiri (ringan, tanpa TensorFlow).

    .venv\\Scripts\\python.exe -m src.cek_rekaman

Menampilkan jumlah sampel per perekam & per huruf, dan memberi peringatan jika
huruf dua tangan banyak terekam hanya dengan satu tangan.
"""
import collections
import json

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DIR = ROOT / "data" / "landmarks" / "rekaman_sendiri"
PATH_DUA = ROOT / "data" / "labels" / "huruf_dua_tangan.json"


def main():
    huruf_dua = set(json.loads(PATH_DUA.read_text(encoding="utf-8"))) if PATH_DUA.exists() else set()
    files = sorted(DIR.glob("*.npz"))
    if not files:
        print("Belum ada rekaman di", DIR)
        return
    for p in files:
        d = np.load(p, allow_pickle=True)
        X, y = d["X"], d["y"]
        print(f"\n== {p.name}: {len(y)} sampel, bentuk {X.shape}")
        if np.isnan(X).any():
            print("  PERINGATAN: ada nilai NaN (data rusak)")
        jumlah = collections.Counter(y)
        if X.ndim == 2:   # huruf: cek dua tangan
            dua = np.any(X[:, :63], 1) & np.any(X[:, 63:], 1)
            for h in sorted(jumlah):
                k = y == h
                catatan = ""
                if h in huruf_dua and dua[k].mean() < 0.7:
                    catatan = "  <- huruf 2 tangan, tapi sering hanya 1 tangan terdeteksi"
                if k.sum() < 15:
                    catatan += "  <- sampel sedikit, tambah rekaman"
                print(f"  {h}: {k.sum():4d} sampel | 2 tangan {dua[k].mean():4.0%}{catatan}")
            kurang = sorted(set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - set(jumlah))
            if kurang:
                print("  Huruf belum direkam:", "".join(kurang))
        else:             # kata
            for k, n in sorted(jumlah.items()):
                print(f"  {k}: {n} rekaman")


if __name__ == "__main__":
    main()
