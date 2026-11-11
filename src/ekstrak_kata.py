"""T3-A - Ekstraksi landmark KATA dari video WL-BISINDO -> (N, 30, 201).

Jalankan (dari folder root proyek):
    .venv\\Scripts\\python.exe -m src.ekstrak_kata
Bisa dihentikan kapan saja dan dijalankan ulang -> otomatis lanjut
(video yang sudah diproses tidak diulang).
"""
import argparse
import csv
import re
from collections import Counter

import numpy as np

from src import lanjut

ROOT = lanjut.ROOT
RAW = ROOT / "data" / "raw" / "kata" / "wl_bisindo"
OUT = ROOT / "data" / "landmarks" / "kata"
REPORTS = ROOT / "reports"

# Peta label WL-BISINDO (PANDUAN bagian 5.2a) + aturan nama 6.5
LABEL_WL = [
    "air", "belajar", "cari", "hari", "ingat", "lagi", "maaf", "makan",
    "motor", "saya", "terima_kasih", "tuli", "apa", "siapa", "kapan", "di_mana",
    "mengapa", "bagaimana", "merah", "kuning", "hijau", "hitam", "dengar", "berangkat",
    "datang", "teman", "keluarga", "rumah", "pagi", "siang", "sore", "malam",
]
POLA = re.compile(r"signer(\d+)_label(\d+)_sample(\d+)\.mp4$", re.IGNORECASE)


def _kerja(path_str):
    from src.features import fitur_video
    try:
        # holistic baru per video agar tracking tidak terbawa dari video lain
        return path_str, fitur_video(path_str)
    except Exception:
        return path_str, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ulang", action="store_true", help="ekstrak ulang walau sudah ada")
    args = ap.parse_args()

    if (OUT / "wl_bisindo.npz").exists() and not args.ulang:
        print("wl_bisindo.npz sudah ada, dilewati (pakai --ulang untuk mengulang).")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    items = []
    for p in sorted(RAW.glob("*.mp4")):
        m = POLA.search(p.name)
        if m:
            s, l, n = map(int, m.groups())
            items.append((p, s, l, n))
    print(f"{len(items)} video | peraga: {sorted({s for _, s, _, _ in items})} "
          f"| label: {len({l for _, _, l, _ in items})}")
    per_sl = Counter((s, l) for _, s, l, _ in items)
    print("video per (peraga, label): min", min(per_sl.values()), "max", max(per_sl.values()))

    hasil = lanjut.jalankan("kata_wl_bisindo", [str(p) for p, *_ in items], _kerja,
                            workers=args.workers, simpan_tiap=40)

    X, y, peraga, sumber, gagal = [], [], [], [], []
    for p, s, l, n in items:
        f = hasil.get(str(p))
        if f is None:
            gagal.append([p.name, s, LABEL_WL[l]])
            continue
        X.append(f)
        y.append(LABEL_WL[l])
        peraga.append(f"wl_s{s}")
        sumber.append(str(p.relative_to(ROOT)))
    X = np.asarray(X, dtype=np.float32).reshape(-1, 30, 201)
    np.savez_compressed(OUT / "wl_bisindo.npz", X=X, y=np.array(y),
                        peraga=np.array(peraga), sumber=np.array(sumber))
    lanjut.hapus_checkpoint("kata_wl_bisindo")

    with open(REPORTS / "ekstraksi_kata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_gagal", "peraga", "label"])
        w.writerows(gagal)
    print(f"Tersimpan wl_bisindo.npz: X={X.shape}, gagal={len(gagal)}, NaN={np.isnan(X).any()}")


if __name__ == "__main__":
    main()
