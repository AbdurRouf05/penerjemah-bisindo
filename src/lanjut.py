"""Pembantu ekstraksi yang BISA DILANJUTKAN (resume) dan paralel.

Cara kerja:
- Hasil tiap file disimpan ke checkpoint data/landmarks/_checkpoint/<nama>.pkl
  setiap `simpan_tiap` item.
- Jika script dihentikan (laptop mati, Ctrl+C), jalankan ulang perintah yang sama:
  file yang sudah ada di checkpoint dilewati.
"""
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DIR_CKPT = ROOT / "data" / "landmarks" / "_checkpoint"


def _path_ckpt(nama):
    return DIR_CKPT / f"{nama}.pkl"


def muat_checkpoint(nama):
    p = _path_ckpt(nama)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return {}


def simpan_checkpoint(nama, hasil):
    DIR_CKPT.mkdir(parents=True, exist_ok=True)
    p = _path_ckpt(nama)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(hasil, f)
    os.replace(tmp, p)  # atomik: checkpoint tidak rusak walau mati mendadak


def hapus_checkpoint(nama):
    p = _path_ckpt(nama)
    if p.exists():
        p.unlink()


def jalankan(nama, daftar_kunci, fungsi_kerja, init_worker=None,
             workers=None, simpan_tiap=200):
    """Proses semua kunci (path file) secara paralel dengan resume.

    fungsi_kerja(kunci) -> (kunci, hasil)   hasil boleh None (gagal)
    Kembalikan dict {kunci: hasil}.
    """
    hasil = muat_checkpoint(nama)
    sisa = [k for k in daftar_kunci if k not in hasil]
    if hasil:
        print(f"[{nama}] lanjut dari checkpoint: {len(hasil)} selesai, {len(sisa)} tersisa")
    if not sisa:
        return hasil

    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    belum_simpan = 0
    t_simpan = time.monotonic()
    try:
        with Pool(workers, initializer=init_worker) as pool:
            for kunci, h in tqdm(pool.imap_unordered(fungsi_kerja, sisa, chunksize=4),
                                 total=len(sisa), desc=nama):
                hasil[kunci] = h
                belum_simpan += 1
                # simpan berkala (per jumlah item atau tiap 60 detik)
                if belum_simpan >= simpan_tiap or time.monotonic() - t_simpan > 60:
                    simpan_checkpoint(nama, hasil)
                    belum_simpan, t_simpan = 0, time.monotonic()
    finally:
        simpan_checkpoint(nama, hasil)
    return hasil
