"""T5 - Training & evaluasi model HURUF (MLP utama, pembanding RF & SVM).

Jalankan (dari folder root proyek):
    .venv\\Scripts\\python.exe -m src.latih_huruf
Bisa dihentikan & dijalankan ulang: MLP lanjut dari epoch terakhir,
model yang sudah selesai (.keras / .joblib) tidak dilatih ulang.

Rekaman sendiri (src.rekam_data --mode huruf) secara bawaan HANYA untuk uji "orang baru".
Agar ikut dilatih (mis. memperbaiki huruf sulit seperti S, T):
    .venv\\Scripts\\python.exe -m src.latih_huruf --pakai-rekaman
(model huruf lama otomatis dilatih ulang)
"""
import argparse
import glob
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.features import cermin_huruf
from src.latih_umum import (MODELS, REPORTS, ROOT, SEED, evaluasi, latih_keras,
                            simpan_label, tulis_tabel)


# Dataset yang DIBUANG setelah analisis (lihat laporan):
# - alphabet_image_data: akurasi silang hanya ±22% -> variasi isyarat/label tidak cocok
# - bisindo_70_30, dataset_of_bisindo: <20 sampel terdeteksi (gambar crop tangan kecil)
DATASET_DIBUANG = {"alphabet_image_data", "bisindo_70_30", "dataset_of_bisindo"}
# Rekaman sendiri digandakan agar cukup berpengaruh (jumlahnya jauh lebih sedikit)
BOBOT_REKAMAN = 3
# Huruf dianggap "dua tangan" jika >= 62% sampelnya terdeteksi dua tangan.
# (Y hanya 59% -> ada variasi 1 tangan & 2 tangan, jadi TIDAK diwajibkan dua tangan)
AMBANG_DUA_TANGAN = 0.62


def muat_huruf(pola, buang=()):
    X, y = [], []
    for p in sorted(glob.glob(str(pola))):
        nama = os.path.basename(p)[:-4]
        if nama in buang:
            print(f"  {nama}: DIBUANG")
            continue
        d = np.load(p, allow_pickle=True)
        X.append(d["X"])
        y.append(d["y"])
        print(f"  {nama}: {len(d['y'])} sampel")
    if not X:
        return None, None
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def dua_tangan(X):
    return np.any(X[:, :63], 1) & np.any(X[:, 63:], 1)


def bersihkan(X, y):
    """Buang sampel huruf dua-tangan yang hanya terdeteksi satu tangan (tidak lengkap)."""
    d = dua_tangan(X)
    huruf_dua = sorted(h for h in set(y) if d[y == h].mean() >= AMBANG_DUA_TANGAN)
    lengkap = ~(np.isin(y, huruf_dua) & ~d)
    print(f"Huruf dua tangan: {''.join(huruf_dua)} | dibuang {np.sum(~lengkap)} sampel tak lengkap")
    return X[lengkap], y[lengkap], huruf_dua


def augmentasi(X, y, rng):
    """Cermin (pengguna kidal) + noise kecil. HANYA untuk data train."""
    Xc = np.stack([cermin_huruf(x) for x in X])
    Xn = X + rng.normal(0, 0.01, X.shape).astype(np.float32)
    return np.concatenate([X, Xc, Xn]), np.concatenate([y, y, y])


def buat_mlp(n_kelas):
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Input((126,)),
        L.Dense(256, activation="relu"), L.Dropout(0.3),
        L.Dense(128, activation="relu"), L.Dropout(0.3),
        L.Dense(n_kelas, activation="softmax"),
    ])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def latih_sklearn(nama, buat, X, y):
    path = MODELS / f"{nama}.joblib"
    if path.exists():
        print(f"[{nama}] sudah terlatih, dimuat")
        return joblib.load(path)
    print(f"[{nama}] training...")
    m = buat().fit(X, y)
    joblib.dump(m, path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pakai-rekaman", action="store_true",
                    help="ikutkan rekaman sendiri ke data latih (model huruf dilatih ulang)")
    pakai_rekaman = ap.parse_args().pakai_rekaman

    print("Memuat data huruf:")
    X, y_str = muat_huruf(ROOT / "data" / "landmarks" / "huruf" / "*.npz", DATASET_DIBUANG)
    Xob, yob_str = muat_huruf(ROOT / "data" / "landmarks" / "rekaman_sendiri" / "huruf_*.npz")
    if pakai_rekaman:
        if Xob is None:
            raise SystemExit("Belum ada rekaman huruf di data/landmarks/rekaman_sendiri/")
        # sisihkan 1 dari 5 sampel rekaman untuk uji, sisanya ikut data latih
        uji = np.arange(len(yob_str)) % 5 == 0
        X_rek, y_rek = np.repeat(Xob[~uji], BOBOT_REKAMAN, 0), np.repeat(yob_str[~uji], BOBOT_REKAMAN)
        Xob, yob_str = Xob[uji], yob_str[uji]
        print(f"Rekaman sendiri: {np.sum(~uji)} sampel latih (x{BOBOT_REKAMAN}), {np.sum(uji)} uji")
        for f in ["huruf_mlp.keras", "huruf_rf.joblib", "huruf_svm.joblib"]:
            (MODELS / f).unlink(missing_ok=True)       # paksa latih ulang
    X, y_str, huruf_dua = bersihkan(X, y_str)
    label = sorted(set(y_str))
    simpan_label("label_huruf.json", label)
    simpan_label("huruf_dua_tangan.json", huruf_dua)  # dipakai app.py
    y = np.array([label.index(s) for s in y_str])
    print(f"Total {len(y)} sampel, {len(label)} kelas")

    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.50, stratify=ytmp,
                                          random_state=SEED)
    rng = np.random.default_rng(SEED)
    if pakai_rekaman:
        # rekaman huruf dua-tangan yang hanya terdeteksi 1 tangan -> buang
        ada = np.isin(y_rek, label) & ~(np.isin(y_rek, huruf_dua) & ~dua_tangan(X_rek))
        Xtr = np.concatenate([Xtr, X_rek[ada]])
        ytr = np.concatenate([ytr, [label.index(s) for s in y_rek[ada]]])
    Xtr_aug, ytr_aug = augmentasi(Xtr, ytr, rng)
    print(f"train {len(ytr)} (+augmentasi = {len(ytr_aug)}) | val {len(yva)} | test {len(yte)}")

    # data orang baru (rekaman sendiri), jika ada
    if Xob is not None:
        ada = np.isin(yob_str, label)
        Xob, yob = Xob[ada], np.array([label.index(s) for s in yob_str[ada]])

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    model = {
        "huruf_mlp": latih_keras("huruf_mlp", lambda: buat_mlp(len(label)), Xtr_aug, ytr_aug,
                                 Xva, yva, epochs=100, batch=64, patience=10),
        "huruf_rf": latih_sklearn("huruf_rf", lambda: RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1), Xtr_aug, ytr_aug),
        # SVM lambat untuk data besar -> dilatih tanpa augmentasi
        "huruf_svm": latih_sklearn("huruf_svm", lambda: SVC(
            kernel="rbf", C=10, probability=True, random_state=SEED), Xtr, ytr),
    }

    baris = []
    for nama, m in model.items():
        hasil = evaluasi(nama, m, Xte, yte, label)
        if Xob is not None and len(yob):
            hasil["akurasi_orang_baru"] = evaluasi(nama + "_orang_baru", m, Xob, yob, label,
                                                   gambar_cm=False)["accuracy"]
        else:
            hasil["akurasi_orang_baru"] = ""
        baris.append({"model": nama.split("_")[1].upper(), **hasil})
    tulis_tabel(REPORTS / "huruf_perbandingan.csv", baris)


if __name__ == "__main__":
    main()
