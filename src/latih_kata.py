"""T6 - Training & evaluasi model KATA (LSTM utama, pembanding GRU & 1D-CNN).

Dua skenario (PANDUAN 6.6):
- sd = signer-dependent  : acak stratified 70/15/15
- si = signer-independent: train peraga 0,1,2 | val peraga 3 | test peraga 4

Dua versi model:
- dasar : augmentasi ringan (cermin, noise, skala), LSTM 128-64
- aug   : augmentasi KUAT acak tiap batch (time-warp, rotasi, skala, geser, noise,
          cermin) + model lebih kecil dengan dropout -> jauh lebih baik untuk orang baru
          (skenario si: LSTM 0.40 -> ±0.72, lihat reports/kata_perbandingan.csv)

Model untuk aplikasi = models/kata_lstm.keras (salinan LSTM versi aug skenario sd,
karena sudah melihat ke-5 peraga -> paling umum untuk orang baru).

Jalankan (dari folder root proyek):
    .venv\\Scripts\\python.exe -m src.latih_kata
Bisa dihentikan & dijalankan ulang: training lanjut dari epoch terakhir.
"""
import glob
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

from src.features import cermin_kata
from src.latih_umum import (MODELS, REPORTS, ROOT, SEED, evaluasi, latih_keras,
                            simpan_label, tulis_tabel)

DIR_KATA = ROOT / "data" / "landmarks" / "kata"
DIR_REKAM = ROOT / "data" / "landmarks" / "rekaman_sendiri"


def muat(pola):
    X, y, peraga = [], [], []
    for p in sorted(glob.glob(str(pola))):
        d = np.load(p, allow_pickle=True)
        X.append(d["X"])
        y.append(d["y"])
        peraga.append(d["peraga"])
        print(f"  {p.split('landmarks')[-1]}: {len(d['y'])} sampel")
    if not X:
        return None, None, None
    return np.concatenate(X).astype(np.float32), np.concatenate(y), np.concatenate(peraga)


def augmentasi(X, y, rng):
    """Cermin + noise kecil + skala acak. HANYA untuk data train."""
    Xc = np.stack([cermin_kata(s) for s in X])
    Xn = X + rng.normal(0, 0.01, X.shape).astype(np.float32)
    Xs = X * rng.uniform(0.9, 1.1, (len(X), 1, 1)).astype(np.float32)
    return np.concatenate([X, Xc, Xn, Xs]), np.concatenate([y, y, y, y])


def aug_kuat(seq, rng):
    """Augmentasi acak 1 sekuens (30, 201): time-warp, rotasi, skala, geser, noise, cermin."""
    T = len(seq)
    # time-warp: potong awal/akhir acak + kecepatan tidak rata, lalu kembali ke 30 frame
    a = rng.uniform(0, 0.15) * T
    b = T - 1 - rng.uniform(0, 0.15) * T
    t = np.sort(np.clip(np.linspace(a, b, T) + rng.normal(0, 0.3, T), 0, T - 1))
    s = seq[np.round(t).astype(int)].reshape(T, -1, 3).copy()   # 67 titik (42 tangan + 25 pose)
    sudut = rng.uniform(-0.25, 0.25)                             # rotasi ±14 derajat
    c, si = np.cos(sudut), np.sin(sudut)
    s[..., :2] = s[..., :2] @ np.array([[c, si], [-si, c]], np.float32)
    s *= rng.uniform(0.85, 1.15)                                 # skala tubuh
    s[:, 42:, :2] += rng.normal(0, 0.05, 2)                      # geser posisi pose
    s = s.reshape(T, -1) + rng.normal(0, 0.02, (T, seq.shape[1]))
    s = s.astype(np.float32)
    s[~np.any(seq[np.round(t).astype(int), :126] != 0, axis=1), :126] = 0  # tangan hilang tetap 0
    return cermin_kata(s) if rng.random() < 0.5 else s


def buat_generator(X, y, rng, batch=32):
    import tensorflow as tf

    class Gen(tf.keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(X) / batch))

        def __getitem__(self, i):
            idx = rng.integers(0, len(X), batch)
            return np.stack([aug_kuat(X[j], rng) for j in idx]), y[idx]

    return Gen()


def buat_model_aug(jenis, n_kelas):
    """Model lebih kecil + regularisasi kuat (untuk versi aug)."""
    import tensorflow as tf
    L = tf.keras.layers
    inp = L.Input((30, 201))
    x = L.GaussianNoise(0.02)(inp)
    if jenis in ("lstm", "gru"):
        RNN = L.LSTM if jenis == "lstm" else L.GRU
        x = RNN(64, return_sequences=True, dropout=0.2)(x)
        x = RNN(64, dropout=0.2)(x)
    else:  # cnn1d
        x = L.Conv1D(64, 5, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.Conv1D(128, 5, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.5)(x)
    x = L.Dense(64, activation="relu")(x)
    x = L.Dropout(0.3)(x)
    m = tf.keras.Model(inp, L.Dense(n_kelas, activation="softmax")(x))
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def buat_model(jenis, n_kelas):
    import tensorflow as tf
    L = tf.keras.layers
    if jenis in ("lstm", "gru"):
        RNN = L.LSTM if jenis == "lstm" else L.GRU
        lapisan = [RNN(128, return_sequences=True), L.Dropout(0.3),
                   RNN(64), L.Dropout(0.3),
                   L.Dense(64, activation="relu")]
    else:  # cnn1d
        lapisan = [L.Conv1D(64, 3, activation="relu"), L.Conv1D(128, 3, activation="relu"),
                   L.GlobalAveragePooling1D(), L.Dropout(0.3),
                   L.Dense(64, activation="relu")]
    m = tf.keras.Sequential([L.Input((30, 201)), *lapisan,
                             L.Dense(n_kelas, activation="softmax")])
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def main():
    print("Memuat data kata:")
    X, y_str, peraga = muat(DIR_KATA / "wl_bisindo.npz")
    # kata tambahan dari rekaman sendiri (kata di luar WL-BISINDO -> ikut training)
    Xr, yr_str, pr = muat(DIR_REKAM / "kata_*.npz")
    kata_wl = set(y_str)
    Xob = yob_str = None
    if Xr is not None:
        tambahan = ~np.isin(yr_str, list(kata_wl))
        X = np.concatenate([X, Xr[tambahan]])
        y_str = np.concatenate([y_str, yr_str[tambahan]])
        peraga = np.concatenate([peraga, pr[tambahan]])
        Xob, yob_str = Xr[~tambahan], yr_str[~tambahan]   # 32 kata WL -> hanya test

    label = sorted(set(y_str))
    simpan_label("label_kata.json", label)
    y = np.array([label.index(s) for s in y_str])
    print(f"Total {len(y)} sampel, {len(label)} kelas")
    rng = np.random.default_rng(SEED)

    # --- skenario sd: acak stratified ---
    idx = np.arange(len(y))
    i_tr, i_tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    i_va, i_te = train_test_split(i_tmp, test_size=0.50, stratify=y[i_tmp], random_state=SEED)
    split = {"sd": (i_tr, i_va, i_te)}

    # --- skenario si: berdasarkan peraga (hanya data WL-BISINDO) ---
    wl = np.char.startswith(peraga.astype(str), "wl_")
    tr = np.isin(peraga, ["wl_s0", "wl_s1", "wl_s2"]) | ~wl  # rekaman tambahan ikut train
    split["si"] = (idx[tr], idx[peraga == "wl_s3"], idx[peraga == "wl_s4"])

    baris = []
    for sk, (a, b, c) in split.items():
        print(f"\n=== Skenario {sk}: train {len(a)} | val {len(b)} | test {len(c)}")
        Xtr, ytr = augmentasi(X[a], y[a], rng)
        for versi in ["dasar", "aug"]:
            for jenis in ["lstm", "gru", "cnn1d"]:
                nama = f"kata_{jenis}_{sk}" + ("_aug" if versi == "aug" else "")
                if versi == "dasar":
                    m = latih_keras(nama, lambda: buat_model(jenis, len(label)), Xtr, ytr,
                                    X[b], y[b], epochs=150, batch=32, patience=15)
                else:
                    m = latih_keras(nama, lambda: buat_model_aug(jenis, len(label)),
                                    buat_generator(X[a], y[a], rng), None, X[b], y[b],
                                    epochs=200, batch=32, patience=30, pantau="val_accuracy")
                hasil = evaluasi(nama, m, X[c], y[c], label)
                baris.append({"model": jenis.upper(), "versi": versi, "skenario": sk, **hasil})
                if Xob is not None and len(Xob):
                    yob = np.array([label.index(s) for s in yob_str])
                    h = evaluasi(nama + "_orang_baru", m, Xob, yob, label, gambar_cm=False)
                    baris.append({"model": jenis.upper(), "versi": versi,
                                  "skenario": sk + "_orang_baru", **h})
    tulis_tabel(REPORTS / "kata_perbandingan.csv", baris)

    shutil.copy(MODELS / "kata_lstm_sd_aug.keras", MODELS / "kata_lstm.keras")
    print("Model aplikasi: models/kata_lstm.keras")


if __name__ == "__main__":
    main()
