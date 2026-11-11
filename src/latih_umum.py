"""Fungsi bersama untuk training & evaluasi (dipakai latih_huruf.py dan latih_kata.py).

Training Keras BISA DILANJUTKAN: bobot & nomor epoch disimpan setiap epoch di
models/_backup/<nama>/. Jika laptop mati, jalankan ulang script yang sama.
"""
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
LABELS = ROOT / "data" / "labels"
SEED = 42


def simpan_label(nama_file, label):
    LABELS.mkdir(parents=True, exist_ok=True)
    with open(LABELS / nama_file, "w", encoding="utf-8") as f:
        json.dump(list(label), f, ensure_ascii=False, indent=1)


def latih_keras(nama, buat_model, Xtr, ytr, Xva, yva, epochs, batch, patience,
                pantau="val_loss"):
    """Latih model Keras dengan resume. Kembalikan model terlatih.
    Jika models/<nama>.keras sudah ada, langsung dimuat (tidak dilatih ulang).
    Xtr boleh berupa generator (PyDataset) -> isi ytr dengan None."""
    import tensorflow as tf
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    path_final = MODELS / f"{nama}.keras"
    if path_final.exists():
        print(f"[{nama}] sudah terlatih, dimuat dari {path_final.name}")
        return tf.keras.models.load_model(path_final)

    dir_backup = MODELS / "_backup" / nama
    path_log = REPORTS / f"{nama}_log.csv"
    if dir_backup.exists():
        print(f"[{nama}] melanjutkan training dari backup terakhir...")
    elif path_log.exists():
        path_log.unlink()  # mulai dari nol -> log lama dibuang

    tf.keras.utils.set_random_seed(SEED)
    model = buat_model()
    cb = [
        tf.keras.callbacks.BackupAndRestore(str(dir_backup)),
        tf.keras.callbacks.EarlyStopping(monitor=pantau, patience=patience,
                                         restore_best_weights=True,
                                         mode="max" if "acc" in pantau else "min"),
        tf.keras.callbacks.CSVLogger(str(path_log), append=True),
    ]
    if ytr is None:   # generator: augmentasi dibuat acak tiap batch
        model.fit(Xtr, validation_data=(Xva, yva), epochs=epochs, callbacks=cb, verbose=2)
    else:
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                  batch_size=batch, callbacks=cb, verbose=2)
    model.save(path_final)
    shutil.rmtree(dir_backup, ignore_errors=True)
    gambar_kurva(nama)
    return model


def gambar_kurva(nama):
    path_log = REPORTS / f"{nama}_log.csv"
    if not path_log.exists():
        return
    with open(path_log, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    ep = range(1, len(rows) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, a in [("loss", ax[0]), ("accuracy", ax[1])]:
        a.plot(ep, [float(r[k]) for r in rows], label="train")
        a.plot(ep, [float(r["val_" + k]) for r in rows], label="val")
        a.set_title(k)
        a.set_xlabel("epoch")
        a.legend()
    fig.suptitle(nama)
    fig.tight_layout()
    fig.savefig(REPORTS / f"{nama}_kurva.png", dpi=120)
    plt.close(fig)


def prediksi_proba(model, X):
    """Prediksi probabilitas untuk model Keras maupun scikit-learn."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.predict(X, verbose=0)


def ukur_ms(model, X, n=100):
    """Rata-rata waktu prediksi 1 sampel (ms)."""
    idx = np.arange(min(n, len(X)))
    keras = not hasattr(model, "predict_proba")
    t0 = time.perf_counter()
    for i in idx:
        x = X[i:i + 1]
        if keras:
            model(x, training=False)
        else:
            model.predict_proba(x)
    return 1000 * (time.perf_counter() - t0) / len(idx)


def evaluasi(nama, model, X, y, label, gambar_cm=True):
    """Hitung metrik pada data uji, simpan laporan & confusion matrix."""
    pred = prediksi_proba(model, X).argmax(1)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    idx = list(range(len(label)))
    teks = classification_report(y, pred, labels=idx, target_names=list(label),
                                 zero_division=0, digits=4)
    (REPORTS / f"{nama}_report.txt").write_text(teks, encoding="utf-8")
    if gambar_cm:
        cm = confusion_matrix(y, pred, labels=idx)
        n = len(label)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.35), max(7, n * 0.32)))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(idx, label, rotation=90, fontsize=7)
        ax.set_yticks(idx, label, fontsize=7)
        ax.set_xlabel("prediksi")
        ax.set_ylabel("sebenarnya")
        for i in idx:
            for j in idx:
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{nama}  (akurasi {acc:.2%})")
        fig.tight_layout()
        fig.savefig(REPORTS / f"{nama}_cm.png", dpi=120)
        plt.close(fig)
        # pasangan yang paling sering tertukar (untuk analisis laporan)
        salah = [(cm[i, j], label[i], label[j]) for i in idx for j in idx if i != j and cm[i, j]]
        salah.sort(reverse=True)
        with open(REPORTS / f"{nama}_tertukar.txt", "w", encoding="utf-8") as f:
            for c, a, b in salah[:15]:
                f.write(f"{a} -> {b}: {c}\n")
    ms = ukur_ms(model, X)
    print(f"[{nama}] acc={acc:.4f} prec={p:.4f} rec={r:.4f} f1={f1:.4f} {ms:.2f} ms/sampel")
    return dict(accuracy=round(acc, 4), precision=round(p, 4), recall=round(r, 4),
                f1=round(f1, 4), ms_per_sampel=round(ms, 3))


def tulis_tabel(path, baris):
    if not baris:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris[0].keys()))
        w.writeheader()
        w.writerows(baris)
    print(f"Tabel tersimpan: {path}")
