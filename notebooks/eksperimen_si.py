"""Eksperimen: meningkatkan akurasi signer-independent (orang baru) model kata."""
import os, sys, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf
from src.features import cermin_kata

d = np.load("data/landmarks/kata/wl_bisindo.npz", allow_pickle=True)
X, ys, pr = d["X"], d["y"], d["peraga"]
L = sorted(set(ys)); y = np.array([L.index(s) for s in ys])
tr = np.isin(pr, ["wl_s0", "wl_s1", "wl_s2"]); va = pr == "wl_s3"; te = pr == "wl_s4"
rng = np.random.default_rng(42)


def aug_kuat(seq):
    """Augmentasi acak 1 sekuens: time-warp, skala, rotasi, geser, noise, cermin."""
    T = len(seq)
    # time warp: potong awal/akhir acak lalu resample ke 30
    a = rng.uniform(0, 0.15) * T; b = T - 1 - rng.uniform(0, 0.15) * T
    t = np.linspace(a, b, T); t = np.clip(t + rng.normal(0, 0.3, T), 0, T - 1)
    t.sort(); s = seq[np.round(t).astype(int)].copy()
    s = s.reshape(T, -1, 3)
    ang = rng.uniform(-0.25, 0.25); c, si = np.cos(ang), np.sin(ang)
    R = np.array([[c, -si], [si, c]], np.float32)
    s[..., :2] = s[..., :2] @ R.T
    s *= rng.uniform(0.85, 1.15)
    s[:, 42:, :2] += rng.normal(0, 0.05, 2)       # geser posisi pose sedikit
    s = s.reshape(T, -1) + rng.normal(0, 0.02, (T, s.shape[1] * 3))
    s = s.astype(np.float32)
    nol = ~np.any(seq[:, :126] != 0, axis=1)     # pertahankan frame tanpa tangan = nol
    s[nol, :126] = 0
    if rng.random() < 0.5:
        s = cermin_kata(s)
    return s


class Gen(tf.keras.utils.PyDataset):
    def __init__(self, X, y, bs=32, **kw):
        super().__init__(**kw); self.X, self.y, self.bs = X, y, bs
    def __len__(self): return int(np.ceil(len(self.X) / self.bs))
    def __getitem__(self, i):
        idx = rng.integers(0, len(self.X), self.bs)
        return np.stack([aug_kuat(self.X[j]) for j in idx]), self.y[idx]


def model(jenis, n):
    Lr = tf.keras.layers
    inp = Lr.Input((30, 201)); x = Lr.GaussianNoise(0.02)(inp)
    if jenis == "lstm":
        x = Lr.LSTM(64, return_sequences=True, dropout=0.2)(x); x = Lr.LSTM(64, dropout=0.2)(x)
    else:
        x = Lr.Conv1D(64, 5, padding="same", activation="relu")(x); x = Lr.BatchNormalization()(x)
        x = Lr.Conv1D(128, 5, padding="same", activation="relu")(x); x = Lr.BatchNormalization()(x)
        x = Lr.GlobalAveragePooling1D()(x)
    x = Lr.Dropout(0.5)(x); x = Lr.Dense(64, activation="relu")(x); x = Lr.Dropout(0.3)(x)
    m = tf.keras.Model(inp, Lr.Dense(n, activation="softmax")(x))
    m.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


for jenis in sys.argv[1:] or ["cnn", "lstm"]:
    tf.keras.utils.set_random_seed(42)
    m = model(jenis, len(L))
    t0 = time.time()
    m.fit(Gen(X[tr], y[tr]), validation_data=(X[va], y[va]), epochs=200, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping("val_accuracy", patience=30,
                                                      restore_best_weights=True, mode="max")])
    acc = (m.predict(X[te], verbose=0).argmax(1) == y[te]).mean()
    va_acc = (m.predict(X[va], verbose=0).argmax(1) == y[va]).mean()
    print(f"{jenis}: val(peraga3)={va_acc:.3f} test(peraga4)={acc:.3f}  {time.time()-t0:.0f}s", flush=True)
