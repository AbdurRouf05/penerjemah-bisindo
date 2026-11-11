"""Fungsi ekstraksi fitur BERSAMA. Semua script WAJIB memakai file ini.

Standar (lihat docs/PANDUAN_PROYEK.md bagian 6):
- Fitur huruf  : 126 angka  (tangan kiri 63 + tangan kanan 63)
- Fitur kata   : 201 angka per frame (tangan 126 + pose 0-24 75), 30 frame
"""
import cv2
import numpy as np
import mediapipe as mp

N_HAND = 63          # 21 titik x 3
N_HANDS = 126        # 2 tangan
POSE_IDX = list(range(25))   # pose badan atas 0..24
N_POSE = 75          # 25 titik x 3
N_FRAME = N_HANDS + N_POSE   # 201
SEQ_LEN = 30

mp_holistic = mp.solutions.holistic

# Pasangan titik pose kiri-kanan (untuk augmentasi cermin)
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def buat_holistic(gambar_diam=False):
    """Buat objek MediaPipe Holistic dengan pengaturan standar proyek."""
    return mp_holistic.Holistic(
        static_image_mode=gambar_diam,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def proses_bgr(holistic, frame_bgr):
    """Jalankan MediaPipe pada frame BGR dari OpenCV. Frame TIDAK di-flip."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    return holistic.process(rgb)


def _fitur_tangan(landmarks):
    """21 titik tangan -> 63 angka, relatif pergelangan & diskalakan."""
    if landmarks is None:
        return np.zeros(N_HAND, dtype=np.float32)
    a = np.array([[p.x, p.y, p.z] for p in landmarks.landmark], dtype=np.float32)
    a = a - a[0]
    skala = np.max(np.linalg.norm(a[:, :2], axis=1))
    if skala > 0:
        a = a / skala
    return a.flatten()


def fitur_huruf(results):
    """Hasil MediaPipe -> vektor 126 angka. Kembalikan None jika tidak ada tangan."""
    if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
        return None
    return np.concatenate([
        _fitur_tangan(results.left_hand_landmarks),
        _fitur_tangan(results.right_hand_landmarks),
    ])


def _fitur_pose(pose_landmarks):
    """Pose 0..24 -> 75 angka, relatif titik tengah bahu & dibagi lebar bahu."""
    if pose_landmarks is None:
        return np.zeros(N_POSE, dtype=np.float32)
    lm = pose_landmarks.landmark
    p = np.array([[lm[i].x, lm[i].y, lm[i].z] for i in POSE_IDX], dtype=np.float32)
    tengah = (p[11] + p[12]) / 2
    lebar = np.linalg.norm(p[11, :2] - p[12, :2])
    p = p - tengah
    if lebar > 0:
        p = p / lebar
    return p.flatten()


def fitur_frame(results):
    """Hasil MediaPipe -> vektor 201 angka untuk 1 frame (selalu ada nilainya)."""
    tangan = np.concatenate([
        _fitur_tangan(results.left_hand_landmarks),
        _fitur_tangan(results.right_hand_landmarks),
    ])
    return np.concatenate([tangan, _fitur_pose(results.pose_landmarks)])


def resample_sekuens(frames, panjang=SEQ_LEN):
    """List/array (T, 201) -> array (30, 201) dengan indeks merata."""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) == 0:
        return np.zeros((panjang, N_FRAME), dtype=np.float32)
    idx = np.linspace(0, len(frames) - 1, panjang).round().astype(int)
    return frames[idx]


def fitur_video(path_video, holistic=None):
    """File video -> array (30, 201). Kembalikan None jika video tak terbaca
    atau tidak ada tangan sama sekali di seluruh video."""
    tutup = holistic is None
    if tutup:
        holistic = buat_holistic(gambar_diam=False)
    cap = cv2.VideoCapture(str(path_video))
    frames, ada_tangan = [], False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = proses_bgr(holistic, frame)
        if res.left_hand_landmarks is not None or res.right_hand_landmarks is not None:
            ada_tangan = True
        frames.append(fitur_frame(res))
    cap.release()
    if tutup:
        holistic.close()
    if not frames or not ada_tangan:
        return None
    return resample_sekuens(frames)


def cermin_huruf(x):
    """Augmentasi cermin untuk fitur huruf (126): tukar kiri-kanan & balik x."""
    x = x.reshape(2, 21, 3).copy()
    x = x[::-1]          # tukar tangan kiri <-> kanan
    x[..., 0] *= -1      # balik sumbu x
    return x.reshape(-1)


def cermin_kata(seq):
    """Augmentasi cermin untuk sekuens kata (30, 201)."""
    seq = seq.copy()
    out = np.empty_like(seq)
    for t in range(seq.shape[0]):
        tangan = cermin_huruf(seq[t, :N_HANDS])
        pose = seq[t, N_HANDS:].reshape(25, 3).copy()
        for a, b in POSE_SWAP:
            pose[[a, b]] = pose[[b, a]]
        pose[:, 0] *= -1
        out[t] = np.concatenate([tangan, pose.reshape(-1)])
    return out
