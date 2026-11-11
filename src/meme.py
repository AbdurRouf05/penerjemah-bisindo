"""Mode MEME: ekspresi wajah & gesture tangan -> gambar meme.

Diadaptasi dari repo gesture-meme-vision (MediaPipe Face Mesh + Hands) agar memakai
hasil MediaPipe Holistic yang sudah ada di aplikasi (tanpa model tambahan).
Berbasis aturan (rule-based), bukan machine learning.
"""
import math
from collections import deque
from pathlib import Path

import cv2

DIR_MEME = Path(__file__).resolve().parents[1] / "assets" / "meme"
FILE_MEME = {
    "default": "normal.jpg",            # posisi santai
    "eyes_closed": "eyes_closed.jpg",   # kedua mata terpejam
    "hand_mouth": "hand_mouth.jpg",     # telapak menutup mulut (kaget)
    "finger_mouth": "finger_mouth.jpg", # telunjuk di depan bibir (shh)
    "cerdas": "cerdas.jpg",             # telunjuk di pelipis/mata (mikir)
    "finger_up": "finger_up.jpg",       # telunjuk ke atas (ide)
}
NAMA_TAMPIL = {
    "default": "santai", "eyes_closed": "mata terpejam", "hand_mouth": "kaget",
    "finger_mouth": "sstt...", "cerdas": "cerdas", "finger_up": "ide!",
}
AMBANG_MATA = 0.15    # rasio buka mata (tinggi/lebar) di bawah ini = terpejam
STABIL_FRAME = 3      # gesture harus muncul 3 frame berturut agar meme tidak berkedip

_gambar = {}


def _muat(nama):
    if nama not in _gambar:
        _gambar[nama] = cv2.imread(str(DIR_MEME / FILE_MEME[nama]))
    return _gambar[nama]


def _jarak(a, b, w, h):
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def mata_terpejam(f, w, h):
    tinggi = (_jarak(f[159], f[145], w, h) + _jarak(f[386], f[374], w, h)) / 2
    lebar = (_jarak(f[33], f[133], w, h) + _jarak(f[362], f[263], w, h)) / 2
    return lebar > 0 and tinggi / lebar < AMBANG_MATA


def tangan_tutup_mulut(t, f, w, h):
    mx, my = (f[13].x + f[14].x) / 2, (f[13].y + f[14].y) / 2
    px = sum(t[i].x for i in (0, 5, 9, 13, 17)) / 5          # tengah telapak
    py = sum(t[i].y for i in (0, 5, 9, 13, 17)) / 5
    d = math.hypot((px - mx) * w, (py - my) * h)
    return d < _jarak(f[234], f[454], w, h) * 0.30 and f[1].y < py < f[152].y


def telunjuk_di_mulut(t, f, w, h):
    mx, my = (f[13].x + f[14].x) / 2, (f[13].y + f[14].y) / 2
    d = math.hypot((t[8].x - mx) * w, (t[8].y - my) * h)
    dekat = d < _jarak(f[234], f[454], w, h) * 0.28
    return dekat and t[8].y < t[6].y and t[12].y > t[10].y


def telunjuk_di_mata(t, f):
    xs = [f[i].x for i in (33, 133, 362, 263)]
    y1 = min(f[159].y, f[386].y) - 0.08
    y2 = max(f[145].y, f[374].y) + 0.05
    return min(xs) - 0.08 <= t[8].x <= max(xs) + 0.08 and y1 <= t[8].y <= y2


def telunjuk_ke_atas(t):
    if t[8].y > t[6].y:
        return False
    return all(t[ujung].y > t[ruas].y for ujung, ruas in ((12, 10), (16, 14), (20, 18)))


def deteksi(res, w, h):
    """Kembalikan nama gesture dari hasil MediaPipe Holistic (urutan = prioritas)."""
    f = res.face_landmarks.landmark if res.face_landmarks else None
    tangan = [x.landmark for x in (res.left_hand_landmarks, res.right_hand_landmarks) if x]
    if f is not None and mata_terpejam(f, w, h):
        return "eyes_closed"
    if f is not None:
        if any(tangan_tutup_mulut(t, f, w, h) for t in tangan):
            return "hand_mouth"
        if any(telunjuk_di_mulut(t, f, w, h) for t in tangan):
            return "finger_mouth"
        if any(telunjuk_di_mata(t, f) for t in tangan):
            return "cerdas"
    if any(telunjuk_ke_atas(t) for t in tangan):
        return "finger_up"
    return "default"


class Penstabil:
    """Ganti meme hanya jika gesture baru muncul beberapa frame berturut-turut."""

    def __init__(self):
        self.aktif = "default"
        self._riwayat = deque(maxlen=STABIL_FRAME)

    def perbarui(self, gesture):
        self._riwayat.append(gesture)
        if len(set(self._riwayat)) == 1 and len(self._riwayat) == STABIL_FRAME:
            self.aktif = gesture
        return self.aktif


def tempel(img, gesture, y_atas):
    """Tempel gambar meme di pojok kanan atas tampilan (lebar ±30% layar)."""
    m = _muat(gesture)
    if m is None:
        return img
    h, w = img.shape[:2]
    lebar = int(w * 0.30)
    tinggi = min(int(m.shape[0] * lebar / m.shape[1]), h - y_atas - int(h * 0.2))
    if tinggi <= 0:
        return img
    m = cv2.resize(m, (lebar, tinggi), interpolation=cv2.INTER_AREA)
    x1, y1 = w - lebar - 15, y_atas + 15
    img[y1:y1 + tinggi, x1:x1 + lebar] = m
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x1 + lebar + 1, y1 + tinggi + 1), (255, 255, 255), 2,
                  cv2.LINE_AA)
    return img
