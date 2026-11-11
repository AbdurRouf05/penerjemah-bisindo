"""T8 - Aplikasi real-time: isyarat BISINDO -> teks -> suara.

    .venv\\Scripts\\python.exe -m src.app

Tombol:
  h = mode HURUF      k = mode KATA
  SPASI = spasi       BACKSPACE = hapus terakhir      c = bersihkan
  ENTER = ucapkan kalimat                              q = keluar
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")    # sembunyikan info TensorFlow
os.environ.setdefault("GLOG_minloglevel", "2")        # sembunyikan info MediaPipe

import json
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from src.features import (N_HANDS, SEQ_LEN, buat_holistic, fitur_frame, fitur_huruf,
                          proses_bgr)
from src.latih_umum import LABELS, MODELS
from src.penyusun_kalimat import PenyusunKalimat
from src.suara import ucapkan

gambar = mp.solutions.drawing_utils
H = mp.solutions.holistic
JUDUL = "Penerjemah BISINDO"
TIAP_N_FRAME_KATA = 5
MIN_FRAME_TANGAN = 10
RATA_FRAME_HURUF = 5    # prediksi huruf = rata-rata 5 frame terakhir


def muat(nama_model, nama_label):
    path = MODELS / nama_model
    if not path.exists():
        print(f"PERINGATAN: {path.name} belum ada (latih dulu). Mode ini dinonaktifkan.")
        return None, None
    with open(LABELS / nama_label, encoding="utf-8") as f:
        label = json.load(f)
    model = tf.keras.models.load_model(path)
    # tf.function: prediksi ±50x lebih cepat dibanding pemanggilan biasa (203 ms -> 4 ms)
    fungsi = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
    fungsi(np.zeros((1, *model.input_shape[1:]), np.float32))  # pemanasan
    return fungsi, label


def prediksi(model, x):
    p = model(np.asarray(x, np.float32)).numpy()[0]
    i = int(p.argmax())
    return i, float(p[i])


def proba(model, x):
    return model(np.asarray(x, np.float32)).numpy()[0]


def teks_top2(label, p):
    """Contoh: 'S (72%)   lalu: D 20%' -> membantu melihat huruf yang tertukar."""
    i, j = np.argsort(p)[::-1][:2]
    return f"{label[i]} ({p[i]:.0%})   lalu: {label[j]} {p[j]:.0%}"


# ---- gaya gambar (warna BGR) ----
FONT = cv2.FONT_HERSHEY_DUPLEX
GAYA_TITIK = gambar.DrawingSpec(color=(255, 255, 255), thickness=-1, circle_radius=3)
GAYA_GARIS_KIRI = gambar.DrawingSpec(color=(80, 200, 120), thickness=2)    # hijau
GAYA_GARIS_KANAN = gambar.DrawingSpec(color=(255, 160, 60), thickness=2)   # biru muda
GAYA_LENGAN = gambar.DrawingSpec(color=(200, 200, 200), thickness=2)
# hanya bahu-siku-pergelangan (tanpa titik wajah / kaki)
LENGAN = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def gambar_kerangka(img, res):
    """Gambar tangan + lengan saja, dengan garis halus."""
    h, w = img.shape[:2]
    if res.pose_landmarks is not None:
        lm = res.pose_landmarks.landmark
        for a, b in LENGAN:
            if lm[a].visibility > 0.5 and lm[b].visibility > 0.5:
                pa = (int(lm[a].x * w), int(lm[a].y * h))
                pb = (int(lm[b].x * w), int(lm[b].y * h))
                cv2.line(img, pa, pb, GAYA_LENGAN.color, 3, cv2.LINE_AA)
    for tangan, gaya in ((res.left_hand_landmarks, GAYA_GARIS_KIRI),
                         (res.right_hand_landmarks, GAYA_GARIS_KANAN)):
        if tangan is not None:
            gambar.draw_landmarks(img, tangan, H.HAND_CONNECTIONS, GAYA_TITIK, gaya)


def tulis(img, teks, xy, skala, warna=(255, 255, 255), tebal=1):
    """Tulisan halus (anti-aliasing) dengan bayangan tipis."""
    x, y = xy
    cv2.putText(img, teks, (x + 2, y + 2), FONT, skala, (0, 0, 0), tebal + 1, cv2.LINE_AA)
    cv2.putText(img, teks, (x, y), FONT, skala, warna, tebal, cv2.LINE_AA)


def panel(img, y1, y2, gelap=0.55):
    """Kotak hitam semi-transparan selebar layar."""
    bagian = img[y1:y2]
    bagian[:] = (bagian * (1 - gelap)).astype(np.uint8)


def susun_tampilan(img, mode, fps, pred_teks, kalimat):
    h, w = img.shape[:2]
    s = w / 1280                                   # skala tulisan ikut lebar gambar
    t = max(1, round(2 * s))
    panel(img, 0, int(95 * s))
    warna_mode = (0, 220, 255) if mode == "huruf" else (255, 200, 0)
    tulis(img, f"MODE {mode.upper()}", (int(20 * s), int(38 * s)), 0.9 * s, warna_mode, t)
    tulis(img, f"{fps:.0f} FPS", (w - int(130 * s), int(38 * s)), 0.7 * s, (180, 180, 180), 1)
    tulis(img, f"Prediksi: {pred_teks}", (int(20 * s), int(78 * s)), 0.8 * s, (255, 255, 255), t)

    panel(img, h - int(120 * s), h, 0.7)
    tulis(img, kalimat or "...", (int(20 * s), h - int(62 * s)), 1.2 * s, (255, 255, 255), t)
    tulis(img, "h huruf  |  k kata  |  SPASI  |  BACKSPACE hapus  |  c bersihkan  |  "
               "ENTER suara  |  q keluar", (int(20 * s), h - int(20 * s)), 0.55 * s,
          (170, 170, 170), 1)
    return img


def buka_kamera():
    """Coba HD (1280x720) dulu; jika gagal/tidak didukung, pakai 640x480."""
    for w, h in [(1280, 720), (640, 480)]:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(10):                      # beri waktu kamera menyala
            ok, frame = cap.read()
            if ok and frame is not None and frame.std() > 2:   # bukan layar hitam
                print(f"Kamera: {frame.shape[1]}x{frame.shape[0]}")
                return cap
        cap.release()
    return None


def main():
    model_huruf, label_huruf = muat("huruf_mlp.keras", "label_huruf.json")
    model_kata, label_kata = muat("kata_lstm.keras", "label_kata.json")
    if model_huruf is None and model_kata is None:
        return
    mode = "kata" if model_kata is not None else "huruf"
    path_dua = LABELS / "huruf_dua_tangan.json"
    huruf_dua = set(json.loads(path_dua.read_text(encoding="utf-8"))) if path_dua.exists() else set()

    cap = buka_kamera()
    if cap is None:
        print("Webcam tidak bisa dibaca. Tutup aplikasi lain yang memakai kamera "
              "(Zoom/Teams/browser) lalu coba lagi.")
        return
    holistic = buat_holistic(gambar_diam=False)
    # jendela bisa diperbesar / dimaksimalkan, rasio gambar tetap
    cv2.namedWindow(JUDUL, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(JUDUL, 1280, 820)
    penyusun = PenyusunKalimat()
    jendela = deque(maxlen=SEQ_LEN)
    ada_tangan = deque(maxlen=SEQ_LEN)
    riwayat_p = deque(maxlen=RATA_FRAME_HURUF)
    n_frame, fps, t_lalu = 0, 0.0, time.monotonic()
    pred_teks = "-"

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam tidak terbaca.")
            break
        n_frame += 1
        res = proses_bgr(holistic, frame)                 # diproses TANPA flip
        f = fitur_frame(res)
        jendela.append(f)
        ada_tangan.append(bool(np.any(f[:N_HANDS])))

        if mode == "huruf" and model_huruf is not None:
            fh = fitur_huruf(res)
            if fh is not None:
                # rata-rata beberapa frame terakhir -> prediksi tidak loncat-loncat
                riwayat_p.append(proba(model_huruf, fh[None]))
                pr = np.mean(riwayat_p, axis=0)
                i, p = int(pr.argmax()), float(pr.max())
                pred_teks = teks_top2(label_huruf, pr)
                dua = res.left_hand_landmarks is not None and res.right_hand_landmarks is not None
                if label_huruf[i] in huruf_dua and not dua:
                    pred_teks += "  - perlu 2 tangan"
                    p = 0.0  # huruf dua tangan tapi hanya 1 tangan terlihat -> abaikan
                penyusun.tambah_prediksi(label_huruf[i], p, "huruf")
            else:
                pred_teks = "- (tangan tidak terlihat)"
                riwayat_p.clear()
                penyusun.tangan_hilang()
        elif mode == "kata" and model_kata is not None and n_frame % TIAP_N_FRAME_KATA == 0:
            if len(jendela) == SEQ_LEN and sum(ada_tangan) >= MIN_FRAME_TANGAN:
                i, p = prediksi(model_kata, np.array(jendela, np.float32)[None])
                pred_teks = f"{label_kata[i].replace('_', ' ')} ({p:.0%})"
                penyusun.tambah_prediksi(label_kata[i], p, "kata")
            elif sum(ada_tangan) < MIN_FRAME_TANGAN:
                pred_teks = "- (tangan tidak terlihat)"
                penyusun.tangan_hilang()

        # ---- tampilan ----
        gambar_kerangka(frame, res)
        sekarang = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(sekarang - t_lalu, 1e-6)
        t_lalu = sekarang
        cv2.imshow(JUDUL, susun_tampilan(cv2.flip(frame, 1), mode, fps, pred_teks,
                                         penyusun.teks()))   # flip HANYA untuk tampilan

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("h") and model_huruf is not None:
            mode = "huruf"
        elif key == ord("k") and model_kata is not None:
            mode, pred_teks = "kata", "-"
            jendela.clear()
            ada_tangan.clear()
        elif key == 32:
            penyusun.spasi()
        elif key == 8:
            penyusun.hapus_terakhir()
        elif key == ord("c"):
            penyusun.bersihkan()
        elif key == 13:
            ucapkan(penyusun.teks())

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
