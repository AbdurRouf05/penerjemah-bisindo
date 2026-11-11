"""T1 - Uji MediaPipe + webcam. Tekan q untuk keluar.

    .venv\\Scripts\\python.exe -m src.cek_webcam
"""
import time

import cv2
import mediapipe as mp

from src.features import buat_holistic, fitur_frame, proses_bgr

gambar = mp.solutions.drawing_utils
H = mp.solutions.holistic


def main():
    cap = cv2.VideoCapture(0)
    holistic = buat_holistic(gambar_diam=False)
    t_lalu = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam tidak terbaca (dipakai aplikasi lain?)")
            break
        res = proses_bgr(holistic, frame)          # diproses TANPA flip
        gambar.draw_landmarks(frame, res.left_hand_landmarks, H.HAND_CONNECTIONS)
        gambar.draw_landmarks(frame, res.right_hand_landmarks, H.HAND_CONNECTIONS)
        gambar.draw_landmarks(frame, res.pose_landmarks, H.POSE_CONNECTIONS)
        tampil = cv2.flip(frame, 1)                # flip HANYA untuk tampilan
        sekarang = time.monotonic()
        fps = 1 / max(sekarang - t_lalu, 1e-6)
        t_lalu = sekarang
        teks = [f"FPS: {fps:.1f}",
                f"Kiri : {'ada' if res.left_hand_landmarks else 'tidak'}",
                f"Kanan: {'ada' if res.right_hand_landmarks else 'tidak'}",
                f"Panjang fitur_frame: {len(fitur_frame(res))}"]
        for i, t in enumerate(teks):
            cv2.putText(tampil, t, (10, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Cek webcam (q = keluar)", tampil)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    holistic.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
