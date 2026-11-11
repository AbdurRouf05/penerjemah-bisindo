"""T7 - Teks ke suara Bahasa Indonesia.

Utama  : edge-tts (id-ID-GadisNeural, butuh internet) -> diputar dengan pygame.
Cadangan: pyttsx3 (offline).
Berjalan di thread terpisah agar video webcam tidak macet.
"""
import asyncio
import os
import tempfile
import threading
import time

SUARA = "id-ID-GadisNeural"
_kunci = threading.Lock()
_terakhir = {"teks": None, "waktu": 0.0}


def _edge_tts(teks):
    import edge_tts
    import pygame
    path = os.path.join(tempfile.gettempdir(), f"bisindo_tts_{time.time_ns()}.mp3")
    asyncio.run(edge_tts.Communicate(teks, SUARA).save(path))
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()
    try:
        os.remove(path)
    except OSError:
        pass


def _pyttsx3(teks):
    import pyttsx3
    mesin = pyttsx3.init()
    for v in mesin.getProperty("voices"):
        if "indonesia" in v.name.lower() or "id" in str(v.languages).lower():
            mesin.setProperty("voice", v.id)
            break
    mesin.say(teks)
    mesin.runAndWait()


def _ucapkan_sinkron(teks):
    with _kunci:  # satu suara pada satu waktu
        try:
            _edge_tts(teks)
        except Exception as e:
            print(f"[suara] edge-tts gagal ({type(e).__name__}), pakai pyttsx3 offline")
            try:
                _pyttsx3(teks)
            except Exception as e2:
                print(f"[suara] pyttsx3 juga gagal: {e2}")


def ucapkan(teks: str) -> None:
    """Ucapkan teks di background. Teks sama dalam 3 detik tidak diulang."""
    teks = (teks or "").strip()
    if not teks:
        return
    sekarang = time.monotonic()
    if teks == _terakhir["teks"] and sekarang - _terakhir["waktu"] < 3:
        return
    _terakhir.update(teks=teks, waktu=sekarang)
    threading.Thread(target=_ucapkan_sinkron, args=(teks,), daemon=True).start()


if __name__ == "__main__":
    _ucapkan_sinkron("Halo, apa kabar?")
