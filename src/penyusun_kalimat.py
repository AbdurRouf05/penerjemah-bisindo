"""T7 - Penyusun kalimat: mengubah aliran prediksi menjadi kalimat yang rapi."""
import time

AMBANG_YAKIN = 0.80
BERTURUT = {"huruf": 6, "kata": 3}       # berapa prediksi berturut-turut sampai diterima
# jika model SANGAT yakin, cukup sedikit frame -> huruf yang benar cepat tercatat
SANGAT_YAKIN = 0.95
BERTURUT_CEPAT = {"huruf": 3, "kata": 2}
JEDA_TOKEN = {"huruf": 0.5, "kata": 1.0}  # detik minimal antar token yang diterima


class PenyusunKalimat:
    def __init__(self):
        # token: ("kata", "saya") atau ("huruf", "S") atau ("spasi", "")
        self.token = []
        self._calon = None
        self._hitung = 0
        self._t_terakhir = -1e9
        self._label_terakhir = None
        self._lepas = True    # label sama baru boleh diulang setelah "dilepas"
        self._yakin_min = 1.0

    def tangan_hilang(self):
        """Dipanggil saat tangan tidak terlihat: isyarat dianggap dilepas."""
        self._calon, self._hitung = None, 0
        self._lepas = True

    def tambah_prediksi(self, label, keyakinan, mode):
        """Kembalikan True jika prediksi diterima sebagai token baru."""
        sekarang = time.monotonic()
        if keyakinan < AMBANG_YAKIN:
            self._calon, self._hitung = None, 0
            return False
        if label == self._calon:
            self._hitung += 1
        else:
            self._calon, self._hitung, self._yakin_min = label, 1, 1.0
        self._yakin_min = min(self._yakin_min, keyakinan)
        perlu = BERTURUT_CEPAT[mode] if self._yakin_min >= SANGAT_YAKIN else BERTURUT[mode]
        if self._hitung < perlu:
            return False
        if label != self._label_terakhir:
            self._lepas = True        # sudah berganti isyarat dengan stabil
        if not self._lepas:
            return False              # isyarat sama masih ditahan -> jangan diketik ulang
        if sekarang - self._t_terakhir < JEDA_TOKEN[mode]:
            return False
        if mode == "kata" and self.token and self.token[-1][0] == "huruf":
            self.token.append(("spasi", ""))  # kata baru memutus ejaan huruf
        self.token.append((mode, label))
        self._t_terakhir, self._label_terakhir = sekarang, label
        self._lepas = False
        self._calon, self._hitung = None, 0
        return True

    def spasi(self):
        if self.token and self.token[-1][0] == "huruf":
            self.token.append(("spasi", ""))
        self._label_terakhir, self._lepas = None, True

    def hapus_terakhir(self):
        while self.token and self.token[-1][0] == "spasi":
            self.token.pop()
        if self.token:
            self.token.pop()
        self._label_terakhir, self._lepas = None, True

    def bersihkan(self):
        self.__init__()

    def kata_kata(self):
        """Gabungkan token menjadi daftar kata (huruf berurutan -> satu kata)."""
        kata, ejaan = [], ""
        for jenis, isi in self.token:
            if jenis == "huruf":
                ejaan += isi.lower()
                continue
            if ejaan:
                kata.append(ejaan)
                ejaan = ""
            if jenis == "kata":
                kata.append(isi.replace("_", " "))
        if ejaan:
            kata.append(ejaan)
        return kata

    def teks(self):
        s = " ".join(self.kata_kata())
        return s[:1].upper() + s[1:]


if __name__ == "__main__":
    # Simulasi tanpa menunggu waktu nyata: ganti jam dengan jam palsu
    jam = [0.0]
    time.monotonic = lambda: jam[0]

    p = PenyusunKalimat()

    def kirim(label, mode, n, yakin=0.95, dt=0.05):
        for _ in range(n):
            jam[0] += dt
            p.tambah_prediksi(label, yakin, mode)

    kirim("saya", "kata", 10)
    jam[0] += 0.3
    kirim("saya", "kata", 40, dt=0.1)        # masih ditahan 4 dtk -> tetap 1x
    kirim("makan", "kata", 10)
    jam[0] += 1.5
    kirim("pagi", "kata", 2)                  # belum 3x berturut -> ditolak
    kirim("pagi", "kata", 1, yakin=0.5)       # tidak yakin -> reset
    kirim("pagi", "kata", 5)
    print("1:", p.teks(), "| harapan: Saya makan pagi")

    p.bersihkan()
    for h in "SITI":
        jam[0] += 1.5
        kirim(h, "huruf", 10)
    p.spasi()
    jam[0] += 1.5
    kirim("terima_kasih", "kata", 5)
    print("2:", p.teks(), "| harapan: Siti terima kasih")

    p.bersihkan()
    kirim("Y", "huruf", 100)                 # ditahan 5 dtk -> tetap 1x (bug "Yyyy")
    kirim("Y", "huruf", 2, yakin=0.3)        # goyang sebentar -> belum dilepas
    kirim("Y", "huruf", 20)
    print("3a:", p.teks(), "| harapan: Y")
    p.bersihkan()
    for h in "AA":
        jam[0] += 1.5
        p.tangan_hilang()                    # tangan diturunkan di antara huruf
        kirim(h, "huruf", 10)
    print("3:", p.teks(), "| harapan: Aa (huruf dobel setelah tangan diturunkan)")
    p.hapus_terakhir()
    print("4:", p.teks(), "| harapan: A")

    p.bersihkan()
    t0 = jam[0]
    for h in "BUDI":                          # model sangat yakin, 20 FPS
        kirim(h, "huruf", 12, yakin=0.98)
    print("5:", p.teks(), f"| harapan: Budi  (4 huruf dalam {jam[0] - t0:.1f} dtk)")

    p.bersihkan()
    kirim("M", "huruf", 5, yakin=0.85)        # kurang yakin: 5 frame belum cukup
    print("6:", repr(p.teks()), "| harapan: '' (perlu 6 frame jika tidak sangat yakin)")
