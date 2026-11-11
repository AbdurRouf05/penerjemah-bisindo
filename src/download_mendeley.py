"""Download dataset BISINDO dari Mendeley Data ke data/raw/.

Pakai: python src/download_mendeley.py [id_dataset ...]
Tanpa argumen = semua dataset. Aman dijalankan ulang: file yang sudah lengkap akan dilewati.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
API = "https://data.mendeley.com/public-api/datasets"

# (id, versi, folder tujuan, folder teratas yang diambil; None = semua)
DATASETS = [
    ("4xnkvr88tk", 1, "huruf/bisindo_7peraga", None),
    ("ywnjpbcz8m", 1, "huruf/alphabet_image_data", ["02. Resized Images"]),
    ("xh6bmx73n6", 1, "huruf/bisindo_recognition", None),
    ("235c78xbmk", 2, "huruf/multimodal_corpus", ["Train", "Test"]),
    ("f33k9w86wr", 3, "campuran/bisindo_video_dataset", None),
    ("m8y229yvrm", 1, "huruf/dataset_of_bisindo", None),
    ("4gtxgjy6xv", 1, "huruf/bisindo_70_30", None),
    ("yzhzkzbnmw", 3, "kata/bisindo_40kata_landmark", None),
]


def get_json(url):
    out = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True)
    return json.loads(out.stdout)


def list_files(ds_id, version, keep):
    folders = get_json(f"{API}/{ds_id}/folders/{version}")
    by_id = {f["id"]: f for f in folders}

    def path_of(folder):
        parts = []
        while folder:
            parts.insert(0, folder["name"])
            folder = by_id.get(folder.get("parent_id"))
        return parts

    jobs = []
    for folder in [None] + folders:
        parts = path_of(folder) if folder else []
        if keep and (not parts or parts[0] not in keep):
            continue
        fid = folder["id"] if folder else "root"
        for f in get_json(f"{API}/{ds_id}/files?folder_id={fid}&version={version}"):
            c = f["content_details"]
            jobs.append((parts + [f["filename"]], c["download_url"], c["size"]))
    return jobs


def download(url, dest, size):
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(3):
        subprocess.run(["curl", "-s", "-L", "-m", "600", "-o", str(dest), url])
        if dest.exists() and dest.stat().st_size == size:
            return "ok"
    return "fail"


def main():
    only = set(sys.argv[1:])
    for ds_id, version, target, keep in DATASETS:
        if only and ds_id not in only:
            continue
        jobs = list_files(ds_id, version, keep)
        total_mb = sum(s for _, _, s in jobs) / 1e6
        print(f"\n== {target}: {len(jobs)} file, {total_mb:.1f} MB", flush=True)
        stats = {"ok": 0, "skip": 0, "fail": 0}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(download, url, RAW / target / Path(*parts), size): parts
                for parts, url, size in jobs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                result = fut.result()
                stats[result] += 1
                if result == "fail":
                    print("   GAGAL:", "/".join(futures[fut]), flush=True)
                if i % 500 == 0 or i == len(jobs):
                    print(f"   {i}/{len(jobs)} {stats}", flush=True)
    print("\nSelesai.")


if __name__ == "__main__":
    sys.exit(main())
