"""
Automated downloader for genuine CIC-IDS2017 MachineLearningCSV files.
Source: Official CIC-IDS2017 dataset mirror hosted on Hugging Face (c01dsnap/CIC-IDS2017).
"""

import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import ORIGINAL_DATA_DIR

HF_BASE_URL = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"

AVAILABLE_FILES = [
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


def download_file(filename: str, max_retries: int = 3):
    dest_path = ORIGINAL_DATA_DIR / filename
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[*] {filename} already exists ({round(dest_path.stat().st_size / (1024*1024), 2)} MB). Skipping.")
        return True

    url = f"{HF_BASE_URL}/{filename}"
    print(f"[*] Downloading {filename} from {url}...")

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as response, open(dest_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                block_size = 1024 * 128
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    if total_size > 0:
                        pct = min(100.0, downloaded * 100 / total_size)
                        mb_down = downloaded / (1024 * 1024)
                        mb_tot = total_size / (1024 * 1024)
                        print(f"\r    Progress: {pct:6.2f}% ({mb_down:6.1f} MB / {mb_tot:6.1f} MB)", end="", flush=True)

            print(f"\n[+] Successfully saved: {filename}")
            return True
        except Exception as e:
            print(f"\n[-] Attempt {attempt} failed for {filename}: {e}")
            if dest_path.exists():
                dest_path.unlink()

    return False


def download_dataset(files=None):
    ORIGINAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets = files or AVAILABLE_FILES
    success_count = 0
    for f in targets:
        if download_file(f):
            success_count += 1

    print(f"\n[+] Downloaded/Verified {success_count}/{len(targets)} dataset files.")
    return success_count > 0


if __name__ == "__main__":
    # If arguments given, download specific files
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    download_dataset(targets)
