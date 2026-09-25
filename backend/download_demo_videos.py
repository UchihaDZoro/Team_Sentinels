"""
Download curated surveillance demo videos with humans and cars
for testing and demonstration of CCTron (IBVAP).
"""
import sys
import time
import urllib.request
from pathlib import Path

DEST_DIR = Path(__file__).parent / "demo_videos"
DEST_DIR.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    {
        "filename": "surveillance_humans_and_cars.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        "desc": "CCTV stream: Humans, cars, and bicycles on roadway",
    },
    {
        "filename": "perimeter_pedestrians.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4",
        "desc": "Surveillance camera: Pedestrians walking past perimeter",
    },
    {
        "filename": "checkpost_vehicles.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "desc": "Overhead CCTV: Multiple cars and vehicle traffic",
    },
    {
        "filename": "restricted_zone_humans.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4",
        "desc": "Security camera: Humans in restricted zone",
    },
    {
        "filename": "checkpoint_single_person.mp4",
        "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4",
        "desc": "Checkpoint CCTV: Single person entry tracking",
    },
]


def download_file(url: str, dest_path: Path, desc: str):
    if dest_path.exists() and dest_path.stat().st_size > 100000:
        print(f"[EXISTS] {dest_path.name} ({dest_path.stat().st_size / 1e6:.1f} MB) — skipping")
        return True

    print(f"\n[DOWNLOADING] {dest_path.name}")
    print(f"  Source: {url}")
    print(f"  Desc:   {desc}")

    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, "wb") as out_file:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 64 * 1024

            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)

                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb = downloaded / (1024 * 1024)
                    tot_mb = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  Progress: {pct:5.1f}% [{mb:.1f} / {tot_mb:.1f} MB]")
                    sys.stdout.flush()

        elapsed = time.time() - start_time
        print(f"\n  [OK] Downloaded successfully in {elapsed:.1f}s ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  [ERROR] Error downloading {dest_path.name}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    print("=" * 60)
    print("  IBVAP — Surveillance Demo Video Downloader")
    print(f"  Target directory: {DEST_DIR}")
    print("=" * 60)

    success_count = 0
    for item in VIDEOS:
        target = DEST_DIR / item["filename"]
        ok = download_file(item["url"], target, item["desc"])
        if ok:
            success_count += 1

    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{len(VIDEOS)} surveillance demo videos.")
    print("Available in demo_videos directory:")
    for f in DEST_DIR.glob("*.mp4"):
        print(f"  • {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
