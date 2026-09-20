import cv2
import hashlib
from pathlib import Path
import csv

# =========================
# DATASET PATH
# =========================
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESULTS_DIR.mkdir(exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}


# =========================
# FILE HASH
# =========================
def get_file_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


# =========================
# VIDEO INFORMATION
# =========================
def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": round(fps, 2),
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": round(duration, 2),
    }


# =========================
# MAIN VERIFICATION
# =========================
def main():

    print("=" * 60)
    print("ACCIDENT DETECTION - DATASET VERIFICATION")
    print("=" * 60)

    all_videos = []

    for class_name in ["accident", "normal"]:

        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"ERROR: Missing folder: {class_dir}")
            continue

        videos = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        ]

        print(f"\n{class_name.upper()} VIDEOS: {len(videos)}")

        for video in videos:
            all_videos.append((class_name, video))

    print(f"\nTOTAL VIDEOS FOUND: {len(all_videos)}")

    # =========================
    # CHECK VIDEOS
    # =========================

    results = []
    hashes = {}

    corrupted = []
    duplicates = []

    for index, (class_name, video_path) in enumerate(all_videos, start=1):

        print(f"[{index}/{len(all_videos)}] Checking: {video_path.name}")

        file_hash = get_file_hash(video_path)

        # Exact duplicate check
        if file_hash in hashes:
            duplicate_of = hashes[file_hash]
            duplicates.append((str(video_path), duplicate_of))
        else:
            hashes[file_hash] = str(video_path)

        info = get_video_info(video_path)

        if info is None:
            corrupted.append(str(video_path))

            results.append({
                "class": class_name,
                "filename": video_path.name,
                "path": str(video_path),
                "status": "CORRUPTED",
                "fps": "",
                "frames": "",
                "width": "",
                "height": "",
                "duration_seconds": "",
                "sha256": file_hash,
            })

            continue

        results.append({
            "class": class_name,
            "filename": video_path.name,
            "path": str(video_path),
            "status": "OK",
            "fps": info["fps"],
            "frames": info["frames"],
            "width": info["width"],
            "height": info["height"],
            "duration_seconds": info["duration"],
            "sha256": file_hash,
        })

    # =========================
    # SAVE CSV
    # =========================

    output_file = RESULTS_DIR / "dataset_verification.csv"

    fieldnames = [
        "class",
        "filename",
        "path",
        "status",
        "fps",
        "frames",
        "width",
        "height",
        "duration_seconds",
        "sha256",
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # =========================
    # SUMMARY
    # =========================

    accident_count = sum(
        1 for c, _ in all_videos if c == "accident"
    )

    normal_count = sum(
        1 for c, _ in all_videos if c == "normal"
    )

    valid_count = sum(
        1 for r in results if r["status"] == "OK"
    )

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    print(f"Accident videos : {accident_count}")
    print(f"Normal videos   : {normal_count}")
    print(f"Total videos    : {len(all_videos)}")
    print(f"Valid videos    : {valid_count}")
    print(f"Corrupted       : {len(corrupted)}")
    print(f"Exact duplicates: {len(duplicates)}")

    if duplicates:
        print("\nEXACT DUPLICATES:")
        for duplicate, original in duplicates:
            print(f"  Duplicate: {duplicate}")
            print(f"  Original : {original}")

    if corrupted:
        print("\nCORRUPTED VIDEOS:")
        for video in corrupted:
            print(f"  {video}")

    print(f"\nDetailed results saved to:")
    print(output_file)

    print("\nDataset verification completed.")


if __name__ == "__main__":
    main()