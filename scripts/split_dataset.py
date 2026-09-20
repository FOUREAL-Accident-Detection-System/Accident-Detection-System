import random
import shutil
from pathlib import Path
import csv

# =========================
# PATHS
# =========================
PROJECT_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_DIR / "dataset_clean"
SPLIT_DIR = PROJECT_DIR / "dataset_split"
RESULTS_DIR = PROJECT_DIR / "results"

# =========================
# SETTINGS
# =========================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

random.seed(SEED)

# =========================
# CREATE FOLDERS
# =========================
for split in ["train", "validation", "test"]:
    for class_name in ["accident", "normal"]:
        (SPLIT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )

# =========================
# SPLIT EACH CLASS
# =========================
split_records = []

for class_name in ["accident", "normal"]:

    source_class_dir = SOURCE_DIR / class_name

    videos = [
        p for p in source_class_dir.iterdir()
        if p.is_file()
    ]

    random.shuffle(videos)

    total = len(videos)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_videos = videos[:train_end]
    val_videos = videos[train_end:val_end]
    test_videos = videos[val_end:]

    split_data = {
        "train": train_videos,
        "validation": val_videos,
        "test": test_videos
    }

    for split_name, video_list in split_data.items():

        for video in video_list:

            destination = (
                SPLIT_DIR
                / split_name
                / class_name
                / video.name
            )

            shutil.copy2(video, destination)

            split_records.append({
                "filename": video.name,
                "class": class_name,
                "split": split_name
            })

# =========================
# SAVE SPLIT RECORD
# =========================
RESULTS_DIR.mkdir(exist_ok=True)

split_file = RESULTS_DIR / "dataset_split.csv"

with open(split_file, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "class", "split"]
    )

    writer.writeheader()
    writer.writerows(split_records)

# =========================
# SUMMARY
# =========================
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)

for split_name in ["train", "validation", "test"]:

    accident_count = len(
        list(
            (SPLIT_DIR / split_name / "accident").glob("*")
        )
    )

    normal_count = len(
        list(
            (SPLIT_DIR / split_name / "normal").glob("*")
        )
    )

    total = accident_count + normal_count

    print(
        f"{split_name.capitalize():12} : "
        f"Accident={accident_count}, "
        f"Normal={normal_count}, "
        f"Total={total}"
    )

print("\nSplit record saved to:")
print(split_file)

print("\nOriginal clean dataset was NOT modified.")