import csv
import shutil
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "dataset"
CLEAN_DIR = PROJECT_DIR / "dataset_clean"
RESULTS_DIR = PROJECT_DIR / "results"

VERIFICATION_FILE = RESULTS_DIR / "dataset_verification.csv"

# Create clean dataset folders
for class_name in ["accident", "normal"]:
    (CLEAN_DIR / class_name).mkdir(parents=True, exist_ok=True)

# Read verification results
with open(VERIFICATION_FILE, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

seen_hashes = set()

copied = 0
skipped_duplicates = 0
skipped_corrupted = 0

for row in rows:

    file_path = Path(row["path"])
    class_name = row["class"]
    file_hash = row["sha256"]
    status = row["status"]

    # Skip corrupted videos
    if status != "OK":
        skipped_corrupted += 1
        continue

    # Skip exact duplicates
    if file_hash in seen_hashes:
        skipped_duplicates += 1
        continue

    seen_hashes.add(file_hash)

    destination = CLEAN_DIR / class_name / file_path.name

    shutil.copy2(file_path, destination)
    copied += 1

print("=" * 60)
print("CLEAN DATASET CREATED")
print("=" * 60)

print(f"Original videos       : {len(rows)}")
print(f"Unique videos copied  : {copied}")
print(f"Duplicates skipped    : {skipped_duplicates}")
print(f"Corrupted skipped     : {skipped_corrupted}")

accident_count = len(list((CLEAN_DIR / "accident").glob("*")))
normal_count = len(list((CLEAN_DIR / "normal").glob("*")))

print(f"\nClean Accident videos : {accident_count}")
print(f"Clean Normal videos   : {normal_count}")
print(f"Clean Total videos    : {accident_count + normal_count}")

print(f"\nClean dataset saved at:")
print(CLEAN_DIR)

print("\nOriginal dataset was NOT modified.")