import os
import cv2
import hashlib
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

TRAIN_FOLDER = "dataset_split/train"
OUTPUT_FILE = "results/training_dataset_audit.csv"

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".MP4",
    ".AVI",
    ".MOV",
    ".MKV"
)


# ==========================================
# CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results", exist_ok=True)


# ==========================================
# CALCULATE FILE HASH
# ==========================================

def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


# ==========================================
# AUDIT DATASET
# ==========================================

records = []

classes = [
    "accident",
    "normal"
]

for class_name in classes:

    class_folder = os.path.join(
        TRAIN_FOLDER,
        class_name
    )

    if not os.path.exists(class_folder):

        print(
            f"ERROR: Folder not found: {class_folder}"
        )

        continue

    files = sorted(
        os.listdir(class_folder)
    )

    for file_name in files:

        if not file_name.lower().endswith(
            VIDEO_EXTENSIONS
        ):
            continue

        file_path = os.path.join(
            class_folder,
            file_name
        )

        print(
            f"Checking: {class_name} / {file_name}"
        )

        file_size_mb = (
            os.path.getsize(file_path)
            / (1024 * 1024)
        )

        file_hash = calculate_hash(
            file_path
        )

        cap = cv2.VideoCapture(
            file_path
        )

        readable = cap.isOpened()

        if readable:

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            frame_count = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            if fps > 0:

                duration = (
                    frame_count / fps
                )

            else:

                duration = 0

        else:

            fps = 0
            frame_count = 0
            width = 0
            height = 0
            duration = 0

        cap.release()

        records.append({

            "class": class_name,

            "file_name": file_name,

            "file_size_mb": round(
                file_size_mb,
                2
            ),

            "fps": round(
                fps,
                2
            ),

            "frame_count": frame_count,

            "width": width,

            "height": height,

            "duration_seconds": round(
                duration,
                2
            ),

            "readable": readable,

            "sha256": file_hash

        })


# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(records)


# ==========================================
# DUPLICATE CHECK
# ==========================================

if not df.empty:

    df["duplicate_hash"] = (
        df["sha256"].duplicated(
            keep=False
        )
    )

else:

    df["duplicate_hash"] = []


# ==========================================
# SAVE AUDIT REPORT
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==========================================
# SUMMARY
# ==========================================

print("\n")
print("=" * 60)
print("TRAINING DATASET AUDIT COMPLETED")
print("=" * 60)

print(
    f"\nTotal training videos: {len(df)}"
)

if not df.empty:

    print("\nVideos by class:")

    print(
        df["class"].value_counts()
    )

    print("\nUnreadable videos:")

    unreadable = df[
        df["readable"] == False
    ]

    print(
        len(unreadable)
    )

    print("\nExact duplicate files:")

    duplicates = df[
        df["duplicate_hash"] == True
    ]

    print(
        len(duplicates)
    )

    print("\nFPS range:")

    print(
        f"Minimum: {df['fps'].min():.2f}"
    )

    print(
        f"Maximum: {df['fps'].max():.2f}"
    )

    print("\nDuration range:")

    print(
        f"Minimum: "
        f"{df['duration_seconds'].min():.2f} seconds"
    )

    print(
        f"Maximum: "
        f"{df['duration_seconds'].max():.2f} seconds"
    )

    print("\nResolution combinations:")

    print(
        df[
            ["width", "height"]
        ].value_counts()
    )


print("\nAudit report saved to:")

print(OUTPUT_FILE)

print("\nNo dataset files were modified.")