import os
import shutil
import random
import pandas as pd

SOURCE_DIR = "dataset_clean"
OUTPUT_DIR = "dataset_split_clean"
REPORT_FILE = "results/clean_split_mapping.csv"

RANDOM_SEED = 42

SPLITS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15
}

CLASSES = ["accident", "normal"]


# ============================================================
# CONFIRMED SOURCE GROUPS
# Every video in a group must remain in the SAME split.
# ============================================================

OVERLAP_GROUPS = {
    1: [
        "WhatsApp Video 2026-09-08 at 3.35.00 PM (2).mp4",
        "WhatsApp Video 2026-09-08 at 3.35.00 PM.mp4"
    ],

    2: [
        "WhatsApp Video 2026-09-08 at 3.22.36 PM.mp4",
        "WhatsApp Video 2026-09-09 at 10.21.43 PM (2).mp4"
    ],

    3: [
        "WhatsApp Video 2026-09-09 at 11.20.42 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.21.04 AM.mp4"
    ],

    4: [
        "WhatsApp Video 2026-09-09 at 11.19.58 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.11 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.36 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.23.28 AM.mp4"
    ],

    5: [
        "WhatsApp Video 2026-09-08 at 9.34.35 PM.mp4",
        "WhatsApp Video 2026-09-08 at 9.36.39 PM.mp4"
    ],

    6: [
        "WhatsApp Video 2026-09-08 at 3.43.48 PM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.02 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.16 AM.mp4"
    ],

    7: [
        "WhatsApp Video 2026-09-09 at 11.23.00 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.23.03 AM.mp4"
    ],

    8: [
        "WhatsApp Video 2026-09-08 at 9.35.21 PM (1).mp4",
        "WhatsApp Video 2026-09-08 at 9.36.31 PM.mp4"
    ],

    9: [
        "WhatsApp Video 2026-09-09 at 11.22.04 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.22.12 AM.mp4"
    ]
}


def get_all_videos():

    videos = []

    for class_name in CLASSES:

        folder = os.path.join(SOURCE_DIR, class_name)

        for filename in os.listdir(folder):

            path = os.path.join(folder, filename)

            if os.path.isfile(path):

                videos.append({
                    "filename": filename,
                    "class": class_name,
                    "source_path": path
                })

    return videos


def main():

    random.seed(RANDOM_SEED)

    print("=" * 80)
    print("BUILDING LEAKAGE-CONTROLLED DATASET SPLIT")
    print("=" * 80)

    all_videos = get_all_videos()

    print(f"\nVideos in dataset_clean: {len(all_videos)}")

    # --------------------------------------------------------
    # Map filenames to their source group
    # --------------------------------------------------------

    filename_to_group = {}

    for group_id, filenames in OVERLAP_GROUPS.items():

        for filename in filenames:
            filename_to_group[filename] = group_id

    # --------------------------------------------------------
    # Build source units
    # A source unit is either:
    #   - one independent video
    #   - one confirmed overlap group
    # --------------------------------------------------------

    used = set()
    units = []

    for group_id, filenames in OVERLAP_GROUPS.items():

        existing = []

        for video in all_videos:

            if video["filename"] in filenames:

                existing.append(video)
                used.add(video["filename"])

        if existing:

            units.append({
                "group_id": group_id,
                "videos": existing,
                "class": existing[0]["class"]
            })

    for video in all_videos:

        if video["filename"] not in used:

            units.append({
                "group_id": None,
                "videos": [video],
                "class": video["class"]
            })

    print(f"Source units created: {len(units)}")

    # --------------------------------------------------------
    # Split separately by class
    # --------------------------------------------------------

    final_rows = []

    for class_name in CLASSES:

        class_units = [
            unit for unit in units
            if unit["class"] == class_name
        ]

        random.shuffle(class_units)

        total_videos = sum(
            len(unit["videos"])
            for unit in class_units
        )

        target_train = round(total_videos * SPLITS["train"])
        target_validation = round(total_videos * SPLITS["validation"])

        counts = {
            "train": 0,
            "validation": 0,
            "test": 0
        }

        for unit in class_units:

            size = len(unit["videos"])

            remaining_train = target_train - counts["train"]
            remaining_validation = target_validation - counts["validation"]

            if remaining_train >= size:
                split = "train"

            elif remaining_validation >= size:
                split = "validation"

            else:
                split = "test"

            counts[split] += size

            for video in unit["videos"]:

                final_rows.append({
                    "filename": video["filename"],
                    "class": class_name,
                    "split": split,
                    "group_id": (
                        unit["group_id"]
                        if unit["group_id"] is not None
                        else "independent"
                    )
                })

        print(f"\nCLASS: {class_name}")
        print(f"Total      : {total_videos}")
        print(f"Train      : {counts['train']}")
        print(f"Validation : {counts['validation']}")
        print(f"Test       : {counts['test']}")

    mapping = pd.DataFrame(final_rows)

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    for split in SPLITS:

        for class_name in CLASSES:

            os.makedirs(
                os.path.join(
                    OUTPUT_DIR,
                    split,
                    class_name
                ),
                exist_ok=True
            )

    # --------------------------------------------------------
    # Copy videos
    # Original dataset_clean remains untouched.
    # --------------------------------------------------------

    for _, row in mapping.iterrows():

        source = os.path.join(
            SOURCE_DIR,
            row["class"],
            row["filename"]
        )

        destination = os.path.join(
            OUTPUT_DIR,
            row["split"],
            row["class"],
            row["filename"]
        )

        shutil.copy2(source, destination)

    # --------------------------------------------------------
    # Save mapping
    # --------------------------------------------------------

    os.makedirs("results", exist_ok=True)

    mapping.to_csv(
        REPORT_FILE,
        index=False
    )

    print("\n" + "=" * 80)
    print("CLEAN SPLIT CREATED")
    print("=" * 80)

    print("\nOverall:")
    print(mapping["split"].value_counts())

    print("\nClass + split:")
    print(
        mapping.groupby(
            ["split", "class"]
        ).size()
    )

    print("\nSaved mapping:")
    print(REPORT_FILE)

    print("\nOutput directory:")
    print(OUTPUT_DIR)

    print("\nIMPORTANT:")
    print("dataset_clean was NOT modified.")
    print("dataset_split was NOT modified.")
    print("Only dataset_split_clean was created.")


if __name__ == "__main__":
    main()