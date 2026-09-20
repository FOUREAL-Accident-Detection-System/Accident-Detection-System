import os
import pandas as pd

GROUP_FILE = "results/confirmed_overlap_groups.csv"
OUTPUT_FILE = "results/overlap_split_mapping.csv"

SPLITS = ["train", "validation", "test"]

def find_video_split(filename):
    matches = []

    for split in SPLITS:
        for class_name in ["accident", "normal"]:
            folder = os.path.join(
                "dataset_split",
                split,
                class_name
            )

            path = os.path.join(folder, filename)

            if os.path.isfile(path):
                matches.append({
                    "split": split,
                    "class": class_name,
                    "path": path
                })

    return matches


def main():

    print("=" * 70)
    print("OVERLAP GROUP → DATASET SPLIT MAPPING")
    print("=" * 70)

    groups = pd.read_csv(GROUP_FILE)

    rows = []

    for group_id, group_df in groups.groupby("group_id"):

        print(f"\nOVERLAP GROUP {group_id}")
        print("-" * 50)

        for _, row in group_df.iterrows():

            filename = row["filename"]
            matches = find_video_split(filename)

            if matches:

                for match in matches:

                    print(
                        f"{match['split'].upper():10} | "
                        f"{match['class'].upper():8} | "
                        f"{filename}"
                    )

                    rows.append({
                        "group_id": group_id,
                        "filename": filename,
                        "split": match["split"],
                        "class": match["class"],
                        "status": "FOUND"
                    })

            else:

                print(
                    f"NOT FOUND  | "
                    f"{filename}"
                )

                rows.append({
                    "group_id": group_id,
                    "filename": filename,
                    "split": "UNKNOWN",
                    "class": "UNKNOWN",
                    "status": "NOT_FOUND"
                })

    result = pd.DataFrame(rows)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nVideos mapped : {len(result)}")

    print("\nBy split:")
    print(result["split"].value_counts())

    print("\nBy class:")
    print(result["class"].value_counts())

    print("\nReport saved to:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("No files were deleted, moved, renamed, or modified.")


if __name__ == "__main__":
    main()