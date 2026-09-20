import pandas as pd
import os

INPUT_FILE = "results/overlap_split_mapping.csv"
OUTPUT_FILE = "results/overlap_leakage_summary.csv"

df = pd.read_csv(INPUT_FILE)

summary = []

for group_id, group in df.groupby("group_id"):

    splits = set(group["split"])

    has_train = "train" in splits
    has_validation = "validation" in splits
    has_test = "test" in splits

    if has_train and has_test:
        leakage_type = "TRAIN_TEST_OVERLAP"
    elif has_train and has_validation:
        leakage_type = "TRAIN_VALIDATION_OVERLAP"
    elif has_validation and has_test:
        leakage_type = "VALIDATION_TEST_OVERLAP"
    else:
        leakage_type = "WITHIN_SPLIT"

    summary.append({
        "group_id": group_id,
        "videos": len(group),
        "train_count": (group["split"] == "train").sum(),
        "validation_count": (group["split"] == "validation").sum(),
        "test_count": (group["split"] == "test").sum(),
        "classes": ", ".join(sorted(group["class"].unique())),
        "leakage_type": leakage_type
    })

result = pd.DataFrame(summary)

result.to_csv(OUTPUT_FILE, index=False)

print("=" * 70)
print("OVERLAP LEAKAGE SUMMARY")
print("=" * 70)

print("\n")
print(result.to_string(index=False))

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nTotal overlap groups : {len(result)}")

print("\nLeakage types:")
print(result["leakage_type"].value_counts())

print("\nReport saved to:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("No files were deleted or modified.")
print("This is an analysis-only report.")