import pandas as pd
import os

INPUT_FILE = "results/overlap_split_mapping.csv"
OUTPUT_FILE = "results/final_overlap_correction_plan.csv"

df = pd.read_csv(INPUT_FILE)

rows = []

for group_id, group in df.groupby("group_id"):

    splits = set(group["split"])

    if "test" in splits and "train" in splits:
        risk = "HIGH - TRAIN/TEST LEAKAGE"
        recommended_action = (
            "Keep this source group in ONE split only. "
            "Do not allow the same source content in train and test."
        )

    elif "validation" in splits and "train" in splits:
        risk = "MEDIUM - TRAIN/VALIDATION LEAKAGE"
        recommended_action = (
            "Keep this source group in ONE split only. "
            "Do not allow the same source content in train and validation."
        )

    else:
        risk = "REVIEW"
        recommended_action = "Manual review required."

    for _, video in group.iterrows():

        rows.append({
            "group_id": group_id,
            "filename": video["filename"],
            "current_split": video["split"],
            "class": video["class"],
            "risk": risk,
            "recommended_action": recommended_action
        })


result = pd.DataFrame(rows)

result.to_csv(OUTPUT_FILE, index=False)

print("=" * 80)
print("FINAL OVERLAP CORRECTION PLAN")
print("=" * 80)

for group_id, group in result.groupby("group_id"):

    print(f"\nGROUP {group_id}")
    print("-" * 80)

    print(f"Risk: {group['risk'].iloc[0]}")
    print(f"Class: {group['class'].iloc[0]}")

    for _, row in group.iterrows():
        print(
            f"{row['current_split'].upper():12} | "
            f"{row['filename']}"
        )

    print("\nAction:")
    print(group["recommended_action"].iloc[0])


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nTotal groups : {result['group_id'].nunique()}")
print(f"Total videos : {len(result)}")

print("\nRisk summary:")
print(result.groupby("group_id")["risk"].first().value_counts())

print("\nReport saved to:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("This script does NOT delete, move, rename, or modify any dataset files.")
print("It only creates the correction plan.")