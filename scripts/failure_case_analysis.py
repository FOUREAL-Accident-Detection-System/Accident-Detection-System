import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"results\unseen_100_predictions_seq32.csv"

OUTPUT_FILE = r"results\failure_case_analysis_seq32.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

print("\nLoading unseen-video results...")

df = pd.read_csv(INPUT_FILE)

print(f"Total videos loaded: {len(df)}")


# ============================================================
# IDENTIFY FAILURE CASES
# ============================================================

# False Negative:
# Actual ACCIDENT but predicted NORMAL

false_negatives = df[
    (df["true_label"] == "ACCIDENT") &
    (df["predicted_label"] == "NORMAL")
].copy()

# False Positive:
# Actual NORMAL but predicted ACCIDENT

false_positives = df[
    (df["true_label"] == "NORMAL") &
    (df["predicted_label"] == "ACCIDENT")
].copy()


# ============================================================
# ADD FAILURE TYPE
# ============================================================

false_negatives["failure_type"] = "FALSE_NEGATIVE"

false_positives["failure_type"] = "FALSE_POSITIVE"


# ============================================================
# COMBINE FAILURE CASES
# ============================================================

failure_cases = pd.concat(
    [
        false_negatives,
        false_positives
    ],
    ignore_index=True
)


# ============================================================
# SORT BY FAILURE TYPE
# ============================================================

failure_cases = failure_cases.sort_values(
    by=["failure_type", "confidence"],
    ascending=[True, False]
)


# ============================================================
# SAVE REPORT
# ============================================================

failure_cases.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FAILURE CASE ANALYSIS")
print("=" * 70)

print(
    f"False Negatives : {len(false_negatives)}"
)

print(
    f"False Positives : {len(false_positives)}"
)

print(
    f"Total failures  : {len(failure_cases)}"
)


# ============================================================
# FALSE NEGATIVES
# ============================================================

print("\n" + "-" * 70)
print("FALSE NEGATIVES")
print("Actual ACCIDENT → Predicted NORMAL")
print("-" * 70)

if len(false_negatives) > 0:

    for _, row in false_negatives.iterrows():

        print(
            f"{row['filename']} | "
            f"Confidence: {row['confidence']:.4f} | "
            f"Max accident probability: "
            f"{row['max_accident_probability']:.4f}"
        )

else:

    print("No false negatives.")


# ============================================================
# FALSE POSITIVES
# ============================================================

print("\n" + "-" * 70)
print("FALSE POSITIVES")
print("Actual NORMAL → Predicted ACCIDENT")
print("-" * 70)

if len(false_positives) > 0:

    for _, row in false_positives.iterrows():

        print(
            f"{row['filename']} | "
            f"Confidence: {row['confidence']:.4f} | "
            f"Max accident probability: "
            f"{row['max_accident_probability']:.4f}"
        )

else:

    print("No false positives.")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("FAILURE ANALYSIS SAVED")
print("=" * 70)

print(
    f"Output file: {OUTPUT_FILE}"
)

print("=" * 70)
