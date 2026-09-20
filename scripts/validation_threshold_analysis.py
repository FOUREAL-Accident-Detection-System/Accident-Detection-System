import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/accident_detection_clean_10fps_160x160_best.keras"
METADATA_PATH = "results/preprocessing_clean_10fps_160x160.csv"

OUTPUT_CSV = "results/validation_threshold_analysis.csv"
OUTPUT_JSON = "results/validation_threshold_best.json"

THRESHOLDS = np.arange(0.10, 0.91, 0.05)

# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)

print("\nLoading validation metadata...")

df = pd.read_csv(METADATA_PATH)

validation_df = df[df["split"] == "validation"].copy()

print(f"Validation sequences: {len(validation_df)}")
print(f"Validation videos: {validation_df['source_video'].nunique()}")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# ============================================================
# PREDICT EACH VALIDATION SEQUENCE
# ============================================================

video_probabilities = {}

print("\nRunning validation predictions...")

for index, row in validation_df.iterrows():

    sequence_path = row["sequence_file"]

    if not os.path.exists(sequence_path):
        print(f"Missing sequence: {sequence_path}")
        continue

    data = np.load(sequence_path)

    sequence = data["frames"].astype(np.float32) / 255.0

    sequence = np.expand_dims(sequence, axis=0)

    probability = float(model.predict(sequence, verbose=0)[0][0])

    video_name = row["source_video"]
    true_label = int(row["label"])

    if video_name not in video_probabilities:
        video_probabilities[video_name] = {
            "true_label": true_label,
            "probabilities": []
        }

    video_probabilities[video_name]["probabilities"].append(probability)

# ============================================================
# CREATE VIDEO-LEVEL DATA
# ============================================================

video_results = []

for video_name, info in video_probabilities.items():

    max_probability = max(info["probabilities"])

    video_results.append({
        "source_video": video_name,
        "true_label": info["true_label"],
        "max_accident_probability": max_probability,
        "sequence_count": len(info["probabilities"])
    })

video_df = pd.DataFrame(video_results)

print(f"\nValidation videos evaluated: {len(video_df)}")

# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for threshold in THRESHOLDS:

    predictions = (
        video_df["max_accident_probability"] >= threshold
    ).astype(int)

    true_labels = video_df["true_label"]

    accuracy = accuracy_score(true_labels, predictions)

    precision = precision_score(
        true_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": round(float(threshold), 2),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

results_df = pd.DataFrame(results)

# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])

print("\n" + "=" * 60)
print("THRESHOLD RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("BEST VALIDATION THRESHOLD")
print("=" * 60)

print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {best_row['accuracy']:.4f}")
print(f"Precision : {best_row['precision']:.4f}")
print(f"Recall    : {best_row['recall']:.4f}")
print(f"F1 Score  : {best_row['f1']:.4f}")
print(f"TN        : {int(best_row['TN'])}")
print(f"FP        : {int(best_row['FP'])}")
print(f"FN        : {int(best_row['FN'])}")
print(f"TP        : {int(best_row['TP'])}")

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs("results", exist_ok=True)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

with open(OUTPUT_JSON, "w") as f:
    json.dump(
        {
            "best_threshold": best_threshold,
            "selection_metric": "F1",
            "validation_videos": len(video_df),
            "accuracy": float(best_row["accuracy"]),
            "precision": float(best_row["precision"]),
            "recall": float(best_row["recall"]),
            "f1": float(best_row["f1"]),
            "TN": int(best_row["TN"]),
            "FP": int(best_row["FP"]),
            "FN": int(best_row["FN"]),
            "TP": int(best_row["TP"])
        },
        f,
        indent=4
    )

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)

print("\nDone.")