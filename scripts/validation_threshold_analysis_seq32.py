import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "accident_detection_clean_10fps_160x160_seq32_best.keras"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "preprocessing_clean_10fps_160x160_seq32.csv"
)

PREPROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessed_clean_seq32"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "validation_threshold_analysis_seq32.csv"
)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT,
    "results",
    "validation_threshold_best_seq32.json"
)

SEQUENCE_LENGTH = 32
BATCH_SIZE = 4


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("SEQ32 VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)

metadata = pd.read_csv(METADATA_PATH)

validation_metadata = metadata[
    metadata["split"].str.lower() == "validation"
].copy()

print(f"Validation sequences: {len(validation_metadata)}")
print(
    f"Validation videos: "
    f"{validation_metadata['source_video'].nunique()}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Seq32 best model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# GENERATE SEQUENCE PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

sequence_probabilities = []
sequence_labels = []
sequence_videos = []

for _, row in validation_metadata.iterrows():

    sequence_path = os.path.join(
        PROJECT_ROOT,
        row["sequence_file"]
    )

    with np.load(sequence_path) as data:
        frames = data["frames"]

    frames = frames.astype(np.float32) / 255.0

    probability = float(
        model.predict(
            np.expand_dims(frames, axis=0),
            verbose=0
        )[0][0]
    )

    sequence_probabilities.append(probability)
    sequence_labels.append(int(row["label"]))
    sequence_videos.append(row["source_video"])


# ============================================================
# AGGREGATE BY VIDEO
# ============================================================

validation_df = pd.DataFrame({
    "video": sequence_videos,
    "label": sequence_labels,
    "probability": sequence_probabilities
})

video_rows = []

for video, group in validation_df.groupby("video"):

    # Maximum accident probability across sequences
    max_probability = group["probability"].max()

    true_label = int(group["label"].iloc[0])

    video_rows.append({
        "video": video,
        "true_label": true_label,
        "max_probability": max_probability
    })

video_df = pd.DataFrame(video_rows)

print(f"Videos evaluated: {len(video_df)}")


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for threshold in np.arange(0.10, 0.91, 0.05):

    y_true = video_df["true_label"].values

    y_pred = (
        video_df["max_probability"].values >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
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
# SELECT BEST THRESHOLD USING VALIDATION F1
# ============================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(best_row["threshold"])


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

best_result = {
    "model": "accident_detection_clean_10fps_160x160_seq32_best.keras",
    "sequence_length": SEQUENCE_LENGTH,
    "aggregation": "max_probability",
    "best_threshold": best_threshold,
    "accuracy": float(best_row["accuracy"]),
    "precision": float(best_row["precision"]),
    "recall": float(best_row["recall"]),
    "f1": float(best_row["f1"]),
    "TN": int(best_row["TN"]),
    "FP": int(best_row["FP"]),
    "FN": int(best_row["FN"]),
    "TP": int(best_row["TP"])
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(best_result, f, indent=4)


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("SEQ32 THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print("\nThreshold results:")
print(results_df.to_string(index=False))

print("\n" + "-" * 70)
print("BEST VALIDATION THRESHOLD")
print("-" * 70)

print(f"Threshold : {best_threshold:.2f}")
print(f"Accuracy  : {best_row['accuracy'] * 100:.2f}%")
print(f"Precision : {best_row['precision'] * 100:.2f}%")
print(f"Recall    : {best_row['recall'] * 100:.2f}%")
print(f"F1 Score  : {best_row['f1'] * 100:.2f}%")
print(
    f"TN={int(best_row['TN'])}, "
    f"FP={int(best_row['FP'])}, "
    f"FN={int(best_row['FN'])}, "
    f"TP={int(best_row['TP'])}"
)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)