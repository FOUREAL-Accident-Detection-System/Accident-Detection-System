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
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "accident_detection_clean_10fps_160x160_seq64_best.keras"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "preprocessing_clean_10fps_160x160_seq64.csv"
)

OUTPUT_CSV = os.path.join(
    PROJECT_ROOT,
    "results",
    "validation_threshold_analysis_seq64.csv"
)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT,
    "results",
    "validation_threshold_best_seq64.json"
)

SEQUENCE_LENGTH = 64


# ============================================================
# START
# ============================================================

print("=" * 70)
print("SEQ64 VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = pd.read_csv(METADATA_PATH)

validation_metadata = metadata[
    metadata["split"].str.lower() == "validation"
].copy()

print(
    f"Validation sequences: "
    f"{len(validation_metadata)}"
)

print(
    f"Validation videos: "
    f"{validation_metadata['source_video'].nunique()}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Seq64 best model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# GENERATE VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

sequence_rows = []

for _, row in validation_metadata.iterrows():

    sequence_path = os.path.join(
        PROJECT_ROOT,
        row["sequence_file"]
    )

    with np.load(sequence_path) as data:
        frames = data["frames"]

    frames = (
        frames.astype(np.float32) / 255.0
    )

    probability = float(
        model.predict(
            np.expand_dims(frames, axis=0),
            verbose=0
        )[0][0]
    )

    sequence_rows.append({
        "video": row["source_video"],
        "true_label": int(row["label"]),
        "probability": probability
    })


sequence_df = pd.DataFrame(sequence_rows)


# ============================================================
# AGGREGATE BY VIDEO
# ============================================================

video_rows = []

for video, group in sequence_df.groupby("video"):

    video_rows.append({
        "video": video,
        "true_label": int(
            group["true_label"].iloc[0]
        ),
        "max_probability": float(
            group["probability"].max()
        )
    })


video_df = pd.DataFrame(video_rows)

print(
    f"Videos evaluated: {len(video_df)}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

results = []

for threshold in np.arange(
    0.10,
    0.91,
    0.05
):

    threshold = round(
        float(threshold),
        2
    )

    y_true = video_df[
        "true_label"
    ].values

    y_pred = (
        video_df["max_probability"].values
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

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
        "threshold": threshold,
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
# BEST THRESHOLD BY VALIDATION F1
# ============================================================

best_row = results_df.loc[
    results_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE BEST THRESHOLD JSON
# ============================================================

best_result = {
    "model":
        "accident_detection_clean_10fps_160x160_seq64_best.keras",

    "sequence_length":
        SEQUENCE_LENGTH,

    "aggregation":
        "max_probability",

    "best_threshold":
        best_threshold,

    "accuracy":
        float(best_row["accuracy"]),

    "precision":
        float(best_row["precision"]),

    "recall":
        float(best_row["recall"]),

    "f1":
        float(best_row["f1"]),

    "TN":
        int(best_row["TN"]),

    "FP":
        int(best_row["FP"]),

    "FN":
        int(best_row["FN"]),

    "TP":
        int(best_row["TP"])
}


with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        best_result,
        f,
        indent=4
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SEQ64 THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)

print("\nThreshold results:")
print(
    results_df.to_string(
        index=False
    )
)

print("\n" + "-" * 70)
print("BEST VALIDATION THRESHOLD")
print("-" * 70)

print(
    f"Threshold : "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy  : "
    f"{best_row['accuracy'] * 100:.2f}%"
)

print(
    f"Precision : "
    f"{best_row['precision'] * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{best_row['recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{best_row['f1'] * 100:.2f}%"
)

print(
    f"TN={int(best_row['TN'])}, "
    f"FP={int(best_row['FP'])}, "
    f"FN={int(best_row['FN'])}, "
    f"TP={int(best_row['TP'])}"
)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_JSON)