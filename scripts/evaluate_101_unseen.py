import os
import cv2
import json
import time
import gc
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

MODEL_PATH = r"models\accident_detection_clean_10fps_160x160_seq32_best.keras"

UNSEEN_ROOT = r"unseen_videos"

FPS = 10
IMAGE_SIZE = (160, 160)
SEQUENCE_LENGTH = 32
STRIDE = 32
THRESHOLD = 0.45

RESULTS_DIR = r"results"

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "unseen_100_predictions_seq32.csv"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "metrics_unseen_100_seq32.json"
)

CONFUSION_FILE = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_unseen_100_seq32.csv"
)


# ============================================================
# FOLDER LABELS
# ============================================================

FOLDER_MAPPING = {
    "positive_videos": "ACCIDENT",
    "negative _videos": "NORMAL"
}


# ============================================================
# VIDEO EXTENSIONS
# ============================================================

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".wmv"
)


# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Seq32 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Seq32 model loaded successfully.")


# ============================================================
# DATASET CHECK
# ============================================================

all_videos = []

for folder_name, label in FOLDER_MAPPING.items():

    folder_path = os.path.join(
        UNSEEN_ROOT,
        folder_name
    )

    if not os.path.isdir(folder_path):

        print(
            f"WARNING: Folder not found: {folder_path}"
        )

        continue

    for filename in os.listdir(folder_path):

        if filename.lower().endswith(VIDEO_EXTENSIONS):

            video_path = os.path.join(
                folder_path,
                filename
            )

            all_videos.append(
                {
                    "path": video_path,
                    "filename": filename,
                    "true_label": label
                }
            )


accident_count = sum(
    1
    for v in all_videos
    if v["true_label"] == "ACCIDENT"
)

normal_count = sum(
    1
    for v in all_videos
    if v["true_label"] == "NORMAL"
)


print("\n" + "=" * 70)
print("DATASET CHECK")
print("=" * 70)

print(
    f"Accident videos : {accident_count}"
)

print(
    f"Normal videos   : {normal_count}"
)

print(
    f"Total videos    : {len(all_videos)}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if len(all_videos) != 100:

    print(
        "\nWARNING: Expected exactly 100 unseen videos."
    )

    print(
        f"Found {len(all_videos)} videos."
    )

    print(
        "Evaluation will NOT start."
    )

    raise SystemExit


if accident_count != 50 or normal_count != 50:

    print(
        "\nWARNING: Expected 50 accident and 50 normal videos."
    )

    print(
        "Evaluation will NOT start."
    )

    raise SystemExit


print(
    "\n100-video dataset verified: 50 ACCIDENT + 50 NORMAL."
)

print(
    "These videos will only be evaluated, not used for training."
)


# ============================================================
# MEMORY-SAFE VIDEO PROCESSING
# ============================================================

def extract_sequences(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        return [], 0, False

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if original_fps <= 0:

        original_fps = FPS

    frame_interval = (
        original_fps / FPS
    )

    frames = []

    frame_index = 0
    next_sample_frame = 0.0

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        if frame_index >= next_sample_frame:

            # Resize immediately to save memory.

            frame = cv2.resize(
                frame,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            frames.append(frame)

            next_sample_frame += frame_interval

        frame_index += 1

    cap.release()

    gc.collect()

    if len(frames) == 0:

        return [], 0, False

    padded = False

    # --------------------------------------------------------
    # Pad short videos to 32 frames
    # --------------------------------------------------------

    if len(frames) < SEQUENCE_LENGTH:

        padded = True

        last_frame = frames[-1]

        while len(frames) < SEQUENCE_LENGTH:

            frames.append(
                last_frame.copy()
            )

    sequences = []

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        sequence = np.array(
            frames[
                start:start + SEQUENCE_LENGTH
            ],
            dtype=np.float32
        )

        sequence = sequence / 255.0

        sequences.append(
            sequence
        )

    del frames

    gc.collect()

    return (
        sequences,
        len(sequences),
        padded
    )


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\n" + "=" * 70)
print("STARTING FINAL 100-VIDEO SEQ32 EVALUATION")
print("=" * 70)

print(
    "\nConfiguration:"
)

print(
    f"FPS              : {FPS}"
)

print(
    f"Resolution       : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
)

print(
    f"Sequence length  : {SEQUENCE_LENGTH}"
)

print(
    f"Stride           : {STRIDE}"
)

print(
    f"Threshold        : {THRESHOLD}"
)

print(
    "Aggregation       : Maximum accident probability"
)


for index, video_info in enumerate(
    all_videos,
    start=1
):

    video_path = video_info["path"]
    filename = video_info["filename"]
    true_label = video_info["true_label"]

    print(
        f"\n[{index}/{len(all_videos)}] {filename}"
    )

    print(
        f"True label: {true_label}"
    )

    start_time = time.time()

    try:

        sequences, sequence_count, padded = (
            extract_sequences(video_path)
        )

        if sequence_count == 0:

            print(
                "WARNING: No usable frames found."
            )

            continue


        # ----------------------------------------------------
        # Predict in small batches
        # ----------------------------------------------------

        predictions = []

        for batch_start in range(
            0,
            len(sequences),
            4
        ):

            batch = np.array(
                sequences[
                    batch_start:batch_start + 4
                ],
                dtype=np.float32
            )

            batch_predictions = model.predict(
                batch,
                verbose=0
            ).reshape(-1)

            predictions.extend(
                batch_predictions.tolist()
            )

            del batch
            del batch_predictions

            gc.collect()


        # ----------------------------------------------------
        # VIDEO-LEVEL AGGREGATION
        # ----------------------------------------------------

        max_accident_probability = float(
            np.max(predictions)
        )

        mean_accident_probability = float(
            np.mean(predictions)
        )


        # ----------------------------------------------------
        # FINAL CLASSIFICATION
        # ----------------------------------------------------

        if (
            max_accident_probability
            >= THRESHOLD
        ):

            predicted_label = "ACCIDENT"

            confidence = (
                max_accident_probability
            )

        else:

            predicted_label = "NORMAL"

            confidence = (
                1.0
                - max_accident_probability
            )


        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        max_index = int(
            np.argmax(predictions)
        )

        timestamp_seconds = (
            max_index
            * STRIDE
            / FPS
        )


        # ----------------------------------------------------
        # INFERENCE TIME
        # ----------------------------------------------------

        inference_time = (
            time.time()
            - start_time
        )


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        print(
            f"Prediction       : {predicted_label}"
        )

        print(
            f"Confidence       : {confidence:.4f}"
        )

        print(
            f"Max accident prob: "
            f"{max_accident_probability:.4f}"
        )

        print(
            f"Mean accident prob: "
            f"{mean_accident_probability:.4f}"
        )

        print(
            f"Sequences        : {sequence_count}"
        )

        print(
            f"Timestamp        : "
            f"{timestamp_seconds:.2f} sec"
        )

        print(
            f"Inference time   : "
            f"{inference_time:.3f} sec"
        )


        if padded:

            print(
                "Note             : "
                "Short video padded to 32 frames"
            )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append(
            {
                "filename":
                    filename,

                "true_label":
                    true_label,

                "predicted_label":
                    predicted_label,

                "confidence":
                    confidence,

                "max_accident_probability":
                    max_accident_probability,

                "mean_accident_probability":
                    mean_accident_probability,

                "timestamp_seconds":
                    timestamp_seconds,

                "sequence_count":
                    sequence_count,

                "padded":
                    padded,

                "inference_time_seconds":
                    inference_time
            }
        )


        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        del sequences
        del predictions

        gc.collect()


    except Exception as e:

        print(
            f"ERROR processing {filename}: {e}"
        )

        gc.collect()

        continue


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    print(
        "\nNo videos were successfully evaluated."
    )

    raise SystemExit


results_df = pd.DataFrame(
    results
)


# ============================================================
# LABEL CONVERSION
# ============================================================

y_true = (
    results_df["true_label"]
    == "ACCIDENT"
).astype(int)

y_pred = (
    results_df["predicted_label"]
    == "ACCIDENT"
).astype(int)


# ============================================================
# METRICS
# ============================================================

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


# ============================================================
# CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
).ravel()


# ============================================================
# OVERALL STATISTICS
# ============================================================

average_accident_probability = float(
    results_df[
        "max_accident_probability"
    ].mean()
)

average_confidence = float(
    results_df[
        "confidence"
    ].mean()
)

average_inference_time = float(
    results_df[
        "inference_time_seconds"
    ].mean()
)


# ============================================================
# SAVE PER-VIDEO RESULTS
# ============================================================

results_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "videos_found":
        len(all_videos),

    "videos_evaluated":
        len(results_df),

    "accident_videos_found":
        accident_count,

    "normal_videos_found":
        normal_count,

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp),

    "average_max_accident_probability":
        average_accident_probability,

    "average_confidence":
        average_confidence,

    "average_inference_time_seconds":
        average_inference_time,

    "model":
        MODEL_PATH,

    "fps":
        FPS,

    "image_size":
        "160x160",

    "sequence_length":
        SEQUENCE_LENGTH,

    "stride":
        STRIDE,

    "threshold":
        THRESHOLD,

    "aggregation":
        "maximum accident probability",

    "short_video_padding":
        True
}


with open(
    METRICS_FILE,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    [
        [tn, fp],
        [fn, tp]
    ],
    index=[
        "Actual NORMAL",
        "Actual ACCIDENT"
    ],
    columns=[
        "Predicted NORMAL",
        "Predicted ACCIDENT"
    ]
)

cm_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)
print("FINAL 100-VIDEO SEQ32 EVALUATION RESULTS")
print("=" * 70)

print(
    f"Videos found       : {len(all_videos)}"
)

print(
    f"Videos evaluated   : {len(results_df)}"
)

print(
    f"Average probability: "
    f"{average_accident_probability:.4f}"
)

print(
    f"Average confidence : "
    f"{average_confidence:.4f}"
)

print(
    f"Average inference  : "
    f"{average_inference_time:.3f} sec"
)

print("\nMETRICS")

print(
    f"Accuracy  : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{f1 * 100:.2f}%"
)

print("\nCONFUSION MATRIX")

print(
    f"TN = {tn}"
)

print(
    f"FP = {fp}"
)

print(
    f"FN = {fn}"
)

print(
    f"TP = {tp}"
)

print("\nFILES SAVED")

print(
    PREDICTIONS_FILE
)

print(
    METRICS_FILE
)

print(
    CONFUSION_FILE
)

print("=" * 70)

print(
    "SEQ32 100-VIDEO EVALUATION COMPLETE"
)

print("=" * 70)