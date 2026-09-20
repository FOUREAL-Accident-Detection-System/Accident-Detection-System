import os
import cv2
import json
import time
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

MODEL_PATH = "models/accident_detection_10fps_160x160_best.keras"
VIDEO_FOLDER = "unseen_videos"

RESULTS_FOLDER = "results"

FPS = 10
IMAGE_SIZE = (160, 160)
SEQUENCE_LENGTH = 16
STRIDE = 16

THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("UNSEEN VIDEO EVALUATION")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# FIND VIDEOS
# ============================================================

videos = []

for filename in os.listdir(VIDEO_FOLDER):

    if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):

        full_path = os.path.join(VIDEO_FOLDER, filename)

        if os.path.isfile(full_path):
            videos.append(filename)

videos.sort()

print(f"\nVideos found: {len(videos)}")

for video in videos:
    print(" -", video)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        return None

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = 30

    frames = []

    frame_index = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_time = frame_index / original_fps

        # Sample approximately at 10 FPS
        if frame_index % max(1, int(round(original_fps / FPS))) == 0:

            frame = cv2.resize(frame, IMAGE_SIZE)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        frame_index += 1

    cap.release()

    # --------------------------------------------------------
    # CREATE SEQUENCES
    # --------------------------------------------------------

    sequences = []
    timestamps = []

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        sequence = frames[
            start:start + SEQUENCE_LENGTH
        ]

        sequences.append(sequence)

        timestamp = start / FPS

        timestamps.append(timestamp)

    if len(sequences) == 0:

        return {
            "probabilities": [],
            "timestamps": []
        }

    X = np.array(sequences, dtype=np.float32) / 255.0

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    start_time = time.time()

    probabilities = model.predict(
        X,
        verbose=0
    ).flatten()

    inference_time = time.time() - start_time

    return {
        "probabilities": probabilities,
        "timestamps": timestamps,
        "inference_time": inference_time
    }


# ============================================================
# EVALUATE ALL VIDEOS
# ============================================================

results = []

for i, video in enumerate(videos, start=1):

    print("\n" + "-" * 60)
    print(f"Processing {i}/{len(videos)}: {video}")

    video_path = os.path.join(
        VIDEO_FOLDER,
        video
    )

    output = process_video(video_path)

    if output is None:
        continue

    probabilities = output["probabilities"]
    timestamps = output["timestamps"]

    if len(probabilities) == 0:

        print("Not enough frames for a sequence.")

        continue

    # --------------------------------------------------------
    # CURRENT AGGREGATION: MAX PROBABILITY
    # --------------------------------------------------------

    max_index = int(
        np.argmax(probabilities)
    )

    max_probability = float(
        probabilities[max_index]
    )

    timestamp = float(
        timestamps[max_index]
    )

    if max_probability >= THRESHOLD:

        prediction = "ACCIDENT"

        confidence = max_probability

    else:

        prediction = "NORMAL"

        confidence = 1 - max_probability

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence:.4f}"
    )

    print(
        f"Accident probability : {max_probability:.4f}"
    )

    print(
        f"Timestamp : {timestamp:.2f} sec"
    )

    print(
        f"Sequences : {len(probabilities)}"
    )

    print(
        f"Inference time : "
        f"{output['inference_time']:.3f} sec"
    )

    # --------------------------------------------------------
    # PRINT ALL SEQUENCE PROBABILITIES
    # --------------------------------------------------------

    print("\nSequence probabilities:")

    for j, probability in enumerate(probabilities):

        print(
            f"Sequence {j + 1:03d} | "
            f"Time {timestamps[j]:6.2f} sec | "
            f"Accident probability = "
            f"{probability:.4f}"
        )

    results.append({

        "video": video,

        "prediction": prediction,

        "confidence": confidence,

        "accident_probability": max_probability,

        "timestamp_sec": timestamp,

        "sequence_count": len(probabilities),

        "inference_time_sec":
            output["inference_time"]

    })


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    RESULTS_FOLDER,
    exist_ok=True
)

results_df = pd.DataFrame(results)

csv_path = os.path.join(
    RESULTS_FOLDER,
    "unseen_video_predictions_retry.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)

print("\n" + "=" * 60)

print(
    "Results saved to:"
)

print(csv_path)

print("=" * 60)