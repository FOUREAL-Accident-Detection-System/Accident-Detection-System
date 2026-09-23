import os
import cv2
import gc
import time
import numpy as np
import tensorflow as tf
from scripts.textbee_sms import send_accident_alert


# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "accident_detection_clean_10fps_160x160_seq32_best.keras"

FPS = 10
IMAGE_SIZE = (160, 160)
SEQUENCE_LENGTH = 32
STRIDE = 32
THRESHOLD = 0.45


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading accident detection model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# EXTRACT SEQUENCES
# ============================================================

def extract_sequences(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        original_fps = FPS

    frame_interval = original_fps / FPS

    frames = []
    frame_index = 0
    next_sample_frame = 0.0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_index >= next_sample_frame:

            frame = cv2.resize(
                frame,
                IMAGE_SIZE,
                interpolation=cv2.INTER_AREA
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

            next_sample_frame += frame_interval

        frame_index += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError("No usable frames found in video.")

    padded = False

    # --------------------------------------------------------
    # Pad short videos to 32 frames
    # --------------------------------------------------------

    if len(frames) < SEQUENCE_LENGTH:

        padded = True

        last_frame = frames[-1]

        while len(frames) < SEQUENCE_LENGTH:
            frames.append(last_frame.copy())

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    sequences = []

    sequence_start_frames = []

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

        sequences.append(sequence)

        sequence_start_frames.append(start)

    del frames
    gc.collect()

    if len(sequences) == 0:
        raise ValueError(
            "Video is too short to create a sequence."
        )

    return (
        sequences,
        sequence_start_frames,
        padded
    )


# ============================================================
# MAIN INFERENCE FUNCTION
# ============================================================

def predict_video(video_path):

    start_time = time.time()

    # --------------------------------------------------------
    # Extract sequences
    # --------------------------------------------------------

    sequences, sequence_start_frames, padded = (
        extract_sequences(video_path)
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Video-level aggregation
    # --------------------------------------------------------

    max_accident_probability = float(
        np.max(predictions)
    )

    mean_accident_probability = float(
        np.mean(predictions)
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if max_accident_probability >= THRESHOLD:

        predicted_label = "ACCIDENT"

        confidence = max_accident_probability

    else:

        predicted_label = "NORMAL"

        confidence = 1.0 - max_accident_probability

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    max_index = int(
        np.argmax(predictions)
    )

    timestamp_seconds = (
        sequence_start_frames[max_index]
        / FPS
    )

    # --------------------------------------------------------
    # Inference time
    # --------------------------------------------------------

    inference_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # AUTOMATIC SMS ALERT
    # --------------------------------------------------------

    sms_sent = False

    if predicted_label == "ACCIDENT":

        sms_sent = send_accident_alert(
            timestamp=timestamp_seconds,
            confidence=confidence
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "prediction":
            predicted_label,

        "confidence":
            round(confidence, 4),

        "accident_probability":
            round(max_accident_probability, 4),

        "mean_accident_probability":
            round(mean_accident_probability, 4),

        "timestamp_seconds":
            round(timestamp_seconds, 2),

        "sequence_count":
            len(sequences),

        "inference_time_seconds":
            round(inference_time, 3),

        "padded":
            padded,

        "sms_sent":
            sms_sent
    }

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del sequences
    del predictions

    gc.collect()

    return result


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    TEST_VIDEO = r"C:\Users\Hamsaveni A R\Desktop\ADP stage 2\unseen_videos\positive_videos\v1.mov"

    print("\nRunning web inference test...")
    print(f"Video: {TEST_VIDEO}")

    result = predict_video(TEST_VIDEO)

    print("\n" + "=" * 60)
    print("AI INFERENCE RESULT")
    print("=" * 60)

    print(f"Prediction          : {result['prediction']}")
    print(f"Confidence          : {result['confidence']:.4f}")

    print(
        f"Accident probability: "
        f"{result['accident_probability']:.4f}"
    )

    print(
        f"Mean probability    : "
        f"{result['mean_accident_probability']:.4f}"
    )

    print(
        f"Timestamp           : "
        f"{result['timestamp_seconds']:.2f} sec"
    )

    print(
        f"Sequence count      : "
        f"{result['sequence_count']}"
    )

    print(
        f"Inference time      : "
        f"{result['inference_time_seconds']:.3f} sec"
    )

    print(f"Padded              : {result['padded']}")

    print(f"SMS sent            : {result['sms_sent']}")

    print("=" * 60)