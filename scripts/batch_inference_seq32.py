import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.environ.get(
    "ADP_MODEL_PATH",
    "models/accident_detection_clean_10fps_160x160_seq32_best.keras"
)

INPUT_FILE = os.environ.get(
    "ADP_INPUT_FILE",
    "input_videos.txt"
)

OUTPUT_FILE = os.environ.get(
    "ADP_OUTPUT_FILE",
    "results/batch_inference_results.csv"
)

TARGET_FPS = int(
    os.environ.get("ADP_TARGET_FPS", "10")
)

IMAGE_WIDTH = int(
    os.environ.get("ADP_IMAGE_WIDTH", "160")
)

IMAGE_HEIGHT = int(
    os.environ.get("ADP_IMAGE_HEIGHT", "160")
)

IMAGE_SIZE = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

SEQUENCE_LENGTH = int(
    os.environ.get("ADP_SEQUENCE_LENGTH", "32")
)

STRIDE = int(
    os.environ.get("ADP_STRIDE", "32")
)

CONFIDENCE_THRESHOLD = float(
    os.environ.get("ADP_THRESHOLD", "0.45")
)

BATCH_SIZE = int(
    os.environ.get("ADP_BATCH_SIZE", "4")
)

LOG_DIRECTORY = os.environ.get(
    "ADP_LOG_DIRECTORY",
    "automation_results/logs"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("BATCH INFERENCE")
print("=" * 60)

print("\nConfiguration:")
print(f"Model              : {MODEL_PATH}")
print(f"Input file         : {INPUT_FILE}")
print(f"Output file        : {OUTPUT_FILE}")
print(f"Log directory      : {LOG_DIRECTORY}")
print(f"Target FPS         : {TARGET_FPS}")
print(f"Image size         : {IMAGE_SIZE}")
print(f"Sequence length    : {SEQUENCE_LENGTH}")
print(f"Stride             : {STRIDE}")
print(f"Threshold          : {CONFIDENCE_THRESHOLD}")
print(f"Batch size         : {BATCH_SIZE}")


# ============================================================
# VALIDATE MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# READ VIDEO LIST
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    video_paths = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"\nVideos listed: {len(video_paths)}"
)


# ============================================================
# CREATE RESULTS FOLDERS
# ============================================================

output_directory = os.path.dirname(
    OUTPUT_FILE
)

if output_directory:
    os.makedirs(
        output_directory,
        exist_ok=True
    )

os.makedirs(
    LOG_DIRECTORY,
    exist_ok=True
)


# ============================================================
# PROCESS VIDEOS
# ============================================================

results = []

for index, video_path in enumerate(
    video_paths,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"Processing video "
        f"{index}/{len(video_paths)}"
    )

    print(
        f"Video: {video_path}"
    )

    if not os.path.exists(video_path):

        print(
            "WARNING: Video file not found."
        )

        continue

    start_time = time.time()

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        print(
            "WARNING: Could not open video."
        )

        continue

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if original_fps <= 0:

        original_fps = TARGET_FPS

    frames = []

    frame_index = 0

    next_sample_time = 0.0


    # ========================================================
    # FRAME SAMPLING
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        current_time = (
            frame_index /
            original_fps
        )

        if (
            current_time >=
            next_sample_time
        ):

            frame = cv2.resize(
                frame,
                IMAGE_SIZE
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                frame
            )

            next_sample_time += (
                1.0 / TARGET_FPS
            )

        frame_index += 1

    cap.release()


    # ========================================================
    # CHECK FRAMES
    # ========================================================

    if len(frames) < SEQUENCE_LENGTH:

        print(
            f"Not enough frames: "
            f"{len(frames)}"
        )

        continue


    frames = np.array(
        frames,
        dtype=np.uint8
    )


    # ========================================================
    # CREATE SEQUENCES
    # ========================================================

    sequences = []

    sequence_times = []

    for start in range(
        0,
        len(frames) -
        SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        sequence = frames[
            start:
            start + SEQUENCE_LENGTH
        ]

        sequences.append(
            sequence
        )

        sequence_times.append(
            start / TARGET_FPS
        )


    sequences = np.array(
        sequences,
        dtype=np.float32
    ) / 255.0


    print(
        f"Frames sampled: "
        f"{len(frames)}"
    )

    print(
        f"Sequences created: "
        f"{len(sequences)}"
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    predictions = model.predict(
        sequences,
        batch_size=BATCH_SIZE,
        verbose=0
    ).flatten()


    # ========================================================
    # MAX PROBABILITY AGGREGATION
    # ========================================================

    max_index = int(
        np.argmax(predictions)
    )

    max_accident_probability = float(
        predictions[max_index]
    )


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    if (
        max_accident_probability
        >= CONFIDENCE_THRESHOLD
    ):

        final_prediction = (
            "ACCIDENT"
        )

        confidence = (
            max_accident_probability
        )

        timestamp = (
            sequence_times[max_index]
        )

    else:

        final_prediction = (
            "NORMAL"
        )

        confidence = (
            1.0 -
            max_accident_probability
        )

        timestamp = ""


    # ========================================================
    # INFERENCE TIME
    # ========================================================

    inference_time = (
        time.time() -
        start_time
    )


    # ========================================================
    # STORE RESULT
    # ========================================================

    results.append({

        "video":
            video_path,

        "prediction":
            final_prediction,

        "confidence":
            round(
                confidence,
                4
            ),

        "accident_probability":
            round(
                max_accident_probability,
                4
            ),

        "timestamp_seconds":
            timestamp,

        "sequences":
            len(sequences),

        "inference_time_seconds":
            round(
                inference_time,
                3
            )
    })


    # ========================================================
    # SAVE INDIVIDUAL VIDEO LOG
    # ========================================================

    video_log = {

        "video":
            video_path,

        "prediction":
            final_prediction,

        "confidence":
            round(
                confidence,
                4
            ),

        "accident_probability":
            round(
                max_accident_probability,
                4
            ),

        "timestamp_seconds":
            timestamp,

        "sequences":
            len(sequences),

        "inference_time_seconds":
            round(
                inference_time,
                3
            ),

        "model":
            MODEL_PATH,

        "fps":
            TARGET_FPS,

        "image_size":
            list(IMAGE_SIZE),

        "sequence_length":
            SEQUENCE_LENGTH,

        "stride":
            STRIDE,

        "threshold":
            CONFIDENCE_THRESHOLD,

        "aggregation":
            "max_probability"
    }


    safe_name = os.path.basename(
        video_path
    )

    safe_name = os.path.splitext(
        safe_name
    )[0]

    log_file = os.path.join(
        LOG_DIRECTORY,
        safe_name + ".json"
    )


    with open(
        log_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            video_log,
            file,
            indent=4
        )


    print(
        f"Log saved: {log_file}"
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print(
        f"Prediction: "
        f"{final_prediction}"
    )

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )

    print(
        f"Accident probability: "
        f"{max_accident_probability:.4f}"
    )

    if timestamp != "":

        print(
            f"Timestamp: "
            f"{timestamp:.2f} seconds"
        )

    print(
        f"Inference time: "
        f"{inference_time:.3f} seconds"
    )


# ============================================================
# SAVE CSV
# ============================================================

if results:

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "BATCH INFERENCE COMPLETED"
    )

    print(
        "=" * 60
    )

    print(
        f"\nVideos processed: "
        f"{len(df)}"
    )

    print(
        f"Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nFinal scorecard:"
    )

    print(
        df.to_string(
            index=False
        )
    )

else:

    print(
        "\nNo videos were successfully processed."
    )

    sys.exit(1)
