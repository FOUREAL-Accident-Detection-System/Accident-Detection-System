import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset_split_clean"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessed_clean_seq64"
)

METADATA_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "preprocessing_clean_10fps_160x160_seq64.csv"
)

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_FPS = 10
IMAGE_SIZE = (160, 160)

SEQUENCE_LENGTH = 64
STRIDE = 64

# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(video_path, output_dir, split, class_name):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        cap.release()
        print(f"WARNING: Invalid FPS: {video_path}")
        return []

    frame_interval = original_fps / TARGET_FPS

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

    metadata_rows = []

    video_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    # ========================================================
    # CREATE SEQUENCES
    # ========================================================

    sequence_index = 0
    start = 0

    while start + SEQUENCE_LENGTH <= len(frames):

        sequence = np.array(
            frames[start:start + SEQUENCE_LENGTH],
            dtype=np.uint8
        )

        output_filename = (
            f"{video_name}_seq_{sequence_index:04d}.npz"
        )

        output_path = os.path.join(
            output_dir,
            output_filename
        )

        np.savez_compressed(
            output_path,
            frames=sequence
        )

        metadata_rows.append({
            "split": split,
            "class": class_name,
            "label": 1 if class_name == "accident" else 0,
            "source_video": os.path.basename(video_path),
            "sequence_index": sequence_index,
            "sequence_file": os.path.relpath(
                output_path,
                PROJECT_ROOT
            ),
            "fps": TARGET_FPS,
            "image_width": IMAGE_SIZE[0],
            "image_height": IMAGE_SIZE[1],
            "sequence_length": SEQUENCE_LENGTH,
            "stride": STRIDE
        })

        sequence_index += 1
        start += STRIDE

    return metadata_rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SEQ64 PREPROCESSING")
    print("=" * 70)

    print(f"Target FPS          : {TARGET_FPS}")
    print(f"Image size          : {IMAGE_SIZE}")
    print(f"Sequence length     : {SEQUENCE_LENGTH}")
    print(f"Sequence stride     : {STRIDE}")

    all_metadata = []

    split_names = [
        "train",
        "validation",
        "test"
    ]

    class_names = [
        "accident",
        "normal"
    ]

    for split in split_names:

        for class_name in class_names:

            input_dir = os.path.join(
                INPUT_DIR,
                split,
                class_name
            )

            output_dir = os.path.join(
                OUTPUT_DIR,
                f"{split}_10fps_160x160",
                class_name
            )

            os.makedirs(
                output_dir,
                exist_ok=True
            )

            if not os.path.exists(input_dir):
                print(
                    f"WARNING: Missing directory: {input_dir}"
                )
                continue

            video_files = [
                f for f in os.listdir(input_dir)
                if f.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                )
            ]

            print(
                f"\nProcessing {split}/{class_name}: "
                f"{len(video_files)} videos"
            )

            for video_file in tqdm(video_files):

                video_path = os.path.join(
                    input_dir,
                    video_file
                )

                rows = process_video(
                    video_path,
                    output_dir,
                    split,
                    class_name
                )

                all_metadata.extend(rows)

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata_df = pd.DataFrame(all_metadata)

    metadata_df.to_csv(
        METADATA_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("SEQ64 PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Videos processed    : "
        f"{metadata_df['source_video'].nunique()}"
    )

    print(
        f"Total sequences     : "
        f"{len(metadata_df)}"
    )

    print(
        f"Image size          : "
        f"{IMAGE_SIZE}"
    )

    print(
        f"Sequence length     : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Sequence stride     : "
        f"{STRIDE}"
    )

    print("\nSequences by split:")

    print(
        metadata_df.groupby(
            ["split", "class"]
        ).size()
    )

    print("\nMetadata saved to:")
    print(METADATA_PATH)


if __name__ == "__main__":
    main()