import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION — SEQ32 EXPERIMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use the verified leakage-controlled split
INPUT_DIR = PROJECT_ROOT / "dataset_split_clean"

# Separate output directory for Seq32
OUTPUT_DIR = PROJECT_ROOT / "preprocessed_clean_seq32"

TARGET_FPS = 10
IMAGE_SIZE = (160, 160)

# Seq32 experiment
SEQUENCE_LENGTH = 32
STRIDE = 32

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ============================================================
# FUNCTIONS
# ============================================================

def get_video_frames(video_path, target_fps):
    """
    Read a video and sample frames at the target FPS.
    Frames are resized immediately to reduce memory usage.
    Returns frames as uint8 RGB images.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return []

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        cap.release()
        return []

    frame_interval = original_fps / target_fps

    frames = []
    frame_number = 0
    next_sample_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_number >= next_sample_frame:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                IMAGE_SIZE
            )

            frames.append(frame)

            next_sample_frame += frame_interval

        frame_number += 1

    cap.release()

    return frames


def create_sequences(frames):
    """
    Convert sampled frames into fixed-length
    sequences of 32 frames.
    """

    sequences = []

    if len(frames) < SEQUENCE_LENGTH:
        return sequences

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        sequence = frames[
            start:start + SEQUENCE_LENGTH
        ]

        if len(sequence) == SEQUENCE_LENGTH:

            sequences.append(
                np.array(
                    sequence,
                    dtype=np.uint8
                )
            )

    return sequences


# ============================================================
# MAIN PREPROCESSING
# ============================================================

def main():

    print("=" * 60)
    print("SEQ32 DATASET PREPROCESSING")
    print("=" * 60)

    print(f"Target FPS       : {TARGET_FPS}")
    print(f"Image size       : {IMAGE_SIZE}")
    print(f"Sequence length  : {SEQUENCE_LENGTH}")
    print(f"Sequence stride  : {STRIDE}")
    print(f"Input directory  : {INPUT_DIR}")
    print(f"Output directory : {OUTPUT_DIR}")

    print("=" * 60)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    total_videos = 0
    total_sequences = 0

    # ========================================================
    # PROCESS TRAIN / VALIDATION / TEST
    # ========================================================

    for split in [
        "train",
        "validation",
        "test"
    ]:

        split_input = INPUT_DIR / split

        split_output = (
            OUTPUT_DIR /
            f"{split}_{TARGET_FPS}fps_"
            f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}"
        )

        split_output.mkdir(
            parents=True,
            exist_ok=True
        )

        for class_name in [
            "accident",
            "normal"
        ]:

            class_input = (
                split_input /
                class_name
            )

            class_output = (
                split_output /
                class_name
            )

            class_output.mkdir(
                parents=True,
                exist_ok=True
            )

            video_files = [
                p for p in class_input.iterdir()
                if p.is_file()
                and p.suffix.lower()
                in VIDEO_EXTENSIONS
            ]

            print(
                f"\n[{split.upper()} - "
                f"{class_name.upper()}]"
            )

            print(
                f"Videos found: "
                f"{len(video_files)}"
            )

            for video_path in tqdm(
                video_files
            ):

                total_videos += 1

                frames = get_video_frames(
                    video_path,
                    TARGET_FPS
                )

                sequences = create_sequences(
                    frames
                )

                label = (
                    1
                    if class_name == "accident"
                    else 0
                )

                for (
                    sequence_index,
                    sequence
                ) in enumerate(sequences):

                    output_file = (
                        class_output /
                        f"{video_path.stem}"
                        f"_seq_{sequence_index:04d}.npz"
                    )

                    np.savez_compressed(
                        output_file,
                        frames=sequence
                    )

                    records.append({
                        "split": split,
                        "class": class_name,
                        "label": label,
                        "source_video": video_path.name,
                        "sequence_index": sequence_index,
                        "sequence_file": str(
                            output_file.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                        "fps": TARGET_FPS,
                        "image_width": IMAGE_SIZE[0],
                        "image_height": IMAGE_SIZE[1],
                        "sequence_length": SEQUENCE_LENGTH,
                        "stride": STRIDE
                    })

                    total_sequences += 1

    # ========================================================
    # SAVE METADATA
    # ========================================================

    metadata = pd.DataFrame(records)

    metadata_file = (
        PROJECT_ROOT /
        "results" /
        f"preprocessing_clean_"
        f"{TARGET_FPS}fps_"
        f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_seq32.csv"
    )

    metadata_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata.to_csv(
        metadata_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("SEQ32 PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Target FPS          : {TARGET_FPS}"
    )

    print(
        f"Videos processed    : {total_videos}"
    )

    print(
        f"Total sequences     : {total_sequences}"
    )

    print(
        f"Image size          : {IMAGE_SIZE}"
    )

    print(
        f"Sequence length     : {SEQUENCE_LENGTH}"
    )

    print(
        f"Sequence stride     : {STRIDE}"
    )

    print("\nSequences by split:")

    if not metadata.empty:

        print(
            metadata
            .groupby(
                ["split", "class"]
            )
            .size()
            .to_string()
        )

    print("\nMetadata saved to:")
    print(metadata_file)

    print("\nProcessed data saved to:")
    print(OUTPUT_DIR)

    print("=" * 60)


if __name__ == "__main__":
    main()