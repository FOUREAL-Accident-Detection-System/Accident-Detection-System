import cv2
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = (
    PROJECT_ROOT
    / "dataset_split_clean"
    / "train"
    / "normal"
    / "WhatsApp Video 2026-09-09 at 11.23.22 AM.mp4"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "preprocessed_clean_seq32"
    / "train_10fps_160x160"
    / "normal"
)

TARGET_FPS = 10
IMAGE_SIZE = (160, 160)
SEQUENCE_LENGTH = 32
STRIDE = 32


def get_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps <= 0:
        cap.release()
        raise RuntimeError("Invalid original FPS.")

    frame_interval = original_fps / TARGET_FPS

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


def main():

    print("=" * 60)
    print("REBUILDING ONE CORRUPTED SEQ32 VIDEO")
    print("=" * 60)

    print(f"Video: {VIDEO_PATH}")

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Remove only this video's existing Seq32 files
    old_files = list(
        OUTPUT_DIR.glob(
            "WhatsApp Video 2026-09-09 at 11.23.22 AM_seq_*.npz"
        )
    )

    print(f"\nExisting files to replace: {len(old_files)}")

    for file in old_files:
        file.unlink()

    print("Old files removed.")

    # Read and sample video
    frames = get_video_frames(VIDEO_PATH)

    print(f"Sampled frames: {len(frames)}")

    if len(frames) < SEQUENCE_LENGTH:
        raise RuntimeError(
            "Video does not contain enough sampled frames "
            "for a 32-frame sequence."
        )

    sequence_count = 0

    for start in range(
        0,
        len(frames) - SEQUENCE_LENGTH + 1,
        STRIDE
    ):

        sequence = frames[
            start:start + SEQUENCE_LENGTH
        ]

        if len(sequence) != SEQUENCE_LENGTH:
            continue

        output_file = (
            OUTPUT_DIR
            / f"WhatsApp Video 2026-09-09 at 11.23.22 AM"
            f"_seq_{sequence_count:04d}.npz"
        )

        np.savez_compressed(
            output_file,
            frames=np.array(
                sequence,
                dtype=np.uint8
            )
        )

        sequence_count += 1

    print(f"\nNew Seq32 files created: {sequence_count}")

    print("=" * 60)
    print("REBUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()