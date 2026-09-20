import os
import cv2
import numpy as np
import pandas as pd
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "dataset_split_clean"
OUTPUT_FILE = "results/source_overlap_audit.csv"

# Number of frames sampled from each video
SAMPLES_PER_VIDEO = 12

# pHash similarity threshold
# Smaller = more visually similar
PHASH_THRESHOLD = 12

# Minimum number of similar frame pairs required
MIN_MATCHES = 2

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm")


# ============================================================
# pHASH FUNCTION
# ============================================================

def compute_phash(frame):
    """
    Calculate a perceptual hash for a video frame.
    Similar-looking frames should have small Hamming distance.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize for DCT
    resized = cv2.resize(gray, (32, 32))

    # Convert to float for DCT
    resized = np.float32(resized)

    # Discrete Cosine Transform
    dct = cv2.dct(resized)

    # Use top-left 8x8 low-frequency components
    dct_low = dct[:8, :8]

    # Remove DC component
    values = dct_low.flatten()[1:]

    median = np.median(values)

    bits = values > median

    return bits


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)


# ============================================================
# SAMPLE FRAMES FROM VIDEO
# ============================================================

def get_video_hashes(video_path):
    """
    Sample frames uniformly from the video and calculate pHashes.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return [], None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total_frames <= 0:
        cap.release()
        return [], fps, width, height

    # Select frame positions uniformly
    frame_positions = np.linspace(
        0,
        total_frames - 1,
        min(SAMPLES_PER_VIDEO, total_frames),
        dtype=int
    )

    hashes = []

    for frame_number in frame_positions:

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        phash = compute_phash(frame)

        hashes.append(phash)

    cap.release()

    return hashes, fps, width, height


# ============================================================
# COLLECT ALL VIDEOS
# ============================================================

def collect_videos():

    videos = []

    # Check train / validation / test
    for split in ["train", "validation", "test"]:

        split_path = os.path.join(DATASET_ROOT, split)

        if not os.path.exists(split_path):
            continue

        for class_name in ["accident", "normal"]:

            class_path = os.path.join(split_path, class_name)

            if not os.path.exists(class_path):
                continue

            for filename in os.listdir(class_path):

                if not filename.lower().endswith(VIDEO_EXTENSIONS):
                    continue

                video_path = os.path.join(class_path, filename)

                videos.append({
                    "path": video_path,
                    "filename": filename,
                    "split": split,
                    "class": class_name.upper()
                })

    return videos


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("SOURCE / NEAR-DUPLICATE VIDEO OVERLAP AUDIT")
    print("=" * 70)

    videos = collect_videos()

    print(f"\nVideos found: {len(videos)}")

    if len(videos) == 0:
        print("No videos found.")
        return

    # --------------------------------------------------------
    # Calculate hashes
    # --------------------------------------------------------

    video_data = []

    print("\nSampling frames and calculating perceptual hashes...\n")

    for index, video in enumerate(videos, start=1):

        print(
            f"[{index}/{len(videos)}] "
            f"{video['split']} | "
            f"{video['class']} | "
            f"{video['filename']}"
        )

        hashes, fps, width, height = get_video_hashes(video["path"])

        video_data.append({
            **video,
            "hashes": hashes,
            "fps": fps,
            "width": width,
            "height": height
        })

    # --------------------------------------------------------
    # Compare every video with every other video
    # --------------------------------------------------------

    print("\nComparing videos for visual overlap...\n")

    results = []

    total_pairs = len(video_data) * (len(video_data) - 1) // 2
    pair_number = 0

    for video1, video2 in combinations(video_data, 2):

        pair_number += 1

        if pair_number % 500 == 0:
            print(f"Compared {pair_number}/{total_pairs} pairs...")

        hashes1 = video1["hashes"]
        hashes2 = video2["hashes"]

        if len(hashes1) == 0 or len(hashes2) == 0:
            continue

        distances = []

        for h1 in hashes1:
            for h2 in hashes2:
                distance = hamming_distance(h1, h2)
                distances.append(distance)

        distances = np.array(distances)

        min_distance = int(np.min(distances))

        matching_pairs = int(
            np.sum(distances <= PHASH_THRESHOLD)
        )

        # Keep only potentially related videos
        if matching_pairs >= MIN_MATCHES:

            same_class = video1["class"] == video2["class"]
            same_split = video1["split"] == video2["split"]

            if not same_class:
                relationship = "CROSS_CLASS_OVERLAP"
            elif not same_split:
                relationship = "CROSS_SPLIT_OVERLAP"
            else:
                relationship = "SAME_SPLIT_SIMILAR"

            results.append({

                "video_1": video1["filename"],
                "class_1": video1["class"],
                "split_1": video1["split"],

                "video_2": video2["filename"],
                "class_2": video2["class"],
                "split_2": video2["split"],

                "min_phash_distance": min_distance,
                "matching_frame_pairs": matching_pairs,

                "same_class": same_class,
                "same_split": same_split,

                "relationship": relationship
            })

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:

        results_df = results_df.sort_values(
            by=["min_phash_distance", "matching_frame_pairs"],
            ascending=[True, False]
        )

    results_df.to_csv(OUTPUT_FILE, index=False)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(f"\nPotential overlap pairs: {len(results_df)}")

    if len(results_df) > 0:

        cross_class = results_df[
            results_df["relationship"] == "CROSS_CLASS_OVERLAP"
        ]

        cross_split = results_df[
            results_df["relationship"] == "CROSS_SPLIT_OVERLAP"
        ]

        print(f"Cross-class overlap : {len(cross_class)}")
        print(f"Cross-split overlap : {len(cross_split)}")

        print("\nPotential overlaps:")
        print(
            results_df[
                [
                    "video_1",
                    "class_1",
                    "split_1",
                    "video_2",
                    "class_2",
                    "split_2",
                    "min_phash_distance",
                    "matching_frame_pairs",
                    "relationship"
                ]
            ].to_string(index=False)
        )

    else:

        print("\nNo strong visual overlap candidates were found.")

    print(f"\nReport saved to:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("This is a candidate overlap audit, not proof of source-video identity.")
    print("Potential matches should be manually verified before changing the dataset.")


if __name__ == "__main__":
    main()