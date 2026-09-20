import os
import pandas as pd

INPUT_FILE = "results/source_overlap_audit.csv"
OUTPUT_FILE = "results/confirmed_overlap_groups.csv"

# ============================================================
# CONFIRMED CROSS-SPLIT OVERLAPS
# These are the pairs we manually verified as SAME.
# ============================================================

CONFIRMED_PAIRS = [
    (
        "WhatsApp Video 2026-09-08 at 3.35.00 PM.mp4",
        "WhatsApp Video 2026-09-08 at 3.35.00 PM (2).mp4"
    ),
    (
        "WhatsApp Video 2026-09-08 at 3.22.36 PM.mp4",
        "WhatsApp Video 2026-09-09 at 10.21.43 PM (2).mp4"
    ),
    (
        "WhatsApp Video 2026-09-08 at 3.43.48 PM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.02 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.21.04 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.42 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.20.36 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.23.28 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.20.11 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.23.28 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.19.58 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.23.28 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-08 at 9.36.39 PM.mp4",
        "WhatsApp Video 2026-09-08 at 9.34.35 PM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.20.16 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.20.02 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.23.00 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.23.03 AM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-08 at 9.35.21 PM (1).mp4",
        "WhatsApp Video 2026-09-08 at 9.36.31 PM.mp4"
    ),
    (
        "WhatsApp Video 2026-09-09 at 11.22.12 AM.mp4",
        "WhatsApp Video 2026-09-09 at 11.22.04 AM.mp4"
    )
]


def build_groups(pairs):
    """
    Group connected videos together.

    If:
        A = B
        B = C

    then A, B, C belong to one overlap group.
    """

    groups = []

    for video_a, video_b in pairs:

        found_groups = []

        for i, group in enumerate(groups):

            if video_a in group or video_b in group:
                found_groups.append(i)

        if not found_groups:

            groups.append({
                video_a,
                video_b
            })

        else:

            # Merge all connected groups
            merged = {video_a, video_b}

            for index in reversed(found_groups):
                merged.update(groups[index])
                groups.pop(index)

            groups.append(merged)

    return groups


def main():

    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("CONFIRMED SOURCE-VIDEO OVERLAP GROUP REPORT")
    print("=" * 70)

    groups = build_groups(CONFIRMED_PAIRS)

    rows = []

    for group_number, group in enumerate(groups, start=1):

        print(f"\nOVERLAP GROUP {group_number}")
        print("-" * 50)

        for filename in sorted(group):

            print(filename)

            rows.append({
                "group_id": group_number,
                "filename": filename
            })

    df = pd.DataFrame(rows)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nConfirmed overlap pairs : {len(CONFIRMED_PAIRS)}")
    print(f"Unique overlap groups   : {len(groups)}")
    print(f"Videos involved         : {len(df)}")

    print(f"\nReport saved to:")
    print(OUTPUT_FILE)

    print("\nIMPORTANT:")
    print("No files were deleted or moved.")
    print("This report only groups the confirmed overlaps.")


if __name__ == "__main__":
    main()