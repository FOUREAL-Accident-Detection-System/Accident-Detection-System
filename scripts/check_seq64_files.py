import os
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SEQ64_DIR = os.path.join(
    PROJECT_ROOT,
    "preprocessed_clean_seq64"
)

EXPECTED_SEQUENCE_LENGTH = 64

bad_files = []
total_files = 0
checked_files = 0

print("=" * 70)
print("SEQ64 NPZ FILE INTEGRITY CHECK")
print("=" * 70)

for root, _, files in os.walk(SEQ64_DIR):

    for filename in files:

        if not filename.lower().endswith(".npz"):
            continue

        total_files += 1

        file_path = os.path.join(root, filename)

        try:
            with np.load(file_path) as data:

                if "frames" not in data:
                    raise ValueError(
                        "Missing 'frames' array"
                    )

                frames = data["frames"]

                if frames.shape[0] != EXPECTED_SEQUENCE_LENGTH:
                    raise ValueError(
                        f"Expected {EXPECTED_SEQUENCE_LENGTH} "
                        f"frames, found {frames.shape[0]}"
                    )

                if frames.ndim != 4:
                    raise ValueError(
                        f"Expected 4D array, found shape {frames.shape}"
                    )

        except Exception as e:

            bad_files.append({
                "file": file_path,
                "error": str(e)
            })

        checked_files += 1

        if checked_files % 50 == 0 or checked_files == total_files:
            print(
                f"Checked: {checked_files}/{total_files}"
            )

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)

print(f"Total files found: {total_files}")
print(f"Bad files: {len(bad_files)}")

if bad_files:

    print("\n❌ BAD FILES:")

    for item in bad_files:
        print(f"\nFile: {item['file']}")
        print(f"Error: {item['error']}")

else:

    print("\n✅ All Seq64 NPZ files are valid.")