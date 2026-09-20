from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "preprocessed_clean_seq32"

files = list(PREPROCESSED_DIR.rglob("*.npz"))

print("=" * 60)
print("CHECKING SEQ32 NPZ FILES")
print("=" * 60)
print(f"Total files found: {len(files)}")

bad_files = []

for i, path in enumerate(files, 1):
    try:
        with np.load(path) as data:
            frames = data["frames"]

            if frames.shape != (32, 160, 160, 3):
                raise ValueError(
                    f"Unexpected shape: {frames.shape}"
                )

        if i % 50 == 0 or i == len(files):
            print(f"Checked: {i}/{len(files)}")

    except Exception as e:
        bad_files.append((path, str(e)))

        print("\n❌ BAD FILE:")
        print(path)
        print("ERROR:", e)

print("\n" + "=" * 60)
print("CHECK COMPLETE")
print("=" * 60)
print(f"Bad files: {len(bad_files)}")

if bad_files:
    print("\nProblematic files:")
    for path, error in bad_files:
        print(f"\n{path}")
        print(error)
else:
    print("✅ All Seq32 NPZ files are valid.")