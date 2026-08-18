from pathlib import Path
import pandas as pd

DATASET_DIR = Path("datasets/CICIDS2017")

files = sorted(DATASET_DIR.glob("*.csv"))

for file in files:
    print("=" * 80)
    print(file.name)

    df = pd.read_csv(file, nrows=5)

    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")

    for i, column in enumerate(df.columns):
        print(f"{i:3}: {column!r}")

    print()