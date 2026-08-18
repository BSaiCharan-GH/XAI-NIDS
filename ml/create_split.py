from pathlib import Path

import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns


INPUT_FILE = Path(
    "datasets/processed/ids_dataset_clean.csv"
)

TRAIN_FILE = Path(
    "datasets/processed/train_v2.csv"
)

TEST_FILE = Path(
    "datasets/processed/test_v2.csv"
)

TRAIN_RATIO = 0.80


def split_source_class(group):
    group = group.sort_index()

    split_point = int(len(group) * TRAIN_RATIO)

    return (
        group.iloc[:split_point],
        group.iloc[split_point:]
    )


def main():
    print("Loading clean dataset...")

    df = pd.read_csv(INPUT_FILE)

    df.columns = normalize_columns(df.columns)

    print(f"Total rows: {len(df):,}")

    train_parts = []
    test_parts = []

    print("\nCreating source-aware split...")

    for (source, label), group in df.groupby(
        ["Source_File", "Label"],
        sort=False
    ):
        train_part, test_part = split_source_class(group)

        train_parts.append(train_part)
        test_parts.append(test_part)

        print(
            f"{source} | {label}: "
            f"{len(train_part):,} train / "
            f"{len(test_part):,} test"
        )

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    print("\nTraining distribution:")
    print(train_df["Label"].value_counts())

    print("\nTesting distribution:")
    print(test_df["Label"].value_counts())

    print("\nSaving datasets...")

    train_df.to_csv(
        TRAIN_FILE,
        index=False
    )

    test_df.to_csv(
        TEST_FILE,
        index=False
    )

    print(f"\nTraining file: {TRAIN_FILE}")
    print(f"Testing file: {TEST_FILE}")

    print("\nSplit complete.")


if __name__ == "__main__":
    main()