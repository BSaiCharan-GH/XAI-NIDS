from pathlib import Path
import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns


INPUT_FILE = Path(
    "datasets/processed/ids_dataset_with_source.csv"
)

OUTPUT_FILE = Path(
    "datasets/processed/ids_dataset_clean.csv"
)


def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)
    df.columns = normalize_columns(df.columns)

    print(f"Original rows: {len(df):,}")

    # Identify feature vectors that occur with multiple labels.
    label_counts = (
        df.groupby(FEATURE_COLUMNS)["Label"]
        .nunique()
    )

    conflicting_features = label_counts[
        label_counts > 1
    ].index

    # Remove every row belonging to a conflicting feature vector.
    if len(conflicting_features) > 0:
        conflict_index = pd.MultiIndex.from_frame(
            df[FEATURE_COLUMNS]
        )

        conflicting_mask = conflict_index.isin(
            conflicting_features
        )

        df = df.loc[~conflicting_mask].copy()

    print(
        f"Rows after removing conflicting labels: "
        f"{len(df):,}"
    )

    # Remove exact duplicate feature rows globally.
    before = len(df)

    df = df.drop_duplicates(
        subset=FEATURE_COLUMNS,
        keep="first"
    )

    duplicates_removed = before - len(df)

    print(
        f"Duplicate rows removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Final clean rows: "
        f"{len(df):,}"
    )

    print("\nClass distribution:")
    print(df["Label"].value_counts())

    print("\nSource distribution:")
    print(df["Source_File"].value_counts())

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()