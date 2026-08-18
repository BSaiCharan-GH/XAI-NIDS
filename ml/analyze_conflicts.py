from pathlib import Path
import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns


INPUT_FILE = Path(
    "datasets/processed/ids_dataset_with_source.csv"
)


def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)
    df.columns = normalize_columns(df.columns)

    df["feature_hash"] = pd.util.hash_pandas_object(
        df[FEATURE_COLUMNS],
        index=False
    )

    label_counts = (
        df.groupby("feature_hash")["Label"]
        .nunique()
    )

    conflicting_hashes = label_counts[
        label_counts > 1
    ].index

    conflicting_rows = df[
        df["feature_hash"].isin(conflicting_hashes)
    ]

    print(f"Total rows: {len(df):,}")
    print(f"Conflicting feature vectors: {len(conflicting_hashes):,}")
    print(f"Rows affected: {len(conflicting_rows):,}")

    print("\nLabels involved:")
    print(
        conflicting_rows["Label"]
        .value_counts()
    )

    print("\nConflicting rows by source file:")
    print(
        conflicting_rows["Source_File"]
        .value_counts()
    )


if __name__ == "__main__":
    main()