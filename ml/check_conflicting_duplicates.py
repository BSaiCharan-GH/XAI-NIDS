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

    print(f"Rows: {len(df):,}")

    print("\nCreating feature fingerprints...")

    df["feature_hash"] = pd.util.hash_pandas_object(
        df[FEATURE_COLUMNS],
        index=False
    )

    label_counts = (
        df.groupby("feature_hash")["Label"]
        .nunique()
    )

    conflicting = label_counts[label_counts > 1]

    print(f"\nUnique feature vectors: {len(label_counts):,}")
    print(f"Conflicting feature vectors: {len(conflicting):,}")

    if len(conflicting) > 0:
        print("\nWARNING: Conflicting labels found.")

        hashes = conflicting.index[:10]

        examples = df[
            df["feature_hash"].isin(hashes)
        ][FEATURE_COLUMNS + ["Label", "Source_File"]]

        print("\nExamples:")
        print(examples.to_string(index=False))
    else:
        print("\nNo conflicting labels found.")
        print("Global duplicate removal is safe.")


if __name__ == "__main__":
    main()