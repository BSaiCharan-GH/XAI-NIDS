from pathlib import Path
import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns


TRAIN_FILE = Path("datasets/processed/train_v2.csv")
TEST_FILE = Path("datasets/processed/test_v2.csv")


def main():
    print("Loading training data...")
    train = pd.read_csv(TRAIN_FILE)

    print("Loading testing data...")
    test = pd.read_csv(TEST_FILE)

    train.columns = normalize_columns(train.columns)
    test.columns = normalize_columns(test.columns)

    print(f"\nTraining rows: {len(train):,}")
    print(f"Testing rows: {len(test):,}")

    print("\nCreating feature fingerprints...")

    train_hashes = pd.util.hash_pandas_object(
        train[FEATURE_COLUMNS],
        index=False
    )

    test_hashes = pd.util.hash_pandas_object(
        test[FEATURE_COLUMNS],
        index=False
    )

    train_hash_set = set(train_hashes)

    overlap = sum(
        value in train_hash_set
        for value in test_hashes
    )

    percentage = (overlap / len(test)) * 100

    print(f"\nExact feature-row overlap: {overlap:,}")
    print(f"Test rows overlapping training: {percentage:.4f}%")

    print("\nTraining labels:")
    print(train["Label"].value_counts())

    print("\nTesting labels:")
    print(test["Label"].value_counts())

    if overlap == 0:
        print("\nPASS: No exact feature overlap detected.")
    else:
        print("\nWARNING: Feature overlap still exists.")


if __name__ == "__main__":
    main()