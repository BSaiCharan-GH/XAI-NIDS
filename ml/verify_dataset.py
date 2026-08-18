from pathlib import Path
import pandas as pd
import numpy as np


TRAIN_FILE = Path("datasets/processed/train.csv")
TEST_FILE = Path("datasets/processed/test.csv")


def verify(file_path):
    print("=" * 60)
    print(file_path)

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    feature_columns = [col for col in df.columns if col != "Label"]

    print(f"Features: {len(feature_columns)}")

    missing = df[feature_columns].isna().sum().sum()
    infinite = np.isinf(df[feature_columns].to_numpy()).sum()

    print(f"Missing values: {missing:,}")
    print(f"Infinite values: {infinite:,}")

    print("\nLabels:")
    print(df["Label"].value_counts())

    print("\nFirst 10 features:")
    for feature in feature_columns[:10]:
        print(f"  {feature}")


def main():
    verify(TRAIN_FILE)
    verify(TEST_FILE)

    print("\nDataset verification complete.")


if __name__ == "__main__":
    main()