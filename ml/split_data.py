from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = Path("datasets/processed/ids_dataset.csv")
OUTPUT_DIR = Path("datasets/processed")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    print("Loading processed dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Total samples: {len(df):,}")
    print("\nClass distribution:")
    print(df["Label"].value_counts())

    X = df.drop(columns=["Label"])
    y = df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["Label"] = y_train

    test_df = X_test.copy()
    test_df["Label"] = y_test

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("\nTrain samples:", len(train_df))
    print("Test samples:", len(test_df))

    print("\nTraining distribution:")
    print(train_df["Label"].value_counts())

    print("\nTest distribution:")
    print(test_df["Label"].value_counts())

    print("\nTrain/test split complete.")


if __name__ == "__main__":
    main()