from pathlib import Path

import joblib
import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns


TEST_FILE = Path(
    "datasets/processed/test_v2.csv"
)

MODEL_FILE = Path(
    "models/random_forest_v2.pkl"
)


def main():
    print("Loading test data...")

    test_df = pd.read_csv(TEST_FILE)
    test_df.columns = normalize_columns(
        test_df.columns
    )

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    X_test = test_df[FEATURE_COLUMNS]

    print("Generating predictions...")

    test_df["Prediction"] = model.predict(X_test)

    dos = test_df[
        test_df["Label"] == "DoS"
    ]

    print("\nDoS performance by source:")

    for source, group in dos.groupby(
        "Source_File"
    ):
        total = len(group)
        correct = (
            group["Prediction"] == "DoS"
        ).sum()

        recall = correct / total

        print("\n" + source)
        print(f"Total DoS: {total:,}")
        print(f"Correct: {correct:,}")
        print(f"Missed: {total - correct:,}")
        print(f"Recall: {recall:.4f}")

        print("Predictions:")
        print(
            group["Prediction"]
            .value_counts()
            .to_string()
        )


if __name__ == "__main__":
    main()