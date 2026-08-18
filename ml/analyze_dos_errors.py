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
    y_test = test_df["Label"]

    print("Generating predictions...")

    predictions = model.predict(X_test)

    test_df["Prediction"] = predictions

    actual_dos = test_df[
        test_df["Label"] == "DoS"
    ]

    missed_dos = actual_dos[
        actual_dos["Prediction"] != "DoS"
    ]

    detected_dos = actual_dos[
        actual_dos["Prediction"] == "DoS"
    ]

    print("\nDoS analysis:")
    print(f"Actual DoS: {len(actual_dos):,}")
    print(f"Correctly detected: {len(detected_dos):,}")
    print(f"Missed: {len(missed_dos):,}")

    print("\nMissed DoS predictions:")
    print(
        missed_dos["Prediction"]
        .value_counts()
    )

    print("\nModel feature importance:")

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print(
        importance.head(20).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()