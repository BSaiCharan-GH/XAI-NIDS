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

    df = pd.read_csv(TEST_FILE)
    df.columns = normalize_columns(df.columns)

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    print("Generating predictions...")

    df["Prediction"] = model.predict(
        df[FEATURE_COLUMNS]
    )

    dos = df[
        df["Label"] == "DoS"
    ].copy()

    detected = dos[
        dos["Prediction"] == "DoS"
    ]

    missed = dos[
        dos["Prediction"] != "DoS"
    ]

    print(f"\nDetected DoS: {len(detected):,}")
    print(f"Missed DoS:   {len(missed):,}")

    print("\nTop feature comparisons:")

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    top_features = importance.sort_values(
        "Importance",
        ascending=False
    ).head(15)["Feature"]

    results = []

    for feature in top_features:
        detected_mean = detected[feature].mean()
        missed_mean = missed[feature].mean()

        results.append({
            "Feature": feature,
            "Detected_Mean": detected_mean,
            "Missed_Mean": missed_mean,
            "Difference_%": (
                abs(
                    detected_mean - missed_mean
                )
                /
                max(
                    abs(detected_mean),
                    1e-9
                )
            ) * 100
        })

    comparison = pd.DataFrame(results)

    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()