from pathlib import Path

import joblib
import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns


RAW_FILE = Path(
    "datasets/CICIDS2017/"
    "Wednesday-workingHours.pcap_ISCX.csv"
)

TEST_FILE = Path(
    "datasets/processed/test_v2.csv"
)

MODEL_FILE = Path(
    "models/random_forest_v2.pkl"
)


def create_hashes(df):
    df.columns = normalize_columns(df.columns)

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return pd.util.hash_pandas_object(
        df[FEATURE_COLUMNS],
        index=False
    )


def main():
    print("Loading test data...")

    test = pd.read_csv(TEST_FILE)
    test.columns = normalize_columns(test.columns)

    print("Loading Random Forest...")

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    test["Prediction"] = model.predict(
        test[FEATURE_COLUMNS]
    )

    # GoldenEye rows that were actually missed.
    goldeneye_missed = test[
        (test["Label"] == "DoS") &
        (test["Prediction"] == "Benign")
    ].copy()

    # Correctly detected GoldenEye.
    # We recover the original subtype below.
    goldeneye_test = test[
        test["Label"] == "DoS"
    ].copy()

    print(
        f"\nTotal DoS test rows: "
        f"{len(goldeneye_test):,}"
    )

    print(
        f"GoldenEye candidates: "
        f"{len(goldeneye_test):,}"
    )

    print(
        f"DoS classified as Benign: "
        f"{len(goldeneye_missed):,}"
    )

    # Load original labels and identify GoldenEye.
    print("\nLoading original Wednesday data...")

    goldeneye_hashes = set()

    for chunk in pd.read_csv(
        RAW_FILE,
        chunksize=100_000
    ):
        chunk.columns = normalize_columns(
            chunk.columns
        )

        chunk["Label"] = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )

        chunk = chunk[
            chunk["Label"] == "DoS GoldenEye"
        ]

        if chunk.empty:
            continue

        hashes = create_hashes(chunk)

        goldeneye_hashes.update(
            hashes.tolist()
        )

    goldeneye_test["feature_hash"] = (
        create_hashes(goldeneye_test)
    )

    goldeneye_test = goldeneye_test[
        goldeneye_test["feature_hash"].isin(
            goldeneye_hashes
        )
    ]

    detected = goldeneye_test[
        goldeneye_test["Prediction"] == "DoS"
    ]

    missed = goldeneye_test[
        goldeneye_test["Prediction"] == "Benign"
    ]

    print(
        f"\nGoldenEye correctly detected: "
        f"{len(detected):,}"
    )

    print(
        f"GoldenEye missed as Benign: "
        f"{len(missed):,}"
    )

    # Compare important features.
    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    top_features = importance.sort_values(
        "Importance",
        ascending=False
    ).head(20)["Feature"]

    results = []

    for feature in top_features:
        detected_mean = detected[feature].mean()
        missed_mean = missed[feature].mean()

        results.append({
            "Feature": feature,
            "Detected_Mean": detected_mean,
            "Missed_Mean": missed_mean
        })

    comparison = pd.DataFrame(results)

    print("\nGoldenEye feature comparison:")

    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()