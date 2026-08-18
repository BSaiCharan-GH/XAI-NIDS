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

    bundle = joblib.load(MODEL_FILE)
    model = bundle["model"]

    test["Prediction"] = model.predict(
        test[FEATURE_COLUMNS]
    )

    test["feature_hash"] = create_hashes(test)

    print("Loading original Wednesday labels...")

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

    test["Original_Attack"] = test[
        "feature_hash"
    ].isin(goldeneye_hashes)

    goldeneye = test[
        test["Original_Attack"]
    ]

    detected = goldeneye[
        goldeneye["Prediction"] == "DoS"
    ]

    missed = goldeneye[
        goldeneye["Prediction"] == "Benign"
    ]

    benign = test[
        test["Label"] == "Benign"
    ]

    print("\nGroup sizes:")
    print(
        f"Detected GoldenEye : {len(detected):,}"
    )
    print(
        f"Missed GoldenEye   : {len(missed):,}"
    )
    print(
        f"Benign              : {len(benign):,}"
    )

    importance = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Importance": model.feature_importances_
    })

    top_features = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(15)["Feature"]
    )

    results = []

    for feature in top_features:

        results.append({
            "Feature": feature,

            "Detected_GoldenEye":
                detected[feature].mean(),

            "Missed_GoldenEye":
                missed[feature].mean(),

            "Benign":
                benign[feature].mean()
        })

    comparison = pd.DataFrame(results)

    print("\nFeature comparison:")

    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()