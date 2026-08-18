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

    test = test[
        test["Label"] == "DoS"
    ].copy()

    print(f"DoS test rows: {len(test):,}")

    test["feature_hash"] = create_hashes(test)

    print("\nLoading Random Forest...")

    bundle = joblib.load(MODEL_FILE)

    model = bundle["model"]

    test["Prediction"] = model.predict(
        test[FEATURE_COLUMNS]
    )

    print("\nLoading original Wednesday dataset...")

    raw_labels = {}

    for chunk in pd.read_csv(
        RAW_FILE,
        chunksize=100_000
    ):

        chunk.columns = normalize_columns(chunk.columns)

        chunk["Label"] = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )

        chunk = chunk[
            chunk["Label"].isin([
                "DoS GoldenEye",
                "DoS Hulk",
                "DoS Slowhttptest",
                "DoS slowloris",
                "Heartbleed"
            ])
        ]

        if chunk.empty:
            continue

        hashes = create_hashes(chunk)

        for feature_hash, label in zip(
            hashes,
            chunk["Label"]
        ):
            if feature_hash not in raw_labels:
                raw_labels[feature_hash] = set()

            raw_labels[feature_hash].add(label)

    test["Original_Attack"] = test[
        "feature_hash"
    ].map(
        lambda x: (
            next(iter(raw_labels[x]))
            if x in raw_labels
            and len(raw_labels[x]) == 1
            else "Ambiguous/Unknown"
        )
    )

    print("\nOriginal DoS subtype distribution:")

    print(
        test["Original_Attack"]
        .value_counts()
        .to_string()
    )

    print("\nDoS recall by original attack subtype:")

    for attack, group in test.groupby(
        "Original_Attack"
    ):

        total = len(group)

        correct = (
            group["Prediction"] == "DoS"
        ).sum()

        missed = total - correct

        recall = correct / total

        print(f"\n{attack}")
        print(f"Total: {total:,}")
        print(f"Correct: {correct:,}")
        print(f"Missed: {missed:,}")
        print(f"Recall: {recall:.4f}")

        print("Predictions:")

        print(
            group["Prediction"]
            .value_counts()
            .to_string()
        )

    print("\nOverall DoS errors by subtype:")

    missed = test[
        test["Prediction"] != "DoS"
    ]

    print(
        missed["Original_Attack"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()