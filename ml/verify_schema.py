from pathlib import Path
import pandas as pd

from schema import FEATURE_COLUMNS, normalize_columns, validate_features


DATASET_FILE = Path(
    "datasets/processed/ids_dataset_with_source.csv"
)


def main():
    df = pd.read_csv(
        DATASET_FILE,
        nrows=5
    )

    original_columns = list(df.columns)

    normalized_columns = normalize_columns(
        original_columns
    )

    feature_columns = [
        column
        for column in normalized_columns
        if column not in ["Label", "Source_File"]
    ]

    validate_features(feature_columns)

    print("Schema validation successful.")
    print(f"Features: {len(feature_columns)}")
    print("\nFeature order:")

    for index, feature in enumerate(feature_columns):
        print(f"{index:2}: {feature}")


if __name__ == "__main__":
    main()