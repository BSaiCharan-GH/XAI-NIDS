from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from schema import FEATURE_COLUMNS, normalize_columns


TRAIN_FILE = Path("datasets/processed/train_v2.csv")
TEST_FILE = Path("datasets/processed/test_v2.csv")

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "random_forest_v2.pkl"


def main():
    print("Loading training data...")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df.columns = normalize_columns(train_df.columns)
    test_df.columns = normalize_columns(test_df.columns)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Label"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["Label"]

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nModel training complete.")

    print("\nEvaluating model...")

    predictions = model.predict(X_test)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "features": FEATURE_COLUMNS
        },
        MODEL_FILE
    )

    print(f"\nModel saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()