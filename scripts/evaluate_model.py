from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.pkl"
DATA_DIR = PROJECT_ROOT / "datasets" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TEST_SIZE = 0.20
RANDOM_STATE = 42

LABEL_NAMES = ["Label", "label", "LABEL"]


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("XAI-NIDS - RANDOM FOREST MODEL EVALUATION")
print("=" * 70)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("[1] Loading model...")

if not MODEL_PATH.exists():
    print("[ERROR] Model not found:")
    print(MODEL_PATH)
    sys.exit(1)

try:
    saved_model = joblib.load(MODEL_PATH)
except Exception as e:
    print("[ERROR] Could not load model:")
    print(e)
    sys.exit(1)


if not isinstance(saved_model, dict):
    print("[ERROR] random_forest.pkl is not in the expected format.")
    print("Expected a dictionary containing 'model' and 'features'.")
    sys.exit(1)


if "model" not in saved_model:
    print("[ERROR] 'model' not found in saved model.")
    sys.exit(1)

if "features" not in saved_model:
    print("[ERROR] 'features' not found in saved model.")
    sys.exit(1)


model = saved_model["model"]
features = list(saved_model["features"])


print(f"    Model    : {type(model).__name__}")
print(f"    Features : {len(features)}")
print()


# ============================================================
# FIND DATASET
# ============================================================

print("[2] Searching for dataset...")

if not DATA_DIR.exists():
    print("[ERROR] Dataset directory does not exist:")
    print(DATA_DIR)
    sys.exit(1)


csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    print("[ERROR] No CSV files found in:")
    print(DATA_DIR)
    sys.exit(1)


print(f"    CSV files found : {len(csv_files)}")
print()


# ============================================================
# FIND COMPATIBLE DATASET
# ============================================================

selected_file = None

for csv_file in csv_files:

    try:
        header = pd.read_csv(csv_file, nrows=0)
        columns = list(header.columns)
    except Exception:
        continue

    label_column = None

    for name in LABEL_NAMES:
        if name in columns:
            label_column = name
            break

    if label_column is None:
        continue

    missing = [
        feature
        for feature in features
        if feature not in columns
    ]

    if not missing:
        selected_file = csv_file
        break


if selected_file is None:
    print("[ERROR] Could not find a compatible dataset.")
    print()
    print("Available CSV files:")

    for file in csv_files:
        print(f"    {file.name}")

    sys.exit(1)


print(f"    Dataset : {selected_file.name}")
print()


# ============================================================
# LOAD DATASET
# ============================================================

print("[3] Loading dataset...")

try:
    df = pd.read_csv(selected_file)
except Exception as e:
    print("[ERROR] Could not read dataset:")
    print(e)
    sys.exit(1)


print(f"    Rows    : {len(df):,}")
print(f"    Columns : {len(df.columns)}")
print()


# ============================================================
# FIND LABEL
# ============================================================

label_column = None

for name in LABEL_NAMES:
    if name in df.columns:
        label_column = name
        break


if label_column is None:
    print("[ERROR] Label column not found.")
    sys.exit(1)


print(f"    Label column : {label_column}")
print()


# ============================================================
# CHECK FEATURES
# ============================================================

print("[4] Checking model features...")

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    print(
        f"[ERROR] {len(missing_features)} "
        "features are missing."
    )

    for feature in missing_features:
        print(f"    {feature}")

    sys.exit(1)


print(f"    Required features : {len(features)}")
print("    Missing features  : 0")
print()


# ============================================================
# PREPARE X AND Y
# ============================================================

X = df[features].copy()
y = df[label_column].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

print("[5] Cleaning data...")

for feature in features:
    X[feature] = pd.to_numeric(
        X[feature],
        errors="coerce"
    )


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


valid = (
    X.notna().all(axis=1)
    &
    y.notna()
)


removed = len(X) - int(valid.sum())


X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

y = y.astype(str).str.strip()


print(f"    Removed invalid rows : {removed:,}")
print(f"    Valid rows           : {len(X):,}")
print()


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("[6] Class distribution")
print("-" * 70)

counts = y.value_counts()

for label, count in counts.items():

    percentage = (
        count / len(y)
    ) * 100

    print(
        f"{label:<20}"
        f"{count:>12,}"
        f"   "
        f"{percentage:>7.2f}%"
    )

print()


classes = sorted(y.unique())

print(f"    Classes : {classes}")
print()


# ============================================================
# CREATE EVALUATION SPLIT
# ============================================================

print("[7] Creating evaluation split...")

try:

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

except Exception as e:

    print("[ERROR] Could not create evaluation split:")
    print(e)
    sys.exit(1)


print(f"    Evaluation rows : {len(X_test):,}")
print(f"    Test size       : {TEST_SIZE * 100:.0f}%")
print()


# ============================================================
# PREDICT
# ============================================================

print("[8] Running predictions...")

try:
    y_pred = model.predict(X_test)
except Exception as e:
    print("[ERROR] Prediction failed:")
    print(e)
    sys.exit(1)


y_pred = pd.Series(y_pred).astype(str)

print("    Prediction complete.")
print()


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

print("=" * 70)
print("OVERALL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision : {precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : {recall * 100:.2f}%"
)

print(
    f"Weighted F1-Score  : {f1 * 100:.2f}%"
)

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

report_text = classification_report(
    y_test,
    y_pred,
    labels=classes,
    zero_division=0
)

print(report_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual {x}"
        for x in classes
    ],
    columns=[
        f"Predicted {x}"
        for x in classes
    ]
)

print(cm_df.to_string())
print()


# ============================================================
# PER-CLASS PERFORMANCE
# ============================================================

print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

report = classification_report(
    y_test,
    y_pred,
    labels=classes,
    output_dict=True,
    zero_division=0
)

for label in classes:

    if label not in report:
        continue

    metrics = report[label]

    print()
    print(label)

    print(
        f"    Precision : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"    Recall    : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"    F1-Score  : "
        f"{metrics['f1-score'] * 100:.2f}%"
    )

    print(
        f"    Samples   : "
        f"{int(metrics['support']):,}"
    )

print()


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_path = REPORT_DIR / "confusion_matrix.csv"

cm_df.to_csv(cm_path)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = REPORT_DIR / "classification_report.csv"

pd.DataFrame(report).transpose().to_csv(
    report_path
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = X_test.copy()

prediction_df["Actual Label"] = y_test.values
prediction_df["Predicted Label"] = y_pred.values


if hasattr(model, "predict_proba"):

    try:

        probabilities = model.predict_proba(X_test)

        for i, class_name in enumerate(model.classes_):

            prediction_df[
                f"Probability {class_name}"
            ] = probabilities[:, i]

    except Exception:
        pass


prediction_path = (
    REPORT_DIR / "evaluation_predictions.csv"
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("=" * 70)
print("EVALUATION FILES SAVED")
print("=" * 70)

print(f"Confusion matrix : {cm_path}")
print(f"Classification   : {report_path}")
print(f"Predictions      : {prediction_path}")

print()

print("=" * 70)
print("PHASE 4 MODEL EVALUATION COMPLETE")
print("=" * 70)
print()