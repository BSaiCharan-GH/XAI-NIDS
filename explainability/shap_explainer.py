from pathlib import Path
import sys

import joblib
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "ml")
)

from schema import FEATURE_COLUMNS, normalize_columns


TEST_FILE = (
    PROJECT_ROOT
    / "datasets"
    / "processed"
    / "test_v2.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_v2.pkl"
)


def explain_sample(
    explainer,
    model,
    sample,
    sample_index
):

    X = sample[FEATURE_COLUMNS]

    prediction = model.predict(X)[0]
    actual = sample.iloc[0]["Label"]

    class_names = list(model.classes_)

    class_index = class_names.index(
        prediction
    )

    shap_values = explainer.shap_values(X)

    values = shap_values[
        0,
        :,
        class_index
    ]

    explanation = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "SHAP": values
    })

    explanation["Absolute_SHAP"] = (
        explanation["SHAP"].abs()
    )

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    )

    print("\n" + "=" * 70)

    print(
        f"Sample: {sample_index}"
    )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Actual: {actual}"
    )

    print(
        f"Prediction class index: "
        f"{class_index}"
    )

    print(
        "\nTop contributing features:"
    )

    print(
        explanation[
            ["Feature", "SHAP"]
        ]
        .head(10)
        .to_string(index=False)
    )


def main():

    print("Loading model...")

    bundle = joblib.load(
        MODEL_FILE
    )

    model = bundle["model"]

    print("Loading test data...")

    df = pd.read_csv(
        TEST_FILE
    )

    df.columns = normalize_columns(
        df.columns
    )

    print(
        f"Test rows: {len(df):,}"
    )

    print(
        "\nCreating SHAP explainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "SHAP explainer ready."
    )

    # --------------------------------------------------
    # 1. Benign
    # --------------------------------------------------

    benign = df[
        df["Label"] == "Benign"
    ].head(1)

    # --------------------------------------------------
    # 2. Correctly detected DoS
    # --------------------------------------------------

    dos = df[
        df["Label"] == "DoS"
    ]

    dos_predictions = model.predict(
        dos[FEATURE_COLUMNS]
    )

    correct_dos = dos[
        dos_predictions == "DoS"
    ].head(1)

    # --------------------------------------------------
    # 3. Correctly detected Port Scan
    # --------------------------------------------------

    port_scan = df[
        df["Label"] == "Port Scan"
    ]

    port_predictions = model.predict(
        port_scan[FEATURE_COLUMNS]
    )

    correct_port_scan = port_scan[
        port_predictions == "Port Scan"
    ].head(1)

    # --------------------------------------------------
    # 4. Missed DoS
    # --------------------------------------------------

    missed_dos = dos[
        dos_predictions == "Benign"
    ].head(1)

    # --------------------------------------------------
    # Generate explanations
    # --------------------------------------------------

    print("\nGenerating explanations...")

    if not benign.empty:
        explain_sample(
            explainer,
            model,
            benign,
            "Benign"
        )

    if not correct_dos.empty:
        explain_sample(
            explainer,
            model,
            correct_dos,
            "Correct DoS"
        )

    if not correct_port_scan.empty:
        explain_sample(
            explainer,
            model,
            correct_port_scan,
            "Correct Port Scan"
        )

    if not missed_dos.empty:
        explain_sample(
            explainer,
            model,
            missed_dos,
            "Missed DoS"
        )

    print(
        "\nSHAP analysis complete."
    )


if __name__ == "__main__":
    main()