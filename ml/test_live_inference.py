import glob
import os
import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.pkl"
CAPTURE_DIR = "datasets/captures"

print("Loading model...")
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
model_features = model_data["features"]

print("Model features:", len(model_features))

# ---------------------------------------------------------
# Find latest capture CSV
# ---------------------------------------------------------

capture_files = glob.glob(os.path.join(CAPTURE_DIR, "*.csv"))

if not capture_files:
    raise FileNotFoundError(
        "No capture CSV found in datasets/captures"
    )

latest_capture = max(capture_files, key=os.path.getmtime)

print("Loading capture data...")
print("Capture file:", latest_capture)

df = pd.read_csv(latest_capture)

if df.empty:
    raise ValueError("Capture CSV contains no flows.")

# ---------------------------------------------------------
# Check feature availability
# ---------------------------------------------------------

missing_features = [
    feature for feature in model_features
    if feature not in df.columns
]

if missing_features:
    print()
    print("Missing features:")
    for feature in missing_features:
        print(" -", feature)

    raise ValueError("Capture data does not contain all model features.")

X = df[model_features]

print()
print("Input features:", X.shape[1])
print("Captured flows:", len(X))

# ---------------------------------------------------------
# Verify feature order
# ---------------------------------------------------------

if list(X.columns) != model_features:
    raise ValueError("Feature order mismatch!")

print("Feature order: OK")

# ---------------------------------------------------------
# Model inference
# ---------------------------------------------------------

print()
print("Running Random Forest inference...")

predictions = model.predict(X)

print()
print("Predictions:")

for i, prediction in enumerate(predictions):
    print(f"Flow {i + 1}: {prediction}")

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

prediction_counts = pd.Series(predictions).value_counts()

print()
print("Prediction summary:")
print(prediction_counts)

print()
print("Live capture inference test: PASS")