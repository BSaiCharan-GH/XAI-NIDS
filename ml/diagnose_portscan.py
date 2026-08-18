import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/random_forest.pkl"
TRAIN_DATA = "datasets/processed/train.csv"

print("=" * 70)
print("XAI-IDS — Port Scan Feature Diagnostics")
print("=" * 70)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\n[1] Loading model...")

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
features = model_data["features"]

print(f"    Model      : {type(model).__name__}")
print(f"    Features   : {len(features)}")
print(f"    Classes    : {list(model.classes_)}")

# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

print("\n[2] Loading training data...")

df = pd.read_csv(TRAIN_DATA)

print(f"    Rows       : {len(df)}")
print(f"    Columns    : {len(df.columns)}")

# ------------------------------------------------------------
# Check labels
# ------------------------------------------------------------

print("\n[3] Class distribution:")

print(df["Label"].value_counts())

# ------------------------------------------------------------
# Extract Port Scan samples
# ------------------------------------------------------------

portscan = df[df["Label"].astype(str).str.lower() == "port scan"].copy()

if portscan.empty:
    print("\n[ERROR] No Port Scan samples found.")
    raise SystemExit(1)

print(f"\n[4] Port Scan samples: {len(portscan)}")

# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

important_features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "FIN Flag Count",
    "Average Packet Size",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]

print("\n[5] Port Scan training statistics")
print("-" * 70)

for feature in important_features:

    if feature not in portscan.columns:
        continue

    values = pd.to_numeric(
        portscan[feature],
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if values.empty:
        continue

    print(
        f"{feature:<32} "
        f"min={values.min():.4f}  "
        f"mean={values.mean():.4f}  "
        f"max={values.max():.4f}"
    )

# ------------------------------------------------------------
# Show actual Port Scan examples
# ------------------------------------------------------------

print("\n[6] First 5 Port Scan samples")
print("-" * 70)

display_features = [
    feature
    for feature in important_features
    if feature in portscan.columns
]

print(
    portscan[
        display_features
    ].head(5).to_string(index=False)
)

# ------------------------------------------------------------
# Save Port Scan reference statistics
# ------------------------------------------------------------

stats = []

for feature in features:

    if feature not in portscan.columns:
        continue

    values = pd.to_numeric(
        portscan[feature],
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if values.empty:
        continue

    stats.append({
        "Feature": feature,
        "PortScan_Min": values.min(),
        "PortScan_Mean": values.mean(),
        "PortScan_Max": values.max(),
        "PortScan_Std": values.std(),
    })

stats_df = pd.DataFrame(stats)

output = "datasets/processed/portscan_reference_stats.csv"

stats_df.to_csv(
    output,
    index=False
)

print("\n[7] Reference statistics saved:")
print(f"    {output}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)