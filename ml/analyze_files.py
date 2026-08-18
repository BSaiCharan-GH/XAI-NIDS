from pathlib import Path
import pandas as pd


DATASET_DIR = Path("datasets/CICIDS2017")

LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "PortScan": "Port Scan",
}


def main():
    files = sorted(DATASET_DIR.glob("*.csv"))

    for file in files:
        counts = {
            "Benign": 0,
            "DoS": 0,
            "Port Scan": 0
        }

        for chunk in pd.read_csv(
            file,
            usecols=[" Label"],
            chunksize=100_000
        ):
            labels = chunk[" Label"].astype(str).str.strip()

            for original, mapped in LABEL_MAP.items():
                counts[mapped] += (labels == original).sum()

        print("=" * 70)
        print(file.name)

        for label, count in counts.items():
            print(f"{label:12}: {count:,}")


if __name__ == "__main__":
    main()