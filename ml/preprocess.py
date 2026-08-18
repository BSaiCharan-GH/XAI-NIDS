from pathlib import Path
import pandas as pd


DATASET_DIR = Path("datasets/CICIDS2017")
OUTPUT_DIR = Path("datasets/processed")

OUTPUT_FILE = OUTPUT_DIR / "ids_dataset_with_source.csv"

CHUNK_SIZE = 100_000


LABEL_MAP = {
    "BENIGN": "Benign",
    "DoS GoldenEye": "DoS",
    "DoS Hulk": "DoS",
    "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS",
    "PortScan": "Port Scan",
}


def process_file(file_path, output_file):
    first_write = not output_file.exists()

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

        chunk.columns = chunk.columns.str.strip()

        chunk["Label"] = chunk["Label"].astype(str).str.strip()

        chunk = chunk[chunk["Label"].isin(LABEL_MAP)]

        if chunk.empty:
            continue

        chunk["Label"] = chunk["Label"].map(LABEL_MAP)

        feature_columns = [
            column for column in chunk.columns
            if column != "Label"
        ]

        for column in feature_columns:
            chunk[column] = pd.to_numeric(
                chunk[column],
                errors="coerce"
            )

        chunk.replace(
            [float("inf"), float("-inf")],
            pd.NA,
            inplace=True
        )

        chunk.dropna(inplace=True)

        chunk.drop_duplicates(inplace=True)

        chunk["Source_File"] = file_path.name

        chunk.to_csv(
            output_file,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    files = sorted(DATASET_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATASET_DIR}"
        )

    print(f"Found {len(files)} dataset files.")

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        process_file(
            file_path,
            OUTPUT_FILE
        )

        print("Completed.")

    print("\nPreprocessing complete.")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()