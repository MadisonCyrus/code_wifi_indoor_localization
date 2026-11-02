
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

# 1. Find the original dataset
ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "datasets" / "archive.zip"
OUTPUT_DIR = ROOT / "data_preparation" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load training and validation CSV files
with ZipFile(ZIP_PATH) as archive:
    csv_files = archive.namelist()

    def load_csv(filename):
        matches = [
            name for name in csv_files
            if Path(name).name.lower() == filename.lower()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one {filename}, found {matches}"
            )
        with archive.open(matches[0]) as file:
            return pd.read_csv(file)

    train = load_csv("trainingData.csv")
    validation = load_csv("validationData.csv")

print("Original training:", train.shape)
print("Original validation:", validation.shape)

# 3. Identify Wi-Fi features
wap_columns = [col for col in train.columns
               if col.startswith("WAP")]

assert len(wap_columns) == 520
assert all(col in validation.columns for col in wap_columns)

# 4. Create consistent building-floor labels
def add_labels(df):
    df = df.copy()

    df["LOCATION_CLASS"] = (
        "B" + df["BUILDINGID"].astype(str)
        + "_F" + df["FLOOR"].astype(str)
    )

    return df

train = add_labels(train)
validation = add_labels(validation)

# 5. Check data quality
print("Training missing values:", train.isna().sum().sum())
print("Validation missing values:", validation.isna().sum().sum())

if train.isna().any().any() or validation.isna().any().any():
    raise ValueError("Missing values found. Review before exporting.")

# 6. Remove exact duplicate training rows
duplicates = train.duplicated().sum()
train = train.drop_duplicates().reset_index(drop=True)

print("Exact duplicate training rows removed:", duplicates)

# Keep official validation rows unchanged

# 7. Handle Wi-Fi sentinel values
# Original dataset uses 100 for undetected access points.
# Replace with -110 dBm as a documented modeling convention.
train[wap_columns] = train[wap_columns].replace(100, -110)
validation[wap_columns] = validation[wap_columns].replace(100, -110)

# 8. Preserve Wi-Fi features and target labels
# Other metadata is retained for auditing and future experiments.
ordered_columns = (
    wap_columns
    + [col for col in train.columns if col not in wap_columns]
)

train = train[ordered_columns]
validation = validation[ordered_columns]

# 9. Verify final data
assert train.columns.tolist() == validation.columns.tolist()
assert not train.isna().any().any()
assert not validation.isna().any().any()
assert (train[wap_columns] != 100).all().all()
assert (validation[wap_columns] != 100).all().all()

print("\nFinal training shape:", train.shape)
print("Final validation shape:", validation.shape)
print("Training classes:", train["LOCATION_CLASS"].nunique())
print("Validation classes:", validation["LOCATION_CLASS"].nunique())

# 10. Export cleaned datasets
train_path = OUTPUT_DIR / "train_cleaned.csv"
validation_path = OUTPUT_DIR / "validation_cleaned.csv"

train.to_csv(train_path, index=False)
validation.to_csv(validation_path, index=False)

print("\nSaved:", train_path)
print("Saved:", validation_path)
print("\nDATA PREPARATION COMPLETE!")
