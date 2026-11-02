# Wi-Fi Indoor Localization — Data Preparation

## Dataset Source
UJIIndoorLoc dataset, provided in `datasets/archive.zip`.

## Prepared Files
- `processed/train_cleaned.csv` — cleaned training observations
- `processed/validation_cleaned.csv` — official held-out validation observations

## Processing Applied
1. Loaded original training and validation data.
2. Verified the 520 WAP feature columns.
3. Created `LOCATION_CLASS` using `BUILDINGID` and `FLOOR`.
4. Checked for missing values.
5. Removed exact duplicate rows from training only.
6. Replaced undetected Wi-Fi readings (`100`) with `-110`.
7. Preserved the original validation observations for final evaluation.

## Important Notes
- WAP-only duplicates are retained.
- Zero-detection samples are retained.
- The official validation dataset should remain held out during model selection.
- Scaling has not been applied. Fit scalers on training data only, after splitting, to avoid leakage.
- For model development, create a stratified training/validation split from `train_cleaned.csv`.
- Use `LOCATION_CLASS` as the classification target.
- Use only the 520 WAP columns as model input features for the initial baseline comparison.

## Reproduce the Files
From the repository root:

`python3 data_preparation/prepare_data.py`

## Models
Prepared for KNN, Random Forest, and Neural Network experiments.