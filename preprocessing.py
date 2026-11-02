import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(train_path: str, val_path: str):
    """Loads raw UJIIndoorLoc training and validation data."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df

def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, drop_zero_waps: bool = True):
    """
    Cleans Wi-Fi RSSI features, handles zero-signal samples, creates combined building-floor 
    labels, imputes missing sentinel values, scales features, and performs stratified splitting.
    """
    # 1. Clean duplicates and zero-signal samples based on EDA findings
    initial_train_len = len(train_df)
    train_df = train_df.drop_duplicates()
    print(f"Dropped {initial_train_len - len(train_df)} full-row duplicate samples from training data.")

    wap_cols = [col for col in train_df.columns if col.startswith('WAP')]

    if drop_zero_waps:
        # Identify samples with 0 detected WAPs (where all WAPs are sentinel value 100)
        detected_mask = (train_df[wap_cols] != 100).sum(axis=1) > 0
        train_df = train_df[detected_mask].reset_index(drop=True)
        print(f"Filtered out samples with 0 detected WAPs. Remaining training samples: {len(train_df)}")

    # 2. Separate Features (X) and create Target Labels (y)
    X_train_raw = train_df[wap_cols].copy()
    X_val_raw = val_df[wap_cols].copy()

    y_train_raw = "B" + train_df["BUILDINGID"].astype(str) + "_F" + train_df["FLOOR"].astype(str)
    y_val_raw = "B" + val_df["BUILDINGID"].astype(str) + "_F" + val_df["FLOOR"].astype(str)

    # Keep coordinate information for regression experiments
    coords_train = train_df[["LONGITUDE", "LATITUDE"]].copy()
    coords_val = val_df[["LONGITUDE", "LATITUDE"]].copy()

    # 3. Impute sentinel +100 values to -104 dBm (background noise floor)
    X_train_raw = X_train_raw.replace(100, -104)
    X_val_raw = X_val_raw.replace(100, -104)

    # 4. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train_df = pd.DataFrame(X_train_scaled, columns=wap_cols)
    X_val_df = pd.DataFrame(X_val_scaled, columns=wap_cols)

    # 5. Stratified Train/Validation split on the training dataset
    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X_train_df, y_train_raw, coords_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train_raw
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "coords_train": c_train,
        "coords_test": c_test,
        "X_holdout_test": X_val_df,
        "y_holdout_test": y_val_raw,
        "coords_holdout": coords_val,
        "scaler": scaler
    }
