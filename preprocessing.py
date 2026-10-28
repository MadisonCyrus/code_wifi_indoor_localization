import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(train_path: str, val_path: str):
    """Loads raw UJIIndoorLoc train and validation CSV files."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    return train_df, val_df

def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Cleans Wi-Fi RSSI features, creates combined building-floor labels,
    imputes missing sentinel values, scales features, and performs stratified splitting.
    """
    # 1. Separate WAP features (first 520 columns)
    wap_cols = [col for col in train_df.columns if col.startswith('WAP')]
    
    X_train_raw = train_df[wap_cols].copy()
    X_val_raw = val_df[wap_cols].copy()
    
    # 2. Create target labels (Combining Building and Floor ID into strings like 'B0_F0')
    y_train_raw = "B" + train_df["BUILDINGID"].astype(str) + "_F" + train_df["FLOOR"].astype(str)
    y_val_raw = "B" + val_df["BUILDINGID"].astype(str) + "_F" + val_df["FLOOR"].astype(str)
    
    # For regression experiments later (Lat/Lon)
    coords_train = train_df[["LONGITUDE", "LATITUDE"]].copy()
    coords_val = val_df[["LONGITUDE", "LATITUDE"]].copy()

    # 3. Impute sentinel +100 values to -104 dBm (background noise floor)
    X_train_raw = X_train_raw.replace(100, -104)
    X_val_raw = X_val_raw.replace(100, -104)
    
    # 4. Feature Scaling (StandardScaler or MinMaxScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    
    # Convert scaled arrays back to DataFrames for convenience
    X_train_df = pd.DataFrame(X_train_scaled, columns=wap_cols)
    X_val_df = pd.DataFrame(X_val_scaled, columns=wap_cols)

    # 5. Stratified 80/20 train/validation split on the training partition
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
