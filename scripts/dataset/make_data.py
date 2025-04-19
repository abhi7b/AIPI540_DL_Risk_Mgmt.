#!/usr/bin/env python
"""
Complete pipeline for data and curation:
1. Download specific files from the Kaggle competition
2. Extract them into a local raw data directory
3. Load, clean, and split the data into train/validation/test sets (60/20/20)
"""

import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

# -------------------------------
# Configuration
# -------------------------------
KAGGLE_CONFIG_DIR = ""  # Enter custom Path to kaggle.json
COMPETITION = "home-credit-default-risk"
TARGET_FILES = ["application_train.csv"]
RAW_DIR = os.path.join("data", "raw", "home_credit")
PROCESSED_DIR = os.path.join("data", "processed")
TRAIN_FILE = os.path.join(RAW_DIR, "application_train.csv")
MISSING_THRESH = 30_000

# -------------------------------
# Part 1: Kaggle Authentication
# -------------------------------
def authenticate_kaggle(config_dir: str = None) -> KaggleApi:
    """
    Authenticate with the Kaggle API.
    
    Args:
        config_dir: Optional path to Kaggle config directory.
    
    Returns:
        An authenticated KaggleApi instance.
    """
    if config_dir:
        os.environ['KAGGLE_CONFIG_DIR'] = config_dir
    api = KaggleApi()
    api.authenticate()
    return api

# -------------------------------
# Part 2: Directory Setup
# -------------------------------
def ensure_directory(path: str):
    """
    Create the directory if it does not exist.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)

# -------------------------------
# Part 3: Download & Extract
# -------------------------------
def download_and_extract_file(
    api: KaggleApi,
    competition: str,
    filename: str,
    dest_dir: str
):
    """
    Download a single file from a Kaggle competition and extract it.
    
    Args:
        api: Authenticated KaggleApi instance.
        competition: Name of the Kaggle competition.
        filename: Name of the file to download.
        dest_dir: Directory where the file and its extraction should go.
    """
    print(f"🔽 Downloading {filename}...")
    api.competition_download_file(competition, file_name=filename, path=dest_dir)

    zip_path = os.path.join(dest_dir, f"{filename}.zip")
    if os.path.exists(zip_path):
        print(f"📦 Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        print(f"✅ Extracted and removed: {os.path.basename(zip_path)}")
    else:
        print(f"❌ Zip file not found for: {filename}")

def download_data():
    """Download all required data files from Kaggle."""
    print("📥 Starting data download process...")
    
    # 1. Authenticate
    api = authenticate_kaggle(config_dir=KAGGLE_CONFIG_DIR)

    # 2. Ensure raw data directory exists
    ensure_directory(RAW_DIR)

    # 3. Download & extract each file
    for fname in TARGET_FILES:
        try:
            download_and_extract_file(api, COMPETITION, fname, RAW_DIR)
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

# -------------------------------
# Part 4: Load & Clean Data
# -------------------------------
def load_and_clean_data():
    """Load and clean the training data."""
    print("⚙️  Loading training data...")
    df = pd.read_csv(TRAIN_FILE)
    print(f"📐 Raw data shape: {df.shape}")

    print(f"🔎  Identifying columns with > {MISSING_THRESH:,} missing values…")
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    cols_to_drop = missing_counts[missing_counts > MISSING_THRESH].index.tolist()

    print(f"🗑  Dropping {len(cols_to_drop)} columns…")
    df = df.drop(columns=cols_to_drop)
    print(f"📐 After column drop: {df.shape}")

    print("🚮  Dropping rows with any remaining nulls…")
    df = df.dropna()
    print(f"📐 Clean data shape: {df.shape}")

    return df

# -------------------------------
# Part 5: Split Data & Save
# -------------------------------
def split_and_save(df):
    """Split the data and save train/validation/test sets."""
    print("✂️  Splitting into train (60%), validation (20%), test (20%)…")
    strat_col = "TARGET" if "TARGET" in df.columns else None

    # First split: train (60%) vs temp (40%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=42,
        stratify=df[strat_col] if strat_col else None
    )

    # Second split: temp → validation & test (each 50% of temp → 20% overall)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df[strat_col] if strat_col else None
    )

    print(f"📊  Train set:      {train_df.shape}")
    print(f"📊  Validation set: {val_df.shape}")
    print(f"📊  Test set:       {test_df.shape}")

    # Save outputs
    ensure_directory(PROCESSED_DIR)
    paths = {
        "train":      os.path.join(PROCESSED_DIR, "train.csv"),
        "validation": os.path.join(PROCESSED_DIR, "validation.csv"),
        "test":       os.path.join(PROCESSED_DIR, "test.csv"),
    }

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)

    for name, path in paths.items():
        print(f"✅  Saved {name}.csv → {path}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    """Run the complete pipeline."""
    print("🚀 Starting Home Credit Default Risk data pipeline")
    
    # Step 1: Download and extract data
    download_data()
    
    # Step 2: Load and clean data
    df_clean = load_and_clean_data()
    
    # Step 3: Split and save data
    split_and_save(df_clean)
    
    print("✨ Pipeline completed successfully!")

if __name__ == "__main__":
    main()


# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 