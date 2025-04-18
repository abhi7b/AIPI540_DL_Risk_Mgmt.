#!/usr/bin/env python
"""

Load, clean, and split the Home Credit Default Risk application data
into train, validation, and test sets (60/20/20 stratified split).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------------
# Configuration
# -------------------------------
RAW_DIR        = os.path.join("data", "raw", "home_credit")
PROCESSED_DIR  = os.path.join("data", "processed")
TRAIN_FILE     = os.path.join(RAW_DIR, "application_train.csv")
MISSING_THRESH = 30_000

os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------------
# Part 1: Load & Clean Data
# -------------------------------
def load_and_clean_data():
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
# Part 2: Split Data & Save
# -------------------------------
def split_and_save(df):
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
    paths = {
        "train":      os.path.join(PROCESSED_DIR, "train.csv"),
        "validation": os.path.join(PROCESSED_DIR, "validation.csv"),
        "test":       os.path.join(PROCESSED_DIR, "test.csv"),
    }

    train_df.to_csv(paths["train"],      index=False)
    val_df.to_csv(paths["validation"],   index=False)
    test_df.to_csv(paths["test"],        index=False)

    for name, path in paths.items():
        print(f"✅  Saved {name}.csv → {path}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    df_clean = load_and_clean_data()
    split_and_save(df_clean)

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 