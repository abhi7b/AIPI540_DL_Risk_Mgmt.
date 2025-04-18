#!/usr/bin/env python
"""

Download specific files from the Home Credit Default Risk Kaggle competition
and extract them into a local raw data directory.
"""

import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# -------------------------------
# Segment 1: Authentication
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
# Segment 2: Directory Setup
# -------------------------------
def ensure_directory(path: str):
    """
    Create the directory if it does not exist.
    
    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)

# -------------------------------
# Segment 3: Download & Extract
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

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # 1. Authenticate
    kaggle_config_dir = "/Users/abhishek/Downloads"
    api = authenticate_kaggle(config_dir=kaggle_config_dir)

    # 2. Ensure raw data directory exists
    raw_data_dir = os.path.join("data", "raw", "home_credit")
    ensure_directory(raw_data_dir)

    # 3. Define files and competition
    competition = "home-credit-default-risk"
    target_files = ["application_train.csv", "application_test.csv"]

    # 4. Download & extract each file
    for fname in target_files:
        try:
            download_and_extract_file(api, competition, fname, raw_data_dir)
        except Exception as e:
            print(f"❌ Failed to process {fname}: {e}")

if __name__ == "__main__":
    main()


# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 