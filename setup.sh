#!/bin/bash
# setup.sh - Setup environment and download datasets using scripts/dataset/download_data.py

# Exit immediately if a command exits with a non-zero status
set -e

# 1. Check for python3
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found. Please install Python 3."
    exit 1
fi

# 2. Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🛠  Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate the virtual environment
echo "▶️  Activating virtual environment..."
source venv/bin/activate

# 4. Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# 5. Install required packages
echo "📥 Installing dependencies..."
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    pip install streamlit pandas numpy scikit-learn imbalanced-learn tensorflow joblib dask kaggle
fi

# 6. Ensure Kaggle API token is present
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "⚠️  ~/.kaggle/kaggle.json not found."
    echo "    Please download your Kaggle API token from your account and place it in ~/.kaggle/kaggle.json"
fi

# 7. Download & preprocess data
echo "🚀 Running data download & preprocessing..."
python scripts/dataset/download_data.py

echo "✅  Datasets downloaded to data/raw and processed to data/processed."
echo "✔️  Setup complete! To reactivate, run: source venv/bin/activate"


# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 