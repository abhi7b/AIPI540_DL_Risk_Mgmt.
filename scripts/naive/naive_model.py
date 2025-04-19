#!/usr/bin/env python
"""

Load processed Home Credit data, train a logistic regression model on numeric features,
evaluate performance (ROC AUC & recall focus), and save the trained model.
"""

import os
import warnings
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score, roc_auc_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# -------------------------------
# 1. Paths & Directories Setup
# -------------------------------
def get_paths():
    processed_dir = os.path.join("data", "processed")
    os.makedirs("models", exist_ok=True)
    return {
        "train": os.path.join(processed_dir, "train.csv"),
        "test":  os.path.join(processed_dir, "test.csv"),
        "model": os.path.join("models", "logistic_regression_model.pkl")
    }

# -------------------------------
# 2. Load Data
# -------------------------------
def load_data(train_path: str, test_path: str, target_col: str = "TARGET"):
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f"  Train shape: {train.shape}")
    print(f"  Test  shape: {test.shape}")

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_test  = test.drop(columns=[target_col])
    y_test  = test[target_col]

    return X_train, y_train, X_test, y_test

# -------------------------------
# 3. Preprocess Numeric Features
# -------------------------------
def select_numeric(X_train: pd.DataFrame, X_test: pd.DataFrame):
    X_train_num = X_train.select_dtypes(include=["int64", "float64"])
    X_test_num  = X_test.select_dtypes(include=["int64", "float64"])
    print("Selected numeric features:")
    print(f"  X_train_num cols: {X_train_num.shape[1]}")
    print(f"  X_test_num  cols: {X_test_num.shape[1]}")
    return X_train_num, X_test_num

# -------------------------------
# 4. Train Logistic Regression
# -------------------------------
def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------
# 5. Evaluate Model
# -------------------------------
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)

    print("\nPerformance Metrics:")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Recall:  {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 6. Save Model
# -------------------------------
def save_model(model, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    paths = get_paths()
    X_train, y_train, X_test, y_test = load_data(paths["train"], paths["test"])
    X_train_num, X_test_num = select_numeric(X_train, X_test)
    model = train_model(X_train_num, y_train)
    evaluate_model(model, X_test_num, y_test)
    save_model(model, paths["model"])

if __name__ == "__main__":
    main()

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 