#!/usr/bin/env python
"""

Load processed Home Credit data, apply resampling (ADASYN & SMOTEENN),
tune RandomForest and AdaBoost via grid search on validation set,
select overall best model, evaluate on test set, and save it.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    recall_score, roc_auc_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
import joblib

# -------------------------------
# 1. Load Processed Datasets
# -------------------------------
def load_datasets(processed_dir: str, target_col: str = "TARGET"):
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(processed_dir, "validation.csv"))
    test  = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col]
    X_val   = val.drop(columns=[target_col])
    y_val   = val[target_col]
    X_test  = test.drop(columns=[target_col])
    y_test  = test[target_col]

    print("Datasets loaded:")
    print(f"  Train:      {X_train.shape}, labels: {y_train.shape}")
    print(f"  Validation: {X_val.shape},   labels: {y_val.shape}")
    print(f"  Test:       {X_test.shape},  labels: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# -------------------------------
# 2. One-Hot Encode & Align
# -------------------------------
def encode_and_align(X_train, X_val, X_test):
    X_train_enc = pd.get_dummies(X_train)
    X_val_enc   = pd.get_dummies(X_val)
    X_test_enc  = pd.get_dummies(X_test)

    X_train_enc, X_val_enc  = X_train_enc.align(X_val_enc,  join="outer", axis=1, fill_value=0)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="outer", axis=1, fill_value=0)

    # Ensure same column order
    X_val_enc  = X_val_enc[X_train_enc.columns]
    X_test_enc = X_test_enc[X_train_enc.columns]

    print("After one-hot encoding & alignment:")
    print(f"  X_train: {X_train_enc.shape}")
    print(f"  X_val:   {X_val_enc.shape}")
    print(f"  X_test:  {X_test_enc.shape}")

    return X_train_enc, X_val_enc, X_test_enc

# -------------------------------
# 3. Resampling Techniques
# -------------------------------
def get_samplers():
    return {
        "ADASYN": ADASYN(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42)
    }

# -------------------------------
# 4. Model Scoring
# -------------------------------
def score_model(model, X, y):
    preds = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    rec   = recall_score(y, preds)
    auc   = roc_auc_score(y, proba)
    return rec, auc, (rec + auc) / 2

# -------------------------------
# 5. Hyperparameter Tuning
# -------------------------------
def tune_rf(X_train, y_train, X_val, y_val):
    best_score, best_model, best_params = -1, None, None
    grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "class_weight": ["balanced"],
        "random_state": [42]
    }
    for params in ParameterGrid(grid):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        rec, auc, comb = score_model(model, X_val, y_val)
        print(f"RF {params} → recall={rec:.4f}, auc={auc:.4f}, combined={comb:.4f}")
        if comb > best_score:
            best_score, best_model, best_params = comb, model, params
    print("Best RF params:", best_params, f"combined={best_score:.4f}")
    return best_model, best_score

def tune_ada(X_train, y_train, X_val, y_val):
    best_score, best_model, best_params = -1, None, None
    grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [1.0, 0.5],
        "random_state": [42]
    }
    for params in ParameterGrid(grid):
        model = AdaBoostClassifier(**params)
        model.fit(X_train, y_train)
        rec, auc, comb = score_model(model, X_val, y_val)
        print(f"Ada {params} → recall={rec:.4f}, auc={auc:.4f}, combined={comb:.4f}")
        if comb > best_score:
            best_score, best_model, best_params = comb, model, params
    print("Best AdaBoost params:", best_params, f"combined={best_score:.4f}")
    return best_model, best_score

# -------------------------------
# 6. Evaluate with Resampling & Tuning
# -------------------------------
def find_best_model(X_train, y_train, X_val, y_val):
    overall_best = {"score": -1}
    samplers = get_samplers()

    for name, sampler in samplers.items():
        print(f"\n--- Resampling: {name} ---")
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        print(" Resampled shape:", X_res.shape)
        print(" Class dist:", pd.Series(y_res).value_counts().to_dict())

        rf_model, rf_score = tune_rf(X_res, y_res, X_val, y_val)
        ada_model, ada_score = tune_ada(X_res, y_res, X_val, y_val)

        if rf_score >= ada_score:
            chosen, score, method = rf_model, rf_score, "RandomForest"
        else:
            chosen, score, method = ada_model, ada_score, "AdaBoost"

        print(f" Best with {name}: {method} (combined={score:.4f})")
        if score > overall_best["score"]:
            overall_best.update({
                "model": chosen,
                "resampling": name,
                "model_type": method,
                "score": score
            })

    print("\nOverall Best:", {
        k: overall_best[k] for k in ("resampling", "model_type", "score")
    })
    return overall_best["model"], overall_best

# -------------------------------
# 7. Test Set Evaluation
# -------------------------------
def evaluate_on_test(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    print("\nTest Set Performance:")
    print(" Recall: ", recall_score(y_test, preds))
    print(" AUROC:  ", roc_auc_score(y_test, proba))
    print(" Precision:", precision_score(y_test, preds))
    print(" F1 Score: ", f1_score(y_test, preds))
    print(" Classification Report:\n", classification_report(y_test, preds))
    print(" Confusion Matrix:\n", confusion_matrix(y_test, preds))

# -------------------------------
# 8. Save Best Model
# -------------------------------
def save_model(model, save_dir="models", fname="traditional_model.pkl"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    proc_dir = os.path.join("data", "processed")
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets(proc_dir)
    X_train_enc, X_val_enc, X_test_enc = encode_and_align(X_train, X_val, X_test)

    best_model, info = find_best_model(X_train_enc, y_train, X_val_enc, y_val)
    evaluate_on_test(best_model, X_test_enc, y_test)
    save_model(best_model)

if __name__ == "__main__":
    main()


# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 