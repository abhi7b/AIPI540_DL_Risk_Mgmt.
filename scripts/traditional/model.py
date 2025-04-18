#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import ADASYN
from imblearn.combine import SMOTEENN
from sklearn.model_selection import ParameterGrid

# -------------------------------
# 1. Load Processed Datasets
# -------------------------------
processed_dir = os.path.join("data", "processed")
train_path = os.path.join(processed_dir, "train.csv")
val_path   = os.path.join(processed_dir, "validation.csv")
test_path  = os.path.join(processed_dir, "test.csv")

print("Loading datasets...")
train_df = pd.read_csv(train_path)
val_df   = pd.read_csv(val_path)
test_df  = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

target_col = "TARGET"

# Separate features and target
X_train_raw = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_val_raw   = val_df.drop(columns=[target_col])
y_val = val_df[target_col]
X_test_raw  = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# -------------------------------
# 2. One-Hot Encode & Align Columns
# -------------------------------
print("\nPerforming one-hot encoding on features...")
X_train = pd.get_dummies(X_train_raw)
X_val = pd.get_dummies(X_val_raw)
X_test = pd.get_dummies(X_test_raw)

# Align columns across datasets (fill missing columns with 0)
X_train, X_val = X_train.align(X_val, join="outer", axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

# Reindex validation and test sets to ensure column order matches X_train
X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

# -------------------------------
# 3. Define Resampling Techniques (ADASYN and SMOTEENN only)
# -------------------------------
samplers = {
    "ADASYN": ADASYN(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42)
}

# -------------------------------
# 4. Evaluation Function (Recall & AUROC)
# -------------------------------
def score_model(model, X_val, y_val):
    """Return recall, AUROC, and combined metric (average) using validation set."""
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    rec = recall_score(y_val, y_pred, pos_label=1)
    auroc = roc_auc_score(y_val, y_proba)
    combined = (rec + auroc) / 2
    return rec, auroc, combined

# -------------------------------
# 5. Hyperparameter Tuning for RandomForest and AdaBoost
# -------------------------------
def tune_rf(X_train, y_train, X_val, y_val):
    best_score = -1
    best_model = None
    best_params = None
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "class_weight": ["balanced"],
        "random_state": [42]
    }
    for params in ParameterGrid(param_grid):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        rec, auroc, combined = score_model(model, X_val, y_val)
        print(f"RF params: {params}, Validation Recall: {rec:.4f}, AUROC: {auroc:.4f}, Combined: {combined:.4f}")
        if combined > best_score:
            best_score = combined
            best_model = model
            best_params = params
    print("Best RF parameters:", best_params)
    print("Best RF Combined Metric: {:.4f}".format(best_score))
    return best_model, best_score

def tune_ada(X_train, y_train, X_val, y_val):
    best_score = -1
    best_model = None
    best_params = None
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [1.0, 0.5],
        "random_state": [42]
    }
    for params in ParameterGrid(param_grid):
        model = AdaBoostClassifier(**params)
        model.fit(X_train, y_train)
        rec, auroc, combined = score_model(model, X_val, y_val)
        print(f"AdaBoost params: {params}, Validation Recall: {rec:.4f}, AUROC: {auroc:.4f}, Combined: {combined:.4f}")
        if combined > best_score:
            best_score = combined
            best_model = model
            best_params = params
    print("Best AdaBoost parameters:", best_params)
    print("Best AdaBoost Combined Metric: {:.4f}".format(best_score))
    return best_model, best_score

# -------------------------------
# 6. Loop Over Resampling Techniques, Tune, and Save Best Model
# -------------------------------
overall_best_score = -1
overall_best_model = None
overall_best_info = {}

for sampler_name, sampler in samplers.items():
    print("\n==============================================")
    print("Resampling using:", sampler_name)
    X_train_res_array, y_train_res = sampler.fit_resample(X_train, y_train)
    X_train_res = pd.DataFrame(X_train_res_array, columns=X_train.columns)
    
    print("Resampled training shape:", X_train_res.shape)
    print("Class distribution after", sampler_name, ":")
    print(pd.Series(y_train_res).value_counts())
    
    print("\nTuning Random Forest with", sampler_name)
    rf_model, rf_score = tune_rf(X_train_res, y_train_res, X_val, y_val)
    
    print("\nTuning AdaBoost with", sampler_name)
    ada_model, ada_score = tune_ada(X_train_res, y_train_res, X_val, y_val)
    
    if rf_score >= ada_score:
        best_model = rf_model
        best_method = "RandomForest"
        best_score = rf_score
    else:
        best_model = ada_model
        best_method = "AdaBoost"
        best_score = ada_score
    
    print(f"\nBest model with {sampler_name} is {best_method} with Combined Metric: {best_score:.4f}")
    if best_score > overall_best_score:
        overall_best_score = best_score
        overall_best_model = best_model
        overall_best_info = {
            "resampling": sampler_name,
            "model_type": best_method,
            "combined_metric": best_score
        }

print("\n==============================================")
print("Overall Best Model Info:")
print(overall_best_info)

# -------------------------------
# 7. Evaluate Best Model on Test Set
# -------------------------------
print("\nEvaluating overall best model on test set...")
y_test_pred = overall_best_model.predict(X_test)
test_recall = recall_score(y_test, y_test_pred, pos_label=1)
y_test_proba = overall_best_model.predict_proba(X_test)[:, 1]
test_auroc = roc_auc_score(y_test, y_test_proba)
test_precision = precision_score(y_test, y_test_pred, pos_label=1)
test_f1 = f1_score(y_test, y_test_pred, pos_label=1)

print("Test Recall: {:.4f}".format(test_recall))
print("Test AUROC: {:.4f}".format(test_auroc))
print(f"Test Precision: {test_precision:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_test_pred)
print(conf_matrix)

# -------------------------------
# 8. Save the Best Model
# -------------------------------
save_dir = "./models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "tradtional_model.pkl")

import joblib
joblib.dump(overall_best_model, model_path)
print(f"Model saved to {model_path}")

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 