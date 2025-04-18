#!/usr/bin/env python
import os
import warnings
import pickle
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Create models directory if it doesn't exist
models_dir = os.path.join(".", "models")
os.makedirs(models_dir, exist_ok=True)

# Define directory paths
processed_dir = os.path.join("data", "processed")
train_path = os.path.join(processed_dir, "train.csv")
test_path = os.path.join(processed_dir, "test.csv")

# Load training and test data
print("Loading training and test datasets...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Define the target column
target_col = "TARGET"

# Separate features and target for training and testing
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# Use only numeric features for logistic regression
X_train = X_train.select_dtypes(include=["int64", "float64"])
X_test = X_test.select_dtypes(include=["int64", "float64"])

# Initialize and train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
print("\nTraining Logistic Regression model...")
model.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate performance metrics with focus on ROC AUC and recall
roc_auc = roc_auc_score(y_test, y_prob)
recall = recall_score(y_test, y_pred)

print("\nPerformance Metrics:")
print("ROC AUC: {:.4f}".format(roc_auc))
print("Recall: {:.4f}".format(recall))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model as a pickle file
model_filename = os.path.join(models_dir, "logistic_regression_model.pkl")
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"\nModel saved to {model_filename}")
