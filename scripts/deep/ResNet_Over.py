#!/usr/bin/env python
"""
Load processed Home Credit data, build & train a ResNet-style classifier
with focal loss and SMOTEENN resampling, then evaluate and save the model.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from math import log
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Add, BatchNormalization, Activation
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_recall_curve, recall_score,
                             precision_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from imblearn.combine import SMOTEENN

# -------------------------------
# 1. Load Processed Datasets
# -------------------------------
def load_datasets(processed_dir: str, target_col: str = "TARGET"):
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(processed_dir, "validation.csv"))
    test  = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train_raw = train.drop(columns=[target_col])
    y_train     = train[target_col]
    X_val_raw   = val.drop(columns=[target_col])
    y_val       = val[target_col]
    X_test_raw  = test.drop(columns=[target_col])
    y_test      = test[target_col]

    return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test

# -------------------------------
# 2. One-Hot Encode & Align
# -------------------------------
def encode_and_align(X_train_raw, X_val_raw, X_test_raw):
    X_train = pd.get_dummies(X_train_raw)
    X_val, X_test = pd.get_dummies(X_val_raw), pd.get_dummies(X_test_raw)
    X_train, X_val   = X_train.align(X_val,   join="outer", axis=1, fill_value=0)
    X_train, X_test  = X_train.align(X_test,  join="outer", axis=1, fill_value=0)
    return X_train, X_val, X_test

# -------------------------------
# 3. Feature Scaling
# -------------------------------
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

# -------------------------------
# 4. Resample with SMOTEENN
# -------------------------------
def resample_smoteenn(X_train_scaled, y_train, sampling_strategy=0.3, random_state=42):
    sampler = SMOTEENN(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
    pos_rate = y_res.mean()
    init_bias = log(pos_rate / (1 - pos_rate))
    return X_res, y_res, init_bias

# -------------------------------
# 5. Focal Loss Definition
# -------------------------------
def focal_loss(gamma=2.0, alpha=0.7):
    def loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        loss1 = -alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())
        loss0 = -(1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon())
        return K.mean(loss1 + loss0)
    return loss

# -------------------------------
# 6. Build Residual Neural Network
# -------------------------------
def residual_block(x, units, dropout_rate=0.3):
    skip = x
    x = Dense(units, kernel_regularizer=l1_l2(1e-5,1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units, kernel_regularizer=l1_l2(1e-5,1e-4))(x)
    x = BatchNormalization()(x)

    if skip.shape[-1] != units:
        skip = Dense(units)(skip)

    x = Add()([x, skip])
    x = Activation("relu")(x)
    return Dropout(dropout_rate)(x)

def build_resnet(input_dim: int, init_bias: float):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation="relu", kernel_regularizer=l1_l2(1e-5,1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = residual_block(x, 128)
    x = residual_block(x, 64)
    x = residual_block(x, 32, dropout_rate=0.2)

    outputs = Dense(1, activation="sigmoid",
                    bias_initializer=tf.keras.initializers.Constant(init_bias))(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss=focal_loss(),
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    return model

# -------------------------------
# 7. Train Model
# -------------------------------
def train_model(model, X_train, y_train, X_val, y_val, class_weight):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=256,
        class_weight=class_weight,
        verbose=0
    )
    return history

# -------------------------------
# 8. Find Optimal Threshold
# -------------------------------
def find_optimal_threshold(model, X_val, y_val, min_prec=0.2, min_rec=0.4):
    probs = model.predict(X_val).ravel()
    precisions, recalls, thresh = precision_recall_curve(y_val, probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
    best_idx = np.argmax(f1s)
    optimal = thresh[best_idx]

    balanced = []
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() in (0, len(preds)): continue
        pr, rc = precision_score(y_val, preds), recall_score(y_val, preds)
        if pr >= min_prec and rc >= min_rec:
            f1 = 2*(pr*rc)/(pr+rc+1e-7)
            balanced.append((t, f1))
    return balanced[0][0] if balanced else optimal

# -------------------------------
# 9. Evaluate Model
# -------------------------------
def evaluate_model(model, X_test, y_test, threshold: float):
    probs = model.predict(X_test).ravel()
    preds = (probs >= threshold).astype(int)

    print(f"Test AUC:    {roc_auc_score(y_test, probs):.4f}")
    print(f"Test Recall: {recall_score(y_test, preds):.4f}")
    print(f"Test F1:     {f1_score(y_test, preds):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# -------------------------------
# 10. Save Model
# -------------------------------
def save_model(model, save_dir="models", fname="resnet_oversample.h5"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    model.save(path)
    print(f"Model saved to {path}")

# -------------------------------
# Main Execution
# -------------------------------
def main():
    processed_dir = os.path.join("data", "processed")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = load_datasets(processed_dir)

    X_train, X_val, X_test = encode_and_align(X_train_raw, X_val_raw, X_test_raw)
    X_train_s, X_val_s, X_test_s = scale_features(X_train, X_val, X_test)
    X_res, y_res, bias = resample_smoteenn(X_train_s, y_train)
    
    print(f"Initial bias (log-odds): {bias:.4f}")
    model = build_resnet(X_res.shape[1], bias)
    class_weights = {0:1.0, 1: (len(y_res)-y_res.sum())/y_res.sum()}
    
    train_model(model, X_res, y_res, X_val_s, y_val, class_weights)
    thresh = find_optimal_threshold(model, X_val_s, y_val)
    print(f"Chosen threshold: {thresh:.4f}")

    evaluate_model(model, X_test_s, y_test, thresh)
    save_model(model)

if __name__ == "__main__":
    main()

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 