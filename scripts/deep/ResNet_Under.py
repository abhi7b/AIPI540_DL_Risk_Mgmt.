#!/usr/bin/env python
"""

Load processed Home Credit data, apply balanced undersampling,
build & train a deep ResNet-style classifier, then evaluate and save the model.
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
from tensorflow.keras.layers import (
    Dense, Dropout, Input, Add,
    BatchNormalization, LeakyReLU
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, recall_score,
    precision_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.under_sampling import RandomUnderSampler

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

    print("Loaded datasets:")
    print(f"  Train:      {X_train_raw.shape}, labels: {y_train.shape}")
    print(f"  Validation: {X_val_raw.shape}, labels: {y_val.shape}")
    print(f"  Test:       {X_test_raw.shape}, labels: {y_test.shape}")

    return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test

# -------------------------------
# 2. One-Hot Encode & Align Columns
# -------------------------------
def encode_and_align(X_train_raw, X_val_raw, X_test_raw):
    X_train = pd.get_dummies(X_train_raw)
    X_val   = pd.get_dummies(X_val_raw)
    X_test  = pd.get_dummies(X_test_raw)

    X_train, X_val   = X_train.align(X_val,   join="outer", axis=1, fill_value=0)
    X_train, X_test  = X_train.align(X_test,  join="outer", axis=1, fill_value=0)

    print("After one-hot & align:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    return X_train, X_val, X_test

# -------------------------------
# 3. Feature Scaling
# -------------------------------
def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print("Features scaled.")
    return X_train_s, X_val_s, X_test_s

# -------------------------------
# 4. Apply Balanced Undersampling
# -------------------------------
def undersample_data(X_train_s, y_train, sampling_strategy: float = 1.0, random_state: int = 42):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train_s, y_train)

    pos_rate = y_res.mean()
    init_bias = log(pos_rate / (1 - pos_rate))

    print("After undersampling:")
    print(f"  Resampled shape: {X_res.shape}")
    print(f"  Class distribution: {pd.Series(y_res).value_counts(normalize=True).to_dict()}")
    print(f"  Bias (log-odds): {init_bias:.4f}")

    return X_res, y_res, init_bias

# -------------------------------
# 5. Build Residual Block & Model
# -------------------------------
def residual_block(x, units, dropout_rate=0.3):
    skip = x
    x = Dense(units, kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units, kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
    x = BatchNormalization()(x)

    if skip.shape[-1] != units:
        skip = Dense(units)(skip)

    x = Add()([x, skip])
    x = LeakyReLU()(x)
    return Dropout(dropout_rate)(x)

def build_resnet(input_dim: int, init_bias: float, lr: float = 7e-4):
    inputs = Input(shape=(input_dim,))
    x = Dense(256, kernel_regularizer=l1_l2(1e-5,1e-4))(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    for units, dr in [(256,0.3), (128,0.3), (64,0.25), (32,0.2)]:
        x = residual_block(x, units, dropout_rate=dr)

    outputs = Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(init_bias)
    )(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    print("Model built and compiled.")
    return model

# -------------------------------
# 6. Train Model with Callbacks
# -------------------------------
def train_model(model, X_train, y_train, X_val, y_val,
                epochs: int = 50, batch_size: int = 256):
    lr_reduce = ReduceLROnPlateau(
        monitor="val_auc", factor=0.5, patience=3, mode="max", verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_auc", patience=5, mode="max", restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_reduce, early_stop],
        verbose=1
    )
    print("Training complete.")
    return history

# -------------------------------
# 7. Find Optimal Threshold
# -------------------------------
def find_optimal_threshold(model, X_val, y_val,
                           min_prec: float = 0.2, min_rec: float = 0.4):
    probs = model.predict(X_val).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-7)
    best_idx = np.argmax(f1s)
    optimal = thresholds[best_idx]

    balanced = []
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        if preds.sum() in (0, len(preds)): continue
        pr = precision_score(y_val, preds)
        rc = recall_score(y_val, preds)
        if pr >= min_prec and rc >= min_rec:
            f1 = 2*(pr*rc)/(pr+rc+1e-7)
            balanced.append((t, f1))
    chosen = balanced[0][0] if balanced else optimal
    print(f"Chosen threshold: {chosen:.4f}")
    return chosen

# -------------------------------
# 8. Evaluate Model
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
# 9. Save Model
# -------------------------------
def save_model(model, save_dir: str = "models", fname: str = "resnet_undersample.h5"):
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
    X_res, y_res, bias = undersample_data(X_train_s, y_train)

    model = build_resnet(input_dim=X_res.shape[1], init_bias=bias)
    train_model(model, X_res, y_res, X_val_s, y_val)

    threshold = find_optimal_threshold(model, X_val_s, y_val)
    evaluate_model(model, X_test_s, y_test, threshold)

    save_model(model)

if __name__ == "__main__":
    main()


# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 