#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from math import log
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Add, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

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

# Original class distribution
print("\nClass distribution in training data:")
print(y_train.value_counts(normalize=True))

# -------------------------------
# 2. One-Hot Encode & Align Columns
# -------------------------------
print("\nPerforming one-hot encoding on features...")
X_train = pd.get_dummies(X_train_raw)
X_val   = pd.get_dummies(X_val_raw)
X_test  = pd.get_dummies(X_test_raw)

# Align columns across datasets (fill missing columns with 0)
X_train, X_val = X_train.align(X_val, join="outer", axis=1, fill_value=0)
X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

# -------------------------------
# 3. Feature Scaling - Important for faster convergence
# -------------------------------
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Apply Balanced Undersampling
# -------------------------------
print("\nApplying Random Undersampling to achieve a perfectly balanced dataset...")
undersampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
X_train_res, y_train_res = undersampler.fit_resample(X_train_scaled, y_train)

print("Resampled training shape:", X_train_res.shape)
print("Class distribution after balanced undersampling:")
print(pd.Series(y_train_res).value_counts(normalize=True))

# Calculate new bias value based on balanced data
pos_rate_resampled = y_train_res.mean()
init_bias = log(pos_rate_resampled/(1-pos_rate_resampled))
print("Balanced data bias value (log odds): {:.4f}".format(init_bias))

# -------------------------------
# 5. Build Residual Neural Network Architecture
# -------------------------------
print("\nBuilding residual neural network model...")

# Define a function to create a residual block with LeakyReLU activation
def residual_block(x, units, dropout_rate=0.3):
    x_skip = x  # preserve input for the skip connection

    # First layer in the block
    x = Dense(units, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer in the block
    x = Dense(units, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    
    # Transform the skip connection if dimensions don't match
    if x_skip.shape[-1] != units:
        x_skip = Dense(units, kernel_initializer='he_normal')(x_skip)
    
    # Add skip connection
    x = Add()([x, x_skip])
    x = LeakyReLU()(x)
    x = Dropout(dropout_rate)(x)
    
    return x

# Build the model using the Functional API
inputs = Input(shape=(X_train_res.shape[1],))

# Initial Dense layer with deeper input dimension and LeakyReLU
x = Dense(256, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(0.3)(x)

# First residual block
x = residual_block(x, 256)

# Second residual block - reducing dimensions
x = residual_block(x, 128)

# Third residual block with further reduced dimensions
x = residual_block(x, 64, dropout_rate=0.25)

# Fourth residual block - an extra block for increased depth
x = residual_block(x, 32, dropout_rate=0.2)

# Output layer with bias initializer set using the balanced data bias (log odds)
outputs = Dense(1, activation='sigmoid', 
                bias_initializer=tf.keras.initializers.Constant(init_bias))(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

print("Using no class weights due to balanced undersampling")

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

# Print model summary
model.summary()

# -------------------------------
# 6. Model Training (Increased Epochs with Early Stopping)
# -------------------------------
print("\nTraining model for 50 epochs (with early stopping and learning rate reduction)...")
lr_reduce = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, 
                              mode='max', min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True)

history = model.fit(X_train_res, y_train_res, 
                    epochs=50,
                    batch_size=256,
                    validation_data=(X_val_scaled, y_val),
                    callbacks=[lr_reduce, early_stop],
                    verbose=1)

# -------------------------------
# 7. Find Optimal Threshold using F1 Score
# -------------------------------
print("\nFinding optimal threshold using F1 score...")
val_probs = model.predict(X_val_scaled).ravel()
print("Validation probability range: min={:.4f}, max={:.4f}".format(val_probs.min(), val_probs.max()))

precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-7)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold (based on F1): {optimal_threshold:.4f}")

balanced_thresholds = []
min_precision = 0.2  
min_recall = 0.4     

for thresh in np.arange(0.1, 0.9, 0.01):
    y_pred = (val_probs >= thresh).astype(int)
    if np.all(y_pred == 0) or np.all(y_pred == 1):
        continue
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    
    if rec >= min_recall and prec >= min_precision:
        f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
        balanced_thresholds.append((thresh, f1, prec, rec))

if balanced_thresholds:
    balanced_thresholds.sort(key=lambda x: x[1], reverse=True)
    best_threshold = balanced_thresholds[0][0]
    print(f"Selected balanced threshold: {best_threshold:.4f}")
    print(f"  - F1: {balanced_thresholds[0][1]:.4f}")
    print(f"  - Precision: {balanced_thresholds[0][2]:.4f}")
    print(f"  - Recall: {balanced_thresholds[0][3]:.4f}")
else:
    best_threshold = optimal_threshold
    print(f"No thresholds met both precision and recall requirements. Using F1-optimal: {best_threshold:.4f}")

# -------------------------------
# 8. Evaluate on Test Set
# -------------------------------
print("\nEvaluating on test set...")
test_probs = model.predict(X_test_scaled).ravel()
y_test_pred = (test_probs >= best_threshold).astype(int)

acc = np.mean(y_test == y_test_pred)
recall = recall_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
auc = roc_auc_score(y_test, test_probs)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Recall (Sensitivity): {recall:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test F1 Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_test_pred)
print(conf_matrix)

# -------------------------------
# 9. Save the Model
# -------------------------------
save_dir = os.path.join(os.getcwd(), "models")
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "credit_risk_deep_model_under.h5")
model.save(model_path)
print(f"\nModel saved to {model_path}")

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 