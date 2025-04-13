#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import shap
from math import log
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Add, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

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

# Save column names for SHAP analysis later
feature_names = X_train.columns.tolist()

# -------------------------------
# 3. Feature Scaling - Important for faster convergence
# -------------------------------
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 4. Resample Using SMOTEENN - Combined oversampling and cleaning
# -------------------------------
print("\nApplying SMOTEENN to the training data...")
# Use SMOTEENN which combines SMOTE with Edited Nearest Neighbors
sampler = SMOTEENN(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = sampler.fit_resample(X_train_scaled, y_train)
print("Resampled training shape:", X_train_res.shape)
print("Class distribution after SMOTEENN:")
print(pd.Series(y_train_res).value_counts(normalize=True))

# Calculate new bias value based on resampled data
pos_rate_resampled = y_train_res.mean()
init_bias = log(pos_rate_resampled/(1-pos_rate_resampled))
print("Resampled data bias value (log odds): {:.4f}".format(init_bias))

# -------------------------------
# 5. Create Focal Loss Function to focus on hard examples
# -------------------------------
def focal_loss(gamma=2.0, alpha=0.7):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# -------------------------------
# 6. Build Residual Neural Network Architecture
# -------------------------------
print("\nBuilding residual neural network model...")

# Define a function to create a residual block
def residual_block(x, units, dropout_rate=0.3):
    # Store input for the skip connection
    x_skip = x
    
    # First layer in the block
    x = Dense(units, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # Second layer in the block
    x = Dense(units, kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    
    # If dimensions don't match, transform skip connection
    if x_skip.shape[-1] != units:
        x_skip = Dense(units, kernel_initializer='he_normal')(x_skip)
    
    # Add skip connection
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    return x

# Build the model using the Functional API
inputs = Input(shape=(X_train_res.shape[1],))

# Initial layer
x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# First residual block
x = residual_block(x, 128)

# Second residual block
x = residual_block(x, 64)

# Third residual block with reduced dimensions
x = residual_block(x, 32, dropout_rate=0.2)

# Output layer
outputs = Dense(1, activation='sigmoid', 
               bias_initializer=tf.keras.initializers.Constant(init_bias))(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Adjusted class weights to reflect the class distribution after SMOTEENN
class_weight = {0: 1.0, 1: 4.0}  # Further increased weight for minority class
print(f"Using class weights: {class_weight}")

# Compile with focal loss
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=focal_loss(gamma=2.0, alpha=0.7),
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.Recall(name='recall')])

# Model summary
model.summary()

# -------------------------------
# 7. Model Training with Callbacks
# -------------------------------
print("\nTraining model...")
early_stop = EarlyStopping(monitor='val_auc', patience=7, mode='max', 
                          restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, 
                             mode='max', min_lr=1e-6, verbose=0)

history = model.fit(X_train_res, y_train_res, 
                    epochs=30,  # Increased max epochs for residual network
                    batch_size=256,
                    validation_data=(X_val_scaled, y_val),
                    class_weight=class_weight,
                    callbacks=[early_stop, lr_reduce],
                    verbose=0)

# -------------------------------
# 8. Find Optimal Threshold using Multiple Criteria
# -------------------------------
print("\nFinding optimal threshold using multiple criteria...")
val_probs = model.predict(X_val_scaled).ravel()
print("Validation probability range: min={:.4f}, max={:.4f}".format(val_probs.min(), val_probs.max()))

# Use precision-recall curve to find optimal F1 threshold
precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-7)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold (based on F1): {optimal_threshold:.4f}")

# Balance precision and recall with minimum requirements
balanced_thresholds = []
min_precision = 0.2  # Set minimum acceptable precision
min_recall = 0.4     # Set minimum acceptable recall

for thresh in np.arange(0.1, 0.9, 0.01):
    y_pred = (val_probs >= thresh).astype(int)
    # Skip if all predictions are the same class
    if np.all(y_pred == 0) or np.all(y_pred == 1):
        continue
    rec = recall_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    
    # Only consider thresholds that meet minimum requirements
    if rec >= min_recall and prec >= min_precision:
        f1 = 2 * (prec * rec) / (prec + rec + 1e-7)
        balanced_thresholds.append((thresh, f1, prec, rec))

if balanced_thresholds:
    # Get threshold with highest F1 among those meeting requirements
    balanced_thresholds.sort(key=lambda x: x[1], reverse=True)
    best_threshold = balanced_thresholds[0][0]
    print(f"Selected balanced threshold: {best_threshold:.4f}")
    print(f"  - F1: {balanced_thresholds[0][1]:.4f}")
    print(f"  - Precision: {balanced_thresholds[0][2]:.4f}")
    print(f"  - Recall: {balanced_thresholds[0][3]:.4f}")
else:
    # If no threshold meets our criteria, use the F1-optimal one
    best_threshold = optimal_threshold
    print(f"No thresholds met both precision and recall requirements. Using F1-optimal: {best_threshold:.4f}")

# -------------------------------
# 9. Evaluate on Test Set
# -------------------------------
print("\nEvaluating on test set...")
test_probs = model.predict(X_test_scaled).ravel()
y_test_pred = (test_probs >= best_threshold).astype(int)

# Calculate metrics
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

# Report on default detection performance
tn, fp, fn, tp = conf_matrix.ravel()
print(f"\nDefault Detection Performance:")
print(f"Correctly identified defaults: {tp} out of {tp + fn} ({tp/(tp+fn):.2%})")
print(f"Falsely flagged as defaults: {fp} out of {tn + fp} ({fp/(tn+fp):.2%})")


# -------------------------------
# 9. Save the Model
# -------------------------------
save_dir = os.path.join(os.getcwd(), "models")
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "credit_risk_deep_model_over.h5")
model.save(model_path)
print(f"\nModel saved to {model_path}")
