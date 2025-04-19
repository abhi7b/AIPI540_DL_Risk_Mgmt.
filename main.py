#!/usr/bin/env python
"""
main.py

Streamlit app to predict Credit default risk using four pre-trained models
and a shared DataProcessor for preprocessing. Shows probabilities and predictions
for each model in tidy tabs, and lets you download all results.
"""

import warnings
warnings.filterwarnings("ignore", message="coroutine 'expire_cache' was never awaited")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
import joblib

from scripts.dataset.processing import DataProcessor

# ────────────────────────────────────────────────────────────────────────────────
# 1. Cached resource loaders
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_processor(path="models/processor.pkl") -> DataProcessor:
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_logistic(path="models/logistic_regression_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_traditional(path="models/traditional_model.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_deep_over(path="models/resnet_oversample.h5"):
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_deep_under(path="models/resnet_undersample.h5"):
    return tf.keras.models.load_model(path, compile=False)

# ────────────────────────────────────────────────────────────────────────────────
# 2. Prediction helper functions
# ────────────────────────────────────────────────────────────────────────────────
def predict_sklearn(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Use numpy arrays to bypass feature-name checks.
    Returns a DataFrame with probability & binary prediction.
    """
    arr = X.values
    proba = model.predict_proba(arr)[:, 1]
    pred  = model.predict(arr)
    return pd.DataFrame({
        "Default Probability": proba,
        "Prediction": pred
    }, index=X.index)

def predict_deep(model, X_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with probability & binary prediction for Keras models.
    """
    proba = model.predict(X_scaled).ravel()
    pred  = (proba >= 0.5).astype(int)
    return pd.DataFrame({
        "Default Probability": proba,
        "Prediction": pred
    }, index=X_scaled.index if hasattr(X_scaled, 'index') else None)

# ────────────────────────────────────────────────────────────────────────────────
# 3. Main Streamlit app
# ────────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Credit Risk", layout="wide")
    st.title("Credit Default Risk Prediction")
    st.markdown(
        "Upload a CSV of applicant data (feature set must match training) "
        "to generate default risk probabilities and predictions from four models."
    )

    # Load preprocessing pipeline & models
    processor        = load_processor()
    lr_model         = load_logistic()
    trad_model       = load_traditional()
    deep_over_model  = load_deep_over()
    deep_under_model = load_deep_under()

    uploaded = st.file_uploader("Upload applicant CSV", type="csv")
    if not uploaded:
        st.info("Please upload a CSV file to continue.")
        return

    df = pd.read_csv(uploaded)
    # Preprocess
    X_num, X_ohe, X_scaled = processor.transform(df)
    # NOTE: X_scaled is numpy array; wrap to DataFrame with same index
    X_scaled_df = pd.DataFrame(X_scaled, index=df.index)

    # Generate predictions per model
    preds = {
        "Logistic Regression": predict_sklearn(lr_model, X_num),
        "Traditional ML (RF/AdaBoost)": predict_sklearn(trad_model, X_ohe),
        "Deep Model (Oversample)": predict_deep(deep_over_model, X_scaled_df),
        "Deep Model (Undersample)": predict_deep(deep_under_model, X_scaled_df),
    }

    # Display each model in its own tab
    tabs = st.tabs(list(preds.keys()))
    for (name, df_out), tab in zip(preds.items(), tabs):
        with tab:
            st.header(name)
            st.dataframe(
                df_out.style.format({"Default Probability": "{:.2%}"}),
                use_container_width=True
            )

    # Combine all predictions into one DataFrame for download
    combined = []
    for model_name, df_out in preds.items():
        tmp = df_out.copy()
        tmp.insert(0, "Model", model_name)
        tmp.insert(1, "Applicant Index", df_out.index)
        combined.append(tmp)
    combined_df = pd.concat(combined, ignore_index=True)

    csv = combined_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download All Model Predictions",
        data=csv,
        file_name="all_model_predictions.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

# Code has been generated using Deepseek, Chatgpt, and Claude.ai then tweaked 