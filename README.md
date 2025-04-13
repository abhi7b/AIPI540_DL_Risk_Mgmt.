
# Exploring Deep-Learning Approaches in Risk Management

## Overview

Financial institutions face significant credit risks related to defaults, fraud, and inaccurate credit scoring. Traditional risk models rely heavily on historical data and predefined rules, which may fail to capture complex patterns in borrower behavior. This project explores the application of deep learning techniques to enhance credit risk assessment, aiming to identify hidden risk indicators and provide more accurate predictions.​

---

## Data Processing Pipeline

### Data Download and Loading

### Data Cleaning and Transformation

### 💾 Data Storage

---

## Modeling Approaches 

### Naïve Rule-Based Classifier

### Traditional Machine Learning

### Deep Learning 

- **ResNet with Oversampling**
This model uses a a residual neural network architecture, incorporating three residual blocks with dropout, batch normalization, and L1/L2 regularization to improve generalization. To address class imbalance, SMOTEENN (a combination of SMOTE oversampling and Edited Nearest Neighbors cleaning) is applied to the training data. The model is trained using a custom focal loss function to focus on improving performance on the minority class. 

- **ResNet with Undersampling**  
This model uses a deeper residual neural network architecture with four residual blocks and LeakyReLU activations to enhance gradient flow and learning capacity. It starts with a 256-unit input layer and progressively reduces dimensionality through residual connections. Binary cross-entropy is used as the loss function to guide learning on the binary classification task. To handle class imbalance, a Random Undersampling technique is applied to create a perfectly balanced dataset, removing the need for class weighting.


---

## Evaluation Metric

The models were primarily evaluated using AUC and recall. AUC (Area Under the ROC Curve) was used to measure the model’s ability to distinguish between defaulters and non-defaulters across all thresholds, providing a robust, threshold-independent assessment. Recall was emphasized to ensure that the model successfully identified as many actual defaulters as possible, which is critical in minimizing credit risk exposure. Precision, F1 score, and threshold optimization are highlighted, but AUC and recall guided the main evaluation.

---

## User Interface

- **Streamlit Web App**: 
WIP

---

## Setup

```bash
./setup.sh
```

This script takes care of setting up your virtual environment if it does not already exist, activating it, installing requirements, pulling the dataset (if not already present in the data directory), pre-processing the data for the traditional model (feature extraction), and traditional model training.

## Running the Streamlit application locally

Assuming your virtual environment is setup and activated, and that the requirements are installed from running `setup.sh`,
you can then run the following to startup a local instance of the Streamlit application.

```bash
python main.py
```

---

## Dataset & License
This repository uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview) dataset from Kaggle Competition, licensed under Kaggle competition rules.

---

## **Ethics Statement**  

This project uses publicly available financial datasets in compliance with their terms of use. All data is handled responsibly, with no collection or use of personally identifiable information (PII). We are committed to preventing misuse, protecting privacy, and avoiding any unauthorized distribution. Our goal is to advance responsible AI in financial services while upholding ethical standards and data integrity.

---

## Presentation Link

View our presentation [HERE](https://docs.google.com/presentation/d/1f10f97H5Tj7s4oodW_kLxO4mKXoLSJzMlBV520TZrPM/edit?usp=sharing).

---

## Streamlit Application

WIP
Access our Streamlit application [HERE]().
