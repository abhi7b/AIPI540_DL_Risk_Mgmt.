
# Exploring Deep-Learning Approaches in Risk Management

## Overview

Financial institutions face significant credit risks related to defaults, fraud, and inaccurate credit scoring. Traditional risk models rely heavily on historical data and predefined rules, which may fail to capture complex patterns in borrower behavior. This project explores the application of deep learning techniques to enhance credit risk assessment, aiming to identify hidden risk indicators and provide more accurate predictions.​

---

## Data Processing Pipeline

The pipeline is designed for memory efficiency and clean dataset preparation.
The pipeline reads the `application_train.csv` file from the Home Credit raw data directory.

### 1. **Raw Data Merging**
Using Dask for memory-efficient processing of large CSV files, the pipeline merges the application training dataset from the Home Credit raw data directory.

### 2. **Data Cleaning**
Performs data preprocessing using pandas, including:
- Dropping columns with excessive missing values
- Removing rows with any remaining nulls
- Saving a cleaned dataset for downstream use

### 3. **Data Splitting**
Implements a stratified split into:
- Training set: 60%
- Validation set: 20%
- Test set: 20%

This ensures that the `TARGET` variable is proportionally distributed across all subsets, preserving label balance for modeling.

### 4. **Memory & File Management**
- Unused intermediate files are deleted to save disk space
- Garbage collection is invoked after each major step to free memory and ensure performance

---

## Modeling Approaches 

### Naïve

This approach implements a baseline logistic regression model. The model is trained exclusively on numeric features from the dataset, using logistic regression with default parameters, except for increased maximum iterations (1000) to ensure convergence. No feature engineering or data resampling techniques are applied, establishing this as a foundational benchmark. 
The model is evaluated primarily through ROC AUC and recall metrics to address potential class imbalance concerns.

### Traditional Machine Learning

This approach uses ensemble models—Random Forest and AdaBoost. It evaluates two resampling techniques - ADASYN (adaptive synthetic oversampling) and SMOTEENN (combined oversampling and cleaning) to address class imbalance. The models are trained using the binary cross-entropy loss (implicitly through sklearn's built-in classifiers) and evaluated using recall and AUC. The validation set is used to tune hyperparameters for each resampling-model pair, and the best model is selected based on a combined recall–AUC validation score and evaluated on test set.

### Deep Learning 

- **ResNet with Oversampling**

This model uses a a residual neural network architecture, incorporating three residual blocks with dropout, batch normalization, and L1/L2 regularization to improve generalization. To address class imbalance, SMOTEENN (a combination of SMOTE oversampling and Edited Nearest Neighbors cleaning) is applied to the training data. The model is trained using a custom focal loss function to focus on improving performance on the minority class. 

- **ResNet with Undersampling**

This model uses a deeper residual neural network architecture with four residual blocks and LeakyReLU activations to enhance gradient flow and learning capacity. It starts with a 256-unit input layer and progressively reduces dimensionality through residual connections. Binary cross-entropy is used as the loss function to guide learning on the binary classification task. To handle class imbalance, a Random Undersampling technique is applied to create a perfectly balanced dataset, removing the need for class weighting.

---

## Evaluation Metric

The models were primarily evaluated using AUC and recall. AUC (Area Under the ROC Curve) was used to measure the model’s ability to distinguish between defaulters and non-defaulters across all thresholds, providing a robust, threshold-independent assessment. Recall was emphasized to ensure that the model successfully identified as many actual defaulters as possible, which is critical in minimizing credit risk exposure. Precision, F1 score, and threshold optimization are highlighted, but AUC and recall guided the main evaluation.

- **Naive Model**
Test ROC AUC: 0.6272
Test Recall: 0.0000

- **Traditional Model**
Overall Best Model Info:
{'resampling': 'SMOTEENN', 'model_type': 'AdaBoost', 'combined_metric': 0.44750224081606577}
Test Recall: 0.2523
Test AUROC: 0.6471

- **Deep Learning Model**

**ResNet with Oversampling**
Test AUC: 0.6962
Test Recall (Sensitivity): 0.4088

**ResNet with Undersampling**
Test AUC: 0.7053
Test Recall (Sensitivity): 0.4461

---

## User Interface

- **Streamlit Web App**: 

WIP

---

## Setup

```bash
./setup.sh
```

This script takes care of setting up your virtual environment if it does not already exist, activating it, installing requirements, pulling the dataset (if not already present in the data directory), pre-processing the data.

## Running the Streamlit application locally

Assuming your virtual environment is setup and activated, and that the requirements are installed from running `setup.sh`,
you can then run the following to startup a local instance of the Streamlit application.

```bash
python main.py
```
---

## Dataset & License
This repository uses the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview) dataset from Kaggle Competition, licensed under Kaggle competition rules.

The Home Credit Default Risk dataset is a comprehensive collection of information aimed at enhancing credit risk assessment models. Provided by Home Credit, the dataset encompasses a wide array of data points, including application details, demographic information, and historical credit behavior. The primary objective is to predict the likelihood of a client defaulting on a loan.

Using only `application_train.csv` file from the Home Credit repository as performance of model deteriorates when other csv files are added in the training.

---

## **Ethics Statement**  

This project uses publicly available financial datasets in compliance with their terms of use. All data is handled responsibly, with no collection or use of personally identifiable information (PII). We are committed to preventing misuse, protecting privacy, and avoiding any unauthorized distribution. Our goal is to advance responsible AI in financial services while upholding ethical standards and data integrity.

---

## Presentation Link

View our presentation [HERE](https://docs.google.com/presentation/d/1XSYJ6GrIr3v_t4enZNBWPNALS4c5bxAqzMUitaKkjZY/edit#slide=id.p).

---

## Streamlit Application

Access our Streamlit application [HERE]().
