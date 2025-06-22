# Credit-Card-Financial-Fraud-Detection

A comprehensive machine learning project that detects fraudulent credit card transactions using both supervised and unsupervised approaches, optimized for imbalanced classification problems. Built using Python and real-world credit card transaction data.

---

## 📌 Table of Contents
- [About the Project](#about-the-project)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Performance Comparison](#performance-comparison)
- [Project Highlights](#project-highlights)
- [How to Run](#how-to-run)
- [Future Work](#future-work)

---

## About the Project

Credit card fraud detection is a classic case of **highly imbalanced classification**, where fraudulent cases represent less than 0.2% of all transactions. This project implements a variety of machine learning models and techniques to deal with the skewed data and maximize the detection of frauds while minimizing false positives.

---

## Problem Statement

> Build a robust, interpretable machine learning model that can accurately identify fraudulent credit card transactions in a highly imbalanced dataset — with an emphasis on maximizing recall and F1-score for the fraud class.

---

## Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: ~284,000 transactions
- **Features**: PCA-transformed components `V1` to `V28`, `Time`, `Amount`, and `Class` (0: normal, 1: fraud)
- **Challenge**: Class imbalance — frauds are ~0.17% of total data

---

## Methodology

- **Data Exploration** and class imbalance analysis
- **Data Scaling** using `StandardScaler`
- **Train-Test Split** with stratified sampling to preserve class distribution
- **Imbalance Handling** using:
  - SMOTE (Synthetic Minority Oversampling Technique)
  - `class_weight='balanced'` for tree-based models
- **Anomaly Detection** using Gaussian distribution modeling
- **Threshold Tuning** using F1-score optimization

---

## Models Implemented

### Supervised:
- **Logistic Regression**
  - With and without SMOTE
- **Random Forest**
  - With `class_weight='balanced'`
- **XGBoost**
  - Hyperparameter tuned with `GridSearchCV`
  - Trained on SMOTE-balanced data

### Unsupervised:
- **Gaussian Anomaly Detection**
  - Trained only on genuine transactions
  - Evaluated on balanced fraud/genuine validation and test sets

---

## Performance Comparison (Class 1 - Fraud)

| Model                            | Precision | Recall | F1-Score |
|----------------------------------|-----------|--------|----------|
| Gaussian Anomaly Detection       | 1.0000    | 0.0569 | 0.1077   |
| XGBoost (SMOTE + GridSearchCV)   | 0.2800    | 0.8700 | 0.4300   |
| Random Forest (class_weight)     | 0.9300    | 0.7900 | 0.8600   |
| Logistic Regression + SMOTE      | 0.0517    | 0.8721 | 0.0975   |
| Logistic Regression (no SMOTE)   | 0.6392    | 0.7209 | 0.6776   |
| Linear Regression (binary use)   | 0.8462    | 0.4064 | 0.5491   |

> 📌 **Best model overall**: Random Forest with `class_weight='balanced'`  
> 📌 **Best recall**: Logistic Regression + SMOTE  
> 📌 **Most conservative**: Gaussian Anomaly Detector (high precision, low recall)

---

## Project Highlights

- ✅ Tackles **severe class imbalance** using SMOTE and cost-sensitive learning
- ✅ Implements **GridSearchCV** to tune XGBoost hyperparameters
- ✅ Evaluates **unsupervised anomaly detection** using Gaussian distribution
- ✅ Uses **threshold tuning** to further optimize F1-score
- ✅ Rich performance metrics: **Confusion Matrix**, **ROC**, **Precision-Recall Curve**

---
