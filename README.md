# Hybrid Ensemble Model for Early Heart Disease Prediction

## Problem Statement
Cardiovascular disease (CVD) remains a leading cause of mortality worldwide. Accurate and early detection of CVD is critical for timely intervention and treatment. Traditional predictive models often struggle with issues such as overfitting, low interpretability, and poor generalization on complex, imbalanced datasets. This project aims to address these challenges through a hybrid ensemble model combining XGBoost and Random Forest classifiers using a soft voting strategy to enhance predictive accuracy and robustness.

---

## Methodology

### 1. Data Loading and Preprocessing

#### Dataset Overview
- **Source**: `cardio_train.csv`, containing anonymized patient health records
- **Target Variable**: `cardio` (1 = CVD present, 0 = CVD absent)

#### Transformations
- Removed `id` column
- Converted `age` from days to years
- Engineered new features:
  - Body Mass Index (BMI)
  - Pulse Pressure (PP)

#### Data Cleaning
- Removed records with:
  - BMI < 10 or > 50
  - Systolic BP > 300 mmHg
  - Diastolic BP > Systolic BP
  - Clinically unrealistic values

#### Outlier and Anomaly Handling
- Removed records such as:
  - CVD-negative with cholesterol level = 3
  - CVD-positive with normal cholesterol, glucose, and healthy BMI
  - CVD-negative with glucose level = 3

#### Correlation Analysis
- Used Pearson correlation heatmap to:
  - Validate engineered features (BMI and PP)
  - Guide feature selection

#### Class Imbalance Handling
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance dataset

#### Data Splitting
- Used `train_test_split` with an **80:20** ratio
- Ensured generalization and evaluation on unseen data

---

### 2. Model Architecture

#### XGBoost Classifier
- Boosting-based model that builds trees sequentially
- Handles regularization and is robust to noise
- Optimized with log loss

#### Random Forest Classifier
- Bagging-based ensemble using parallel decision trees
- Reduces variance through averaging
- Offers feature importance for model interpretability

> Both models use `n_estimators = 50` for optimal performance vs efficiency

---

### 3. Ensemble Strategy

#### Soft Voting Ensemble
- Combines predictions of XGBoost and Random Forest
- Averages predicted probabilities for better calibration
- Captures both low-bias (RF) and low-variance (XGBoost) strengths

---

### 4. Model Training and Evaluation

#### Training Phase
- Performed on SMOTE-balanced 80% training data
- Used features like `age`, `ap_hi`, `ap_lo`, `cholesterol`, `BMI`, and `pulse pressure`

#### Testing Phase
- Predictions made on 20% hold-out test set
- Compared against true labels

#### Evaluation Metrics
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC
- Visuals: Confusion matrix, ROC curve

---

### 5. Cross-Validation
- Used **5-fold cross-validation** to validate generalization
- Measured consistent performance across multiple data splits

---

## Installation & Requirements

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
