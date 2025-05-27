# Hybrid Ensemble Model for Early Heart Disease Prediction

## Problem Statement
Cardiovascular disease (CVD) remains a leading cause of mortality worldwide. Accurate and early detection of CVD is critical for timely intervention and treatment. Traditional predictive models often struggle with issues such as overfitting, low interpretability, and poor generalization on complex, imbalanced datasets. This project aims to address these challenges through a hybrid ensemble model combining XGBoost and Random Forest classifiers using a soft voting strategy to enhance predictive accuracy and robustness.

---

## Methodology

### 1. Data Loading & Preprocessing

- **Dataset**: `cardio_train.csv` with `cardio` as target (1 = CVD, 0 = No CVD)
- **Transformations**:
  - Removed `id`, converted `age` to years
  - Created BMI and Pulse Pressure features
- **Cleaning**:
  - Dropped unrealistic BMI (<10 or >50), systolic >300, diastolic > systolic
  - Removed anomalies (e.g., CVD- with cholesterol=3)
- **Correlation Analysis**:
  - Used Pearson heatmap to validate BMI, PP
- **Balancing**:
  - Applied SMOTE to address class imbalance
- **Splitting**:
  - 80:20 train-test split using `train_test_split`

### 2. Model Architecture

- **XGBoost**: Sequential boosting, robust with regularization  
- **Random Forest**: Parallel trees via bagging, reduces variance  
- Both models used `n_estimators = 50`

### 3. Ensemble Strategy

- **Soft Voting**: Combined probabilities from both models  
- Benefits: Improved calibration, blends low-bias & low-variance models

### 4. Training & Evaluation

- Trained on SMOTE-balanced data using selected features  
- Tested on hold-out set  
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- **Visuals**: Confusion matrix, ROC curve

### 5. Cross-Validation

- **5-Fold CV** for reliable generalization across multiple data splits

---

## Research & Comparison

Our proposed hybrid ensemble model was compared with baseline models such as Logistic Regression, Random Forest, and XGBoost. The hybrid model consistently outperformed individual models across all metrics.

**Results**  
Accuracy: 0.8141  
Precision: 0.8155  
Recall: 0.8113  
F1-Score: 0.8134  
AUC-ROC: 0.9020

---

## Installation & Requirements

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
