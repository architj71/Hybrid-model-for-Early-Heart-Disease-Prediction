# Hybrid Model for Early Heart Disease Prediction

## Problem Statement
Heart disease remains a leading cause of mortality, necessitating accurate and early detection. Existing models face challenges such as overfitting, poor interpretability, and handling complex data. This project aims to develop a hybrid model that leverages a Transformer-based approach for feature extraction and XGBoost for classification, with a stacking ensemble to enhance prediction accuracy and robustness.

---

## Methodology
Our approach integrates deep learning and machine learning techniques to create a robust predictive model. The following steps outline our methodology:

### Data Preprocessing

1. Load Dataset – Load `cardio_train.csv` containing cardiovascular health records.  
2. Convert Target Variable to Binary – Encode the target variable:  
   - `0`: No heart disease (HD)  
   - `1`: Presence of heart disease (HD)  
3. Handle Missing Values – Use KNN Imputer to fill missing values.  
4. Balance Data – Apply Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance.  
5. Normalize Features – Use RobustScaler to normalize feature distributions.  
6. Feature Selection – Compute feature importance using:  
   - Extra Trees Importance  
   - Mutual Information  
   - Select the Top 10 features.  

### Model Training

7. Split Data – Allocate 80% for training and 20% for testing.  
8. Train Transformer Model:  
   - Linear Embedding → Transformer Encoder → Fully Connected Layer → Probability Scores  
9. Train XGBoost Model:  
   - Gradient Boosting → Log Loss Optimization → Probability Scores  

### Model Stacking & Evaluation

10. Stacking Ensemble:  
    - Combine Transformer & XGBoost outputs  
    - Train Logistic Regression as a meta-classifier  
    - Generate final predictions  
11. Evaluate Model Performance:  
    - Accuracy  
    - Precision  
    - Recall  
    - F1-Score  
    - AUC-ROC Curve  

---

## Research & Comparison

- We have reviewed five research papers to compare our approach with existing methods.  
- Moving forward, we will compare our model’s performance with a benchmark research paper.  
- Methodology and pseudo-code refinements are ongoing to enhance accuracy and efficiency.  

---

## Future Enhancements

- Experiment with different feature selection techniques.  
- Fine-tune hyperparameters for Transformer and XGBoost models.  
- Implement SHAP for explainability.  
- Optimize computational efficiency for real-time applications.  

---

## Installation & Requirements

To run this project, install the following dependencies:

```bash
pip install pandas numpy scikit-learn xgboost transformers imbalanced-learn
