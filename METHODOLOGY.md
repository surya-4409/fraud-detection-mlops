# Methodology: Fraud Detection System

## 1. Data Analysis and Exploration (EDA)
The dataset utilized is highly imbalanced, consisting of 284,807 transactions where only 492 (0.172%) are fraudulent. This extreme skewness dictates the entire modeling approach, as standard accuracy would be highly misleading. The features `V1` through `V28` are anonymized via PCA, leaving `Time` and `Amount` as the primary interpretable features.

## 2. Feature Engineering
To provide the models with stronger predictive signals, three new features were engineered:
* **`Time_Hour`**: Converted elapsed seconds into a 24-hour cycle to capture specific times of day when fraud might spike.
* **`Amount_Log`**: Applied a logarithmic transformation (`log1p`) to the `Amount` feature to handle its extreme right-skewness and compress outlier transaction values.
* **`Is_High_Amount`**: A binary indicator (1 or 0) for transactions exceeding $200, as fraudsters often test cards with higher-value purchases.

## 3. Handling Class Imbalance
To address the 99.8% vs 0.17% class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) was applied. 
* **Crucial MLOps Practice**: The dataset was split into an 80/20 train/test split using **stratification** to maintain the exact fraud ratio in both sets. *Then*, SMOTE was only applied to the training set within the cross-validation pipeline to prevent data leakage and ensure the model was evaluated on strictly unseen, real-world data.

## 4. Model Selection and Evaluation
Two models were trained and tracked using MLflow:
1. **Logistic Regression (Baseline)**: Tended to over-predict fraud, resulting in high Recall but extremely poor Precision (many false positives).
2. **XGBoost (Advanced)**: A tree-based boosting algorithm that effectively handled the non-linear relationships in the tabular data.

**Primary Metric**: PR-AUC (Precision-Recall Area Under Curve) was chosen as the primary evaluation metric because it focuses explicitly on the minority (fraud) class, unlike standard ROC-AUC which can be overly optimistic on heavily imbalanced datasets.
* **Result**: XGBoost significantly outperformed the baseline, achieving a high F1-score and a PR-AUC of ~0.86. It was automatically registered and saved as the `best_model.joblib` artifact for production deployment.