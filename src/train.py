import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score

def load_data():
    """Loads the processed training and testing datasets."""
    print("Loading processed data...")
    train = pd.read_csv('data/processed/train_smote.csv')
    test = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train.drop('Class', axis=1)
    y_train = train['Class']
    X_test = test.drop('Class', axis=1)
    y_test = test['Class']
    
    return X_train, y_train, X_test, y_test

def train_and_log_model(model_name, model, X_train, y_train, X_test, y_test):
    """Trains a model, evaluates it, and logs everything to MLflow."""
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions on the UNSEEN test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_proba)
        
        # Log metrics to MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("pr_auc", pr_auc)
        
        # Log the model artifact
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
            
        print(f"{model_name} Metrics -> F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | PR-AUC: {pr_auc:.4f}")
        
        return model, pr_auc

if __name__ == "__main__":
    # Ensure we are in the root directory context
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    X_train, y_train, X_test, y_test = load_data()
    
    # Set up MLflow experiment
    mlflow.set_experiment("Fraud_Detection_Experiment")
    
    # 1. Train Logistic Regression (Baseline)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_trained, lr_pr_auc = train_and_log_model("Logistic_Regression", lr_model, X_train, y_train, X_test, y_test)
    
    # 2. Train XGBoost (Advanced Tree-based)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_trained, xgb_pr_auc = train_and_log_model("XGBoost", xgb_model, X_train, y_train, X_test, y_test)
    
    # 3. Model Selection and Versioning Strategy
    print("\nSelecting the best model based on PR-AUC...")
    os.makedirs('models', exist_ok=True)
    
    if xgb_pr_auc > lr_pr_auc:
        print("XGBoost selected as the best model. Saving artifact...")
        joblib.dump(xgb_trained, 'models/best_model.joblib')
    else:
        print("Logistic Regression selected as the best model. Saving artifact...")
        joblib.dump(lr_trained, 'models/best_model.joblib')
        
    print("Training phase completed successfully! MLflow tracking data saved in 'mlruns' directory.")