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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def load_data():
    """Loads the processed training and testing datasets."""
    print("Loading processed data...")
    # NOTE: To prevent data leakage, load the PRE-SMOTE training data here.
    # SMOTE will be applied dynamically inside the cross-validation pipeline.
    train = pd.read_csv('data/processed/train.csv') 
    test = pd.read_csv('data/processed/test_data.csv')
    
    X_train = train.drop('Class', axis=1)
    y_train = train['Class']
    X_test = test.drop('Class', axis=1)
    y_test = test['Class']
    
    return X_train, y_train, X_test, y_test

def train_and_log_model(model_name, model, param_grid, X_train, y_train, X_test, y_test):
    """Trains a model using hyperparameter tuning and CV, evaluates it, and logs to MLflow."""
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining and tuning {model_name}...")
        
        # Create an imblearn Pipeline to apply SMOTE correctly during cross-validation
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        # Map parameter grid keys to match the pipeline's classifier step
        pipeline_param_grid = {f'classifier__{key}': value for key, value in param_grid.items()}
        
        # Set up Stratified K-Fold cross-validation to preserve extreme class imbalance
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Set up the randomized search, optimizing for average precision (PR-AUC)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=pipeline_param_grid,
            n_iter=3, # Reduced to 3 to speed up the training time significantly
            scoring='average_precision', 
            cv=cv,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model and find the best hyperparameters
        search.fit(X_train, y_train)
        
        # Extract the best pipeline and the underlying model
        best_pipeline = search.best_estimator_
        best_model = best_pipeline.named_steps['classifier']
        
        # Log best parameters to MLflow
        mlflow.log_params(search.best_params_)
        print(f"Best Parameters: {search.best_params_}")
        
        # Make predictions on the UNSEEN test set using the best pipeline
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
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
            mlflow.xgboost.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")
            
        print(f"{model_name} Metrics -> F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | PR-AUC: {pr_auc:.4f}")
        
        return best_model, pr_auc

if __name__ == "__main__":
    # Ensure we are in the root directory context
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    X_train, y_train, X_test, y_test = load_data()
    
    # Set up MLflow experiment
    mlflow.set_experiment("Fraud_Detection_Experiment")
    
    # 1. Train Logistic Regression (Baseline)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'liblinear']
    }
    lr_trained, lr_pr_auc = train_and_log_model("Logistic_Regression", lr_model, lr_param_grid, X_train, y_train, X_test, y_test)
    
    # 2. Train XGBoost (Advanced Tree-based)
    # tree_method='hist' massively speeds up training on large datasets
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, tree_method='hist')
    
    # Smaller grid to guarantee it finishes quickly on a local machine
    xgb_param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [50, 100]
    }
    xgb_trained, xgb_pr_auc = train_and_log_model("XGBoost", xgb_model, xgb_param_grid, X_train, y_train, X_test, y_test)
    
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