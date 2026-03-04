import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_data():
    # Ensure we are in the project root directory
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    print("Loading raw data...")
    # Load the original raw dataset
    df = pd.read_csv('data/raw/creditcard.csv')

    print("Engineering features...")
    # Create the 3 new features to match your API requirements
    df['Time_Hour'] = (df['Time'] / 3600) % 24
    df['Amount_Log'] = np.log1p(df['Amount'])
    df['Is_High_Amount'] = (df['Amount'] > 200).astype(int)

    # Drop the original Time and Amount columns
    df = df.drop(['Time', 'Amount'], axis=1)

    # Reorder columns so the model receives features in the exact same order as the API
    cols = [f'V{i}' for i in range(1, 29)] + ['Time_Hour', 'Amount_Log', 'Is_High_Amount', 'Class']
    df = df[cols]

    print("Splitting data...")
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Perform the 80/20 split, maintaining the fraud imbalance ratio using stratify
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Ensure the processed directory exists
    os.makedirs('data/processed', exist_ok=True)

    print("Saving pre-SMOTE training data (train.csv)...")
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv('data/processed/train.csv', index=False)

    print("Saving test data (test_data.csv)...")
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv('data/processed/test_data.csv', index=False)
    
    print("Data preparation complete! You can now run python src/train.py")

if __name__ == "__main__":
    prepare_data()