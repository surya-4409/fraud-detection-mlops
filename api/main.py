import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.schemas import TransactionIn, PredictionOut

# 1. Setup Structured Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("fraud_api")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time API for credit card fraud detection.",
    version="1.0.0"
)

# Global variable to hold our model
model = None

@app.on_event("startup")
def load_model():
    """Loads the machine learning model on application startup."""
    global model
    try:
        logger.info("Loading XGBoost model from models/best_model.joblib...")
        model = joblib.load("models/best_model.joblib")
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Could not load the model. Check the models directory.")

@app.post("/predict", response_model=PredictionOut)
async def predict_fraud(transaction: TransactionIn):
    """
    Endpoint to predict if a transaction is fraudulent.
    """
    try:
        # 1. Log incoming request (without sensitive features if this were real, but we log amount for debugging)
        logger.info(f"Received prediction request for transaction amount: {transaction.Amount}")

        # 2. Recreate the Feature Engineering from Phase 1
        time_hour = (transaction.Time / 3600) % 24
        amount_log = np.log1p(transaction.Amount)
        is_high_amount = int(transaction.Amount > 200)

        # 3. Assemble features in the EXACT order the model was trained on:
        # V1 through V28, Time_Hour, Amount_Log, Is_High_Amount
        features = [
            transaction.V1, transaction.V2, transaction.V3, transaction.V4, transaction.V5,
            transaction.V6, transaction.V7, transaction.V8, transaction.V9, transaction.V10,
            transaction.V11, transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19, transaction.V20,
            transaction.V21, transaction.V22, transaction.V23, transaction.V24, transaction.V25,
            transaction.V26, transaction.V27, transaction.V28,
            time_hour, amount_log, is_high_amount
        ]

        # Convert to numpy array and reshape for prediction (1 sample, n features)
        input_data = np.array(features).reshape(1, -1)

        # 4. Make Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Convert numpy types to native Python types for JSON serialization
        is_fraud = bool(prediction)
        prob_float = float(probability)

        logger.info(f"Prediction made. Fraud: {is_fraud}, Probability: {prob_float:.4f}")

        return PredictionOut(is_fraud=is_fraud, probability=prob_float)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "API is running and model is loaded."}