
# Real-Time Fraud Detection System

**Author:** Billakurti Venkata Suryanarayana  
**Roll Number:** 23MH1A4409  

## Project Overview
This repository contains a production-ready, real-time machine learning pipeline for credit card fraud detection. It encapsulates the complete ML lifecycle, from handling heavily imbalanced data (using SMOTE) to deploying an optimized XGBoost model via a FastAPI REST interface, all containerized within Docker.

## System Architecture
* **Modeling:** XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Experiment Tracking:** MLflow
* **API Framework:** FastAPI, Uvicorn, Pydantic (Data Validation)
* **Deployment:** Multi-stage Docker Build
* **Testing:** Pytest

## Setup Instructions

### 1. Local Development Setup
Ensure you have Python 3.8+ installed.
```bash
# Install dependencies
pip install -r requirements.txt

# Run the training pipeline (tracks via MLflow)
python src/train.py

# Start the API locally
uvicorn api.main:app --reload

```

### 2. Docker Deployment (Production)

```bash
# Build the optimized multi-stage image
docker build -t fraud-detection-api:latest .

# Run the container
docker run -d -p 8000:8000 fraud-detection-api:latest

```

## API Usage Example

To make a real-time prediction, send a POST request to the `/predict` endpoint.

**cURL Request:**

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Time": 0.0, "V1": -1.3598, "V2": -0.0727, "V3": 2.5363, "V4": 1.3781,
  "V5": -0.3383, "V6": 0.4623, "V7": 0.2395, "V8": 0.0986, "V9": 0.3637,
  "V10": 0.0907, "V11": -0.5516, "V12": -0.6178, "V13": -0.9913, "V14": -0.3111,
  "V15": 1.4681, "V16": -0.4704, "V17": 0.2079, "V18": 0.0257, "V19": 0.4039,
  "V20": 0.2514, "V21": -0.0183, "V22": 0.2778, "V23": -0.1104, "V24": 0.0669,
  "V25": 0.1285, "V26": -0.1891, "V27": 0.1335, "V28": -0.0210, "Amount": 149.62
}'

```

**Response:**

```json
{
  "is_fraud": false,
  "probability": 0.000021
}

```

## Running Tests

To verify functionality, validation, and API latency (<100ms):

```bash
pytest tests/test_api.py -v -s

```