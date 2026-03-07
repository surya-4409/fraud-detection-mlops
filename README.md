
# Real-Time Fraud Detection System

**Author:** Billakurti Venkata Suryanarayana  
**Roll Number:** 23MH1A4409  

## 🎥 Video Demonstration
**[Watch the Project Walkthrough & API Demo Here](https://drive.google.com/file/d/1sHJo4-6xvngS4Q5nkG-BNYvNY1V0A8Jl/view?usp=sharing)**

---

## 📌 Project Overview
This repository contains a production-ready, real-time machine learning pipeline for credit card fraud detection. It encapsulates the complete ML lifecycle, from handling heavily imbalanced data (using SMOTE) to deploying an optimized XGBoost model via a FastAPI REST interface, all containerized within Docker.

## 🏗️ System Architecture
* **Modeling:** XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE)
* **Experiment Tracking:** MLflow
* **API Framework:** FastAPI, Uvicorn, Pydantic (Data Validation)
* **Deployment:** Multi-stage Docker Build
* **Testing:** Pytest (Requests)

---

## ⚙️ Prerequisites: Dataset Setup
This project uses the standard Credit Card Fraud Detection dataset from Kaggle. Because of its large size (150MB+), it is not included in this repository. 

**Before running any training commands, you must download the dataset:**
1. Download the dataset from Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Extract the downloaded archive.
3. Place the `creditcard.csv` file inside the `data/raw/` directory of this project.

*(Ensure the path is exactly: `data/raw/creditcard.csv`)*

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Clone the Repository
```bash
git clone [https://github.com/surya-4409/fraud-detection-mlops.git](https://github.com/surya-4409/fraud-detection-mlops.git)
cd fraud-detection-mlops

```

### Step 2: Setup Local Environment

Ensure you have Python 3.8+ installed. Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

```

### Step 3: Data Preparation & Model Training

First, process the raw dataset to engineer features and split the data. Then, train the ML models (tracked via MLflow).

```bash
# 1. Prepare data (Engineers features and creates train.csv / test_data.csv)
python src/prepare_data.py

# 2. Train models and select the best one based on PR-AUC
python src/train.py

```

*To view the MLflow UI and experiment logs, run `mlflow ui` and visit `http://127.0.0.1:5000`.*

### Step 4: Docker Deployment (Production API)

Build the highly optimized, multi-stage Docker image and start the containerized FastAPI server:

```bash
# Build the production image
docker build -t fraud-detection-api:latest .

# Run the container in the background
docker run -d -p 8000:8000 fraud-detection-api:latest

```

### Step 5: Automated Testing

Verify the live API functionality, input validation, and latency (p95 < 100ms) by running the Pytest suite against the Docker container:

```bash
pytest tests/test_api.py -v -s

```

---

## 🤖 Automated Evaluation (`submission.yml`)

For automated CI/CD or evaluation environments, the entire pipeline is mapped in the `submission.yml` file. Evaluators can execute the following commands in sequence (assuming the dataset has been placed in `data/raw/`):

1. **Setup:** `pip install -r requirements.txt`
2. **Train:** `python src/prepare_data.py && python src/train.py`
3. **Start API:** `docker build -t fraud-detection-api:latest . && docker run -d -p 8000:8000 fraud-detection-api:latest`
4. **Test API:** `pytest tests/test_api.py -v -s`

---

## 🌐 API Usage Example

Once the Docker container is running, you can make a real-time prediction by sending a `POST` request to the `/predict` endpoint.

**cURL Request:**

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
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

**Expected JSON Response:**

```json
{
  "is_fraud": false,
  "probability": 0.000021
}

```

```

