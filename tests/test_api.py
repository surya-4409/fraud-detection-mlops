import time
import pytest
import requests

# Point directly to the live Docker container port
BASE_URL = "http://127.0.0.1:8000"

# Fixture to wait for the Docker container to be ready before running tests
@pytest.fixture(scope="session", autouse=True)
def wait_for_api():
    """Pings the health endpoint until the API is live, or times out."""
    max_retries = 10
    delay = 1  # second
    
    for _ in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                return  # API is ready!
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    
    pytest.fail("Docker API did not start in time.")

# A valid transaction payload based on normal credit card data
VALID_PAYLOAD = {
    "Time": 0.0,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
    "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
    "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
    "Amount": 149.62
}

def test_health_check():
    """Test if the API is running and model is loaded."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is running and model is loaded."}

def test_predict_success():
    """Test if a valid payload returns a successful prediction."""
    response = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert isinstance(data["is_fraud"], bool)
    assert isinstance(data["probability"], float)

def test_predict_validation_error():
    """Test if Pydantic properly catches missing fields."""
    invalid_payload = VALID_PAYLOAD.copy()
    del invalid_payload["Amount"]  # Remove a required field

    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_predict_type_error():
    """Test if Pydantic catches wrong data types."""
    invalid_payload = VALID_PAYLOAD.copy()
    invalid_payload["Amount"] = "one hundred"  # String instead of float

    response = requests.post(f"{BASE_URL}/predict", json=invalid_payload)
    assert response.status_code == 422

def test_latency_p95():
    """Simulate load and ensure 95th percentile latency is under 100ms."""
    num_requests = 100
    latencies = []

    for _ in range(num_requests):
        start_time = time.perf_counter()
        response = requests.post(f"{BASE_URL}/predict", json=VALID_PAYLOAD)
        end_time = time.perf_counter()
        
        assert response.status_code == 200
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate the 95th percentile
    latencies.sort()
    p95_index = int(0.95 * len(latencies))
    p95_latency = latencies[p95_index]

    print(f"\n[Performance] p95 Latency: {p95_latency:.2f}ms")
    assert p95_latency < 100.0, f"p95 latency is {p95_latency:.2f}ms, which is >= 100ms!"