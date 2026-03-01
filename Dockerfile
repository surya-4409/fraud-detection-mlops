# ==========================================
# STAGE 1: Builder
# ==========================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install system build dependencies required for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a Python virtual environment
RUN python -m venv /opt/venv

# Make sure we use the virtualenv
ENV PATH="/opt/venv/bin:$PATH"

# Copy just the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Runtime (Production Image)
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Copy ONLY the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set the environment variable to use the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy our application code and the trained model
# We do not copy notebooks, raw data, or training scripts to keep the image small
COPY api/ ./api/
COPY models/ ./models/

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to start the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]