# ML Inference API (FastAPI + scikit-learn)

A REST API built with FastAPI and scikit-learn that serves real-time Iris flower species predictions with Pydantic input validation and automated `pytest` tests.

## Features
- **Health Check (`GET /health`):** Confirms the service is running and the ML model is loaded.
- **Prediction Endpoint (`POST /predict`):** Accepts flower measurements, validates inputs with Pydantic (`gt=0`), and returns the predicted class with a confidence score.
- **Interactive Docs:** Auto-generated Swagger UI available at `/docs`.
- **Automated Testing:** Endpoint and validation tests built with `pytest` and FastAPI's `TestClient`.

## Tech Stack
- Python, FastAPI, Uvicorn, Pydantic
- scikit-learn, Joblib
- pytest

## Setup & Run Locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and save the model
python train.py

# 4. Start the API server
uvicorn main:app --reload