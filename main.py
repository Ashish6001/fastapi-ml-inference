from contextlib import asynccontextmanager
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once when the server starts
    ml_artifacts["iris"] = joblib.load("model.joblib")
    yield
    ml_artifacts.clear()

app = FastAPI(title="ML Inference API", lifespan=lifespan)

class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, examples=[5.1])
    sepal_width: float = Field(..., gt=0, examples=[3.5])
    petal_length: float = Field(..., gt=0, examples=[1.4])
    petal_width: float = Field(..., gt=0, examples=[0.2])

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": "iris" in ml_artifacts}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    artifact = ml_artifacts.get("iris")
    if not artifact:
        raise HTTPException(status_code=503, detail="Model not loaded")

    row = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]]

    model = artifact["model"]
    target_names = artifact["target_names"]

    pred_idx = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return PredictionResponse(
        predicted_class=target_names[pred_idx],
        confidence=round(float(max(probabilities)), 4),
    )