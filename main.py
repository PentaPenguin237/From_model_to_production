import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Import your existing logic (assuming we rename your old script to 'pipeline_logic.py' or keep it in same file)
# For simplicity, I will re-implement the core logic here to keep it self-contained in the API.

# --- CONFIG ---
DATA_DIR = "/data"
RESULTS_DIR = "/results"
MODEL_PATH = os.path.join(RESULTS_DIR, "isolation_forest_model.joblib")

# --- DATA MODELS (Pydantic) ---
class SensorReading(BaseModel):
    temperature_k: float
    rotational_speed_rpm: float

class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    status: str

# --- LIFECYCLE ---
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        ml_models["isolation_forest"] = joblib.load(MODEL_PATH)
    else:
        print("Model not found! Please run the training script first or ensure persistence.")
        # In a real scenario, we might trigger training here, but for now we warn.
        ml_models["isolation_forest"] = None
    yield
    # Clean up on shutdown
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# --- HELPER FUNCTIONS ---
def engineer_single_row(temp_k, rpm):
    """
    Replicates the feature engineering logic for a single data point.
    """
    # 1. Create DataFrame
    df = pd.DataFrame([[temp_k, rpm]], columns=['Air temperature [K]', 'Rotational speed [rpm]'])
    
    # 2. Rename
    df.rename(columns={'Rotational speed [rpm]': 'sound_volume_proxy'}, inplace=True)
    
    # 3. Kelvin to Celsius
    df['temperature_celsius'] = df['Air temperature [K]'] - 273.15
    df.drop(columns=['Air temperature [K]'], inplace=True)
    
    # 4. Synthetic Humidity (The Multiparametric Function)
    base_humidity = 45.0
    # We use hardcoded means here based on your training data observations 
    # (In production, these would be loaded from a config file)
    temp_mean = 26.85 
    rpm_mean = 1538.0
    
    temp_weight = -0.5
    rpm_weight = 0.005
    
    humidity = (
        base_humidity + 
        (df['temperature_celsius'] - temp_mean) * temp_weight + 
        (df['sound_volume_proxy'] - rpm_mean) * rpm_weight
    )
    
    # Add slight random noise (simulating sensor jitter)
    noise = np.random.uniform(-0.5, 0.5)
    df['humidity'] = (humidity + noise).clip(0, 100)
    
    # Reorder to match training: ['sound_volume_proxy', 'temperature_celsius', 'humidity']
    return df[['sound_volume_proxy', 'temperature_celsius', 'humidity']]

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "running", "model_loaded": ml_models["isolation_forest"] is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(reading: SensorReading):
    if ml_models["isolation_forest"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # 1. Process Data
    try:
        processed_data = engineer_single_row(reading.temperature_k, reading.rotational_speed_rpm)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature engineering failed: {str(e)}")

    # 2. Inference
    model = ml_models["isolation_forest"]
    prediction = model.predict(processed_data)
    score = model.decision_function(processed_data)
    
    is_anomaly = int(prediction[0] == -1)
    
    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(float(score[0]), 4),
        "status": "ALERT" if is_anomaly else "OK"
    }
