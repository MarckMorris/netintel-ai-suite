"""
FastAPI endpoint for Network Anomaly Detection
Real-time inference API, intended to be consumed by a NOC dashboard or alerting pipeline
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
from datetime import datetime
import sys
sys.path.append('..')
from models.lstm_autoencoder import NetworkAnomalyDetector

app = FastAPI(title="NetIntel Anomaly Detection API", version="1.0.0")

detector = NetworkAnomalyDetector()

class NetworkMetrics(BaseModel):
    timestamp: datetime
    latency_ms: float = Field(..., ge=0, le=1000)
    jitter_ms: float = Field(..., ge=0, le=100)
    packet_loss_pct: float = Field(..., ge=0, le=100)
    bandwidth_util_pct: float = Field(..., ge=0, le=100)
    connection_count: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)

@app.on_event("startup")
async def load_model():
    try:
        detector.load_model(path='../models/')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠ Model not found: {e}")

@app.get("/")
async def root():
    return {"service": "NetIntel Anomaly Detection API", "status": "operational"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector.autoencoder is not None}

@app.post("/predict")
async def predict_anomaly(metrics: NetworkMetrics):
    if detector.autoencoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    df = pd.DataFrame([metrics.dict()])
    
    try:
        result = detector.predict(df)
        return {
            "timestamp": metrics.timestamp,
            "anomaly_score": float(result['anomaly_score'].iloc[0]),
            "is_anomaly": bool(result['is_anomaly'].iloc[0]),
            "severity": result['severity'].iloc[0],
            "root_cause": result['root_cause'].iloc[0]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
