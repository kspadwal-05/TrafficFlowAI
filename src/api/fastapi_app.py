#!/usr/bin/env python3
"""
FastAPI Application for TrafficFlow AI Platform
Real-time traffic analytics API with ML predictions
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio
import json
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="TrafficFlow AI API",
    description="Intelligent Traffic Analytics Platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TrafficPredictionRequest(BaseModel):
    volume: float
    speed_limit: float
    average_speed: float
    road_type: str
    time_of_day: int
    day_of_week: int

class TrafficPredictionResponse(BaseModel):
    predicted_volume: float
    confidence: float
    anomaly_score: float
    is_anomaly: bool

class TrafficMetrics(BaseModel):
    location: str
    volume: float
    average_speed: float
    congestion_level: str
    timestamp: datetime

# WebSocket connections
websocket_connections: List[WebSocket] = []

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "TrafficFlow AI API",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

@app.post("/api/v1/predict", response_model=TrafficPredictionResponse)
async def predict_traffic_volume(request: TrafficPredictionRequest):
    """Predict traffic volume using ML models"""
    try:
        # Simulate ML prediction (replace with actual model)
        base_prediction = request.volume * 1.1
        confidence = 0.85
        anomaly_score = 0.1 if request.average_speed > request.speed_limit * 1.2 else 0.3
        is_anomaly = anomaly_score < 0.2
        
        return TrafficPredictionResponse(
            predicted_volume=base_prediction,
            confidence=confidence,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_data():
    """Get dashboard analytics data"""
    return {
        "total_locations": 50,
        "average_volume": 2500,
        "anomalies_detected": 3,
        "system_health": "excellent",
        "last_updated": datetime.now()
    }

@app.get("/api/v1/realtime/{location}")
async def get_realtime_metrics(location: str):
    """Get real-time metrics for a specific location"""
    return {
        "location": location,
        "volume": 1200,
        "average_speed": 45,
        "congestion_level": "low",
        "timestamp": datetime.now()
    }

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        while True:
            # Send real-time data
            data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_volume": 50000,
                    "average_speed": 45,
                    "anomalies": 2
                }
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(5)  # Update every 5 seconds
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)