#!/usr/bin/env python3
"""
Machine Learning Models for TrafficFlow AI
XGBoost and LightGBM models for traffic prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficPredictor:
    """Traffic volume prediction using XGBoost and LightGBM"""
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.feature_columns = [
            'speed_limit', 'average_speed', '85th_percentile_speed',
            'time_of_day', 'day_of_week', 'road_type_encoded'
        ]
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        # Encode categorical variables
        df['road_type_encoded'] = df['road_type'].map({
            'Residential': 1, 'Collector': 2, 'Arterial': 3, 'Highway': 4
        })
        
        # Time-based features
        df['time_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        return df
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train XGBoost model"""
        try:
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            self.xgb_model.fit(X_train, y_train)
            logger.info("XGBoost model trained successfully")
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train LightGBM model"""
        try:
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self.lgb_model.fit(X_train, y_train)
            logger.info("LightGBM model trained successfully")
        except Exception as e:
            logger.error(f"LightGBM training failed: {e}")
    
    def predict_volume(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Predict traffic volume using ensemble of models"""
        if not self.xgb_model or not self.lgb_model:
            return {"predicted_volume": 0.0, "confidence": 0.0}
        
        try:
            # Prepare feature vector
            feature_vector = np.array([
                features.get('speed_limit', 50),
                features.get('average_speed', 45),
                features.get('85th_percentile_speed', 55),
                features.get('time_of_day', 12),
                features.get('day_of_week', 1),
                features.get('road_type_encoded', 2)
            ]).reshape(1, -1)
            
            # Get predictions from both models
            xgb_pred = self.xgb_model.predict(feature_vector)[0]
            lgb_pred = self.lgb_model.predict(feature_vector)[0]
            
            # Ensemble prediction (average)
            ensemble_pred = (xgb_pred + lgb_pred) / 2
            
            # Calculate confidence based on model agreement
            confidence = 1.0 - abs(xgb_pred - lgb_pred) / max(xgb_pred, lgb_pred, 1)
            
            return {
                "predicted_volume": float(ensemble_pred),
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"predicted_volume": 0.0, "confidence": 0.0}
    
    def save_models(self, filepath: str) -> None:
        """Save trained models"""
        try:
            joblib.dump({
                'xgb_model': self.xgb_model,
                'lgb_model': self.lgb_model,
                'feature_columns': self.feature_columns
            }, filepath)
            logger.info(f"Models saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save models: {e}")

class AnomalyDetector:
    """Anomaly detection using Isolation Forest"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = ['volume', 'average_speed', 'speed_limit']
        
    def train(self, df: pd.DataFrame) -> None:
        """Train anomaly detection model"""
        try:
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # Prepare features
            X = df[self.feature_columns].fillna(df[self.feature_columns].mean())
            self.model.fit(X)
            logger.info("Anomaly detection model trained successfully")
        except Exception as e:
            logger.error(f"Anomaly detection training failed: {e}")
    
    def detect_anomalies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in traffic data"""
        if not self.model:
            return {"is_anomaly": False, "anomaly_score": 0.0}
        
        try:
            # Prepare feature vector
            feature_vector = np.array([
                data.get('volume', 0),
                data.get('average_speed', 0),
                data.get('speed_limit', 50)
            ]).reshape(1, -1)
            
            # Get anomaly score
            anomaly_score = self.model.decision_function(feature_vector)[0]
            is_anomaly = self.model.predict(feature_vector)[0] == -1
            
            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score)
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"is_anomaly": False, "anomaly_score": 0.0}

if __name__ == "__main__":
    # Example usage
    predictor = TrafficPredictor()
    detector = AnomalyDetector()
    
    # Sample data
    sample_data = {
        'volume': 1200,
        'average_speed': 45,
        'speed_limit': 50,
        'road_type': 'Residential'
    }
    
    # Make prediction
    prediction = predictor.predict_volume(sample_data)
    anomaly = detector.detect_anomalies(sample_data)
    
    print(f"Prediction: {prediction}")
    print(f"Anomaly: {anomaly}")