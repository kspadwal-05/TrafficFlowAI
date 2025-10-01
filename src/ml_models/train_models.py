"""
TrafficFlow AI - ML Model Training Pipeline
Automated model training and evaluation pipeline
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import joblib
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from traffic_predictor import TrafficPredictor, AnomalyDetector
from etl_main_enhanced import EnhancedAccessExtractor, EnhancedConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Automated ML model training pipeline"""
    
    def __init__(self):
        self.models = {}
        self.training_results = {}
        self.config = EnhancedConfig.from_env()
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and prepare training data"""
        logger.info("Loading training data...")
        
        # Extract data using existing pipeline
        extractor = EnhancedAccessExtractor(self.config)
        df = extractor.extract()
        
        logger.info(f"Loaded {len(df)} records for training")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for training"""
        logger.info("Preparing features...")
        
        # Initialize predictor to get feature engineering
        predictor = TrafficPredictor()
        
        # Prepare features
        X = predictor.prepare_features(df)
        
        # Use volume as target variable
        y = df['volume'].values if 'volume' in df.columns else df['Volume (vpd)'].values
        
        # Handle missing values
        valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        return X, y
    
    def train_traffic_predictor(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train traffic volume prediction model"""
        logger.info("Training traffic predictor...")
        
        predictor = TrafficPredictor(model_type="xgboost")
        metrics = predictor.train(X, y)
        
        # Save model
        os.makedirs("/app/models", exist_ok=True)
        predictor.save_model("/app/models/traffic_model.pkl")
        
        self.models['traffic_predictor'] = predictor
        self.training_results['traffic_predictor'] = metrics
        
        logger.info(f"Traffic predictor trained - R²: {metrics['test_r2']:.3f}")
        return metrics
    
    def train_anomaly_detector(self, X: np.ndarray) -> Dict:
        """Train anomaly detection model"""
        logger.info("Training anomaly detector...")
        
        detector = AnomalyDetector()
        detector.fit(X)
        
        # Save model
        joblib.dump(detector, "/app/models/anomaly_detector.pkl")
        
        self.models['anomaly_detector'] = detector
        
        # Evaluate anomaly detection
        anomalies, scores = detector.detect_anomalies(X)
        anomaly_rate = anomalies.sum() / len(anomalies)
        
        metrics = {
            'anomaly_rate': float(anomaly_rate),
            'total_samples': len(X),
            'anomalies_detected': int(anomalies.sum())
        }
        
        self.training_results['anomaly_detector'] = metrics
        
        logger.info(f"Anomaly detector trained - Anomaly rate: {anomaly_rate:.3f}")
        return metrics
    
    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        # Evaluate traffic predictor
        if 'traffic_predictor' in self.models:
            predictor = self.models['traffic_predictor']
            predictions = predictor.predict(X)
            
            # Calculate additional metrics
            mae = np.mean(np.abs(y - predictions))
            rmse = np.sqrt(np.mean((y - predictions) ** 2))
            r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            evaluation_results['traffic_predictor'] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'mean_absolute_percentage_error': float(np.mean(np.abs((y - predictions) / y)) * 100)
            }
        
        # Evaluate anomaly detector
        if 'anomaly_detector' in self.models:
            detector = self.models['anomaly_detector']
            anomalies, scores = detector.detect_anomalies(X)
            
            evaluation_results['anomaly_detector'] = {
                'anomaly_rate': float(anomalies.sum() / len(anomalies)),
                'mean_anomaly_score': float(np.mean(scores)),
                'std_anomaly_score': float(np.std(scores))
            }
        
        return evaluation_results
    
    def save_training_report(self, results: Dict):
        """Save comprehensive training report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_results': self.training_results,
            'evaluation_results': results,
            'model_info': {
                'traffic_predictor': {
                    'type': 'XGBoost',
                    'features': 13,
                    'saved_path': '/app/models/traffic_model.pkl'
                },
                'anomaly_detector': {
                    'type': 'IsolationForest',
                    'features': 13,
                    'saved_path': '/app/models/anomaly_detector.pkl'
                }
            }
        }
        
        os.makedirs("/app/data/processed", exist_ok=True)
        with open("/app/data/processed/training_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Training report saved to /app/data/processed/training_report.json")
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        logger.info("Starting ML model training pipeline...")
        
        try:
            # 1. Load data
            df = self.load_training_data()
            
            # 2. Prepare features
            X, y = self.prepare_features(df)
            
            if len(X) < 50:
                logger.warning("Insufficient data for training. Need at least 50 samples.")
                return
            
            # 3. Train models
            self.train_traffic_predictor(X, y)
            self.train_anomaly_detector(X)
            
            # 4. Evaluate models
            evaluation_results = self.evaluate_models(X, y)
            
            # 5. Save report
            self.save_training_report(evaluation_results)
            
            logger.info("ML model training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise

def main():
    """Main training function"""
    trainer = ModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()
