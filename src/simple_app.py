#!/usr/bin/env python3
"""
TrafficFlow AI - Simplified Application
A working version of the TrafficFlow AI platform
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data() -> pd.DataFrame:
    """Load sample traffic data"""
    sample_data = {
        'ID': ['A-001', 'A-002', 'A-003', 'A-004', 'A-005'],
        'Location - Street Name1': ['Main St', 'Oak Ave', 'Pine St', 'Elm St', 'Cedar Ave'],
        'From - Street Name2': ['First Ave', 'Second St', 'Third Ave', 'Fourth St', 'Fifth Ave'],
        'To - Street Name3': ['Second Ave', 'Third St', 'Fourth Ave', 'Fifth St', 'Sixth Ave'],
        'Road Classification': ['Residential', 'Collector', 'Arterial', 'Residential', 'Collector'],
        'Status': ['Active', 'Pending', 'Active', 'Completed', 'Active'],
        'Volume (vpd)': [1200, 2800, 4500, 800, 3200],
        'Posted Speed Limit (km/h)': [50, 40, 60, 30, 45],
        'Average Speed (km/h)': [45, 38, 55, 28, 42],
        '85th Percentile Speed (km/h)': [55, 47, 65, 35, 52]
    }
    return pd.DataFrame(sample_data)

def calculate_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ML features for traffic data"""
    df['speed_compliance'] = df['Average Speed (km/h)'] / df['Posted Speed Limit (km/h)']
    df['congestion_level'] = np.where(df['speed_compliance'] < 0.8, 'High', 
                                     np.where(df['speed_compliance'] < 1.0, 'Medium', 'Low'))
    df['volume_category'] = pd.cut(df['Volume (vpd)'], bins=[0, 1000, 3000, 5000, float('inf')], 
                                  labels=['Low', 'Medium', 'High', 'Very High'])
    return df

def predict_traffic_volume(features: Dict) -> Dict:
    """Simple traffic volume prediction"""
    base_volume = features.get('volume', 1000)
    road_type_multiplier = {
        'Residential': 0.8,
        'Collector': 1.2,
        'Arterial': 1.5,
        'Highway': 2.0
    }.get(features.get('road_type', 'Residential'), 1.0)
    
    time_factor = 1.0 + 0.2 * np.sin(2 * np.pi * features.get('time_of_day', 12) / 24)
    
    predicted_volume = base_volume * road_type_multiplier * time_factor
    
    return {
        'predicted_volume': int(predicted_volume),
        'confidence': 0.85,
        'model_type': 'Simple Linear Model',
        'timestamp': datetime.now().isoformat()
    }

def detect_anomalies(df: pd.DataFrame) -> List[Dict]:
    """Simple anomaly detection"""
    anomalies = []
    
    # Check for speed violations
    speed_violations = df[df['Average Speed (km/h)'] > df['Posted Speed Limit (km/h)'] * 1.2]
    for _, row in speed_violations.iterrows():
        anomalies.append({
            'location': f"{row['Location - Street Name1']} & {row['From - Street Name2']}",
            'type': 'Speed Violation',
            'severity': 'High',
            'details': f"Speed {row['Average Speed (km/h)']} km/h exceeds limit {row['Posted Speed Limit (km/h)']} km/h"
        })
    
    # Check for unusual volume patterns
    mean_volume = df['Volume (vpd)'].mean()
    volume_anomalies = df[df['Volume (vpd)'] > mean_volume * 2]
    for _, row in volume_anomalies.iterrows():
        anomalies.append({
            'location': f"{row['Location - Street Name1']} & {row['From - Street Name2']}",
            'type': 'Volume Anomaly',
            'severity': 'Medium',
            'details': f"Volume {row['Volume (vpd)']} vpd is unusually high"
        })
    
    return anomalies

def generate_dashboard_data() -> Dict:
    """Generate dashboard data"""
    df = load_sample_data()
    df = calculate_ml_features(df)
    
    # Calculate metrics
    total_locations = len(df)
    avg_volume = df['Volume (vpd)'].mean()
    avg_speed = df['Average Speed (km/h)'].mean()
    congestion_level = (df['congestion_level'] == 'High').sum() / len(df)
    
    # Detect anomalies
    anomalies = detect_anomalies(df)
    
    # Generate predictions
    predictions = []
    for _, row in df.iterrows():
        pred = predict_traffic_volume({
            'volume': row['Volume (vpd)'],
            'road_type': row['Road Classification'],
            'time_of_day': 14  # 2 PM
        })
        predictions.append({
            'location': f"{row['Location - Street Name1']} & {row['From - Street Name2']}",
            'current_volume': row['Volume (vpd)'],
            'predicted_volume': pred['predicted_volume'],
            'confidence': pred['confidence']
        })
    
    return {
        'overview': {
            'total_locations': total_locations,
            'average_volume': int(avg_volume),
            'average_speed': round(avg_speed, 1),
            'congestion_level': f"{congestion_level:.1%}",
            'anomalies_detected': len(anomalies)
        },
        'anomalies': anomalies,
        'predictions': predictions,
        'timestamp': datetime.now().isoformat()
    }

def main():
    """Main application"""
    logger.info("🚀 TrafficFlow AI - Simplified Version Starting...")
    
    # Generate dashboard data
    dashboard_data = generate_dashboard_data()
    
    # Save to file
    os.makedirs("/app/data/processed", exist_ok=True)
    with open("/app/data/processed/dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f, indent=2)
    
    logger.info("📊 Dashboard data generated successfully!")
    logger.info(f"Total locations: {dashboard_data['overview']['total_locations']}")
    logger.info(f"Average volume: {dashboard_data['overview']['average_volume']} vpd")
    logger.info(f"Anomalies detected: {dashboard_data['overview']['anomalies_detected']}")
    
    # Print sample predictions
    logger.info("🔮 Sample Predictions:")
    for pred in dashboard_data['predictions'][:3]:
        logger.info(f"  {pred['location']}: {pred['current_volume']} → {pred['predicted_volume']} vpd (confidence: {pred['confidence']:.2f})")
    
    logger.info("✅ TrafficFlow AI analysis complete!")

if __name__ == "__main__":
    main()
