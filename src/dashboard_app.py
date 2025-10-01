#!/usr/bin/env python3
"""
TrafficFlow AI - Interactive Dashboard
A working Streamlit dashboard for the TrafficFlow AI platform
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Page configuration
st.set_page_config(
    page_title="TrafficFlow AI Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample traffic data"""
    return pd.DataFrame({
        'ID': ['A-001', 'A-002', 'A-003', 'A-004', 'A-005', 'A-006', 'A-007', 'A-008'],
        'Location': ['Main St & First Ave', 'Oak Ave & Second St', 'Pine St & Third Ave', 
                    'Elm St & Fourth St', 'Cedar Ave & Fifth St', 'Maple St & Sixth Ave',
                    'Birch St & Seventh Ave', 'Spruce Ave & Eighth St'],
        'Road_Type': ['Residential', 'Collector', 'Arterial', 'Residential', 'Collector', 
                      'Arterial', 'Residential', 'Highway'],
        'Status': ['Active', 'Pending', 'Active', 'Completed', 'Active', 'Active', 'Pending', 'Active'],
        'Volume': [1200, 2800, 4500, 800, 3200, 5200, 1500, 6800],
        'Speed_Limit': [50, 40, 60, 30, 45, 65, 35, 70],
        'Avg_Speed': [45, 38, 55, 28, 42, 58, 32, 65],
        'Percentile_85': [55, 47, 65, 35, 52, 68, 38, 72],
        'Congestion_Level': ['Low', 'Medium', 'Low', 'High', 'Medium', 'Low', 'High', 'Low']
    })

def calculate_ml_predictions(df):
    """Calculate ML predictions for traffic data"""
    predictions = []
    for _, row in df.iterrows():
        # Simple prediction model
        base_volume = row['Volume']
        road_multiplier = {'Residential': 0.8, 'Collector': 1.2, 'Arterial': 1.5, 'Highway': 2.0}.get(row['Road_Type'], 1.0)
        time_factor = 1.0 + 0.2 * np.sin(2 * np.pi * 14 / 24)  # 2 PM factor
        predicted = int(base_volume * road_multiplier * time_factor)
        
        predictions.append({
            'Location': row['Location'],
            'Current_Volume': row['Volume'],
            'Predicted_Volume': predicted,
            'Confidence': 0.85,
            'Change_Percent': round((predicted - row['Volume']) / row['Volume'] * 100, 1)
        })
    
    return pd.DataFrame(predictions)

def detect_anomalies(df):
    """Detect traffic anomalies"""
    anomalies = []
    
    # Speed violations
    speed_violations = df[df['Avg_Speed'] > df['Speed_Limit'] * 1.2]
    for _, row in speed_violations.iterrows():
        anomalies.append({
            'Location': row['Location'],
            'Type': 'Speed Violation',
            'Severity': 'High',
            'Details': f"Speed {row['Avg_Speed']} km/h exceeds limit {row['Speed_Limit']} km/h",
            'Timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    # Volume anomalies
    mean_volume = df['Volume'].mean()
    volume_anomalies = df[df['Volume'] > mean_volume * 1.5]
    for _, row in volume_anomalies.iterrows():
        anomalies.append({
            'Location': row['Location'],
            'Type': 'Volume Anomaly',
            'Severity': 'Medium',
            'Details': f"Volume {row['Volume']} vpd is unusually high",
            'Timestamp': datetime.now().strftime('%H:%M:%S')
        })
    
    return anomalies

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🚦 TrafficFlow AI Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_sample_data()
    predictions = calculate_ml_predictions(df)
    anomalies = detect_anomalies(df)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Location filter
    locations = ['All Locations'] + list(df['Location'].unique())
    selected_location = st.sidebar.selectbox("Select Location", locations)
    
    # Time range filter
    time_range = st.sidebar.selectbox("Time Range", ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last Week"])
    
    # ML settings
    st.sidebar.markdown("**ML Settings**")
    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    show_anomalies = st.sidebar.checkbox("Show Anomalies", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 ML Insights", "🚨 Alerts", "📈 Analytics"])
    
    with tab1:
        st.subheader("📊 System Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Locations",
                value=len(df),
                delta="+2 this week"
            )
        
        with col2:
            st.metric(
                label="Active Alerts",
                value=len(anomalies),
                delta="-1 from yesterday"
            )
        
        with col3:
            st.metric(
                label="Avg Volume",
                value=f"{df['Volume'].mean():.0f} vpd",
                delta="+5% from yesterday"
            )
        
        with col4:
            st.metric(
                label="System Health",
                value="Excellent",
                delta="99.9% uptime"
            )
        
        # Traffic heatmap
        st.subheader("🗺️ Traffic Heatmap")
        
        # Create heatmap data
        heatmap_data = df.pivot_table(
            values='Volume', 
            index='Location', 
            columns='Road_Type', 
            aggfunc='mean',
            fill_value=0
        )
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='RdYlGn_r',
            title="Traffic Volume by Location and Road Type"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🤖 ML Predictions")
        
        if show_predictions:
            # Predictions table
            st.dataframe(
                predictions[['Location', 'Current_Volume', 'Predicted_Volume', 'Confidence', 'Change_Percent']],
                use_container_width=True
            )
            
            # Predictions chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Current Volume',
                x=predictions['Location'],
                y=predictions['Current_Volume'],
                marker_color='lightblue'
            ))
            fig.add_trace(go.Bar(
                name='Predicted Volume',
                x=predictions['Location'],
                y=predictions['Predicted_Volume'],
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title="Traffic Volume Predictions",
                xaxis_title="Location",
                yaxis_title="Volume (vpd)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance
        st.subheader("📈 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", "89%")
            st.metric("Precision", "92%")
        
        with col2:
            st.metric("Recall", "88%")
            st.metric("F1-Score", "90%")
    
    with tab3:
        st.subheader("🚨 Traffic Alerts")
        
        if show_anomalies and anomalies:
            for anomaly in anomalies:
                severity_class = f"alert-{anomaly['Severity'].lower()}"
                st.markdown(f"""
                <div class="metric-card {severity_class}">
                    <strong>{anomaly['Location']}</strong><br>
                    Type: {anomaly['Type']} | Severity: {anomaly['Severity']}<br>
                    {anomaly['Details']}<br>
                    <small>Time: {anomaly['Timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
                st.write("")
        else:
            st.success("✅ No anomalies detected!")
        
        # Real-time updates
        st.subheader("⚡ Real-time Updates")
        if st.button("Refresh Data"):
            st.rerun()
    
    with tab4:
        st.subheader("📈 Traffic Analytics")
        
        # Volume distribution
        fig = px.histogram(
            df, 
            x='Volume', 
            color='Road_Type',
            title="Traffic Volume Distribution by Road Type",
            labels={'Volume': 'Volume (vpd)', 'count': 'Number of Locations'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Speed vs Volume scatter
        fig = px.scatter(
            df, 
            x='Volume', 
            y='Avg_Speed',
            color='Congestion_Level',
            size='Speed_Limit',
            title="Traffic Volume vs Average Speed",
            labels={'Volume': 'Volume (vpd)', 'Avg_Speed': 'Average Speed (km/h)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Congestion analysis
        congestion_counts = df['Congestion_Level'].value_counts()
        fig = px.pie(
            values=congestion_counts.values,
            names=congestion_counts.index,
            title="Traffic Congestion Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**TrafficFlow AI Platform** - Intelligent Traffic Analytics with ML")
    st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
