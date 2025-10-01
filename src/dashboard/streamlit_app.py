#!/usr/bin/env python3
"""
Streamlit Dashboard for TrafficFlow AI
Interactive traffic analytics dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json

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
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Load sample traffic data"""
    return pd.DataFrame({
        'Location': ['Main St & First Ave', 'Oak Ave & Second St', 'Pine St & Third Ave'],
        'Volume': [1200, 2800, 4500],
        'Speed_Limit': [50, 40, 60],
        'Average_Speed': [45, 38, 55],
        'Road_Type': ['Residential', 'Collector', 'Arterial'],
        'Congestion_Level': ['Low', 'Medium', 'Low']
    })

def get_api_data():
    """Get data from FastAPI"""
    try:
        response = requests.get("http://localhost:8000/api/v1/analytics/dashboard")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def main():
    """Main dashboard application"""
    # Header
    st.markdown('<h1 class="main-header">🚦 TrafficFlow AI Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Load data
    df = load_sample_data()
    api_data = get_api_data()
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Locations", len(df))
    
    with col2:
        st.metric("Average Volume", f"{df['Volume'].mean():.0f} vpd")
    
    with col3:
        st.metric("System Health", "Excellent")
    
    with col4:
        st.metric("Active Alerts", "2")
    
    # Charts
    st.subheader("📊 Traffic Volume Distribution")
    fig = px.bar(df, x='Location', y='Volume', color='Road_Type', 
                 title="Traffic Volume by Location")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🚗 Speed Analysis")
    fig = px.scatter(df, x='Volume', y='Average_Speed', 
                     color='Congestion_Level', size='Speed_Limit',
                     title="Volume vs Speed Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Real-time updates
    if st.button("Refresh Data"):
        st.rerun()

if __name__ == "__main__":
    main()