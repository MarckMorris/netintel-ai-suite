"""
NetIntel AI Suite - Unified Dashboard
Real-time monitoring for network operations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="NetIntel AI Suite",
    page_icon="🌐",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {font-size: 48px; font-weight: bold; color: #0071ce;}
    .sub-header {font-size: 24px; color: #555;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🌐 NetIntel AI Suite</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Network Intelligence Platform</p>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select View", [
        "🏠 Overview",
        "🔍 Anomaly Detection",
        "🔔 Alert Correlation",
        "🛣️ Path Optimization"
    ])
    
    st.divider()
    st.subheader("Filters")
    time_range = st.selectbox("Time Range", ["Last Hour", "Last 24 Hours", "Last 7 Days"])

if page == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stores Monitored", "4,723", "+12")
    with col2:
        st.metric("Active Alerts", "127", "-35 (70%)")
    with col3:
        st.metric("Avg Latency", "28ms", "-5ms")
    with col4:
        st.metric("Path Failures", "3", "-8")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Network Health Score")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=94,
            delta={'reference': 90},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 75], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 85}},
            title={'text': "Overall Health"}
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Anomalies Detected (24h)")
        data = pd.DataFrame({
            'hour': range(24),
            'anomalies': [2,1,0,1,3,5,8,12,15,20,18,16,14,12,10,8,15,22,25,20,15,10,5,3]
        })
        fig = px.area(data, x='hour', y='anomalies', title="Hourly Anomaly Detection")
        st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Anomaly Detection":
    st.subheader("Real-Time Anomaly Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info("🤖 LSTM Autoencoder analyzing 4,723 stores @ 1-minute intervals")
        
        st.subheader("Anomaly Timeline")
        anomalies = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-07 10:00', periods=60, freq='min'),
            'score': [30 + (i%10)*5 + (50 if i in [15,35,48] else 0) for i in range(60)],
            'threshold': [70]*60
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['score'], 
                                 name='Anomaly Score', fill='tozeroy'))
        fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['threshold'], 
                                 name='Threshold', line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Recent Anomalies")
        recent = [
            {"time": "10:48 AM", "store": "Store-1234", "severity": "🔴 CRITICAL", "cause": "Path Failure"},
            {"time": "10:35 AM", "store": "Store-5678", "severity": "🟡 WARNING", "cause": "High Latency"},
            {"time": "10:15 AM", "store": "Store-9012", "severity": "🔴 CRITICAL", "cause": "Packet Loss"}
        ]
        for r in recent:
            with st.container():
                st.markdown(f"**{r['time']}** - {r['store']}")
                st.markdown(f"{r['severity']} - {r['cause']}")
                st.divider()

elif page == "🔔 Alert Correlation":
    st.subheader("Intelligent Alert Correlation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Raw Alerts (24h)", "8,542")
        st.metric("After Correlation", "1,281", "-7,261 (85%)")
    
    with col2:
        st.metric("False Positives Eliminated", "7,261")
        st.metric("Engineer Hours Saved", "48 hrs/week")
    
    st.divider()
    st.subheader("Correlated Incident Groups")
    
    incidents = pd.DataFrame({
        'Group ID': ['CORR-001', 'CORR-002', 'CORR-003'],
        'Root Cause': ['SDWAN Path Failure', 'Bandwidth Saturation', 'DDoS Attack'],
        'Alert Count': [47, 23, 156],
        'Affected Stores': [12, 5, 3],
        'Severity': ['🔴 Critical', '🟡 Warning', '🔴 Critical']
    })
    st.dataframe(incidents, use_container_width=True)

elif page == "🛣️ Path Optimization":
    st.subheader("SDWAN Path Optimization (RL Agent)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Paths", "18,892")
    with col2:
        st.metric("Optimizations/hour", "1,247")
    with col3:
        st.metric("Latency Improvement", "-25%")
    
    st.divider()
    
    st.subheader("Path Performance Comparison")
    paths = pd.DataFrame({
        'ISP': ['Comcast', 'AT&T', 'Verizon', '5G Backup'],
        'Latency (ms)': [25, 30, 28, 40],
        'Packet Loss (%)': [0.05, 0.08, 0.06, 0.15],
        'Cost ($/GB)': [0.10, 0.12, 0.11, 0.20],
        'Utilization (%)': [65, 45, 55, 15]
    })
    
    fig = px.bar(paths, x='ISP', y=['Latency (ms)', 'Packet Loss (%)', 'Cost ($/GB)'], 
                 title="Multi-ISP Performance Metrics", barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("🤖 DQN Agent making real-time path selection decisions")

st.divider()
st.caption("NetIntel AI Suite v1.0.0 ")
