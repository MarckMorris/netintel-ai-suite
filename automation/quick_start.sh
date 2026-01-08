#!/bin/bash
# Quick start helper script

echo "🚀 Starting NetIntel AI Suite..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies if needed
if ! pip show streamlit > /dev/null 2>&1; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Generate data if needed
if [ ! -f "project1_anomaly_detection/data/training_data_clean.csv" ]; then
    echo "Generating demo data..."
    python project1_anomaly_detection/data/synthetic_network_logs.py
fi

# Launch dashboard
echo "✓ Launching dashboard at http://localhost:8501"
streamlit run dashboard/streamlit_app.py
