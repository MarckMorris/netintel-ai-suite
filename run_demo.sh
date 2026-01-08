#!/bin/bash
# Quick demo launcher

echo "🚀 NetIntel AI Suite - Demo Launcher"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating environment..."
source venv/bin/activate

# Check if packages installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -q -r requirements.txt
fi

# Generate data if needed
if [ ! -f "project1_anomaly_detection/data/training_data_clean.csv" ]; then
    echo "Generating demo data..."
    python project1_anomaly_detection/data/synthetic_network_logs.py
fi

echo ""
echo "✓ Setup complete!"
echo "✓ Launching dashboard at http://localhost:8501"
echo ""
streamlit run dashboard/streamlit_app.py
