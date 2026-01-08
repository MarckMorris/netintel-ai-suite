# 🚀 NetIntel AI Suite - Deployment Guide

## Prerequisites

- Python 3.9+
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

---

## Quick Deployment

### Option 1: Local Development

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate demo data
cd project1_anomaly_detection/data
python synthetic_network_logs.py
cd ../..

# 4. Launch dashboard
streamlit run dashboard/streamlit_app.py
```

### Option 2: Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up -d

# Access services
# Dashboard:    http://localhost:8501
# API Anomaly:  http://localhost:8001/docs
# API Correlation: http://localhost:8002/docs
# API Optimizer: http://localhost:8003/docs
```

---

## Testing the Deployment

```bash
# Test API health
curl http://localhost:8001/health

# Make a prediction
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-07T10:00:00",
    "latency_ms": 45.5,
    "jitter_ms": 5.2,
    "packet_loss_pct": 0.1,
    "bandwidth_util_pct": 65.0,
    "connection_count": 500,
    "error_rate": 0.01
  }'
```

---

## Troubleshooting

### Port Already in Use
```bash
# Find and kill process
lsof -i :8501
kill -9 <PID>
```

### Module Not Found
```bash
pip install --force-reinstall -r requirements.txt
```

### Out of Memory
Reduce batch size in `lstm_autoencoder.py` line 87:
```python
batch_size=16  # instead of 32
```

---

## Production Deployment (Walmart)

1. Configure Versa SDWAN credentials in `.env`
2. Set up NOC webhook endpoints
3. Deploy to Kubernetes cluster
4. Enable Prometheus monitoring
5. Configure alerting to PagerDuty/Slack

