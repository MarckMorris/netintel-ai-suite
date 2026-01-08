# API Reference - NetIntel AI Suite

## Anomaly Detection API

**Base URL**: `http://localhost:8001`

### Endpoints

#### GET /health
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### POST /predict
Predict anomaly for network metrics

**Request Body**:
```json
{
  "timestamp": "2024-01-07T10:00:00",
  "latency_ms": 45.5,
  "jitter_ms": 5.2,
  "packet_loss_pct": 0.1,
  "bandwidth_util_pct": 65.0,
  "connection_count": 500,
  "error_rate": 0.01
}
```

**Response**:
```json
{
  "timestamp": "2024-01-07T10:00:00",
  "anomaly_score": 23.4,
  "is_anomaly": false,
  "severity": "NORMAL",
  "root_cause": "NORMAL"
}
```

---

## Alert Correlation API

**Base URL**: `http://localhost:8002`

## SDWAN Optimizer API

**Base URL**: `http://localhost:8003`
