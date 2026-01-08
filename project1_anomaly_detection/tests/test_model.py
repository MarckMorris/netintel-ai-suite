"""
Unit tests for Network Anomaly Detector
"""

import pytest
import pandas as pd
import sys
sys.path.append('..')
from models.lstm_autoencoder import NetworkAnomalyDetector
from data.synthetic_network_logs import WalmartNetworkDataGenerator

@pytest.fixture
def detector():
    return NetworkAnomalyDetector(sequence_length=60)

@pytest.fixture
def sample_data():
    generator = WalmartNetworkDataGenerator(seed=42)
    return generator.generate_normal_data(days=1)

def test_model_initialization(detector):
    assert detector.sequence_length == 60
    assert detector.contamination == 0.01

def test_feature_engineering(detector, sample_data):
    features = detector._engineer_features(sample_data)
    assert features.shape[0] == sample_data.shape[0]
    assert features.shape[1] > sample_data.shape[1]

def test_training(detector, sample_data):
    history = detector.train(sample_data, epochs=2, batch_size=32)
    assert detector.autoencoder is not None
    assert detector.threshold is not None

def test_prediction(detector, sample_data):
    detector.train(sample_data, epochs=2, batch_size=32)
    results = detector.predict(sample_data)
    
    assert 'anomaly_score' in results.columns
    assert 'is_anomaly' in results.columns
    assert len(results) == len(sample_data) - detector.sequence_length

if __name__ == "__main__":
    pytest.main([__file__])
