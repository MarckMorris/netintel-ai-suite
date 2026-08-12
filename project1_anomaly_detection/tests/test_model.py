"""LSTM autoencoder tests.

Skipped unless TensorFlow is installed. It is an optional extra here so that a
clean clone, and CI, can run the rest of the suite in seconds instead of pulling
a ~600 MB wheel. Install it with:

    pip install -r requirements-anomaly.txt
"""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow is an optional extra")

from lstm_autoencoder import NetworkAnomalyDetector  # noqa: E402
from synthetic_network_logs import NetworkTelemetryGenerator  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def sample_data():
    return NetworkTelemetryGenerator(seed=42).generate_normal_data(days=1)


@pytest.fixture
def detector():
    return NetworkAnomalyDetector(sequence_length=60)


def test_initialisation(detector):
    assert detector.sequence_length == 60
    assert detector.contamination == 0.01


def test_feature_engineering_adds_derived_columns(detector, sample_data):
    features = detector._engineer_features(sample_data)
    assert features.shape[0] == sample_data.shape[0]
    assert features.shape[1] > sample_data.shape[1]


def test_training_sets_a_threshold(detector, sample_data):
    detector.train(sample_data, epochs=2, batch_size=32)
    assert detector.autoencoder is not None
    assert detector.threshold is not None


def test_prediction_labels_every_complete_sequence(detector, sample_data):
    detector.train(sample_data, epochs=2, batch_size=32)
    results = detector.predict(sample_data)

    assert {"anomaly_score", "is_anomaly"} <= set(results.columns)
    # The first `sequence_length` rows cannot form a full window.
    assert len(results) == len(sample_data) - detector.sequence_length
