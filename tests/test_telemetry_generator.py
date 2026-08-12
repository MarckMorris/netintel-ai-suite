"""The synthetic telemetry generator.

Labelled data is only useful if the labels are right and the output is
reproducible. These tests check both, plus that the traffic shape the model is
supposed to learn is actually present.
"""

from __future__ import annotations

import pandas as pd
import pytest

from synthetic_network_logs import REFERENCE_START, NetworkTelemetryGenerator

COLUMNS = {
    "timestamp",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "bandwidth_util_pct",
    "connection_count",
    "error_rate",
}


@pytest.fixture
def generator():
    return NetworkTelemetryGenerator(seed=42)


class TestBaseline:
    def test_one_row_per_minute(self, generator):
        assert len(generator.generate_normal_data(days=1)) == 24 * 60

    def test_every_expected_column_is_present(self, generator):
        assert COLUMNS <= set(generator.generate_normal_data(days=1).columns)

    def test_output_is_reproducible(self):
        """Without a fixed reference start this was impossible: the timestamps
        moved with the wall clock on every call."""
        first = NetworkTelemetryGenerator(42).generate_normal_data(days=1)
        second = NetworkTelemetryGenerator(42).generate_normal_data(days=1)
        pd.testing.assert_frame_equal(first, second)

    def test_a_different_seed_gives_different_data(self):
        first = NetworkTelemetryGenerator(42).generate_normal_data(days=1)
        second = NetworkTelemetryGenerator(7).generate_normal_data(days=1)
        assert not first["latency_ms"].equals(second["latency_ms"])

    def test_the_start_date_can_be_overridden(self, generator):
        start = pd.Timestamp("2020-06-15")
        frame = generator.generate_normal_data(days=1, start_date=start)
        assert frame["timestamp"].iloc[0] == start

    def test_the_default_start_is_the_fixed_reference(self, generator):
        assert generator.generate_normal_data(days=1)["timestamp"].iloc[0] == REFERENCE_START

    def test_the_global_random_state_is_left_alone(self):
        """Generating data must not perturb any other RNG consumer in the process."""
        import numpy as np

        np.random.seed(0)
        expected = np.random.random()
        np.random.seed(0)
        NetworkTelemetryGenerator(42).generate_normal_data(days=1)
        assert np.random.random() == expected


class TestValueRanges:
    @pytest.mark.parametrize(
        "column,low,high",
        [
            ("latency_ms", 10, 100),
            ("jitter_ms", 0.5, 20),
            ("packet_loss_pct", 0, 0.5),
            ("bandwidth_util_pct", 10, 80),
            ("error_rate", 0, 0.1),
        ],
    )
    def test_clean_traffic_stays_inside_its_band(self, generator, column, low, high):
        series = generator.generate_normal_data(days=1)[column]
        assert series.min() >= low
        assert series.max() <= high

    def test_connection_counts_are_positive_integers(self, generator):
        counts = generator.generate_normal_data(days=1)["connection_count"]
        assert (counts > 0).all()
        assert counts.dtype.kind in "iu"


class TestTrafficShape:
    def test_peak_hours_carry_more_traffic_than_the_small_hours(self, generator):
        """This daily shape is the signal an unsupervised detector has to learn;
        if it is absent the whole exercise is meaningless."""
        frame = generator.generate_normal_data(days=7)
        peak = frame[frame["timestamp"].dt.hour.between(10, 14)]["bandwidth_util_pct"].mean()
        quiet = frame[frame["timestamp"].dt.hour.between(2, 5)]["bandwidth_util_pct"].mean()
        assert peak > quiet

    def test_weekends_are_busier_than_weekdays(self, generator):
        frame = generator.generate_normal_data(days=14)
        weekend = frame[frame["timestamp"].dt.dayofweek >= 5]["connection_count"].mean()
        weekday = frame[frame["timestamp"].dt.dayofweek < 5]["connection_count"].mean()
        assert weekend > weekday


class TestFaultInjection:
    def test_labels_are_produced(self, generator):
        frame, _ = generator.generate_dataset_with_anomalies(days=2, anomaly_count=3)
        assert set(frame["is_anomaly"].unique()) <= {0, 1}
        assert frame["is_anomaly"].sum() > 0

    def test_one_location_is_recorded_per_injected_fault(self, generator):
        _, locations = generator.generate_dataset_with_anomalies(days=3, anomaly_count=4)
        assert len(locations) == 4
        assert all({"type", "start", "duration"} <= set(loc) for loc in locations)

    def test_the_labelled_windows_match_the_recorded_locations(self, generator):
        frame, locations = generator.generate_dataset_with_anomalies(days=3, anomaly_count=2)
        for location in locations:
            window = frame.loc[location["start"] : location["start"] + location["duration"]]
            assert (window["is_anomaly"] == 1).all()

    def test_faults_actually_degrade_the_metrics(self, generator):
        """A label with no corresponding signal trains a detector on nothing."""
        frame, _ = generator.generate_dataset_with_anomalies(days=3, anomaly_count=4)
        faulty = frame[frame["is_anomaly"] == 1]
        clean = frame[frame["is_anomaly"] == 0]
        assert faulty["latency_ms"].mean() > clean["latency_ms"].mean()
        assert faulty["packet_loss_pct"].mean() > clean["packet_loss_pct"].mean()

    def test_most_of_the_series_stays_clean(self, generator):
        frame, _ = generator.generate_dataset_with_anomalies(days=7, anomaly_count=5)
        assert frame["is_anomaly"].mean() < 0.10

    def test_injection_is_reproducible(self):
        first, first_locations = NetworkTelemetryGenerator(9).generate_dataset_with_anomalies(
            days=2, anomaly_count=3
        )
        second, second_locations = NetworkTelemetryGenerator(9).generate_dataset_with_anomalies(
            days=2, anomaly_count=3
        )
        pd.testing.assert_frame_equal(first, second)
        assert first_locations == second_locations

