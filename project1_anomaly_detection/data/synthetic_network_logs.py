"""Synthetic network telemetry with injectable faults.

Produces a per-minute time series of latency, jitter, loss, utilisation,
connection count and error rate, with a daily and weekly shape: two traffic
peaks on weekdays, a broader lift at weekends. Faults can then be injected at
known indices, which gives labelled data for evaluating an unsupervised
detector without needing a real capture.

The traffic model is a plausible shape, not a measurement of any real network.
Anything trained on it should be re-validated on real telemetry before it is
trusted.
"""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd

#: Fixed epoch for reproducible output. Overridable per call.
REFERENCE_START = datetime(2026, 1, 1, 0, 0, 0)


class NetworkTelemetryGenerator:
    """Generates clean baseline telemetry and injects labelled faults into it."""

    def __init__(self, seed=42):
        # Local generators rather than the global numpy/random state, so
        # generating data does not perturb anything else in the process and two
        # generators with the same seed always agree.
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)

    def generate_normal_data(self, days=7, interval_minutes=1, start_date=None):
        """A clean baseline series.

        ``start_date`` defaults to a fixed reference instant rather than
        ``datetime.now()`` so that two generators with the same seed produce
        byte-identical frames. Pass a real date when you want the series to line
        up with something.
        """
        if start_date is None:
            start_date = REFERENCE_START

        timestamps = pd.date_range(
            start_date,
            periods=days * 24 * 60 // interval_minutes,
            freq=f"{interval_minutes}min",
        )

        data = []
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.dayofweek

            # Two peaks a day: late morning and early evening. Weekends lift
            # the whole curve rather than adding a third peak.
            is_peak = (10 <= hour <= 14) or (17 <= hour <= 20)
            is_weekend = day_of_week >= 5

            base_multiplier = 2.5 if is_peak else 1.0
            if is_weekend:
                base_multiplier *= 1.3

            latency = self.rng.normal(25, 5) * (1 + 0.1 * base_multiplier)
            jitter = self.rng.exponential(3) + 1
            packet_loss = self.rng.exponential(0.05)
            bandwidth_util = self.rng.normal(45, 10) * base_multiplier
            connection_count = int(self.rng.normal(500, 100) * base_multiplier)
            error_rate = self.rng.exponential(0.01)

            data.append(
                {
                    "timestamp": ts,
                    "latency_ms": round(float(np.clip(latency, 10, 100)), 2),
                    "jitter_ms": round(float(np.clip(jitter, 0.5, 20)), 2),
                    "packet_loss_pct": round(float(np.clip(packet_loss, 0, 0.5)), 3),
                    "bandwidth_util_pct": round(float(np.clip(bandwidth_util, 10, 80)), 2),
                    "connection_count": connection_count,
                    "error_rate": round(float(np.clip(error_rate, 0, 0.1)), 4),
                }
            )

        return pd.DataFrame(data)

    def inject_anomaly_path_failure(self, df, start_idx, duration_minutes=30):
        """Model an uplink failing over.

        The first fifth of the window is the failure itself: latency and loss
        spike, throughput collapses. The rest is a linear recovery as the
        secondary path takes over, which is what the shape looks like in a real
        capture and is harder for a detector than a simple step change.
        """
        end_idx = min(start_idx + duration_minutes, len(df))

        for i in range(start_idx, end_idx):
            progress = (i - start_idx) / duration_minutes

            if progress < 0.2:
                df.loc[i, "latency_ms"] *= self.rng.uniform(3, 8)
                df.loc[i, "packet_loss_pct"] = self.rng.uniform(5, 15)
                df.loc[i, "jitter_ms"] *= self.rng.uniform(4, 10)
                df.loc[i, "bandwidth_util_pct"] *= 0.3
            else:
                recovery = 1 - ((progress - 0.2) / 0.8)
                df.loc[i, "latency_ms"] *= 1 + recovery * 2
                df.loc[i, "packet_loss_pct"] = recovery * 2
                df.loc[i, "jitter_ms"] *= 1 + recovery

        return df

    def generate_dataset_with_anomalies(self, days=7, anomaly_count=5):
        """Baseline plus faults, with an ``is_anomaly`` label column.

        Returns the frame and the list of injected locations, so an evaluation
        can report per-incident detection as well as per-sample metrics.
        """
        df = self.generate_normal_data(days=days)

        anomaly_locations = []
        data_points = len(df)

        for _ in range(anomaly_count):
            start_idx = self.random.randint(200, data_points - 500)
            duration = self.random.randint(20, 60)

            self.inject_anomaly_path_failure(df, start_idx, duration)
            anomaly_locations.append(
                {"type": "path_failure", "start": start_idx, "duration": duration}
            )

        df["is_anomaly"] = 0
        for anomaly in anomaly_locations:
            start = anomaly["start"]
            df.loc[start : start + anomaly["duration"], "is_anomaly"] = 1

        return df, anomaly_locations

    def save_datasets(self, output_dir=""):
        """Write the two CSVs the training scripts expect.

        These files are not committed to the repository: they are ~900 KB of
        output that this function reproduces byte-for-byte from the seed.
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        train_data = self.generate_normal_data(days=7)
        train_data.to_csv(f"{output_dir}training_data_clean.csv", index=False)
        print(f"training data: {len(train_data)} samples")

        test_data, anomalies = self.generate_dataset_with_anomalies(days=3, anomaly_count=7)
        test_data.to_csv(f"{output_dir}test_data_with_anomalies.csv", index=False)
        print(f"test data:     {len(test_data)} samples, {len(anomalies)} injected faults")

        return train_data, test_data


#: Kept so older notebooks and scripts that imported the previous name keep
#: working. New code should use NetworkTelemetryGenerator.
WalmartNetworkDataGenerator = NetworkTelemetryGenerator


if __name__ == "__main__":
    print("Network telemetry generator - NetIntel AI Suite")
    print("=" * 60)
    NetworkTelemetryGenerator(seed=42).save_datasets()
