"""
Synthetic Network Data Generator for Walmart SDWAN Scenarios
Generates realistic network telemetry data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

class WalmartNetworkDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_normal_data(self, days=7, interval_minutes=1):
        start_date = datetime.now() - timedelta(days=days)
        timestamps = pd.date_range(start_date, periods=days*24*60//interval_minutes, freq=f'{interval_minutes}min')
        
        data = []
        for ts in timestamps:
            hour = ts.hour
            day_of_week = ts.dayofweek
            
            is_peak = (10 <= hour <= 14) or (17 <= hour <= 20)
            is_weekend = day_of_week >= 5
            
            base_multiplier = 1.0
            if is_peak:
                base_multiplier = 2.5
            if is_weekend:
                base_multiplier *= 1.3
            
            latency = np.random.normal(25, 5) * (1 + 0.1 * base_multiplier)
            jitter = np.random.exponential(3) + 1
            packet_loss = np.random.exponential(0.05)
            bandwidth_util = np.random.normal(45, 10) * base_multiplier
            connection_count = int(np.random.normal(500, 100) * base_multiplier)
            error_rate = np.random.exponential(0.01)
            
            latency = np.clip(latency, 10, 100)
            jitter = np.clip(jitter, 0.5, 20)
            packet_loss = np.clip(packet_loss, 0, 0.5)
            bandwidth_util = np.clip(bandwidth_util, 10, 80)
            error_rate = np.clip(error_rate, 0, 0.1)
            
            data.append({
                'timestamp': ts,
                'latency_ms': round(latency, 2),
                'jitter_ms': round(jitter, 2),
                'packet_loss_pct': round(packet_loss, 3),
                'bandwidth_util_pct': round(bandwidth_util, 2),
                'connection_count': connection_count,
                'error_rate': round(error_rate, 4)
            })
        
        return pd.DataFrame(data)
    
    def inject_anomaly_path_failure(self, df, start_idx, duration_minutes=30):
        end_idx = min(start_idx + duration_minutes, len(df))
        
        for i in range(start_idx, end_idx):
            failover_progress = (i - start_idx) / duration_minutes
            
            if failover_progress < 0.2:
                df.loc[i, 'latency_ms'] *= np.random.uniform(3, 8)
                df.loc[i, 'packet_loss_pct'] = np.random.uniform(5, 15)
                df.loc[i, 'jitter_ms'] *= np.random.uniform(4, 10)
                df.loc[i, 'bandwidth_util_pct'] *= 0.3
            else:
                recovery_factor = 1 - ((failover_progress - 0.2) / 0.8)
                df.loc[i, 'latency_ms'] *= (1 + recovery_factor * 2)
                df.loc[i, 'packet_loss_pct'] = recovery_factor * 2
                df.loc[i, 'jitter_ms'] *= (1 + recovery_factor)
        
        return df
    
    def generate_dataset_with_anomalies(self, days=7, anomaly_count=5):
        df = self.generate_normal_data(days=days)
        
        anomaly_locations = []
        data_points = len(df)
        
        for _ in range(anomaly_count):
            start_idx = random.randint(200, data_points - 500)
            duration = random.randint(20, 60)
            
            self.inject_anomaly_path_failure(df, start_idx, duration)
            
            anomaly_locations.append({
                'type': 'path_failure',
                'start': start_idx,
                'duration': duration
            })
        
        df['is_anomaly'] = 0
        for anomaly in anomaly_locations:
            start = anomaly['start']
            end = start + anomaly['duration']
            df.loc[start:end, 'is_anomaly'] = 1
        
        return df, anomaly_locations
    
    def save_datasets(self, output_dir=''):
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        train_data = self.generate_normal_data(days=7)
        train_data.to_csv(f'{output_dir}training_data_clean.csv', index=False)
        print(f"✓ Generated training data: {len(train_data)} samples")
        
        test_data, anomalies = self.generate_dataset_with_anomalies(days=3, anomaly_count=7)
        test_data.to_csv(f'{output_dir}test_data_with_anomalies.csv', index=False)
        print(f"✓ Generated test data: {len(test_data)} samples, {len(anomalies)} anomalies")
        
        return train_data, test_data

if __name__ == "__main__":
    print("Walmart Network Data Generator - NetIntel AI Suite")
    print("=" * 60)
    
    generator = WalmartNetworkDataGenerator(seed=42)
    train, test = generator.save_datasets()
    
    print("\n✓ Synthetic data ready for model training")
