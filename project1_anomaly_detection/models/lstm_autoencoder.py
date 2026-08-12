"""
Network Anomaly Detection - LSTM Autoencoder
Unsupervised anomaly detection on SD-WAN telemetry
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime

class NetworkAnomalyDetector:
    def __init__(self, sequence_length=60, contamination=0.01):
        self.sequence_length = sequence_length
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.isolation_forest = None
        self.threshold = None
        
        self.critical_thresholds = {
            'latency_ms': 150,
            'jitter_ms': 30,
            'packet_loss_pct': 1.0,
            'bandwidth_util_pct': 85
        }
    
    def build_autoencoder(self, input_dim):
        encoder_inputs = keras.Input(shape=(self.sequence_length, input_dim))
        
        encoded = keras.layers.LSTM(64, activation='relu', return_sequences=True)(encoder_inputs)
        encoded = keras.layers.LSTM(32, activation='relu', return_sequences=False)(encoded)
        encoded = keras.layers.RepeatVector(self.sequence_length)(encoded)
        
        decoded = keras.layers.LSTM(32, activation='relu', return_sequences=True)(encoded)
        decoded = keras.layers.LSTM(64, activation='relu', return_sequences=True)(decoded)
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(input_dim))(decoded)
        
        autoencoder = keras.Model(encoder_inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i+self.sequence_length])
        return np.array(sequences)
    
    def _engineer_features(self, df):
        features = df.copy()
        
        for col in ['latency_ms', 'jitter_ms', 'packet_loss_pct']:
            features[f'{col}_rolling_mean_5'] = df[col].rolling(5).mean()
            features[f'{col}_rolling_std_5'] = df[col].rolling(5).std()
        
        features['latency_change_rate'] = df['latency_ms'].diff()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            features['hour'] = df['timestamp'].dt.hour
            features['day_of_week'] = df['timestamp'].dt.dayofweek
            features['is_peak_hour'] = ((features['hour'] >= 10) & (features['hour'] <= 14)) | \
                                       ((features['hour'] >= 17) & (features['hour'] <= 20))
        
        features = features.fillna(method='bfill').fillna(0)
        numeric_features = features.select_dtypes(include=[np.number])
        
        return numeric_features
    
    def train(self, training_data, epochs=50, batch_size=32):
        print(f"[{datetime.now()}] Training NetIntel Anomaly Detector...")
        
        features = self._engineer_features(training_data)
        scaled_data = self.scaler.fit_transform(features)
        sequences = self.create_sequences(scaled_data)
        
        print(f"Created {len(sequences)} training sequences")
        
        input_dim = features.shape[1]
        self.autoencoder = self.build_autoencoder(input_dim)
        
        history = self.autoencoder.fit(
            sequences, sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=3)
            ]
        )
        
        reconstructions = self.autoencoder.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1,2))
        self.threshold = np.percentile(mse, 99)
        
        self.isolation_forest = IsolationForest(contamination=self.contamination, random_state=42)
        self.isolation_forest.fit(mse.reshape(-1, 1))
        
        print(f"✓ Training complete. Threshold: {self.threshold:.6f}")
        return history
    
    def predict(self, test_data):
        features = self._engineer_features(test_data)
        scaled_data = self.scaler.transform(features)
        sequences = self.create_sequences(scaled_data)
        
        reconstructions = self.autoencoder.predict(sequences)
        mse = np.mean(np.square(sequences - reconstructions), axis=(1,2))
        
        lstm_anomalies = (mse > self.threshold).astype(int)
        isolation_anomalies = (self.isolation_forest.predict(mse.reshape(-1, 1)) == -1).astype(int)
        critical_violations = self._check_critical_thresholds(test_data)
        
        combined_score = (lstm_anomalies + isolation_anomalies + critical_violations) / 3
        
        results = pd.DataFrame({
            'timestamp': test_data['timestamp'].iloc[self.sequence_length:].values,
            'reconstruction_error': mse,
            'anomaly_score': combined_score * 100,
            'is_anomaly': (combined_score >= 0.5).astype(int),
            'severity': self._classify_severity(combined_score, mse)
        })
        
        results['root_cause'] = results.apply(
            lambda row: self._identify_root_cause(
                test_data.iloc[row.name + self.sequence_length] if row['is_anomaly'] else None
            ), axis=1
        )
        
        return results
    
    def _check_critical_thresholds(self, df):
        violations = np.zeros(len(df) - self.sequence_length)
        
        for i in range(len(df) - self.sequence_length):
            window = df.iloc[i:i+self.sequence_length]
            if any([
                window['latency_ms'].max() > self.critical_thresholds['latency_ms'],
                window['jitter_ms'].max() > self.critical_thresholds['jitter_ms'],
                window['packet_loss_pct'].max() > self.critical_thresholds['packet_loss_pct'],
                window['bandwidth_util_pct'].max() > self.critical_thresholds['bandwidth_util_pct']
            ]):
                violations[i] = 1
        
        return violations
    
    def _classify_severity(self, scores, mse):
        severity = []
        for score in scores:
            if score >= 0.8:
                severity.append('CRITICAL')
            elif score >= 0.5:
                severity.append('WARNING')
            else:
                severity.append('NORMAL')
        return severity
    
    def _identify_root_cause(self, data_point):
        if data_point is None:
            return 'NORMAL'
        
        causes = []
        if data_point['latency_ms'] > self.critical_thresholds['latency_ms']:
            causes.append('HIGH_LATENCY')
        if data_point['packet_loss_pct'] > self.critical_thresholds['packet_loss_pct']:
            causes.append('PACKET_LOSS')
        if data_point['bandwidth_util_pct'] > self.critical_thresholds['bandwidth_util_pct']:
            causes.append('BANDWIDTH_SATURATION')
        
        return ','.join(causes) if causes else 'UNKNOWN_PATTERN'
    
    def save_model(self, path='./'):
        self.autoencoder.save(f'{path}lstm_autoencoder.h5')
        joblib.dump(self.scaler, f'{path}scaler.pkl')
        joblib.dump(self.isolation_forest, f'{path}isolation_forest.pkl')
        joblib.dump({'threshold': self.threshold}, f'{path}config.pkl')
        print(f"✓ Models saved to {path}")
    
    def load_model(self, path='./'):
        self.autoencoder = keras.models.load_model(f'{path}lstm_autoencoder.h5')
        self.scaler = joblib.load(f'{path}scaler.pkl')
        self.isolation_forest = joblib.load(f'{path}isolation_forest.pkl')
        config = joblib.load(f'{path}config.pkl')
        self.threshold = config['threshold']
        print(f"✓ Models loaded from {path}")

if __name__ == "__main__":
    print("NetIntel Anomaly Detection - demo")
    print("Ready for integration with NOC alerting systems")
