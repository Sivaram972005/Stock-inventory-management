import numpy as np
import pandas as pd

class AnomalyDetector:
    def __init__(self, threshold=3):
        self.threshold = threshold

    def detect_anomalies(self, series):
        """
        Detects anomalies using Z-score method.
        Returns a boolean mask where True indicates an anomaly.
        """
        if len(series) < 2:
            return np.zeros(len(series), dtype=bool)
            
        mean = np.mean(series)
        std = np.std(series)
        
        if std == 0:
            return np.zeros(len(series), dtype=bool)
            
        z_scores = np.abs((series - mean) / std)
        return z_scores > self.threshold

    def flag_recent_anomalies(self, df, window=3):
        """Flags anomalies in the most recent 'window' periods."""
        recent_data = df.tail(window)
        anomalies = self.detect_anomalies(recent_data['sales'].values)
        
        flags = []
        if any(anomalies):
            for i, is_anomaly in enumerate(anomalies):
                if is_anomaly:
                    month = recent_data.iloc[i]['month']
                    val = recent_data.iloc[i]['sales']
                    flags.append(f"Anomaly detected in {month}: Sold {val} units.")
        
        return flags
