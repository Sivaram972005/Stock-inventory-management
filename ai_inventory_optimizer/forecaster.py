import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings("ignore")

class Forecaster:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.lstm_model = None
        self.history = None

    def train_arima(self, series):
        """Trains an ARIMA model and returns predictions."""
        # Use auto_arima for best parameters, disable seasonality since we have limited data (12 points)
        model = auto_arima(series, seasonal=False, suppress_warnings=True, error_action="ignore")
        return model

    def build_lstm(self, input_shape):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        self.lstm_model = model
        return model

    def prepare_lstm_data(self, data, n_steps=30):
        """Prepares sequences for LSTM."""
        X, y = [], []
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        for i in range(len(scaled_data) - n_steps):
            X.append(scaled_data[i:i + n_steps, 0])
            y.append(scaled_data[i + n_steps, 0])
            
        return np.array(X).reshape(-1, n_steps, 1), np.array(y), scaled_data

    def forecast_hybrid(self, df, forecast_periods=3):
        """
        Combines ARIMA and LSTM for hybrid forecasting.
        """
        series = df['sales'].values
        
        # 1. ARIMA Forecast
        arima_model = self.train_arima(series)
        arima_forecast = arima_model.predict(n_periods=forecast_periods)
        
        # 2. LSTM Forecast
        n_steps = 3
        
        # If series is too short even for n_steps, or TF is not available, fallback to ARIMA entirely
        if len(series) <= n_steps or not TF_AVAILABLE:
            hybrid_forecast = arima_forecast
            lstm_forecast = np.zeros(forecast_periods)
        else:
            X, y, scaled_data = self.prepare_lstm_data(series, n_steps)

        
            if self.lstm_model is None:
                self.build_lstm((n_steps, 1))
                self.lstm_model.fit(X, y, epochs=50, batch_size=4, verbose=0)
                
            # Recursive LSTM Forecast
            lstm_forecast_scaled = []
            last_sequence = scaled_data[-n_steps:]
            
            current_seq = last_sequence.copy().reshape(1, n_steps, 1)
            for _ in range(forecast_periods):
                pred = self.lstm_model.predict(current_seq, verbose=0)
                lstm_forecast_scaled.append(pred[0, 0])
                # Update sequence: shift left and add prediction at the end
                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1, 0] = pred[0, 0]
                
            lstm_forecast = self.scaler.inverse_transform(np.array(lstm_forecast_scaled).reshape(-1, 1)).flatten()
            
            # 3. Hybrid Ensemble (Simple Weighted Average)
            hybrid_forecast = (arima_forecast * 0.4) + (lstm_forecast * 0.6)
            
        # Calculate Confidence Intervals (simplified)
        std_dev = np.std(series) if len(series) > 0 else 0
        upper_bound = hybrid_forecast + (1.96 * std_dev)
        lower_bound = np.maximum(0, hybrid_forecast - (1.96 * std_dev))
        
        return {
            'forecast': hybrid_forecast,
            'arima': arima_forecast,
            'lstm': lstm_forecast,
            'upper': upper_bound,
            'lower': lower_bound,
            'trend': 'increasing' if hybrid_forecast[-1] > hybrid_forecast[0] else 'decreasing'
        }

    def evaluate(self, actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
