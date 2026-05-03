import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class Visualizer:
    def __init__(self, output_dir='static'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_demand_forecast(self, historical_df, forecast_results, product_name):
        """
        Generates a plot showing historical sales, forecast, and confidence intervals.
        """
        plt.figure(figsize=(12, 6))
        
        # Historical Data
        hist_subset = historical_df.tail(12) # Since it's monthly
        plt.plot(hist_subset['month_index'], hist_subset['sales'], label='Historical Sales', color='blue', alpha=0.6, marker='o')
        
        # Forecast Data
        last_index = historical_df['month_index'].max()
        forecast_dates = [last_index + i + 1 for i in range(len(forecast_results['forecast']))]
        
        plt.plot(forecast_dates, forecast_results['forecast'], label='AI Hybrid Forecast', color='red', linewidth=2, marker='o')
        
        # Confidence Intervals
        plt.fill_between(forecast_dates, forecast_results['lower'], forecast_results['upper'], 
                         color='red', alpha=0.1, label='95% Confidence Interval')
        
        plt.title(f'Demand Forecast: {product_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Month Index')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        file_path = os.path.join(self.output_dir, f'forecast_{product_name.replace(" ", "_")}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path

    def plot_actual_vs_predicted(self, actual, predicted, product_name):
        """
        Scatter plot for actual vs predicted to evaluate model performance visually.
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(actual, predicted, alpha=0.5, color='green')
        
        # 45-degree line
        max_val = max(max(actual), max(predicted))
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        
        plt.title(f'Actual vs Predicted: {product_name}', fontsize=14)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        file_path = os.path.join(self.output_dir, f'eval_{product_name.replace(" ", "_")}.png')
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        return file_path
