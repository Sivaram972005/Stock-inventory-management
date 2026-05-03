import pandas as pd
import numpy as np
from ai_inventory_optimizer import (
    DataManager, Forecaster, ProductClassifier, 
    InventoryOptimizer, AnomalyDetector, Visualizer
)
import os

def run_demo():
    print("=== AI Inventory Optimizer Demo ===")
    
    # 1. Data Management
    dm = DataManager()
    print("Generating synthetic historical sales data (2 years)...")
    sales_df = dm.generate_synthetic_data(num_products=5)
    
    # 2. Product Classification
    classifier = ProductClassifier()
    print("Classifying products based on sales patterns...")
    product_classes = classifier.classify_products(sales_df)
    print(product_classes[['product_id', 'name', 'classification']])
    
    # 3. Forecasting & Optimization for a specific product
    forecaster = Forecaster()
    optimizer = InventoryOptimizer(service_level=0.95)
    detector = AnomalyDetector()
    visualizer = Visualizer()
    
    # Let's pick 'Rice Bag' (ID 1)
    target_product_id = 1
    product_name = "Rice Bag"
    print(f"\nProcessing {product_name}...")
    
    # Preprocess
    df_prod = dm.preprocess_for_forecasting(target_product_id)
    
    # Anomaly Detection
    anomalies = detector.flag_recent_anomalies(df_prod)
    if anomalies:
        for a in anomalies:
            print(f"[ALERT] {a}")
    else:
        print("No recent anomalies detected.")
        
    # Hybrid Forecast
    print("Running Hybrid LSTM + ARIMA Forecasting (90 days)...")
    forecast_results = forecaster.forecast_hybrid(df_prod, forecast_days=90)
    
    # Performance Evaluation (on last 30 days of historical data)
    actual_recent = df_prod['quantity_sold'].tail(30).values
    # For demo, we'll just show the metrics for the training fit or a validation split
    # Here we'll just print the trend
    print(f"Forecast Trend: {forecast_results['trend'].upper()}")
    
    # 4. Inventory Optimization
    current_stock = 250  # Hypothetical current stock
    std_dev = df_prod['quantity_sold'].std()
    
    opt_results = optimizer.optimize(
        current_stock=current_stock,
        predicted_demand_3m=np.sum(forecast_results['forecast']),
        historical_std_dev=std_dev,
        lead_time_days=7,
        storage_capacity=1000
    )
    
    print("\n--- Optimization Recommendations ---")
    print(f"Predicted 3-Month Demand: {np.sum(forecast_results['forecast']):.2f}")
    print(f"Calculated Safety Stock: {opt_results['safety_stock']}")
    print(f"Reorder Point: {opt_results['reorder_point']}")
    print(f"Recommended Reorder Quantity: {opt_results['reorder_qty']}")
    print(f"Status: {opt_results['status']}")
    
    # 5. Visualization
    print("\nGenerating visualization plots in 'static/' directory...")
    visualizer.plot_demand_forecast(df_prod, forecast_results, product_name)
    print(f"Plots generated: static/forecast_{product_name.replace(' ', '_')}.png")
    
    print("\nDemo Completed Successfully!")

if __name__ == "__main__":
    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    run_demo()
