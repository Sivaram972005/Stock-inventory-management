import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class DataManager:
    def __init__(self, data_path='data_set/data.csv'):
        self.data_path = data_path
        self.raw_df = None
        self.sales_df = None

    def load_data(self):
        """Loads existing data if available."""
        if os.path.exists(self.data_path):
            self.raw_df = pd.read_csv(self.data_path)
            self.sales_df = self.raw_df
            return self.raw_df
        return None

    def generate_synthetic_data(self, num_products=5, days=730):
        """
        Generates synthetic historical sales data for multiple products.
        Includes seasonality, trends, and holidays.
        """
        np.random.seed(42)
        start_date = datetime.now() - timedelta(days=days)
        date_range = [start_date + timedelta(days=i) for i in range(days)]
        
        categories = ['Grocery', 'Dairy', 'Bakery', 'Beverage', 'Snacks']
        products = [
            {'id': 1, 'name': 'Rice Bag', 'category': 'Grocery', 'base_demand': 50, 'trend': 0.01},
            {'id': 2, 'name': 'Milk Pack', 'category': 'Dairy', 'base_demand': 70, 'trend': 0.005},
            {'id': 3, 'name': 'Bread Loaf', 'category': 'Bakery', 'base_demand': 60, 'trend': 0.002},
            {'id': 4, 'name': 'Tea Powder', 'category': 'Beverage', 'base_demand': 25, 'trend': 0.008},
            {'id': 5, 'name': 'Biscuit Pack', 'category': 'Snacks', 'base_demand': 65, 'trend': 0.015}
        ]

        all_sales = []
        
        for prod in products[:num_products]:
            for i, date in enumerate(date_range):
                # Base demand + Trend
                demand = prod['base_demand'] + (i * prod['trend'])
                
                # Seasonality (Monthly)
                month_effect = 1.0 + 0.2 * np.sin(2 * np.pi * date.month / 12)
                
                # Day of week effect (Weekends have higher sales)
                dow_effect = 1.2 if date.weekday() >= 5 else 1.0
                
                # Holiday effect (Random spikes for "festivals")
                holiday_effect = 1.5 if np.random.random() < 0.02 else 1.0
                
                # Noise
                noise = np.random.normal(0, 5)
                
                total_sold = max(0, int(demand * month_effect * dow_effect * holiday_effect + noise))
                
                all_sales.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'product_id': prod['id'],
                    'product_name': prod['name'],
                    'category': prod['category'],
                    'quantity_sold': total_sold,
                    'is_holiday': 1 if holiday_effect > 1.0 else 0
                })
        
        self.sales_df = pd.DataFrame(all_sales)
        self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
        return self.sales_df

    def preprocess_for_forecasting(self, product_name):
        """Prepares multivariate time-series features for a specific product."""
        if self.sales_df is None:
            return None
        
        df = self.sales_df[self.sales_df['product_name'] == product_name].copy()
        df = df.sort_values('month_index')
        
        # Lag features (shortened for monthly data)
        df['lag_1'] = df['sales'].shift(1)
        df['lag_2'] = df['sales'].shift(2)
        df['lag_3'] = df['sales'].shift(3)
        
        # Rolling averages
        df['rolling_mean_3'] = df['sales'].rolling(window=3).mean()
        
        df = df.fillna(0) # Forward fill missing due to lag, or zero since it's a short sequence
        return df

if __name__ == "__main__":
    dm = DataManager()
    sales = dm.load_data()
    print(f"Loaded {len(sales)} sales records.")
    
    product_features = dm.preprocess_for_forecasting('Laptop')
    print("\nFeature Engineering for Laptop:")
    print(product_features.head())
