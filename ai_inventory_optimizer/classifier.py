import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class ProductClassifier:
    def __init__(self):
        self.categories = {}

    def classify_products(self, sales_df):
        """
        Classifies products based on velocity and variability.
        Fast-moving: High mean volume
        Slow-moving: Low mean volume
        Seasonal: High coefficient of variation or strong monthly correlation
        """
        product_stats = sales_df.groupby('product_name').agg({
            'sales': ['mean', 'std', 'sum'],
            'category': 'first'
        })
        
        product_stats.columns = ['mean_sales', 'std_sales', 'total_sales', 'category']
        product_stats['cv'] = product_stats['std_sales'] / product_stats['mean_sales']
        
        # Simple threshold-based classification
        mean_threshold = product_stats['mean_sales'].median()
        cv_threshold = 0.5  # High variability indicates potential seasonality or erratic demand
        
        classifications = []
        for pname, row in product_stats.iterrows():
            if row['cv'] > cv_threshold:
                label = 'Seasonal'
            elif row['mean_sales'] > mean_threshold:
                label = 'Fast-moving'
            else:
                label = 'Slow-moving'
            
            classifications.append({
                'product_name': pname,
                'category': row['category'],
                'classification': label,
                'mean_sales': row['mean_sales'],
                'cv': row['cv']
            })
            
        return pd.DataFrame(classifications)

    def get_strategy(self, classification):
        strategies = {
            'Fast-moving': 'High safety stock, frequent reordering, high service level.',
            'Slow-moving': 'Minimal safety stock, order on demand, focus on reducing holding costs.',
            'Seasonal': 'Proactive pre-season stock building, aggressive clearance post-season.'
        }
        return strategies.get(classification, 'Standard inventory management.')
