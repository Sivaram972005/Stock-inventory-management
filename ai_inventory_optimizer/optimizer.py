import numpy as np
import pandas as pd

class InventoryOptimizer:
    def __init__(self, service_level=0.95):
        """
        service_level: The probability of not having a stockout (e.g., 0.95 for 95%)
        Z-score for 95% is ~1.65
        """
        from scipy.stats import norm
        self.z_score = norm.ppf(service_level)

    def calculate_safety_stock(self, std_dev_demand, lead_time_days):
        """
        Safety Stock = Z * std_dev(demand) * sqrt(lead_time)
        """
        return self.z_score * std_dev_demand * np.sqrt(lead_time_days)

    def optimize(self, current_stock, predicted_demand_3m, historical_std_dev, 
                 lead_time_days=7, storage_capacity=1000, unit_cost=10, budget=5000):
        """
        Calculates reorder quantity based on predictions and constraints.
        predicted_demand_3m: total predicted demand over next 90 days.
        """
        # Daily demand variability
        safety_stock = self.calculate_safety_stock(historical_std_dev, lead_time_days)
        
        # Total needed for the period (lead time + buffer)
        # We usually want enough stock to cover lead time plus some cycle stock
        # For simplicity, let's look at next 30 days demand for immediate reordering
        predicted_demand_30d = predicted_demand_3m / 3
        
        reorder_point = (predicted_demand_30d / 30 * lead_time_days) + safety_stock
        
        reorder_qty = 0
        status = "Optimal"
        
        if current_stock <= reorder_point:
            reorder_qty = (predicted_demand_30d + safety_stock) - current_stock
            status = "Reorder Now"
        elif current_stock > (predicted_demand_30d * 2):
            status = "Overstocked"
        
        # Apply Constraints
        # 1. Storage Capacity
        if current_stock + reorder_qty > storage_capacity:
            reorder_qty = max(0, storage_capacity - current_stock)
            status = "Constraint: Storage Limit"
            
        # 2. Budget
        if reorder_qty * unit_cost > budget:
            reorder_qty = budget / unit_cost
            status = "Constraint: Budget Limit"
            
        return {
            'reorder_qty': round(reorder_qty, 2),
            'safety_stock': round(safety_stock, 2),
            'reorder_point': round(reorder_point, 2),
            'status': status
        }
