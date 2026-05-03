import pandas as pd
import os
from datetime import datetime, timedelta

def parse_inventory_dates(series):
    """Parse inventory date columns from uploaded CSV files using common formats."""
    return pd.to_datetime(series, format='%d-%m-%Y', errors='coerce')

def load_inventory_data(data_path='data_set/data.csv'):
    """Load inventory data from CSV file"""
    try:
        print(f"Loading inventory data from: {data_path}")
        if not os.path.exists(data_path):
            print(f"Data file not found: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows of data")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading inventory data: {str(e)}")
        return None

def get_low_stock_products(df, threshold=300):
    """Get products with low stock levels"""
    # Stock levels are not tracked in the new sales schema, returning empty
    return []

def get_near_expiry_products(df, days_threshold=7):
    """Get products nearing expiry"""
    # Expiry is not tracked in the new sales schema, returning empty
    return []

def calculate_inventory_metrics(df):
    """Calculate comprehensive inventory metrics based on new sales schema"""
    try:
        df = df.copy()
        print(f"Calculating metrics for {len(df)} records")
        metrics = {}
        
        # Basic counts
        metrics['total_products'] = int(df['product_name'].nunique()) if 'product_name' in df.columns else 0
        metrics['low_stock_count'] = 0
        
        # Stock levels (N/A)
        metrics['average_stock_level'] = 0.0
        metrics['total_stock_value'] = 0.0
        
        # Expiry analysis (N/A)
        metrics['near_expiry_count'] = 0
        
        # Sales metrics
        metrics['total_revenue'] = float(df['sales'].sum()) if 'sales' in df.columns else 0.0
        metrics['average_order_value'] = float(df['sales'].mean()) if 'sales' in df.columns else 0.0
        
        print(f"Calculated metrics: {metrics}")
        return metrics
    except Exception as e:
        print(f"Error calculating inventory metrics: {str(e)}")
        return {
            'total_products': 0,
            'low_stock_count': 0,
            'average_stock_level': 0.0,
            'total_stock_value': 0.0,
            'near_expiry_count': 0,
            'total_revenue': 0.0,
            'average_order_value': 0.0
        }

def generate_inventory_report(data_path='data_set/data.csv'):
    """Generate comprehensive inventory report"""
    try:
        df = load_inventory_data(data_path)
        if df is None:
            return None
        
        metrics = calculate_inventory_metrics(df)
        low_stock = get_low_stock_products(df)
        near_expiry = get_near_expiry_products(df)
        
        report = {
            'metrics': metrics,
            'low_stock_products': low_stock,
            'near_expiry_products': near_expiry,
            'generated_at': datetime.now().isoformat(),
            'total_products_analyzed': len(df)
        }
        
        return report
    except Exception as e:
        print(f"Error generating inventory report: {str(e)}")
        return None

def validate_csv_data(df):
    """Validate CSV data structure and content"""
    try:
        required_columns = [
            'product_name', 'category', 'month', 'month_index', 'sales'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {missing_columns}"
        
        # Check for empty values in critical columns
        if df['product_name'].isnull().any():
            return False, "Product name cannot be null"
        
        # Check for negative sales
        if (df['sales'] < 0).any():
            return False, "Sales cannot be negative"
        
        return True, "Data validation passed"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_stock_alerts(df):
    """Get stock alerts and recommendations"""
    # Not applicable for the new schema
    return []

def format_currency(amount):
    """Format amount as currency"""
    try:
        return f"${amount:,.2f}"
    except:
        return str(amount)

def format_number(number):
    """Format number with commas"""
    try:
        return f"{number:,.0f}"
    except:
        return str(number)
