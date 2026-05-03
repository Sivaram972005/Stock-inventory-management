from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Keep Flask requests from touching a GUI backend during chart generation/training.
import matplotlib.pyplot as plt
import os
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from utils import generate_inventory_report, get_low_stock_products, get_near_expiry_products, parse_inventory_dates
from ai_inventory_optimizer import (
    DataManager, Forecaster, ProductClassifier, 
    InventoryOptimizer, AnomalyDetector, Visualizer
)

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
app.secret_key = 'inventory-management-system-secret-key'

# Configuration
app.config['UPLOAD_FOLDER'] = 'data_set'
app.config['MODEL_PATH'] = 'trained_model.pkl'
app.config['DATA_PATH'] = 'data_set/data.csv'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

PUBLIC_ENDPOINTS = {'login', 'logout', 'static'}

@app.before_request
def require_login():
    """Keep the existing site behind a lightweight login gate."""
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None

    if not session.get('is_authenticated'):
        return redirect(url_for('login'))

    return None

# Load the pickled model
def load_trained_model():
    """Load the trained model with proper error handling"""
    try:
        import os
        if not os.path.exists(app.config['MODEL_PATH']):
            return None
        with open(app.config['MODEL_PATH'], 'rb') as model_file:
            try:
                model_data = pickle.load(model_file)
            except Exception as e:
                print(f"Error unpickling model (possibly TF missing): {e}")
                return None
        
        # Check if the loaded data is a model or a dictionary containing a model
        if hasattr(model_data, 'predict'):
            print("Model loaded successfully!")
            return model_data
        elif isinstance(model_data, dict) and 'model' in model_data:
            print("Model loaded successfully from dictionary!")
            return model_data['model']
        else:
            print("Loaded data is not a valid model")
            return None
    except FileNotFoundError:
        print(f"Model file not found at {app.config['MODEL_PATH']}")
        return None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def simple_prediction(quantity1, quantity2, quantity3):
    """Simple prediction using weighted average"""
    try:
        # Simple weighted average prediction
        weights = [0.2, 0.3, 0.5]  # Give more weight to recent data
        prediction = (quantity1 * weights[0] + quantity2 * weights[1] + quantity3 * weights[2])
        return prediction
    except Exception as e:
        print(f"Simple prediction error: {str(e)}")
        return None

model = load_trained_model()

def predict_next_period(quantity1, quantity2, quantity3):
    """Predict the next period using a simple linear trend on the original input scale."""
    try:
        # Treat the three form inputs as three consecutive periods.
        periods = np.array([1, 2, 3]).reshape(-1, 1)
        quantities = np.array([quantity1, quantity2, quantity3], dtype=float)

        # Use a simple regression line so the prediction follows the visible trend.
        regression_model = LinearRegression()
        regression_model.fit(periods, quantities)

        # Predict period 4 directly in the same scale as the user inputs.
        prediction = float(regression_model.predict([[4]])[0])

        # Use R^2 as a lightweight confidence proxy instead of a hardcoded UI value.
        confidence = float(max(0.0, min(1.0, regression_model.score(periods, quantities))))

        return {
            "prediction": prediction,
            "confidence": confidence,
            "method": "linear_regression_trend"
        }
    except Exception as e:
        print(f"Trend prediction error: {str(e)}")
        return None

def build_sales_chart_data(dataframe):
    """Build chart labels/data directly from the uploaded CSV."""
    chart_df = dataframe.copy()

    if 'month' in chart_df.columns:
        monthly_sales = chart_df.groupby(['month_index', 'month'])['sales'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values('month_index')
        labels = monthly_sales['month'].tolist()
        values = monthly_sales['sales'].astype(float).tolist()
    else:
        chart_df = chart_df.sort_values('product_name')
        labels = chart_df['product_name'].astype(str).tolist()
        values = chart_df['sales'].astype(float).tolist()

    return {
        'labels': labels,
        'values': values
    }



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Simple login page for staff access."""
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        staff_id = request.form.get('staff_id', '').strip()
        password = request.form.get('password', '').strip()

        # Fast login flow: require all three fields before opening the dashboard.
        if not name or not staff_id or not password:
            return render_template(
                'login.html',
                error='Please enter name, staff ID, and password.'
            )

        session['is_authenticated'] = True
        session['staff_name'] = name
        session['staff_id'] = staff_id
        return redirect(url_for('home'))

    if session.get('is_authenticated'):
        return redirect(url_for('home'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Clear the session and return to the login page."""
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with improved error handling"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file part"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No selected file"
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                "success": False,
                "error": "Please upload a CSV file"
            }), 400
        
        # Save the file to the data directory
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
        file.save(file_path)
        
        return jsonify({
            "success": True,
            "message": "File uploaded successfully!"
        }), 200
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error saving file: {str(e)}"
        }), 500

@app.route('/inventory')
def inventory():
    """Display inventory with restocking and expiry recommendations"""
    try:
        # Check if data file exists
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', 
                                 error="Data file not found. Please upload a CSV file first.")
        
        # Read data from CSV file
        df = pd.read_csv(app.config['DATA_PATH'])
        
        # Get recommendations for restocking and near expiry products
        low_stock_recommendations = get_low_stock_products(df)
        near_expiry_recommendations = get_near_expiry_products(df)
        
        # Get inventory metrics
        from utils import calculate_inventory_metrics
        metrics = calculate_inventory_metrics(df)
        
        return render_template('inventory.html', 
                             restock_recommendations=low_stock_recommendations,
                             near_expiry_recommendations=near_expiry_recommendations,
                             metrics=metrics)
    except Exception as e:
        return render_template('error.html', error=f"Error loading inventory: {str(e)}")

@app.route('/api/products')
def get_products():
    """Returns a list of products from the inventory."""
    try:
        if not os.path.exists(app.config['DATA_PATH']):
            return jsonify([])
        df = pd.read_csv(app.config['DATA_PATH'])
        products = df[['product_name', 'category']].drop_duplicates().to_dict('records')
        return jsonify(products)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/product-data/<product_name>')
def get_product_data(product_name):
    """Returns historical sales data for a specific product."""
    try:
        dm = DataManager(app.config['DATA_PATH'])
        # Try to load real data
        df = dm.load_data()
        if df is not None and 'sales' in df.columns:
            # Group by product
            prod_df = df[df['product_name'] == product_name]
            return jsonify(prod_df.to_dict('records'))
        return jsonify([])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=["GET", "POST"])
def predict():
    """Handle prediction requests with improved error handling"""
    if request.method == "POST":
        try:
            # Extract input data from the request
            data = request.get_json()
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400
            
            product_name = data.get('product_name')
            
            # --- ADVANCED FORECASTING PATH ---
            if product_name:
                dm = DataManager(app.config['DATA_PATH'])
                sales_df = dm.load_data()
                
                df_prod = dm.preprocess_for_forecasting(product_name)
                if df_prod is None or len(df_prod) < 3:
                    return jsonify({"success": False, "error": "Not enough historical data for this product."}), 400
                
                # Hybrid Forecast
                forecaster = Forecaster()
                forecast_results = forecaster.forecast_hybrid(df_prod, forecast_periods=3)
                
                # Classification
                classifier = ProductClassifier()
                classification_df = classifier.classify_products(sales_df)
                prod_class = classification_df[classification_df['product_name'] == product_name]
                class_label = prod_class['classification'].values[0] if not prod_class.empty else "Unknown"
                
                # Optimization
                optimizer = InventoryOptimizer()
                current_stock = 0 # Dummy value since current stock is not provided
                
                opt_results = optimizer.optimize(
                    current_stock=current_stock,
                    predicted_demand_3m=np.sum(forecast_results['forecast']),
                    historical_std_dev=df_prod['sales'].std()
                )
                
                # Anomaly Detection
                detector = AnomalyDetector()
                anomalies = detector.flag_recent_anomalies(df_prod)
                
                # Visualization
                visualizer = Visualizer()
                plot_path = visualizer.plot_demand_forecast(df_prod, forecast_results, product_name)
                
                return jsonify({
                    "success": True,
                    "advanced": True,
                    "product_name": product_name,
                    "classification": class_label,
                    "forecast": forecast_results['forecast'].tolist()[:3], # Next 3 periods (approx months if daily data is aggregated)
                    "full_forecast": forecast_results['forecast'].tolist(),
                    "trend": forecast_results['trend'],
                    "confidence": 0.92, # Placeholder or calculated from metrics
                    "recommendations": opt_results,
                    "anomalies": anomalies,
                    "plot_url": f"static/forecast_{product_name.replace(' ', '_')}.png"
                })

            # --- MANUAL INPUT PATH (BACKWARD COMPATIBILITY) ---
            quantity1 = float(data.get('quantity1', 0))
            quantity2 = float(data.get('quantity2', 0))
            quantity3 = float(data.get('quantity3', 0))

            # Validate input data
            if quantity1 < 0 or quantity2 < 0 or quantity3 < 0:
                return jsonify({
                    "success": False,
                    "error": "Quantities must be non-negative"
                }), 400

            # IMPORTANT:
            # The previously saved LSTM model was trained on normalized sequences, but this
            # endpoint was sending raw values like [100, 110, 150] directly into the model
            # and returning the raw output without inverse-scaling it. That is why values
            # such as 2.78 appeared even though the input scale was ~100-150.
            #
            # For the 3-input form we use a simple linear trend forecast instead. This keeps
            # the prediction on the same scale as the user inputs and avoids shape/scaling bugs.
            trend_result = predict_next_period(quantity1, quantity2, quantity3)
            if trend_result is not None:
                return jsonify({
                    "success": True,
                    "prediction": float(trend_result["prediction"]),
                    "confidence": float(trend_result["confidence"]),
                    "method": trend_result["method"]
                })

            # Fallback to simple prediction
            prediction_value = simple_prediction(quantity1, quantity2, quantity3)
            if prediction_value is not None:
                return jsonify({
                    "success": True,
                    "prediction": float(prediction_value),
                    "confidence": None,
                    "method": "simple_weighted_average"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to make prediction"
                }), 500

        except ValueError as e:
            return jsonify({
                "success": False,
                "error": f"Invalid input data: {str(e)}"
            }), 400
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Failed to make prediction: {str(e)}"
            }), 500

    elif request.method == "GET":
        products = []
        try:
            if os.path.exists(app.config['DATA_PATH']):
                df = pd.read_csv(app.config['DATA_PATH'])
                products = df[['product_name', 'category']].drop_duplicates().to_dict('records')
        except Exception as e:
            print(f"Prediction page data error: {str(e)}")

        return render_template("prediction.html", 
                             products=products)

@app.route('/analytics')
def sales_analytics():
    """Display sales analytics with improved error handling"""
    try:
        # Check if data file exists
        if not os.path.exists(app.config['DATA_PATH']):
            return render_template('error.html', 
                                 error="Data file not found. Please upload a CSV file first.")
        
        # Load data from CSV file
        data = pd.read_csv(app.config['DATA_PATH'])
        
        # Calculate total sales and average order value
        total_sales = float(data["sales"].sum())
        average_order_value = float(data["sales"].mean())

        # Group by product_name to get top selling
        product_sales = data.groupby('product_name')['sales'].sum().reset_index()
        top_selling_products = product_sales.nlargest(5, "sales")
        bottom_selling_products = product_sales.nsmallest(5, "sales")

        # Convert DataFrames to dictionaries and ensure JSON serializable
        top_selling_dict = []
        for _, row in top_selling_products.iterrows():
            top_selling_dict.append({
                'product_name': str(row['product_name']),
                'total_revenue': float(row['sales']) # keeping the name total_revenue for UI compatibility
            })

        bottom_selling_dict = []
        for _, row in bottom_selling_products.iterrows():
            bottom_selling_dict.append({
                'product_name': str(row['product_name']),
                'total_revenue': float(row['sales'])
            })

        # Create sales trend plot
        try:
            monthly_sales = data.groupby(['month_index', 'month'])['sales'].sum().reset_index()
            monthly_sales = monthly_sales.sort_values('month_index')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(monthly_sales['month'], monthly_sales['sales'], marker='o', linestyle='-', linewidth=2, markersize=4)
            ax.set_title('Overall Sales Trend by Month', fontsize=16, fontweight='bold')
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Total Sales', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('white')
            fig.tight_layout()

            # Save the plot to a static file
            sales_trend_file_path = "static/sales_trend.png"
            fig.savefig(sales_trend_file_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
        except Exception as e:
            print(f"Error creating sales trend plot: {e}")

        chart_data = build_sales_chart_data(data)

        return render_template('analytics.html', 
                             total_sales=total_sales,
                             average_order_value=average_order_value,
                             top_selling_products=top_selling_dict,
                             bottom_selling_products=bottom_selling_dict,
                             chart_labels=chart_data['labels'],
                             chart_values=chart_data['values'])
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        return render_template('error.html', error=f"Error loading analytics: {str(e)}")

@app.route('/train', methods=['POST'])
def train_model():
    """Train the prediction model"""
    try:
        from Prediction import main as train_prediction_model
        print("Starting model training process...")
        success = train_prediction_model()
        
        if success:
            # Reload the model
            global model
            model = load_trained_model()
            if model is not None:
                return jsonify({
                    "success": True,
                    "message": "Model trained successfully!"
                }), 200
            else:
                return jsonify({
                    "success": False,
                    "error": "Model training completed but failed to load the model."
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": "Model training failed. Check the logs for details."
            }), 500
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error training model: {str(e)}"
        }), 500

@app.route('/api/inventory-summary')
def inventory_summary():
    """API endpoint for inventory summary"""
    try:
        print(f"Checking for data file at: {app.config['DATA_PATH']}")
        
        if not os.path.exists(app.config['DATA_PATH']):
            print("Data file not found, returning default metrics")
            return jsonify({
                "metrics": {
                    "total_products": 0,
                    "low_stock_count": 0,
                    "average_stock_level": 0,
                    "total_stock_value": 0,
                    "near_expiry_count": 0,
                    "total_revenue": 0,
                    "average_order_value": 0
                },
                "message": "No data file found. Please upload a CSV file first."
            }), 200
        
        print("Data file found, generating report...")
        report = generate_inventory_report(app.config['DATA_PATH'])
        
        if report:
            print(f"Report generated successfully: {report}")
            return jsonify(report)
        else:
            print("Failed to generate report")
            return jsonify({
                "metrics": {
                    "total_products": 0,
                    "low_stock_count": 0,
                    "average_stock_level": 0,
                    "total_stock_value": 0,
                    "near_expiry_count": 0,
                    "total_revenue": 0,
                    "average_order_value": 0
                },
                "error": "Failed to generate report"
            }), 200
    except Exception as e:
        print(f"Error in inventory summary: {str(e)}")
        return jsonify({
            "metrics": {
                "total_products": 0,
                "low_stock_count": 0,
                "average_stock_level": 0,
                "total_stock_value": 0,
                "near_expiry_count": 0,
                "total_revenue": 0,
                "average_order_value": 0
            },
            "error": str(e)
        }), 200

@app.route('/api/model-status')
def model_status():
    """Return real model/data status for the prediction page."""
    try:
        data_exists = os.path.exists(app.config['DATA_PATH'])
        model_exists = os.path.exists(app.config['MODEL_PATH'])

        row_count = 0
        if data_exists:
            try:
                row_count = int(len(pd.read_csv(app.config['DATA_PATH'])))
            except Exception:
                row_count = 0

        return jsonify({
            "success": True,
            "model_ready": bool(model is not None and model_exists),
            "model_file_exists": model_exists,
            "data_file_exists": data_exists,
            "row_count": row_count
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "model_ready": False,
            "model_file_exists": False,
            "data_file_exists": False,
            "row_count": 0
        }), 200

if __name__ == '__main__':
    # Disable the auto-reloader because TensorFlow/matplotlib teardown is unstable under
    # Flask's debug child-process restart flow on Windows. This keeps /train reliable.
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    
