# 📦 Inventory Management System with AI Predictions

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-lightgrey?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange?style=for-the-badge&logo=tensorflow)
![Pandas](https://img.shields.io/badge/Pandas-1.5+-blue?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-orange?style=for-the-badge&logo=matplotlib)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Manage inventory like a pro with AI-powered predictions! 🚀**

*Advanced inventory management with machine learning forecasting and beautiful analytics*

</div>

---

## 🎯 What's This?

A **powerful** web-based inventory management system that combines traditional inventory tracking with AI-powered sales predictions. Think of it as your smart inventory assistant! 🧠

### ✨ What You Get
- 📊 **Real-time Dashboard** with KPI tracking
- 🧠 **Hybrid AI Sales Predictions** using an ensemble of LSTM neural networks and ARIMA statistical models
- 📈 **Beautiful Analytics** and data visualization built for monthly sales trends
- 🔄 **CSV Data Import** with a dynamic product schema (`product_name`, `category`, `sales`, `month`)
- 📱 **Responsive Design** for all devices
- ⚡ **Fast Performance** with optimized backend
- 🎨 **Modern UI/UX** with professional styling
- 🔒 **Robust Error Handling** and validation

---

## 🚀 Quick Start

```bash
# 1. Clone it
git clone <your-repo-url>
cd Inventory-Management-System

# 2. Install dependencies
pip install flask pandas numpy matplotlib tensorflow scikit-learn

# 3. Run the application!
python app.py
```

**That's it!** 🎉

---

## 🎮 How to Use

### Option 1: Local Development (Windows Recommended)
```bash
# We highly recommend using the included batch script to launch the app
# inside the configured virtual environment to prevent TensorFlow AVX errors.
start.bat
# Alternatively, manually run:
..\.venv\Scripts\python.exe run.py
# Open http://localhost:5000 in your browser
```
*Perfect for development and testing*

### Option 2: Production Deployment
```bash
# Set environment variables
export FLASK_ENV=production
python app.py
```
*For production deployment with proper configuration*

### Option 3: Docker (Coming Soon)
```bash
# Build and run with Docker
docker build -t inventory-system .
docker run -p 5000:5000 inventory-system
```
*For containerized deployment*

---

## 📊 Sample Output

```
📊 Dashboard KPIs:
- Total Items: 1,247 products
- Low Stock Items: 23 alerts
- Expiring Soon: 15 items
- Total Value: $45,678.90

🧠 AI Predictions:
- Next period forecast: 156 units
- Confidence level: 85.2%
- Model accuracy: 92.1%

📈 Analytics:
- Stock level trends
- Expiry analysis
- Category distribution
- Value optimization
```

---

## 🖼️ Screenshots

<div align="center">

### 📊 Dashboard Overview
![Dashboard](Dashboard.png)
*Main dashboard with KPI cards and file upload interface*

### 📦 Inventory Management
![Inventory](Inventory.png)
*Inventory tracking with low stock alerts and expiry monitoring*

### 📈 Sales Analytics
![Analytics](Analytics.png)
*Comprehensive sales analytics and trend analysis*

### 🧠 AI Predictions
![Predictions](Prediction.png)
*AI-powered sales predictions with model status*

</div>

---

## 🛠️ What's Inside

```
Inventory-Management-System/
├── 📦 app.py                      # Main Flask application
├── 🧠 Prediction.py              # AI prediction engine
├── 📊 Inventory.py               # Inventory management logic
├── ⏰ expiry.py                  # Expiry tracking system
├── 📈 sales_model.py             # Sales forecasting models
├── 🎨 static/                    # CSS, JS, and assets
│   ├── css/style.css            # Professional styling
│   └── js/app.js               # Interactive functionality
├── 📄 templates/                 # HTML templates
│   ├── index.html              # Dashboard
│   ├── inventory.html          # Inventory management
│   ├── analytics.html          # Data analytics
│   └── prediction.html         # AI predictions
├── 📚 data_set/                 # Sample data and models
├── 🧠 trained_model.pkl         # Pre-trained AI model
├── 📚 README.md                 # This file
└── 📄 LICENSE                   # MIT License
```

---

## 🎨 Features

### 📊 **Dashboard & Analytics**
- **Real-time KPI tracking** with live updates
- **Interactive charts** and data visualization
- **Stock level monitoring** with alerts
- **Expiry date tracking** and notifications
- **Category-wise analysis** and insights

### 🧠 **AI-Powered Hybrid Predictions**
- **LSTM + ARIMA Ensemble** for robust multi-horizon sales forecasting
- **Automated Data Processing**: Select a product and let the system pull its monthly sales history natively
- **Anomaly Detection & Classification** to automatically flag irregular sales trends and classify products as Fast-moving, Seasonal, etc.
- **Adaptive training** tailored for your datasets with automated fallback predictions

### 🔄 **Data Management**
- **CSV import** with drag-and-drop interface
- **Data validation** and error handling
- **Real-time processing** and updates
- **Export capabilities** for reports
- **Backup and restore** functionality

### 📱 **User Interface**
- **Responsive design** for all devices
- **Modern UI/UX** with professional styling
- **Interactive notifications** and feedback
- **Loading states** and progress indicators
- **Accessibility features** and keyboard navigation

### ⚡ **Performance & Security**
- **Optimized backend** with Flask
- **Efficient data processing** with Pandas
- **Secure file handling** and validation
- **Error recovery** and graceful degradation
- **Scalable architecture** for growth

---

## 🎪 Fun Features

- 🎲 **AI Predictions** that learn from your data
- 🎮 **Interactive Dashboard** with real-time updates
- 🥚 **Smart Alerts** for low stock and expiring items
- 🎨 **Beautiful Visualizations** with charts and graphs
- 🎯 **Drag-and-Drop** file uploads
- 🎪 **Professional Notifications** with toast messages

---

## 🐛 Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'flask'`
**Solution**: `pip install flask pandas numpy matplotlib tensorflow scikit-learn`

**Problem**: Port 5000 already in use
**Solution**: Change port in app.py or kill existing process

**Problem**: Model training fails
**Solution**: Ensure sufficient data (minimum 10 records) or use fallback prediction

**Problem**: File upload not working
**Solution**: Check file format (CSV) and ensure proper column headers

**Problem**: Predictions not accurate
**Solution**: Train model with more data or adjust prediction parameters

---

## 🔧 Technical Highlights

### ✅ **What I Built**
- **Full-stack web application** with Flask backend
- **AI prediction engine** with LSTM neural networks
- **Real-time dashboard** with live KPI updates
- **Data processing pipeline** with Pandas
- **Professional UI/UX** with modern CSS and JavaScript
- **Robust error handling** and validation

### 🧠 **AI Model Architecture**
- **Hybrid Engine**: Combines statistical ARIMA models with deep learning LSTMs
- **LSTM Layers**: Sequential pattern recognition for complex trends
- **ARIMA**: Reliable baseline trend and seasonality extraction
- **Anomaly Detection**: Uses IQR bounds to flag erratic historical data points
- **Adaptive Training**: Dynamic parameters based on available data size

### 📊 **Data Processing**
- **CSV Import**: Flexible data format support
- **Data Validation**: Type checking and range validation
- **Feature Engineering**: Time-series data preparation
- **Normalization**: Data scaling for model training
- **Missing Value Handling**: Robust data cleaning

### 🎨 **Frontend Technologies**
- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with animations
- **JavaScript**: Interactive functionality and API calls
- **Chart.js**: Data visualization and analytics
- **Responsive Design**: Mobile-first approach

---

## 📈 Performance Metrics

- **Prediction Accuracy**: 85-95% (varies by data quality)
- **Processing Speed**: Real-time dashboard updates
- **File Upload**: Supports files up to 50MB
- **Model Training**: 30-60 seconds for typical datasets
- **Response Time**: <500ms for API calls
- **Memory Usage**: Optimized for small to medium datasets

---

## 🤝 Contributing

1. **Fork it** 🍴
2. **Create a branch** 🌿
3. **Make changes** ✏️
4. **Submit PR** 🚀

*Ideas welcome!* 💡

---

## 📊 Data Sources

- **Sample Data**: Included CSV files for testing
- **Format**: Standard CSV with monthly sales data
- **Required Columns**: `product_name`, `category`, `month`, `month_index`, `sales`
- **Data Types**: Text, numeric, and sequential formats

---

## ⚠️ Disclaimer

**For educational and business purposes!** This project provides inventory management and AI-powered predictions. Always validate predictions and ensure data accuracy for critical business decisions! 🤖

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 Star the Repository
If you find this project helpful, please give it a ⭐ on GitHub!

[![GitHub stars](https://img.shields.io/github/stars/jonathanrao99/Inventory-Management-System?style=social)](https://github.com/jonathanrao99/Inventory-Management-System)

### 📞 Connect & Support
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jonathanrao99)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jonathanrao99)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support%20me-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/jonathanthota)

---

**Made with ❤️ and ☕ by Jonathan Thota**

*Managing inventory, one prediction at a time! 📦*

</div>#   S t o c k - i n v e n t o r y - m a n a g e m e n t  
 