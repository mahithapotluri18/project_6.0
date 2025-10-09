# 🌍 ClimateScope - Global Weather Analytics Platform

Interactive climate data analysis platform with automated data processing and real-time visualization dashboard.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
python climatescope_dashboard.py
# Access at: http://127.0.0.1:8062

# Or run notebook analysis
jupyter notebook ClimateScope.ipynb
```

## 📊 Features

### 🎯 **Interactive Dashboard**
- **Global Weather Map**: 191 countries with choropleth visualization
- **Regional Analytics**: 7 geographic regions with 97K+ weather records
- **Advanced Visualizations**: Time series, correlations, seasonal patterns
- **Air Quality Index**: Composite environmental health indicators
- **Dark/Light Themes**: Complete theme switching with adaptive charts
- **Smart Insights**: 4-tab system (Statistics, Regional, Top Performers, Trends)

### 🔬 **Jupyter Notebook Analysis**
- **Milestone 1**: Automated Kaggle data pipeline (6-hour refresh)
- **Milestone 2**: Statistical analysis and correlation discovery
- **Milestone 3**: Geographic region enhancement and data quality

### 🌡️ **Climate Metrics**
Temperature, Humidity, Wind Speed, Pressure, UV Index, Air Quality (PM2.5, PM10, CO, O3, NO2, SO2), Visibility

## 📁 Project Structure

```
project_6.0/
├── climatescope_dashboard.py    # Main dashboard app
├── ClimateScope.ipynb          # Analysis notebook
├── requirements.txt            # Dependencies
└── data/
    ├── raw/enhanced_weather_with_regions.csv  # Main dataset (97,824 records)
    └── clean/                  # Processed data by year/month
```

## 🔧 Requirements

- Python 3.8+
- 4GB+ RAM recommended
- Modern browser with JavaScript
- Optional: Kaggle credentials for auto-refresh

## 📈 Dataset

**97,824 weather observations** from **191 countries** across **7 regions** with multi-year historical data and automated updates from Kaggle's Global Weather Repository.
