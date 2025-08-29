"""
Utility functions for ClimateScope Dashboard
Additional analysis and data processing functions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def calculate_weather_severity_index(df):
    """
    Calculate a weather severity index based on multiple factors
    """
    # Normalize factors to 0-1 scale
    temp_extreme = np.abs(df['temperature_celsius'] - df['temperature_celsius'].mean()) / df['temperature_celsius'].std()
    wind_factor = df['wind_kph'] / df['wind_kph'].max()
    humidity_extreme = np.abs(df['humidity'] - 50) / 50  # 50% is considered optimal
    
    # Combine factors (higher = more severe weather)
    severity_index = (temp_extreme * 0.4 + wind_factor * 0.3 + humidity_extreme * 0.3)
    return severity_index

def identify_climate_anomalies(df, threshold=2):
    """
    Identify locations with anomalous weather patterns
    """
    anomalies = []
    
    # Temperature anomalies
    temp_z_scores = np.abs((df['temperature_celsius'] - df['temperature_celsius'].mean()) / df['temperature_celsius'].std())
    temp_anomalies = df[temp_z_scores > threshold]
    
    for _, row in temp_anomalies.iterrows():
        anomalies.append({
            'location': f"{row['location_name']}, {row['country']}",
            'type': 'Temperature',
            'value': row['temperature_celsius'],
            'severity': 'High' if temp_z_scores[row.name] > 3 else 'Moderate'
        })
    
    # Humidity anomalies
    humidity_z_scores = np.abs((df['humidity'] - df['humidity'].mean()) / df['humidity'].std())
    humidity_anomalies = df[humidity_z_scores > threshold]
    
    for _, row in humidity_anomalies.iterrows():
        anomalies.append({
            'location': f"{row['location_name']}, {row['country']}",
            'type': 'Humidity',
            'value': row['humidity'],
            'severity': 'High' if humidity_z_scores[row.name] > 3 else 'Moderate'
        })
    
    return pd.DataFrame(anomalies)

def generate_weather_recommendations(df):
    """
    Generate weather-based recommendations for different activities
    """
    recommendations = {
        'outdoor_activities': [],
        'travel_warnings': [],
        'health_alerts': []
    }
    
    # Outdoor activities (good weather conditions)
    good_weather = df[
        (df['temperature_celsius'].between(15, 28)) &
        (df['humidity'] < 70) &
        (df['wind_kph'] < 20) &
        (df['condition_text'].str.contains('Clear|Sunny|Partly', case=False, na=False))
    ]
    
    for _, row in good_weather.head(5).iterrows():
        recommendations['outdoor_activities'].append(
            f"🌞 {row['location_name']}, {row['country']}: Perfect for outdoor activities "
            f"({row['temperature_celsius']:.1f}°C, {row['condition_text']})"
        )
    
    # Travel warnings (extreme conditions)
    extreme_weather = df[
        (df['temperature_celsius'] > 40) |
        (df['temperature_celsius'] < -10) |
        (df['wind_kph'] > 50) |
        (df['condition_text'].str.contains('Storm|Hurricane|Blizzard', case=False, na=False))
    ]
    
    for _, row in extreme_weather.head(5).iterrows():
        recommendations['travel_warnings'].append(
            f"⚠️ {row['location_name']}, {row['country']}: Extreme conditions "
            f"({row['temperature_celsius']:.1f}°C, {row['condition_text']})"
        )
    
    # Health alerts (air quality or extreme conditions)
    if 'air_quality_PM2.5' in df.columns:
        unhealthy_air = df[df['air_quality_PM2.5'] > 35]  # WHO guideline
        
        for _, row in unhealthy_air.head(3).iterrows():
            recommendations['health_alerts'].append(
                f"🏥 {row['location_name']}, {row['country']}: Poor air quality "
                f"(PM2.5: {row['air_quality_PM2.5']:.1f} μg/m³)"
            )
    
    return recommendations

def create_climate_comparison_chart(df, locations):
    """
    Create a detailed comparison chart for specific locations
    """
    if not locations:
        return None
    
    comparison_data = df[df['location_name'].isin(locations)]
    
    if comparison_data.empty:
        return None
    
    # Create subplot with multiple metrics
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature', 'Humidity', 'Wind Speed', 'Pressure'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1[:len(locations)]
    
    for i, location in enumerate(locations):
        loc_data = comparison_data[comparison_data['location_name'] == location].iloc[0]
        color = colors[i % len(colors)]
        
        # Temperature
        fig.add_trace(
            go.Bar(x=[location], y=[loc_data['temperature_celsius']], 
                   name=f'{location} Temp', marker_color=color, showlegend=False),
            row=1, col=1
        )
        
        # Humidity
        fig.add_trace(
            go.Bar(x=[location], y=[loc_data['humidity']], 
                   name=f'{location} Humidity', marker_color=color, showlegend=False),
            row=1, col=2
        )
        
        # Wind Speed
        fig.add_trace(
            go.Bar(x=[location], y=[loc_data['wind_kph']], 
                   name=f'{location} Wind', marker_color=color, showlegend=False),
            row=2, col=1
        )
        
        # Pressure
        fig.add_trace(
            go.Bar(x=[location], y=[loc_data['pressure_mb']], 
                   name=f'{location} Pressure', marker_color=color, showlegend=False),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Location Weather Comparison",
        height=600,
        showlegend=False
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Humidity (%)", row=1, col=2)
    fig.update_yaxes(title_text="Wind Speed (km/h)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (mb)", row=2, col=2)
    
    return fig

def calculate_comfort_index(df):
    """
    Calculate a comfort index based on temperature, humidity, and wind
    Heat Index approximation for comfort assessment
    """
    temp_f = df['temperature_fahrenheit']
    humidity = df['humidity']
    
    # Simplified heat index calculation
    heat_index = (
        -42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
        - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f**2
        - 5.481717e-2 * humidity**2 + 1.22874e-3 * temp_f**2 * humidity
        + 8.5282e-4 * temp_f * humidity**2 - 1.99e-6 * temp_f**2 * humidity**2
    )
    
    # Convert back to Celsius and normalize to 0-100 scale
    heat_index_c = (heat_index - 32) * 5/9
    
    # Create comfort categories
    comfort_conditions = []
    for hi in heat_index_c:
        if hi < 27:
            comfort_conditions.append("Comfortable")
        elif hi < 32:
            comfort_conditions.append("Caution")
        elif hi < 41:
            comfort_conditions.append("Extreme Caution")
        elif hi < 54:
            comfort_conditions.append("Danger")
        else:
            comfort_conditions.append("Extreme Danger")
    
    return comfort_conditions

def export_dashboard_data(df, filters_applied):
    """
    Export filtered data for further analysis
    """
    export_data = {
        'summary_stats': df.describe(),
        'filter_info': filters_applied,
        'export_timestamp': datetime.now().isoformat(),
        'total_locations': len(df),
        'countries_covered': df['country'].nunique(),
        'continents_covered': df['continent'].nunique() if 'continent' in df.columns else 0
    }
    
    return export_data

def create_weather_timeline(df, location_name):
    """
    Create a timeline view for a specific location (if historical data available)
    """
    location_data = df[df['location_name'] == location_name]
    
    if location_data.empty:
        return None
    
    # Since we have snapshot data, create a mock timeline for demonstration
    # In real implementation, this would use historical data
    
    fig = go.Figure()
    
    # Mock historical data (in real scenario, this would come from time-series data)
    dates = pd.date_range(start='2024-05-01', end='2024-05-16', freq='D')
    base_temp = location_data['temperature_celsius'].iloc[0]
    
    # Add some realistic variation
    temps = [base_temp + np.random.normal(0, 3) for _ in range(len(dates))]
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=temps,
        mode='lines+markers',
        name='Temperature',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f'Temperature Timeline for {location_name}',
        xaxis_title='Date',
        yaxis_title='Temperature (°C)',
        height=400,
        showlegend=True
    )
    
    return fig
