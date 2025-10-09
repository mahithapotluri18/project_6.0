#!/usr/bin/env python3
"""
ClimateScope - Global Weather Patterns Visualization Dashboard

A comprehensive Plotly Dash application for visualizing global weather patterns
and extreme events with interactive controls.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
from datetime import datetime, date
import json

# Performance optimization: Set pandas options
pd.options.mode.chained_assignment = None

# Initialize the Dash app with Bootstrap theme and FontAwesome
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
])
app.title = "ClimateScope - Global Weather Analytics"

# Configure callback timeout for performance
app.config.suppress_callback_exceptions = True

# Add custom CSS for dropdown theming
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Light theme dropdown styles */
            .light-theme .Select-control {
                background-color: white !important;
                color: black !important;
                border-color: #ccc !important;
            }
            
            .light-theme .Select-menu-outer {
                background-color: white !important;
                border-color: #ccc !important;
            }
            
            .light-theme .Select-option {
                background-color: white !important;
                color: black !important;
            }
            
            .light-theme .Select-option:hover {
                background-color: #f5f5f5 !important;
                color: black !important;
            }
            
            .light-theme .Select-option.is-selected {
                background-color: #667eea !important;
                color: white !important;
            }
            
            .light-theme .Select-multi-value-wrapper {
                color: black !important;
            }
            
            .light-theme .Select-value-label {
                color: black !important;
            }
            
            /* Dark theme dropdown styles */
            .dark-theme .Select-control {
                background-color: #34495e !important;
                color: white !important;
                border-color: #54616e !important;
            }
            
            .dark-theme .Select-menu-outer {
                background-color: #34495e !important;
                border-color: #54616e !important;
                box-shadow: 0 4px 15px rgba(255,255,255,0.1) !important;
            }
            
            .dark-theme .Select-option {
                background-color: #34495e !important;
                color: white !important;
            }
            
            .dark-theme .Select-option:hover {
                background-color: #54616e !important;
                color: white !important;
            }
            
            .dark-theme .Select-option.is-selected {
                background-color: #667eea !important;
                color: white !important;
            }
            
            .dark-theme .Select-multi-value-wrapper {
                color: white !important;
            }
            
            .dark-theme .Select-value-label {
                color: white !important;
            }
            
            .dark-theme .Select-input input {
                color: white !important;
            }
            
            .dark-theme .Select-placeholder {
                color: #bdc3c7 !important;
            }
            
            .dark-theme .Select-arrow-zone {
                color: white !important;
            }
            
            /* DatePicker styles for dark theme */
            .dark-theme .DateInput {
                background-color: #34495e !important;
                color: white !important;
            }
            
            .dark-theme .DateInput__input {
                background-color: #34495e !important;
                color: white !important;
                border-color: #54616e !important;
            }
            
            .dark-theme .DateRangePickerInput {
                background-color: #34495e !important;
                border-color: #54616e !important;
            }
            
            /* Multi-value tags in dark theme */
            .dark-theme .Select-value {
                background-color: #667eea !important;
                color: white !important;
                border: 1px solid #5a6fd8 !important;
            }
            
            .dark-theme .Select-value-icon {
                border-right-color: #5a6fd8 !important;
            }
            
            .dark-theme .Select-value-icon:hover {
                background-color: #5a6fd8 !important;
                color: white !important;
            }
            
            /* Tab styles for light theme */
            .light-theme .nav-tabs .nav-link {
                color: #495057 !important;
                background-color: #f8f9fa !important;
                border-color: #dee2e6 !important;
            }
            
            .light-theme .nav-tabs .nav-link:hover {
                color: #667eea !important;
                background-color: #e9ecef !important;
            }
            
            .light-theme .nav-tabs .nav-link.active {
                color: white !important;
                background-color: #667eea !important;
                border-color: #667eea !important;
            }
            
            /* Tab styles for dark theme */
            .dark-theme .nav-tabs .nav-link {
                color: #ffffff !important;
                background-color: #34495e !important;
                border-color: #54616e !important;
            }
            
            .dark-theme .nav-tabs .nav-link:hover {
                color: #ffffff !important;
                background-color: #54616e !important;
                border-color: #667eea !important;
            }
            
            .dark-theme .nav-tabs .nav-link.active {
                color: white !important;
                background-color: #667eea !important;
                border-color: #667eea !important;
            }
            
            /* Tab content styles */
            .dark-theme .tab-content {
                background-color: #34495e !important;
                color: white !important;
            }
            
            /* Sidebar animation styles */
            .controls-sidebar {
                position: fixed;
                top: 0;
                left: 0;
                height: 100vh;
                width: 350px;
                z-index: 1000;
                transform: translateX(-100%);
                transition: transform 0.3s ease-in-out;
                overflow-y: auto;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }
            
            .controls-sidebar.show {
                transform: translateX(0);
            }
            
            .main-content {
                transition: margin-left 0.3s ease-in-out;
                margin-left: 0;
            }
            
            .main-content.shifted {
                margin-left: 350px;
            }
            
            .controls-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.3);
                z-index: 999;
                opacity: 0;
                visibility: hidden;
                transition: opacity 0.3s ease-in-out, visibility 0.3s ease-in-out;
            }
            
            .controls-overlay.show {
                opacity: 1;
                visibility: visible;
            }
            
            /* Dark theme sidebar styles */
            .dark-theme .controls-sidebar {
                background-color: #2c3e50 !important;
                color: white !important;
                box-shadow: 2px 0 10px rgba(255,255,255,0.1);
            }
            
            /* Light theme sidebar styles */
            .light-theme .controls-sidebar {
                background-color: #ffffff !important;
                color: black !important;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }
            
            .light-theme .tab-content {
                background-color: white !important;
                color: black !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Load and prepare data
def load_data():
    """Load and preprocess the weather data"""
    try:
        print("📊 Loading data from ./data/raw/enhanced_weather_with_regions.csv...")
        # Load the enhanced weather data with regions
        df = pd.read_csv('./data/raw/enhanced_weather_with_regions.csv')
        print(f"✅ Data loaded successfully: {len(df)} records")
        
        # Convert date column to datetime
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df['date'] = df['last_updated'].dt.date
        df['year'] = df['last_updated'].dt.year
        df['month'] = df['last_updated'].dt.month
        df['month_name'] = df['last_updated'].dt.strftime('%B')
        
        # Handle missing values
        numeric_columns = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'uv_index']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Create precipitation column (if not available, use humidity as proxy)
        if 'precipitation' not in df.columns:
            df['precipitation'] = df['humidity'] * 0.1  # Simple proxy
        
        # Create wind_speed column (alias for wind_kph)
        df['wind_speed'] = df['wind_kph']
        
        print(f"📍 Countries: {df['normalized_country'].nunique()}")
        print(f"🌍 Regions: {df['geographic_region'].nunique()}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("🔍 Please check if the file exists and has the correct format")
        return pd.DataFrame()

# Load data
print("🌍 Initializing ClimateScope Dashboard...")
df = load_data()

# Get unique values for dropdowns
if not df.empty:
    countries = sorted(df['normalized_country'].unique())
    regions = sorted(df['geographic_region'].unique())
    print(f"✅ Loaded {len(countries)} countries and {len(regions)} regions")
else:
    countries = ['No Data Available']
    regions = ['No Data Available']
    print("⚠️ No data loaded - using fallback values")

metrics = ['temperature_celsius', 'humidity', 'wind_speed', 'precipitation']
metric_labels = {
    'temperature_celsius': 'Temperature (°C)',
    'humidity': 'Humidity (%)',
    'wind_speed': 'Wind Speed (km/h)',
    'precipitation': 'Precipitation (proxy)'
}

# Get date range
if not df.empty:
    min_date = df['date'].min()
    max_date = df['date'].max()
else:
    min_date = date(2024, 1, 1)
    max_date = date(2024, 12, 31)

# App layout
app.layout = html.Div([
    # Store for theme state
    dcc.Store(id='theme-store', data={'theme': 'light'}),
    # Store for controls visibility
    dcc.Store(id='controls-store', data={'visible': False}),
    
    # Controls Overlay
    html.Div(id='controls-overlay', className='controls-overlay'),
    
    # Collapsible Controls Sidebar
    html.Div([
        dbc.Container([
            # Sidebar Header
            html.Div([
                html.Div([
                    html.H5("📊 Interactive Controls", className="mb-2", style={'fontWeight': 'bold'}),
                    dbc.Button(
                        html.I(className="fas fa-times"),
                        id="close-controls-btn",
                        color="light",
                        size="sm",
                        className="float-end",
                        style={'border': 'none'}
                    )
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
                html.P("Customize your climate data analysis", className="mb-3", style={'fontSize': '0.9rem', 'opacity': '0.8'})
            ], style={'padding': '20px 20px 10px 20px', 'borderBottom': '1px solid rgba(0,0,0,0.1)'}),
            
            # Controls Content
            html.Div([
                # Date Selection Mode Toggle
                html.Div([
                    html.Label("📅 Date Selection Mode", className="fw-bold mb-3", style={'fontSize': '1rem'}),
                    dbc.RadioItems(
                        id='date-mode-toggle',
                        options=[
                            {'label': ' Date Range Analysis', 'value': 'range'},
                            {'label': ' Single Day Focus', 'value': 'single'}
                        ],
                        value='range',
                        style={'marginBottom': '20px'}
                    )
                ], style={'marginBottom': '25px'}),
                
                # Date Controls
                html.Div([
                    # Date Range Picker
                    html.Div([
                        html.Label("📅 Date Range", className="fw-bold mb-2", style={'fontSize': '0.9rem'}),
                        dcc.DatePickerRange(
                            id='date-picker-range',
                            start_date=min_date,
                            end_date=max_date,
                            display_format='YYYY-MM-DD',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            style={'width': '100%', 'marginBottom': '15px'}
                        )
                    ], id='date-range-container'),
                    
                    # Single Date Picker (initially hidden)
                    html.Div([
                        html.Label("📅 Select Date", className="fw-bold mb-2", style={'fontSize': '0.9rem'}),
                        dcc.DatePickerSingle(
                            id='single-date-picker',
                            date=min_date,
                            display_format='YYYY-MM-DD',
                            min_date_allowed=min_date,
                            max_date_allowed=max_date,
                            style={'width': '100%', 'marginBottom': '15px'}
                        )
                    ], id='single-date-container', style={'display': 'none'})
                ], style={'marginBottom': '25px'}),
                
                # Region Multi-select
                html.Div([
                    html.Label("🌍 Geographic Regions", className="fw-bold mb-2", style={'fontSize': '0.9rem'}),
                    dcc.Dropdown(
                        id='region-dropdown',
                        options=[{'label': f"🌍 {region}", 'value': region} for region in regions],
                        value=regions[:2] if len(regions) >= 2 else regions,
                        multi=True,
                        placeholder="🔍 Select regions to analyze...",
                        style={'fontSize': '0.85rem', 'marginBottom': '15px'}
                    )
                ], style={'marginBottom': '25px'}),
                
                # Country Multi-select
                html.Div([
                    html.Label("🏳️ Countries", className="fw-bold mb-2", style={'fontSize': '0.9rem'}),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': f"🏳️ {country}", 'value': country} for country in countries],
                        value=countries[:3] if len(countries) >= 3 else countries,
                        multi=True,
                        placeholder="🔍 Select countries to focus on...",
                        style={'fontSize': '0.85rem', 'marginBottom': '15px'}
                    )
                ], style={'marginBottom': '25px'}),
                
                # Metric Selector
                html.Div([
                    html.Label("📈 Primary Climate Metric", className="fw-bold mb-2", style={'fontSize': '0.9rem'}),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[{'label': f"📊 {metric_labels[metric]}", 'value': metric} for metric in metrics],
                        value='temperature_celsius',
                        clearable=False,
                        style={'fontSize': '0.85rem', 'marginBottom': '15px'}
                    )
                ], style={'marginBottom': '25px'}),
                

                
            ], style={'padding': '20px', 'height': 'calc(100vh - 120px)', 'overflowY': 'auto'})
        ], fluid=True)
    ], id='controls-sidebar', className='controls-sidebar'),
    
    # Main Content Area
    dbc.Container([
    
    # Header with gradient background and controls
    dbc.Row([
        dbc.Col([
            html.Div([
                # Top section with logo, title, and controls
                html.Div([
                    html.Div([
                        html.Span("🌍", style={'fontSize': '3rem', 'marginRight': '15px'}),
                        html.Span("ClimateScope Weather Dashboard", 
                                 style={'fontSize': '2.5rem', 'fontWeight': 'bold', 'color': 'white'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}),
                    
                    # Controls section with theme toggle, controls toggle, and report button
                    html.Div([
                        # Dark mode toggle and controls toggle
                        html.Div([
                            # Theme toggle row
                            html.Div([
                                html.I(className="fas fa-sun", id="theme-icon", 
                                      style={'color': 'white', 'fontSize': '1.2rem', 'marginRight': '10px'}),
                                dbc.Switch(
                                    id="theme-switch",
                                    value=False,
                                    style={'transform': 'scale(1.2)', 'marginRight': '20px'}
                                ),
                                html.I(className="fas fa-moon", 
                                      style={'color': 'white', 'fontSize': '1.2rem'})
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
                            
                            # Controls toggle button row
                            html.Div([
                                dbc.Button([
                                    html.I(className="fas fa-sliders-h", style={'marginRight': '8px'}),
                                    "Show Controls"
                                ], 
                                id="toggle-controls-btn", 
                                color="light", 
                                size="sm",
                                style={
                                    'fontWeight': 'bold',
                                    'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                                })
                            ])
                        ], style={
                            'position': 'absolute',
                            'top': '15px',
                            'left': '20px'
                        }),
                        
                        # Generate Report button positioned at top right
                        dbc.Button(
                            "📄 Generate Report", 
                            id="btn-report", 
                            color="light", 
                            size="sm",
                            style={
                                'position': 'absolute',
                                'top': '15px',
                                'right': '20px',
                                'fontWeight': 'bold',
                                'boxShadow': '0 2px 8px rgba(0,0,0,0.2)'
                            }
                        )
                    ])
                ], style={'position': 'relative', 'marginBottom': '10px'}),
                
                html.P("Advanced Global Weather Analytics & Interactive Visualizations",
                      style={'fontSize': '1.2rem', 'color': 'white', 'textAlign': 'center', 'margin': '0'})
            ], id='header-div', style={
                'background': 'linear-gradient(135deg, #4a90e2 0%, #f093fb 100%)',
                'padding': '30px 20px',
                'borderRadius': '10px',
                'marginBottom': '20px'
            })
        ])
    ]),
    
    # Download component for report
    dcc.Download(id="download-report"),
    
    # Statistics Cards Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='total-locations', children=f"{len(df):,}" if not df.empty else "0", 
                           style={'color': '#4a90e2', 'fontWeight': 'bold', 'margin': '0'}),
                    html.P("Total Locations", style={'color': '#666', 'margin': '0', 'fontSize': '0.9rem'})
                ])
            ], id='stats-card-1', style={'textAlign': 'center', 'border': 'none', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='avg-temperature', children=f"{df['temperature_celsius'].mean():.1f}°C" if not df.empty else "0°C", 
                           style={'color': '#e74c3c', 'fontWeight': 'bold', 'margin': '0'}),
                    html.P("Average Temperature", style={'color': '#666', 'margin': '0', 'fontSize': '0.9rem'})
                ])
            ], id='stats-card-2', style={'textAlign': 'center', 'border': 'none', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='avg-humidity', children=f"{df['humidity'].mean():.1f}%" if not df.empty else "0%", 
                           style={'color': '#27ae60', 'fontWeight': 'bold', 'margin': '0'}),
                    html.P("Average Humidity", style={'color': '#666', 'margin': '0', 'fontSize': '0.9rem'})
                ])
            ], id='stats-card-3', style={'textAlign': 'center', 'border': 'none', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='avg-windspeed', children=f"{df['wind_kph'].mean():.1f}" if not df.empty else "0", 
                           style={'color': '#f39c12', 'fontWeight': 'bold', 'margin': '0'}),
                    html.P("Avg Wind Speed (km/h)", style={'color': '#666', 'margin': '0', 'fontSize': '0.9rem'})
                ])
            ], id='stats-card-4', style={'textAlign': 'center', 'border': 'none', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3(id='avg-uv-index', children=f"{df['uv_index'].mean():.1f}" if not df.empty else "0", 
                           style={'color': '#9b59b6', 'fontWeight': 'bold', 'margin': '0'}),
                    html.P("Average UV Index", style={'color': '#666', 'margin': '0', 'fontSize': '0.9rem'})
                ])
            ], id='stats-card-5', style={'textAlign': 'center', 'border': 'none', 'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'})
        ], width=3)
    ], id='stats-row', className="mb-4"),
    
    # Date Validation Message Area
    html.Div(id='date-validation-message', className="mb-3"),
    
    # Main Visualizations Layout
    
    # Full-width Global Choropleth Map
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("🗺️ Global Climate Distribution Map", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'})
                ], id='map-header', style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='world-map', config={'displayModeBar': True}, style={'height': '500px'})
                ], style={'padding': '20px'})
            ], id='map-card', style={'border': 'none', 'boxShadow': '0 8px 25px rgba(0,0,0,0.15)'})
        ], width=12)
    ], className="mb-4"),
    
    # Secondary Visualizations Row
    dbc.Row([
        # Time Series
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("📈 Climate Trends Over Time", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'})
                ], id='timeseries-header', style={'background': 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='time-series', config={'displayModeBar': False})
                ])
            ], id='timeseries-card', style={'border': 'none', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], width=6),
        
        # Air Quality Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("🌬️ Air Quality Index", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'})
                ], id='airquality-header', style={'background': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='air-quality-chart', config={'displayModeBar': False})
                ])
            ], id='airquality-card', style={'border': 'none', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], width=6)
    ], className="mb-4"),
    
    # Additional Analysis Row
    dbc.Row([
        # Correlation Analysis
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("🔍 Climate Correlation Analysis", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("X-Axis:", style={'font-size': '0.8em', 'color': 'white'}),
                            dcc.Dropdown(
                                id='scatter-x-dropdown',
                                options=[{'label': metric_labels[metric], 'value': metric} for metric in metrics],
                                value='temperature_celsius',
                                clearable=False,
                                style={'font-size': '0.8em'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Y-Axis:", style={'font-size': '0.8em', 'color': 'white'}),
                            dcc.Dropdown(
                                id='scatter-y-dropdown',
                                options=[{'label': metric_labels[metric], 'value': metric} for metric in metrics],
                                value='humidity',
                                clearable=False,
                                style={'font-size': '0.8em'}
                            )
                        ], width=6)
                    ], className="mt-2")
                ], id='correlation-header', style={'background': 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='scatter-plot', config={'displayModeBar': False})
                ])
            ], id='correlation-card', style={'border': 'none', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], width=6),
        
        # Seasonality Heatmap
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("🔥 Seasonal Climate Patterns", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'})
                ], id='seasonality-header', style={'background': 'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='seasonality-heatmap', config={'displayModeBar': False})
                ])
            ], id='seasonality-card', style={'border': 'none', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], width=6)
    ], className="mb-4"),
    
    # Additional Visualizations Section
    dbc.Row([
        # Regional Climate Distribution Box Plot
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("📦 Regional Climate Distribution", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'})
                ], id='boxplot-header', style={'background': 'linear-gradient(135deg, #74b9ff 0%, #0984e3 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='regional-boxplot', config={'displayModeBar': False})
                ])
            ], id='boxplot-card', style={'border': 'none', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], width=6),
        
        # Climate Profile Radar Chart
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H6("🎯 Country Climate Profiles", className="mb-0", style={'color': 'white', 'fontWeight': 'bold'})
                ], id='radar-header', style={'background': 'linear-gradient(135deg, #fd79a8 0%, #e84393 100%)', 'border': 'none'}),
                dbc.CardBody([
                    dcc.Graph(id='climate-radar-chart', config={'displayModeBar': False})
                ])
            ], id='radar-card', style={'border': 'none', 'boxShadow': '0 4px 15px rgba(0,0,0,0.1)'})
        ], width=6)
    ], className="mb-4"),
    
    # Download component for report only
    dcc.Download(id="download-report"),
    
    # Insights Panel with Enhanced Interactive Design
    dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H4([
                    html.I(className="fas fa-lightbulb me-2"),
                    "Key Climate Insights"
                ], className="mb-0", style={'color': 'white', 'fontWeight': 'bold'}),
                html.P("Interactive analysis of your selected data", 
                      className="mb-0", style={'color': 'rgba(255,255,255,0.8)', 'fontSize': '0.9rem'})
            ])
        ], id='insights-header', style={
            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'border': 'none',
            'borderRadius': '15px 15px 0 0'
        }),
        dbc.CardBody([
            # Interactive Insights Tabs
            dbc.Tabs([
                dbc.Tab(
                    label="📊 Statistical Overview", 
                    tab_id="stats-tab",
                    label_style={'color': 'inherit', 'fontWeight': 'bold'},
                    active_label_style={'background-color': '#667eea', 'color': 'white', 'fontWeight': 'bold'}
                ),
                dbc.Tab(
                    label="🌍 Regional Analysis", 
                    tab_id="regional-tab",
                    label_style={'color': 'inherit', 'fontWeight': 'bold'},
                    active_label_style={'background-color': '#667eea', 'color': 'white', 'fontWeight': 'bold'}
                ),
                dbc.Tab(
                    label="☀️ Highest Temperature", 
                    tab_id="top-tab",
                    label_style={'color': 'inherit', 'fontWeight': 'bold'},
                    active_label_style={'background-color': '#667eea', 'color': 'white', 'fontWeight': 'bold'}
                ),
                dbc.Tab(
                    label="📈 Trends & Patterns", 
                    tab_id="trends-tab",
                    label_style={'color': 'inherit', 'fontWeight': 'bold'},
                    active_label_style={'background-color': '#667eea', 'color': 'white', 'fontWeight': 'bold'}
                )
            ], id="insights-tabs", active_tab="stats-tab", className="mb-3"),
            
            html.Br(),
            
            # Dynamic Content Area
            html.Div(id='insights-content', className="p-3")
        ], style={'padding': '25px'})
    ], id='insights-card', className="mb-4", style={
        'border': 'none', 
        'boxShadow': '0 15px 35px rgba(0,0,0,0.1)',
        'borderRadius': '15px'
    }),
    
    # Footer
    html.Hr(),
    html.P("ClimateScope Dashboard - by Mahitha Potluri", 
           className="text-center text-muted", style={'font-size': '0.9em'})
    ], id='main-container', fluid=True, className='light-theme', style={'background-color': 'white', 'min-height': '100vh'})
    
], id='main-content', className='main-content')

# Callback to toggle between date range and single date modes
@app.callback(
    [Output('date-range-container', 'style'),
     Output('single-date-container', 'style')],
    [Input('date-mode-toggle', 'value')]
)
def toggle_date_mode(mode):
    """Toggle between date range and single date selection"""
    if mode == 'range':
        return {'display': 'block'}, {'display': 'none'}
    else:
        return {'display': 'none'}, {'display': 'block'}

# Callback for date validation
@app.callback(
    Output('date-validation-message', 'children'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('single-date-picker', 'date'),
     Input('date-mode-toggle', 'value')]
)
def validate_dates(start_date, end_date, single_date, mode):
    """Validate selected dates and show appropriate messages"""
    if df.empty:
        return dbc.Alert("⚠️ No data available for validation", color="warning", className="mb-3")
    
    # Get available date range
    available_min = min_date
    available_max = max_date
    
    if mode == 'range':
        if start_date and end_date:
            start_date_obj = pd.to_datetime(start_date).date()
            end_date_obj = pd.to_datetime(end_date).date()
            
            # Check if dates are outside available range
            if start_date_obj < available_min or end_date_obj > available_max:
                return dbc.Alert(
                    [
                        html.Strong("📅 Date Range Error: "),
                        f"Data is only available from {available_min} to {available_max}. "
                        f"Please select dates within this range."
                    ],
                    color="danger",
                    className="mb-3"
                )
            
            # Check if start date is after end date
            if start_date_obj > end_date_obj:
                return dbc.Alert(
                    [
                        html.Strong("📅 Date Range Error: "),
                        "Start date cannot be after end date. Please adjust your selection."
                    ],
                    color="danger",
                    className="mb-3"
                )
            
            # Valid range - no message needed
            return html.Div()
    
    else:  # single date mode
        if single_date:
            single_date_obj = pd.to_datetime(single_date).date()
            
            # Check if date is outside available range
            if single_date_obj < available_min or single_date_obj > available_max:
                return dbc.Alert(
                    [
                        html.Strong("📅 Date Error: "),
                        f"Data is only available from {available_min} to {available_max}. "
                        f"Please select a date within this range."
                    ],
                    color="danger",
                    className="mb-3"
                )
            
            # Valid single date - no message needed
            return html.Div()
    
    return html.Div()  # No message

# Theme switching callbacks
@app.callback(
    [Output('theme-store', 'data'),
     Output('theme-icon', 'className'),
     Output('main-container', 'className')],
    [Input('theme-switch', 'value')]
)
def update_theme_store(dark_mode):
    """Update theme store based on switch value"""
    theme = 'dark' if dark_mode else 'light'
    icon_class = "fas fa-moon" if dark_mode else "fas fa-sun"
    container_class = 'dark-theme' if dark_mode else 'light-theme'
    return {'theme': theme}, icon_class, container_class

@app.callback(
    [Output('header-div', 'style'),
     Output('main-container', 'style'),
     Output('stats-card-1', 'style'),
     Output('stats-card-2', 'style'),
     Output('stats-card-3', 'style'),
     Output('stats-card-4', 'style'),
     Output('stats-card-5', 'style'),
     Output('map-card', 'style'),
     Output('timeseries-card', 'style'),
     Output('airquality-card', 'style'),
     Output('correlation-card', 'style'),
     Output('seasonality-card', 'style'),
     Output('boxplot-card', 'style'),
     Output('radar-card', 'style'),
     Output('insights-card', 'style')],
    [Input('theme-store', 'data')]
)
def update_theme_styles(theme_data):
    """Update component styles based on theme"""
    theme = theme_data.get('theme', 'light')
    
    if theme == 'dark':
        # Dark theme styles
        header_style = {
            'background': 'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
            'padding': '30px 20px',
            'borderRadius': '10px',
            'marginBottom': '20px'
        }
        
        container_style = {
            'background-color': '#2c3e50', 
            'min-height': '100vh',
            'color': 'white'
        }
        
        card_base_style = {
            'textAlign': 'center',
            'border': 'none',
            'boxShadow': '0 2px 10px rgba(255,255,255,0.1)',
            'background': '#34495e',
            'color': 'white'
        }
        
        main_card_style = {
            'border': 'none',
            'boxShadow': '0 8px 25px rgba(255,255,255,0.1)',
            'background': '#34495e',
            'color': 'white'
        }
        
        viz_card_style = {
            'border': 'none',
            'boxShadow': '0 4px 15px rgba(255,255,255,0.1)',
            'background': '#34495e',
            'color': 'white'
        }
        
        insights_card_style = {
            'border': 'none',
            'boxShadow': '0 15px 35px rgba(255,255,255,0.1)',
            'borderRadius': '15px',
            'background': '#34495e',
            'color': 'white'
        }
        
    else:
        # Light theme styles (original)
        header_style = {
            'background': 'linear-gradient(135deg, #4a90e2 0%, #f093fb 100%)',
            'padding': '30px 20px',
            'borderRadius': '10px',
            'marginBottom': '20px'
        }
        
        container_style = {
            'background-color': 'white', 
            'min-height': '100vh',
            'color': 'black'
        }
        
        card_base_style = {
            'textAlign': 'center',
            'border': 'none',
            'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
            'background': 'white',
            'color': 'black'
        }
        
        main_card_style = {
            'border': 'none',
            'boxShadow': '0 8px 25px rgba(0,0,0,0.15)',
            'background': 'white',
            'color': 'black'
        }
        
        viz_card_style = {
            'border': 'none',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)',
            'background': 'white',
            'color': 'black'
        }
        
        insights_card_style = {
            'border': 'none',
            'boxShadow': '0 15px 35px rgba(0,0,0,0.1)',
            'borderRadius': '15px',
            'background': 'white',
            'color': 'black'
        }
    
    return (header_style, container_style, card_base_style, card_base_style, card_base_style, 
            card_base_style, card_base_style, main_card_style,
            viz_card_style, viz_card_style, viz_card_style, viz_card_style,
            viz_card_style, viz_card_style, insights_card_style)

# Callback for controls sidebar toggle
@app.callback(
    [Output('controls-sidebar', 'className'),
     Output('controls-overlay', 'className'),
     Output('main-content', 'className'),
     Output('toggle-controls-btn', 'children')],
    [Input('toggle-controls-btn', 'n_clicks'),
     Input('close-controls-btn', 'n_clicks'),
     Input('controls-overlay', 'n_clicks')],
    [State('controls-sidebar', 'className'),
     State('theme-store', 'data')],
    prevent_initial_call=True
)
def toggle_controls_sidebar(toggle_clicks, close_clicks, overlay_clicks, current_sidebar_class, theme_data):
    """Toggle the controls sidebar visibility"""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        # Default state - sidebar hidden
        theme_class = 'dark-theme' if theme_data.get('theme') == 'dark' else 'light-theme'
        return 'controls-sidebar', 'controls-overlay', f'main-content {theme_class}', [
            html.I(className="fas fa-sliders-h", style={'marginRight': '8px'}),
            "Show Controls"
        ]
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determine current state
    is_sidebar_visible = 'show' in current_sidebar_class
    
    # Toggle logic
    if trigger_id == 'toggle-controls-btn':
        show_sidebar = not is_sidebar_visible
    elif trigger_id in ['close-controls-btn', 'controls-overlay']:
        show_sidebar = False
    else:
        show_sidebar = is_sidebar_visible
    
    # Update classes and button text
    theme_class = 'dark-theme' if theme_data.get('theme') == 'dark' else 'light-theme'
    
    if show_sidebar:
        sidebar_class = f'controls-sidebar show {theme_class}'
        overlay_class = 'controls-overlay show'
        main_class = f'main-content shifted {theme_class}'
        button_content = [
            html.I(className="fas fa-times", style={'marginRight': '8px'}),
            "Hide Controls"
        ]
    else:
        sidebar_class = f'controls-sidebar {theme_class}'
        overlay_class = 'controls-overlay'
        main_class = f'main-content {theme_class}'
        button_content = [
            html.I(className="fas fa-sliders-h", style={'marginRight': '8px'}),
            "Show Controls"
        ]
    
    return sidebar_class, overlay_class, main_class, button_content

# Callback for updating all visualizations
@app.callback(
    [Output('world-map', 'figure'),
     Output('time-series', 'figure'),
     Output('air-quality-chart', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('seasonality-heatmap', 'figure'),
     Output('total-locations', 'children'),
     Output('avg-temperature', 'children'),
     Output('avg-humidity', 'children'),
     Output('avg-windspeed', 'children'),
     Output('avg-uv-index', 'children')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('single-date-picker', 'date'),
     Input('date-mode-toggle', 'value'),
     Input('region-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('scatter-x-dropdown', 'value'),
     Input('scatter-y-dropdown', 'value'),
     Input('theme-store', 'data')]
)
def update_visualizations(start_date, end_date, single_date, date_mode, selected_regions, selected_countries, 
                         selected_metric, scatter_x, scatter_y, theme_data):
    """Update all visualizations based on filter selections and theme"""
    
    # Get theme
    theme = theme_data.get('theme', 'light')
    
    # Define theme-specific colors
    if theme == 'dark':
        plot_bg = '#2c3e50'
        paper_bg = '#34495e'
        font_color = 'white'
        grid_color = '#7f8c8d'
        line_color = '#3498db'
        map_colors = ['#FFF5B7', '#FFD93D', '#FF8C42', '#FF6B35', '#C73E1D']  # Keep warm colors
    else:
        plot_bg = 'white'
        paper_bg = 'white'
        font_color = 'black'
        grid_color = '#ecf0f1'
        line_color = '#2E86AB'
        map_colors = ['#FFF5B7', '#FFD93D', '#FF8C42', '#FF6B35', '#C73E1D']  # Keep warm colors
    
    if df.empty:
        # Return empty figures if no data
        empty_fig = go.Figure().add_annotation(
            text="No data available", 
            xref="paper", yref="paper", x=0.5, y=0.5, 
            showarrow=False, font_size=16
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "0", "0°C", "0%", "0", "0"
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    # Date filtering based on mode
    if date_mode == 'range':
        # Date range filtering
        if start_date and end_date:
            start_date_obj = pd.to_datetime(start_date).date()
            end_date_obj = pd.to_datetime(end_date).date()
            
            # Validate dates are within available range
            if start_date_obj < min_date or end_date_obj > max_date:
                # Return empty result for invalid dates
                empty_fig = go.Figure().add_annotation(
                    text="Selected dates are outside available data range", 
                    xref="paper", yref="paper", x=0.5, y=0.5, 
                    showarrow=False, font_size=14
                )
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "0", "0°C", "0%", "0", "0"
            
            filtered_df = filtered_df[(filtered_df['date'] >= start_date_obj) & 
                                     (filtered_df['date'] <= end_date_obj)]
    else:
        # Single date filtering
        if single_date:
            single_date_obj = pd.to_datetime(single_date).date()
            
            # Validate date is within available range
            if single_date_obj < min_date or single_date_obj > max_date:
                # Return empty result for invalid date
                empty_fig = go.Figure().add_annotation(
                    text="Selected date is outside available data range", 
                    xref="paper", yref="paper", x=0.5, y=0.5, 
                    showarrow=False, font_size=14
                )
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "0", "0°C", "0%", "0", "0"
            
            filtered_df = filtered_df[filtered_df['date'] == single_date_obj]
    
    # Region filtering
    if selected_regions:
        filtered_df = filtered_df[filtered_df['geographic_region'].isin(selected_regions)]
    
    # Country filtering
    if selected_countries:
        filtered_df = filtered_df[filtered_df['normalized_country'].isin(selected_countries)]
    
    if filtered_df.empty:
        empty_fig = go.Figure().add_annotation(
            text="No data matches the selected filters", 
            xref="paper", yref="paper", x=0.5, y=0.5, 
            showarrow=False, font_size=14
        )
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "0", "0°C", "0%", "0", "0"
    
    # Performance optimization: Sample large datasets for visualizations
    viz_sample_size = min(5000, len(filtered_df))  # Max 5000 points for performance
    viz_df = filtered_df.sample(n=viz_sample_size) if len(filtered_df) > viz_sample_size else filtered_df
    
    # 1. World Map (Choropleth) - Use aggregated data for performance
    try:
        country_agg = filtered_df.groupby('normalized_country')[selected_metric].mean().reset_index()
        # Show all countries from filtered data - no artificial limitations
        
        world_map = px.choropleth(
            country_agg,
            locations='normalized_country',
            color=selected_metric,
            locationmode='country names',  # Using country names as per our data format
            title=f"Global {metric_labels[selected_metric]} Distribution",
            color_continuous_scale=map_colors,  # Use theme-aware colors
            hover_name='normalized_country',
            hover_data={selected_metric: ':.2f'}
        )
        world_map.update_layout(
            geo=dict(showframe=False, showcoastlines=True, bgcolor=plot_bg),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            title_font_color=font_color
        )
    except Exception as e:
        world_map = go.Figure().add_annotation(
            text=f"Error creating world map: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font_color=font_color
        )
        world_map.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
    
    # 2. Time Series - Use sampled data for performance
    try:
        # Use sampled data for time series to improve performance
        sample_df = viz_df.copy()
        sample_df['year_month'] = sample_df['last_updated'].dt.to_period('M')
        time_series_data = sample_df.groupby('year_month')[selected_metric].mean().reset_index()
        time_series_data['year_month'] = time_series_data['year_month'].dt.to_timestamp()
        
        time_series = px.line(
            time_series_data,
            x='year_month',
            y=selected_metric,
            title=f"{metric_labels[selected_metric]} Trend Over Time",
            markers=True
        )
        time_series.update_layout(
            xaxis_title="Date",
            yaxis_title=metric_labels[selected_metric],
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            title_font_color=font_color,
            xaxis=dict(gridcolor=grid_color, color=font_color),
            yaxis=dict(gridcolor=grid_color, color=font_color)
        )
        time_series.update_traces(line_color=line_color)
    except Exception as e:
        time_series = go.Figure().add_annotation(
            text=f"Error creating time series: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font_color=font_color
        )
        time_series.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
    
    # 3. Air Quality Chart - Use sampled data for performance
    try:
        # Use smaller sample for air quality calculation
        aqi_df = viz_df.copy()
        
        # Create air quality index using available metrics with better error handling
        required_columns = ['humidity', 'wind_kph', 'uv_index', 'pressure_mb']
        available_columns = [col for col in required_columns if col in aqi_df.columns]
        
        if len(available_columns) >= 2:  # Need at least 2 metrics
            # Initialize composite score
            composite_score = pd.Series(0, index=aqi_df.index)
            
            # Add components based on available data
            if 'humidity' in aqi_df.columns:
                composite_score += aqi_df['humidity'].fillna(50) * 0.3
            if 'wind_kph' in aqi_df.columns:
                composite_score += (100 - aqi_df['wind_kph'].fillna(10)) * 0.2  # Inverted: higher wind = better air quality
            if 'uv_index' in aqi_df.columns:
                composite_score += aqi_df['uv_index'].fillna(5) * 0.3
            if 'pressure_mb' in aqi_df.columns:
                normalized_pressure = (aqi_df['pressure_mb'].fillna(1013) - 1000) * 0.2
                composite_score += normalized_pressure
            
            # Normalize to 0-100 scale
            if composite_score.std() > 0:  # Avoid division by zero
                min_score = composite_score.min()
                max_score = composite_score.max()
                aqi_df['normalized_aqi'] = ((composite_score - min_score) / (max_score - min_score)) * 100
            else:
                aqi_df['normalized_aqi'] = 50  # Default middle value
            
            # Create categories
            def get_aqi_category(value):
                if pd.isna(value):
                    return "Unknown"
                elif value <= 20:
                    return "Excellent"
                elif value <= 40:
                    return "Good"
                elif value <= 60:
                    return "Moderate"
                elif value <= 80:
                    return "Poor"
                else:
                    return "Very Poor"
            
            aqi_df['aqi_category'] = aqi_df['normalized_aqi'].apply(get_aqi_category)
            
            # Count by category
            aqi_counts = aqi_df['aqi_category'].value_counts()
            
            # Create bar chart with improved styling
            colors = {
                'Excellent': '#00e400',
                'Good': '#ffff00', 
                'Moderate': '#ff7e00',
                'Poor': '#ff0000',
                'Very Poor': '#8f3f97',
                'Unknown': '#cccccc'
            }
            
            air_quality_chart = go.Figure()
            
            categories = ['Excellent', 'Good', 'Moderate', 'Poor', 'Very Poor', 'Unknown']
            for category in categories:
                if category in aqi_counts.index:
                    air_quality_chart.add_trace(go.Bar(
                        x=[category],
                        y=[aqi_counts[category]],
                        name=category,
                        marker_color=colors[category],
                        text=[f"{aqi_counts[category]}<br>({aqi_counts[category]/len(aqi_df)*100:.1f}%)"],
                        textposition='auto',
                        textfont=dict(color='white', size=12, family='Arial Black')
                    ))
            
            air_quality_chart.update_layout(
                title=dict(
                    text="Air Quality Distribution (Composite Index)",
                    font=dict(size=16, color=font_color)
                ),
                xaxis_title="Air Quality Category",
                yaxis_title="Number of Locations",
                showlegend=False,
                height=400,
                margin=dict(l=0, r=0, t=40, b=0),
                plot_bgcolor=plot_bg,
                paper_bgcolor=paper_bg,
                font_color=font_color,
                xaxis=dict(gridcolor=grid_color, color=font_color),
                yaxis=dict(gridcolor=grid_color, color=font_color)
            )
        else:
            # Fallback chart when insufficient data
            air_quality_chart = go.Figure().add_annotation(
                text="Insufficient data for air quality analysis<br>Need humidity, wind, UV, or pressure data", 
                xref="paper", yref="paper", x=0.5, y=0.5, 
                showarrow=False, font_size=14, font_color=font_color
            )
            air_quality_chart.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
        
    except Exception as e:
        air_quality_chart = go.Figure().add_annotation(
            text=f"Air quality data processing error:<br>{str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, 
            showarrow=False, font_size=12, font_color=font_color
        )
        air_quality_chart.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
    
    # 4. Scatter Plot - Use pre-sampled data
    try:
        scatter_plot = px.scatter(
            viz_df,  # Already sampled for performance
            x=scatter_x,
            y=scatter_y,
            color='geographic_region',
            title=f"{metric_labels[scatter_x]} vs {metric_labels[scatter_y]}",
            hover_data=['normalized_country'],
            opacity=0.7
        )
        scatter_plot.update_layout(
            xaxis_title=metric_labels[scatter_x],
            yaxis_title=metric_labels[scatter_y],
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            title_font_color=font_color,
            xaxis=dict(gridcolor=grid_color, color=font_color),
            yaxis=dict(gridcolor=grid_color, color=font_color)
        )
    except Exception as e:
        scatter_plot = go.Figure().add_annotation(
            text=f"Error creating scatter plot: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font_color=font_color
        )
        scatter_plot.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
    
    # 5. Seasonality Heatmap - Use sampled data
    try:
        # Use sampled data for heatmap
        heatmap_df = viz_df.copy()
        heatmap_data = heatmap_df.pivot_table(
            values=selected_metric,
            index='month',
            columns='year',
            aggfunc='mean'
        )
        
        # Choose colorscale based on theme
        heatmap_colorscale = 'RdYlBu_r' if theme == 'light' else 'Viridis'
        
        seasonality_heatmap = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=[datetime(2024, i, 1).strftime('%B') for i in heatmap_data.index],
            color_continuous_scale=heatmap_colorscale,
            title=f"{metric_labels[selected_metric]} Seasonality Pattern",
            aspect='auto'
        )
        seasonality_heatmap.update_layout(
            xaxis_title="Year",
            yaxis_title="Month",
            height=400,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            title_font_color=font_color,
            xaxis=dict(color=font_color),
            yaxis=dict(color=font_color)
        )
    except Exception as e:
        seasonality_heatmap = go.Figure().add_annotation(
            text=f"Error creating heatmap: {str(e)}", 
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font_color=font_color
        )
        seasonality_heatmap.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
    
    # 6. Calculate Statistics for Cards
    try:
        total_locations = f"{len(filtered_df):,}"
        avg_temperature = f"{filtered_df['temperature_celsius'].mean():.1f}°C"
        avg_humidity = f"{filtered_df['humidity'].mean():.1f}%"
        avg_windspeed = f"{filtered_df['wind_kph'].mean():.1f}"
        avg_uv_index = f"{filtered_df['uv_index'].mean():.1f}"
    except Exception as e:
        total_locations = "0"
        avg_temperature = "0°C"
        avg_humidity = "0%"
        avg_windspeed = "0"
        avg_uv_index = "0"
    
    return world_map, time_series, air_quality_chart, scatter_plot, seasonality_heatmap, total_locations, avg_temperature, avg_humidity, avg_windspeed, avg_uv_index

# Callback for insights tabs
@app.callback(
    Output('insights-content', 'children'),
    [Input('insights-tabs', 'active_tab'),
     Input('region-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('metric-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('single-date-picker', 'date'),
     Input('date-mode-toggle', 'value'),
     Input('theme-store', 'data')]
)
def update_insights_content(active_tab, selected_regions, selected_countries, selected_metric, 
                           start_date, end_date, single_date, date_mode, theme_data):
    """Update insights content based on selected tab, filters, and theme"""
    
    # Get theme
    theme = theme_data.get('theme', 'light')
    
    if df.empty:
        return html.Div("No data available for insights", className="text-center text-muted")
    
    # Apply same filtering logic as main callback
    filtered_df = df.copy()
    
    # Date filtering
    if date_mode == 'range' and start_date and end_date:
        start_date_obj = pd.to_datetime(start_date).date()
        end_date_obj = pd.to_datetime(end_date).date()
        filtered_df = filtered_df[(filtered_df['date'] >= start_date_obj) & 
                                 (filtered_df['date'] <= end_date_obj)]
    elif date_mode == 'single' and single_date:
        single_date_obj = pd.to_datetime(single_date).date()
        filtered_df = filtered_df[filtered_df['date'] == single_date_obj]
    
    # Region and country filtering
    if selected_regions:
        filtered_df = filtered_df[filtered_df['geographic_region'].isin(selected_regions)]
    if selected_countries:
        filtered_df = filtered_df[filtered_df['normalized_country'].isin(selected_countries)]
    
    if filtered_df.empty:
        return html.Div("No data matches current filters", className="text-center text-muted")
    
    try:
        return generate_interactive_insights(filtered_df, selected_metric, active_tab, theme)
    except Exception as e:
        return html.Div(f"Error generating insights: {str(e)}", className="text-danger")

def generate_interactive_insights(data, metric, tab_type, theme='light'):
    """Generate interactive insights based on tab selection and theme"""
    
    if tab_type == "stats-tab":
        return generate_statistical_insights(data, metric, theme)
    elif tab_type == "regional-tab":
        return generate_regional_insights(data, metric, theme)
    elif tab_type == "top-tab":
        return generate_top_performers_insights(data, metric, theme)
    elif tab_type == "trends-tab":
        return generate_trends_insights(data, metric, theme)
    else:
        return generate_statistical_insights(data, metric, theme)

def generate_statistical_insights(data, metric, theme='light'):
    """Generate statistical overview insights with theme support"""
    stats = data[metric].describe()
    
    # Theme-aware gradient backgrounds
    if theme == 'dark':
        gradients = [
            'linear-gradient(135deg, #34495e 0%, #2c3e50 100%)',
            'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)',
            'linear-gradient(135deg, #34495e 0%, #2c3e50 100%)',
            'linear-gradient(135deg, #2c3e50 0%, #34495e 100%)'
        ]
    else:
        gradients = [
            'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)',
            'linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)',
            'linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%)',
            'linear-gradient(135deg, #fff3e0 0%, #ffcc02 100%)'
        ]
    
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{stats['mean']:.2f}", className="text-primary fw-bold"),
                    html.P("Average", className="mb-0 text-muted small")
                ])
            ], className="text-center border-0", style={'background': gradients[0]})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{stats['std']:.2f}", className="text-success fw-bold"),
                    html.P("Std Deviation", className="mb-0 text-muted small")
                ])
            ], className="text-center border-0", style={'background': gradients[1]})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{stats['min']:.2f}", className="text-info fw-bold"),
                    html.P("Minimum", className="mb-0 text-muted small")
                ])
            ], className="text-center border-0", style={'background': gradients[2]})
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5(f"{stats['max']:.2f}", className="text-warning fw-bold"),
                    html.P("Maximum", className="mb-0 text-muted small")
                ])
            ], className="text-center border-0", style={'background': gradients[3]})
        ], width=3)
    ])

def generate_regional_insights(data, metric, theme='light'):
    """Generate regional analysis insights with theme support"""
    try:
        regional_stats = data.groupby('geographic_region')[metric].agg(['mean', 'count']).sort_values('mean', ascending=False)
        
        cards = []
        
        # Theme-aware colors
        if theme == 'dark':
            colors = ['#34495e', '#2c3e50', '#34495e', '#2c3e50', '#34495e', '#2c3e50', '#34495e']
        else:
            colors = ['#e3f2fd', '#e8f5e8', '#fff3e0', '#fce4ec', '#e0f2f1', '#f3e5f5', '#e1f5fe']
        
        for i, (region, stats) in enumerate(regional_stats.head(7).iterrows()):
            color = colors[i % len(colors)]
            cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(region, className="fw-bold text-dark"),
                            html.H5(f"{stats['mean']:.2f}", className="text-primary"),
                            html.Small(f"{stats['count']} locations", className="text-muted")
                        ])
                    ], style={'background': f'linear-gradient(135deg, {color} 0%, {color}88 100%)', 'border': 'none'})
                ], width=6, className="mb-3")
            )
        
        return dbc.Row(cards)
    except Exception as e:
        return html.Div(f"Error generating regional insights: {str(e)}", className="text-danger")

def generate_top_performers_insights(data, metric, theme='light'):
    """Generate top performers insights with theme support"""
    try:
        # Define metric labels locally if not available
        local_metric_labels = {
            'temperature_celsius': 'Temperature (°C)',
            'humidity': 'Humidity (%)',
            'wind_speed': 'Wind Speed (km/h)',
            'precipitation': 'Precipitation (proxy)'
        }
        
        # Get top 10 locations with highest values for the selected metric
        top_locations = data.nlargest(10, metric)[['normalized_country', 'location_name', metric]]
        
        # Theme-aware border color
        border_color = '#34495e' if theme == 'dark' else '#667eea'
        
        performers = []
        for i, (_, row) in enumerate(top_locations.iterrows()):
            performers.append(
                dbc.ListGroupItem([
                    html.Div([
                        html.H6(f"#{i+1} {row['normalized_country']}", className="mb-1 fw-bold"),
                        html.P(f"{row['location_name']}", className="mb-1 text-muted small"),
                        html.Small(f"{local_metric_labels.get(metric, metric)}: {row[metric]:.2f}", className="text-primary fw-bold")
                    ])
                ], style={'border-left': f'4px solid {border_color}'})
            )
        
        return html.Div([
            html.H5(f"🏆 Top 10 Locations - {local_metric_labels.get(metric, metric)}", className="mb-3"),
            dbc.ListGroup(performers, flush=True)
        ])
    except Exception as e:
        return html.Div(f"Error generating highest temperature data: {str(e)}", className="text-danger")

def generate_trends_insights(data, metric, theme='light'):
    """Generate trends and patterns insights with theme support"""
    try:
        # Monthly trends
        monthly_trend = data.groupby(data['last_updated'].dt.month)[metric].mean()
        peak_month = monthly_trend.idxmax()
        peak_month_name = datetime(2024, peak_month, 1).strftime('%B')
        
        # Regional diversity
        regional_diversity = data.groupby('geographic_region')[metric].std().mean()
        
        return dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5("📈 Seasonal Peak", className="alert-heading"),
                    html.P(f"{peak_month_name} shows the highest average {metric_labels[metric].lower()} ({monthly_trend.max():.2f})")
                ], color="info")
            ], width=6),
            dbc.Col([
                dbc.Alert([
                    html.H5("🌍 Regional Diversity", className="alert-heading"),
                    html.P(f"Average regional variation: {regional_diversity:.2f}")
                ], color="success")
            ], width=6)
        ])
    except Exception as e:
        return html.Div(f"Error generating trends insights: {str(e)}", className="text-danger")

def generate_insights(data, metric, tab_type="stats", theme='light'):
    """Legacy function for backward compatibility"""
    return generate_interactive_insights(data, metric, f"{tab_type}-tab", theme)

# Callback to update country dropdown based on region selection
@app.callback(
    Output('country-dropdown', 'options'),
    [Input('region-dropdown', 'value')]
)
def update_country_options(selected_regions):
    """Update country dropdown based on selected regions"""
    if not selected_regions or df.empty:
        return [{'label': country, 'value': country} for country in countries]
    
    filtered_countries = df[df['geographic_region'].isin(selected_regions)]['normalized_country'].unique()
    filtered_countries = sorted(filtered_countries)
    return [{'label': country, 'value': country} for country in filtered_countries]

# Callback for Regional Box Plot
@app.callback(
    Output('regional-boxplot', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('single-date-picker', 'date'),
     Input('date-mode-toggle', 'value'),
     Input('metric-dropdown', 'value'),
     Input('theme-store', 'data')]
)
def update_regional_boxplot(selected_regions, selected_countries, start_date, end_date, 
                           single_date, date_mode, selected_metric, theme_data):
    """Update regional box plot visualization"""
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Theme settings
    theme = theme_data.get('theme', 'light')
    if theme == 'dark':
        plot_bg = '#2c3e50'
        paper_bg = '#34495e'
        font_color = 'white'
        grid_color = '#54616e'
    else:
        plot_bg = 'white'
        paper_bg = 'white'
        font_color = 'black'
        grid_color = '#e5e5e5'
    
    try:
        # Apply filtering
        filtered_df = df.copy()
        
        if selected_regions:
            filtered_df = filtered_df[filtered_df['geographic_region'].isin(selected_regions)]
        if selected_countries:
            filtered_df = filtered_df[filtered_df['normalized_country'].isin(selected_countries)]
        
        # Date filtering
        if date_mode == "single" and single_date:
            filtered_df = filtered_df[filtered_df['date'] == pd.to_datetime(single_date).date()]
        elif date_mode == "range" and start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.to_datetime(start_date).date()) &
                (filtered_df['date'] <= pd.to_datetime(end_date).date())
            ]
        
        if filtered_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data matches current filters", xref="paper", yref="paper", 
                              x=0.5, y=0.5, showarrow=False, font_color=font_color)
            fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
            return fig
        
        # Create box plot by region
        fig = px.box(
            filtered_df, 
            x='geographic_region', 
            y=selected_metric,
            title=f"{metric_labels[selected_metric]} Distribution by Region",
            color='geographic_region',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Update layout for theme
        fig.update_layout(
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            font_color=font_color,
            title_font_color=font_color,
            height=400,
            margin=dict(l=0, r=0, t=50, b=100),
            showlegend=False,
            xaxis=dict(
                title="Geographic Region",
                tickangle=45,
                gridcolor=grid_color,
                color=font_color
            ),
            yaxis=dict(
                title=metric_labels[selected_metric],
                gridcolor=grid_color,
                color=font_color
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating box plot: {str(e)}", xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False, font_color=font_color)
        fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
        return fig

# Callback for Climate Radar Chart
@app.callback(
    Output('climate-radar-chart', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('country-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('single-date-picker', 'date'),
     Input('date-mode-toggle', 'value'),
     Input('theme-store', 'data')]
)
def update_climate_radar_chart(selected_regions, selected_countries, start_date, end_date, 
                              single_date, date_mode, theme_data):
    """Update climate radar chart visualization"""
    
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Theme settings
    theme = theme_data.get('theme', 'light')
    if theme == 'dark':
        plot_bg = '#2c3e50'
        paper_bg = '#34495e'
        font_color = 'white'
        grid_color = '#54616e'
    else:
        plot_bg = 'white'
        paper_bg = 'white'
        font_color = 'black'
        grid_color = '#e5e5e5'
    
    try:
        # Apply filtering
        filtered_df = df.copy()
        
        if selected_regions:
            filtered_df = filtered_df[filtered_df['geographic_region'].isin(selected_regions)]
        if selected_countries:
            filtered_df = filtered_df[filtered_df['normalized_country'].isin(selected_countries)]
        
        # Date filtering
        if date_mode == "single" and single_date:
            filtered_df = filtered_df[filtered_df['date'] == pd.to_datetime(single_date).date()]
        elif date_mode == "range" and start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.to_datetime(start_date).date()) &
                (filtered_df['date'] <= pd.to_datetime(end_date).date())
            ]
        
        # Use selected_countries for radar chart, limit to 5 countries for readability
        if selected_countries and len(selected_countries) > 0:
            final_countries = selected_countries[:5]  # Limit to 5 countries
            final_countries = [c for c in final_countries if c in filtered_df['normalized_country'].unique()]
        else:
            # Use first 3 countries from filtered data
            final_countries = filtered_df['normalized_country'].unique()[:3].tolist()
        
        if len(final_countries) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No countries selected or available", xref="paper", yref="paper", 
                              x=0.5, y=0.5, showarrow=False, font_color=font_color)
            fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
            return fig
        
        # Filter data to selected countries
        radar_df = filtered_df[filtered_df['normalized_country'].isin(final_countries)]
        
        if radar_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data matches current filters", xref="paper", yref="paper", 
                              x=0.5, y=0.5, showarrow=False, font_color=font_color)
            fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
            return fig
        
        # Aggregate data by country
        required_columns = ['temperature_celsius', 'humidity', 'wind_kph', 'uv_index', 'pressure_mb']
        country_data = radar_df.groupby('normalized_country')[required_columns].mean()
        
        # Ensure we have data for all countries
        if country_data.empty:
            fig = go.Figure()
            fig.add_annotation(text="Insufficient data for radar chart", xref="paper", yref="paper", 
                              x=0.5, y=0.5, showarrow=False, font_color=font_color)
            fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
            return fig
        
        # Normalize values to 0-100 scale for radar chart
        normalized_data = pd.DataFrame(index=country_data.index)
        
        for col in required_columns:
            if col in country_data.columns:
                col_data = country_data[col]
                min_val = col_data.min()
                max_val = col_data.max()
                
                if max_val > min_val and not pd.isna(min_val) and not pd.isna(max_val):
                    normalized_data[col] = ((col_data - min_val) / (max_val - min_val)) * 100
                else:
                    normalized_data[col] = 50  # Default middle value if no variation
            else:
                normalized_data[col] = 50
        
        # Create radar chart
        fig = go.Figure()
        
        categories = ['Temperature', 'Humidity', 'Wind Speed', 'UV Index', 'Pressure']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for i, (country, values) in enumerate(normalized_data.iterrows()):
            color = colors[i % len(colors)]
            
            # Convert hex color to RGB for fillcolor
            hex_color = color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Prepare data for radar (close the shape)
            radar_values = values.tolist()
            radar_categories = categories.copy()
            
            # Close the radar shape
            radar_values.append(radar_values[0])
            radar_categories.append(radar_categories[0])
            
            fig.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=radar_categories,
                fill='toself',
                name=country,
                line=dict(color=color, width=2),
                fillcolor=f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.3)',
                marker=dict(size=6, color=color)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=grid_color,
                    color=font_color,
                    tickfont=dict(color=font_color, size=10)
                ),
                angularaxis=dict(
                    gridcolor=grid_color,
                    color=font_color,
                    tickfont=dict(color=font_color, size=10)
                ),
                bgcolor=plot_bg
            ),
            showlegend=True,
            title=dict(
                text="Climate Profile Comparison",
                font=dict(color=font_color, size=14),
                x=0.5
            ),
            font_color=font_color,
            plot_bgcolor=plot_bg,
            paper_bgcolor=paper_bg,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                font=dict(color=font_color),
                bgcolor='rgba(255,255,255,0.1)' if theme == 'dark' else 'rgba(0,0,0,0.1)',
                bordercolor=grid_color,
                borderwidth=1
            )
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating radar chart: {str(e)}", xref="paper", yref="paper", 
                          x=0.5, y=0.5, showarrow=False, font_color=font_color)
        fig.update_layout(plot_bgcolor=plot_bg, paper_bgcolor=paper_bg)
        return fig

# Report export callback function

@app.callback(
    Output("download-report", "data"),
    Input("btn-report", "n_clicks"),
    [State('region-dropdown', 'value'),
     State('country-dropdown', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('single-date-picker', 'date'),
     State('date-mode-toggle', 'value')],
    prevent_initial_call=True
)
def export_report(n_clicks, selected_regions, selected_countries, start_date, end_date, single_date, date_mode):
    """Export comprehensive markdown report"""
    if n_clicks:
        # Apply same filtering logic
        filtered_df = df.copy()
        
        if selected_regions:
            filtered_df = filtered_df[filtered_df['geographic_region'].isin(selected_regions)]
        if selected_countries:
            filtered_df = filtered_df[filtered_df['normalized_country'].isin(selected_countries)]
        
        # Date filtering
        if date_mode == "single" and single_date:
            filtered_df = filtered_df[filtered_df['date'] == pd.to_datetime(single_date).date()]
        elif date_mode == "range" and start_date and end_date:
            filtered_df = filtered_df[
                (filtered_df['date'] >= pd.to_datetime(start_date).date()) &
                (filtered_df['date'] <= pd.to_datetime(end_date).date())
            ]
        
        # Generate comprehensive report
        report = generate_comprehensive_report(filtered_df, selected_regions, selected_countries, date_mode)
        
        return dict(
            content=report,
            filename=f"climatescope_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )

def generate_comprehensive_report(df, selected_regions, selected_countries, date_mode):
    """Generate a comprehensive climate report"""
    report = f"""# 🌍 ClimateScope Dashboard - Comprehensive Analysis Report

## 📊 Executive Summary

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Export Mode**: Filtered Analysis

### Applied Filters
- **Regions**: {', '.join(selected_regions) if selected_regions else 'All Regions'}
- **Countries**: {', '.join(selected_countries[:5]) if selected_countries else 'All Countries'}{' (and more...)' if selected_countries and len(selected_countries) > 5 else ''}
- **Date Mode**: {date_mode.title()}

### Dataset Overview
- **Total Locations**: {len(df):,}
- **Countries Covered**: {df['normalized_country'].nunique() if 'normalized_country' in df.columns and len(df) > 0 else 'N/A'}
- **Regions Covered**: {df['geographic_region'].nunique() if 'geographic_region' in df.columns and len(df) > 0 else 'N/A'}
- **Date Range**: {df['date'].min()} to {df['date'].max() if 'date' in df.columns and len(df) > 0 else 'N/A'}

## 🌡️ Climate Analysis

### Temperature Statistics
"""
    
    if len(df) > 0:
        report += f"""
- **Global Average**: {df['temperature_celsius'].mean():.1f}°C
- **Temperature Range**: {df['temperature_celsius'].min():.1f}°C to {df['temperature_celsius'].max():.1f}°C
- **Standard Deviation**: {df['temperature_celsius'].std():.1f}°C

### Humidity Analysis
- **Average Humidity**: {df['humidity'].mean():.1f}%
- **Humidity Range**: {df['humidity'].min():.1f}% to {df['humidity'].max():.1f}%

### Wind Patterns
- **Average Wind Speed**: {df['wind_kph'].mean():.1f} km/h
- **Maximum Wind Speed**: {df['wind_kph'].max():.1f} km/h

### UV Index
- **Average UV Index**: {df['uv_index'].mean():.1f}
- **Maximum UV Index**: {df['uv_index'].max():.1f}

## 🏆 Notable Locations

### Climate Extremes
- **Hottest Location**: {df.loc[df['temperature_celsius'].idxmax(), 'location_name']}, {df.loc[df['temperature_celsius'].idxmax(), 'normalized_country']} ({df['temperature_celsius'].max():.1f}°C)
- **Coldest Location**: {df.loc[df['temperature_celsius'].idxmin(), 'location_name']}, {df.loc[df['temperature_celsius'].idxmin(), 'normalized_country']} ({df['temperature_celsius'].min():.1f}°C)
- **Most Humid**: {df.loc[df['humidity'].idxmax(), 'location_name']}, {df.loc[df['humidity'].idxmax(), 'normalized_country']} ({df['humidity'].max():.1f}%)
- **Windiest**: {df.loc[df['wind_kph'].idxmax(), 'location_name']}, {df.loc[df['wind_kph'].idxmax(), 'normalized_country']} ({df['wind_kph'].max():.1f} km/h)

## 📈 Distribution Analysis

### Temperature Distribution
"""
        
        # Temperature distribution
        hot_locations = len(df[df['temperature_celsius'] > 30])
        cold_locations = len(df[df['temperature_celsius'] < 10])
        moderate_locations = len(df[(df['temperature_celsius'] >= 10) & (df['temperature_celsius'] <= 30)])
        
        report += f"""
- **Hot Locations (>30°C)**: {hot_locations} ({hot_locations/len(df)*100:.1f}%)
- **Cold Locations (<10°C)**: {cold_locations} ({cold_locations/len(df)*100:.1f}%)
- **Moderate Locations (10-30°C)**: {moderate_locations} ({moderate_locations/len(df)*100:.1f}%)

### Humidity Patterns
"""
        
        # Humidity analysis
        high_humidity = len(df[df['humidity'] > 70])
        low_humidity = len(df[df['humidity'] < 40])
        
        report += f"""
- **High Humidity (>70%)**: {high_humidity} ({high_humidity/len(df)*100:.1f}%)
- **Low Humidity (<40%)**: {low_humidity} ({low_humidity/len(df)*100:.1f}%)

### Wind Analysis
"""
        
        # Wind analysis
        windy_locations = len(df[df['wind_kph'] > 20])
        calm_locations = len(df[df['wind_kph'] < 10])
        
        report += f"""
- **Windy Locations (>20 km/h)**: {windy_locations} ({windy_locations/len(df)*100:.1f}%)
- **Calm Locations (<10 km/h)**: {calm_locations} ({calm_locations/len(df)*100:.1f}%)

## 🌍 Regional Breakdown
"""
        
        # Regional analysis
        if 'geographic_region' in df.columns:
            regional_temps = df.groupby('geographic_region')['temperature_celsius'].mean().sort_values(ascending=False)
            report += "\n### Average Temperature by Region\n"
            for region, temp in regional_temps.head(10).items():
                report += f"- **{region}**: {temp:.1f}°C\n"
            
            regional_humidity = df.groupby('geographic_region')['humidity'].mean().sort_values(ascending=False)
            report += "\n### Average Humidity by Region\n"
            for region, humidity in regional_humidity.head(5).items():
                report += f"- **{region}**: {humidity:.1f}%\n"
        
        report += f"""

## 🔍 Data Quality Assessment

- **Complete Temperature Records**: {df['temperature_celsius'].notna().sum():,} ({df['temperature_celsius'].notna().sum()/len(df)*100:.1f}%)
- **Complete Humidity Records**: {df['humidity'].notna().sum():,} ({df['humidity'].notna().sum()/len(df)*100:.1f}%)
- **Complete Wind Records**: {df['wind_kph'].notna().sum():,} ({df['wind_kph'].notna().sum()/len(df)*100:.1f}%)

## 💡 Key Insights

1. **Climate Diversity**: The filtered dataset shows a temperature range of {df['temperature_celsius'].max() - df['temperature_celsius'].min():.1f}°C, indicating significant climate diversity.

2. **Comfort Zones**: {moderate_locations} locations ({moderate_locations/len(df)*100:.1f}%) fall within the moderate temperature range (10-30°C).

3. **Extreme Conditions**: {hot_locations + cold_locations} locations ({(hot_locations + cold_locations)/len(df)*100:.1f}%) experience extreme temperatures.

4. **Humidity Patterns**: {high_humidity} locations have high humidity, which may affect comfort and weather patterns.

## 📊 Statistical Summary

| Metric | Mean | Min | Max | Std Dev |
|--------|------|-----|-----|---------|
| Temperature (°C) | {df['temperature_celsius'].mean():.1f} | {df['temperature_celsius'].min():.1f} | {df['temperature_celsius'].max():.1f} | {df['temperature_celsius'].std():.1f} |
| Humidity (%) | {df['humidity'].mean():.1f} | {df['humidity'].min():.1f} | {df['humidity'].max():.1f} | {df['humidity'].std():.1f} |
| Wind Speed (km/h) | {df['wind_kph'].mean():.1f} | {df['wind_kph'].min():.1f} | {df['wind_kph'].max():.1f} | {df['wind_kph'].std():.1f} |
| UV Index | {df['uv_index'].mean():.1f} | {df['uv_index'].min():.1f} | {df['uv_index'].max():.1f} | {df['uv_index'].std():.1f} |

"""
    else:
        report += "\n**No data available for analysis with current filters.**\n"
    
    report += f"""
## 🔬 Methodology

This report was generated from the ClimateScope Dashboard using filtered weather data based on user selections:

1. **Data Processing**: Applied regional and country filters as specified
2. **Statistical Analysis**: Calculated descriptive statistics for all metrics
3. **Extreme Value Analysis**: Identified locations with maximum and minimum values
4. **Distribution Analysis**: Categorized locations by climate characteristics
5. **Regional Comparison**: Analyzed average values by geographic region

## 📝 Recommendations

Based on the current analysis:

1. **Travel Planning**: Consider the moderate climate locations for comfortable travel experiences
2. **Climate Monitoring**: Monitor locations with extreme temperatures for potential weather events
3. **Health Considerations**: Be aware of high UV index locations requiring sun protection
4. **Regional Insights**: Use regional averages to understand broader climate patterns

---

*Report generated by ClimateScope Dashboard - Advanced Weather Analytics Platform*  
*Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Dashboard Version: Enhanced with Export Features*
"""
    
    return report

# Run the app
if __name__ == '__main__':
    print("🌍 Starting Enhanced ClimateScope Dashboard...")
    print(f"📊 Loaded data with {len(df)} records from {len(df['normalized_country'].unique()) if not df.empty else 0} countries")
    print("🚀 Enhanced features: Comprehensive report generation")
    print("🚀 Access the dashboard at: http://127.0.0.1:8062")
    app.run(debug=True, host='127.0.0.1', port=8062)
