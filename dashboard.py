import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ClimateScope Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sector-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2a5298;
        border-bottom: 2px solid #2a5298;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        border: 2px solid #2a5298;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .insight-text {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #333;
        margin: 0.5rem 0;
    }
    .insight-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2a5298;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the weather data"""
    data_path = Path('data/raw/GlobalWeatherRepository.csv')
    if data_path.exists():
        df = pd.read_csv(data_path)
        # Data preprocessing
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df['continent'] = df['country'].map(get_continent_mapping())
        return df
    else:
        st.error("Data file not found. Please ensure GlobalWeatherRepository.csv exists in data/raw/")
        return pd.DataFrame()

def get_continent_mapping():
    """Map countries to continents"""
    continent_map = {
        'Afghanistan': 'Asia', 'Albania': 'Europe', 'Algeria': 'Africa', 'Andorra': 'Europe',
        'Angola': 'Africa', 'Antigua and Barbuda': 'North America', 'Argentina': 'South America',
        'Armenia': 'Asia', 'Australia': 'Oceania', 'Austria': 'Europe', 'Azerbaijan': 'Asia',
        'Bahamas': 'North America', 'Bahrain': 'Asia', 'Bangladesh': 'Asia', 'Barbados': 'North America',
        'Belarus': 'Europe', 'Belgium': 'Europe', 'Belize': 'North America', 'Benin': 'Africa',
        'Bhutan': 'Asia', 'Bolivia': 'South America', 'Bosnia and Herzegovina': 'Europe',
        'Botswana': 'Africa', 'Brazil': 'South America', 'Brunei': 'Asia', 'Bulgaria': 'Europe',
        'Burkina Faso': 'Africa', 'Burundi': 'Africa', 'Cambodia': 'Asia', 'Cameroon': 'Africa',
        'Canada': 'North America', 'Cape Verde': 'Africa', 'Central African Republic': 'Africa',
        'Chad': 'Africa', 'Chile': 'South America', 'China': 'Asia', 'Colombia': 'South America',
        'Comoros': 'Africa', 'Congo': 'Africa', 'Costa Rica': 'North America', 'Croatia': 'Europe',
        'Cuba': 'North America', 'Cyprus': 'Europe', 'Czech Republic': 'Europe', 'Denmark': 'Europe',
        'Djibouti': 'Africa', 'Dominica': 'North America', 'Dominican Republic': 'North America',
        'Ecuador': 'South America', 'Egypt': 'Africa', 'El Salvador': 'North America',
        'Equatorial Guinea': 'Africa', 'Eritrea': 'Africa', 'Estonia': 'Europe', 'Ethiopia': 'Africa',
        'Fiji': 'Oceania', 'Finland': 'Europe', 'France': 'Europe', 'Gabon': 'Africa',
        'Gambia': 'Africa', 'Georgia': 'Asia', 'Germany': 'Europe', 'Ghana': 'Africa',
        'Greece': 'Europe', 'Grenada': 'North America', 'Guatemala': 'North America',
        'Guinea': 'Africa', 'Guinea-Bissau': 'Africa', 'Guyana': 'South America', 'Haiti': 'North America',
        'Honduras': 'North America', 'Hungary': 'Europe', 'Iceland': 'Europe', 'India': 'Asia',
        'Indonesia': 'Asia', 'Iran': 'Asia', 'Iraq': 'Asia', 'Ireland': 'Europe', 'Israel': 'Asia',
        'Italy': 'Europe', 'Jamaica': 'North America', 'Japan': 'Asia', 'Jordan': 'Asia',
        'Kazakhstan': 'Asia', 'Kenya': 'Africa', 'Kiribati': 'Oceania', 'Kuwait': 'Asia',
        'Kyrgyzstan': 'Asia', 'Laos': 'Asia', 'Latvia': 'Europe', 'Lebanon': 'Asia',
        'Lesotho': 'Africa', 'Liberia': 'Africa', 'Libya': 'Africa', 'Liechtenstein': 'Europe',
        'Lithuania': 'Europe', 'Luxembourg': 'Europe', 'Madagascar': 'Africa', 'Malawi': 'Africa',
        'Malaysia': 'Asia', 'Maldives': 'Asia', 'Mali': 'Africa', 'Malta': 'Europe',
        'Marshall Islands': 'Oceania', 'Mauritania': 'Africa', 'Mauritius': 'Africa',
        'Mexico': 'North America', 'Micronesia': 'Oceania', 'Moldova': 'Europe', 'Monaco': 'Europe',
        'Mongolia': 'Asia', 'Montenegro': 'Europe', 'Morocco': 'Africa', 'Mozambique': 'Africa',
        'Myanmar': 'Asia', 'Namibia': 'Africa', 'Nauru': 'Oceania', 'Nepal': 'Asia',
        'Netherlands': 'Europe', 'New Zealand': 'Oceania', 'Nicaragua': 'North America',
        'Niger': 'Africa', 'Nigeria': 'Africa', 'North Korea': 'Asia', 'North Macedonia': 'Europe',
        'Norway': 'Europe', 'Oman': 'Asia', 'Pakistan': 'Asia', 'Palau': 'Oceania',
        'Palestine': 'Asia', 'Panama': 'North America', 'Papua New Guinea': 'Oceania',
        'Paraguay': 'South America', 'Peru': 'South America', 'Philippines': 'Asia',
        'Poland': 'Europe', 'Portugal': 'Europe', 'Qatar': 'Asia', 'Romania': 'Europe',
        'Russia': 'Europe', 'Rwanda': 'Africa', 'Saint Kitts and Nevis': 'North America',
        'Saint Lucia': 'North America', 'Saint Vincent and the Grenadines': 'North America',
        'Samoa': 'Oceania', 'San Marino': 'Europe', 'Sao Tome and Principe': 'Africa',
        'Saudi Arabia': 'Asia', 'Senegal': 'Africa', 'Serbia': 'Europe', 'Seychelles': 'Africa',
        'Sierra Leone': 'Africa', 'Singapore': 'Asia', 'Slovakia': 'Europe', 'Slovenia': 'Europe',
        'Solomon Islands': 'Oceania', 'Somalia': 'Africa', 'South Africa': 'Africa',
        'South Korea': 'Asia', 'South Sudan': 'Africa', 'Spain': 'Europe', 'Sri Lanka': 'Asia',
        'Sudan': 'Africa', 'Suriname': 'South America', 'Sweden': 'Europe', 'Switzerland': 'Europe',
        'Syria': 'Asia', 'Taiwan': 'Asia', 'Tajikistan': 'Asia', 'Tanzania': 'Africa',
        'Thailand': 'Asia', 'Timor-Leste': 'Asia', 'Togo': 'Africa', 'Tonga': 'Oceania',
        'Trinidad and Tobago': 'North America', 'Tunisia': 'Africa', 'Turkey': 'Asia',
        'Turkmenistan': 'Asia', 'Tuvalu': 'Oceania', 'Uganda': 'Africa', 'Ukraine': 'Europe',
        'United Arab Emirates': 'Asia', 'United Kingdom': 'Europe', 'United States': 'North America',
        'Uruguay': 'South America', 'Uzbekistan': 'Asia', 'Vanuatu': 'Oceania',
        'Vatican City': 'Europe', 'Venezuela': 'South America', 'Vietnam': 'Asia',
        'Yemen': 'Asia', 'Zambia': 'Africa', 'Zimbabwe': 'Africa'
    }
    return continent_map

def create_global_overview(df):
    """Create global overview visualizations"""
    st.markdown('<div class="sector-header">🌍 Global Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_temp = df['temperature_celsius'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>🌡️ Global Avg Temp</h3>
            <h2>{avg_temp:.1f}°C</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_temp = df['temperature_celsius'].max()
        hottest_location = df.loc[df['temperature_celsius'].idxmax(), 'location_name']
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔥 Hottest Location</h3>
            <h2>{max_temp:.1f}°C</h2>
            <p>{hottest_location}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        min_temp = df['temperature_celsius'].min()
        coldest_location = df.loc[df['temperature_celsius'].idxmin(), 'location_name']
        st.markdown(f"""
        <div class="metric-card">
            <h3>❄️ Coldest Location</h3>
            <h2>{min_temp:.1f}°C</h2>
            <p>{coldest_location}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_humidity = df['humidity'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>💧 Global Avg Humidity</h3>
            <h2>{avg_humidity:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # World map
    fig_map = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='temperature_celsius',
        size='humidity',
        hover_name='location_name',
        hover_data=['country', 'condition_text', 'wind_kph'],
        color_continuous_scale='RdYlBu_r',
        title='Global Temperature Distribution',
        height=500
    )
    
    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        title_font_size=16,
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

def create_regional_analysis(df):
    """Create regional analysis visualizations"""
    st.markdown('<div class="sector-header">🌏 Regional Analysis</div>', unsafe_allow_html=True)
    
    # Regional temperature comparison
    regional_stats = df.groupby('continent').agg({
        'temperature_celsius': ['mean', 'min', 'max'],
        'humidity': 'mean',
        'pressure_mb': 'mean',
        'wind_kph': 'mean'
    }).round(2)
    
    regional_stats.columns = ['Avg_Temp', 'Min_Temp', 'Max_Temp', 'Avg_Humidity', 'Avg_Pressure', 'Avg_Wind']
    regional_stats = regional_stats.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_temp = px.bar(
            regional_stats,
            x='continent',
            y='Avg_Temp',
            color='Avg_Temp',
            color_continuous_scale='RdYlBu_r',
            title='Average Temperature by Continent',
            labels={'Avg_Temp': 'Temperature (°C)', 'continent': 'Continent'}
        )
        fig_temp.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        fig_humidity = px.bar(
            regional_stats,
            x='continent',
            y='Avg_Humidity',
            color='Avg_Humidity',
            color_continuous_scale='Blues',
            title='Average Humidity by Continent',
            labels={'Avg_Humidity': 'Humidity (%)', 'continent': 'Continent'}
        )
        fig_humidity.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_humidity, use_container_width=True)
    
    # Temperature range by continent
    fig_range = go.Figure()
    
    for continent in regional_stats['continent']:
        continent_data = df[df['continent'] == continent]
        fig_range.add_trace(go.Box(
            y=continent_data['temperature_celsius'],
            name=continent,
            boxpoints='outliers'
        ))
    
    fig_range.update_layout(
        title='Temperature Distribution by Continent',
        yaxis_title='Temperature (°C)',
        xaxis_title='Continent',
        height=400
    )
    
    st.plotly_chart(fig_range, use_container_width=True)

def create_weather_patterns(df):
    """Create weather patterns visualizations"""
    st.markdown('<div class="sector-header">🌤️ Weather Patterns</div>', unsafe_allow_html=True)
    
    # Weather condition distribution
    condition_counts = df['condition_text'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_conditions = px.pie(
            values=condition_counts.values,
            names=condition_counts.index,
            title='Top 10 Weather Conditions Globally',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_conditions.update_traces(textposition='inside', textinfo='percent+label')
        fig_conditions.update_layout(height=400)
        st.plotly_chart(fig_conditions, use_container_width=True)
    
    with col2:
        # Temperature vs Humidity scatter
        fig_scatter = px.scatter(
            df.sample(1000),  # Sample for performance
            x='temperature_celsius',
            y='humidity',
            color='continent',
            size='wind_kph',
            hover_data=['location_name', 'condition_text'],
            title='Temperature vs Humidity Relationship',
            labels={'temperature_celsius': 'Temperature (°C)', 'humidity': 'Humidity (%)'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation heatmap
    numeric_cols = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 'visibility_km', 'uv_index']
    correlation_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Weather Variables Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

def create_air_quality_monitor(df):
    """Create air quality monitoring visualizations"""
    st.markdown('<div class="sector-header">🏭 Air Quality Monitor</div>', unsafe_allow_html=True)
    
    # Air quality metrics
    air_quality_cols = [
        'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
        'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10'
    ]
    
    # Check if air quality data exists
    if all(col in df.columns for col in air_quality_cols):
        col1, col2 = st.columns(2)
        
        with col1:
            # Average air quality by continent
            aq_by_continent = df.groupby('continent')[air_quality_cols].mean()
            
            fig_aq = px.bar(
                aq_by_continent.reset_index().melt(id_vars='continent'),
                x='continent',
                y='value',
                color='variable',
                title='Average Air Quality Pollutants by Continent',
                labels={'value': 'Concentration', 'variable': 'Pollutant'}
            )
            fig_aq.update_layout(height=400)
            st.plotly_chart(fig_aq, use_container_width=True)
        
        with col2:
            # PM2.5 vs Temperature
            fig_pm = px.scatter(
                df.sample(1000),
                x='temperature_celsius',
                y='air_quality_PM2.5',
                color='continent',
                title='PM2.5 Levels vs Temperature',
                labels={'temperature_celsius': 'Temperature (°C)', 'air_quality_PM2.5': 'PM2.5 (μg/m³)'}
            )
            fig_pm.update_layout(height=400)
            st.plotly_chart(fig_pm, use_container_width=True)
        
        # Air quality index distribution
        if 'air_quality_us-epa-index' in df.columns:
            epa_counts = df['air_quality_us-epa-index'].value_counts().sort_index()
            
            fig_epa = px.bar(
                x=epa_counts.index,
                y=epa_counts.values,
                title='US EPA Air Quality Index Distribution',
                labels={'x': 'EPA Index', 'y': 'Number of Locations'},
                color=epa_counts.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig_epa.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_epa, use_container_width=True)
    else:
        st.info("Air quality data not available in the current dataset.")

def create_climate_insights(df):
    """Create climate insights and notable findings"""
    st.markdown('<div class="sector-header">🔍 Climate Insights</div>', unsafe_allow_html=True)
    
    # Notable findings
    insights = []
    
    # Temperature extremes
    hottest = df.loc[df['temperature_celsius'].idxmax()]
    coldest = df.loc[df['temperature_celsius'].idxmin()]
    temp_range = df['temperature_celsius'].max() - df['temperature_celsius'].min()
    
    insights.append({
        'title': '🌡️ Temperature Extremes',
        'content': f"Global temperature varies by {temp_range:.1f}°C, from {coldest['temperature_celsius']:.1f}°C in {coldest['location_name']}, {coldest['country']} to {hottest['temperature_celsius']:.1f}°C in {hottest['location_name']}, {hottest['country']}"
    })
    
    # Humidity patterns
    high_humidity = df[df['humidity'] > 80]
    low_humidity = df[df['humidity'] < 30]
    
    insights.append({
        'title': '💧 Humidity Distribution',
        'content': f"{len(high_humidity)} locations have humidity >80% (very humid), while {len(low_humidity)} locations have humidity <30% (very dry). Average global humidity is {df['humidity'].mean():.1f}%"
    })
    
    # Wind patterns
    windiest = df.loc[df['wind_kph'].idxmax()]
    avg_wind_by_continent = df.groupby('continent')['wind_kph'].mean().sort_values(ascending=False)
    
    insights.append({
        'title': '💨 Wind Analysis',
        'content': f"Windiest location is {windiest['location_name']}, {windiest['country']} with {windiest['wind_kph']:.1f} km/h. {avg_wind_by_continent.index[0]} has the highest average wind speeds ({avg_wind_by_continent.iloc[0]:.1f} km/h)"
    })
    
    # Weather conditions
    most_common_condition = df['condition_text'].mode()[0]
    condition_percentage = (df['condition_text'] == most_common_condition).mean() * 100
    unique_conditions = df['condition_text'].nunique()
    
    insights.append({
        'title': '☁️ Weather Patterns',
        'content': f"'{most_common_condition}' is the most common weather condition, occurring in {condition_percentage:.1f}% of locations. Total of {unique_conditions} different weather conditions observed globally"
    })
    
    # Pressure analysis
    high_pressure = df[df['pressure_mb'] > 1020]
    low_pressure = df[df['pressure_mb'] < 1000]
    
    insights.append({
        'title': '📊 Atmospheric Pressure',
        'content': f"{len(high_pressure)} locations have high pressure (>1020 mb) indicating stable weather, while {len(low_pressure)} locations have low pressure (<1000 mb) suggesting potential storms"
    })
    
    # UV Index
    if 'uv_index' in df.columns:
        high_uv = df[df['uv_index'] > 7]
        moderate_uv = df[(df['uv_index'] >= 3) & (df['uv_index'] <= 7)]
        insights.append({
            'title': '☀️ UV Exposure Levels',
            'content': f"{len(high_uv)} locations have high UV index (>7) requiring sun protection, {len(moderate_uv)} have moderate UV levels (3-7)"
        })
    
    # Display insights in a more structured way
    col1, col2 = st.columns(2)
    
    for i, insight in enumerate(insights):
        if i % 2 == 0:
            with col1:
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-text">{insight['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(f"""
                <div class="insight-box">
                    <div class="insight-title">{insight['title']}</div>
                    <div class="insight-text">{insight['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Regional comparison chart
    regional_comparison = df.groupby('continent').agg({
        'temperature_celsius': 'mean',
        'humidity': 'mean',
        'wind_kph': 'mean',
        'pressure_mb': 'mean'
    }).round(2)
    
    fig_radar = go.Figure()
    
    for continent in regional_comparison.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=[
                regional_comparison.loc[continent, 'temperature_celsius'],
                regional_comparison.loc[continent, 'humidity'],
                regional_comparison.loc[continent, 'wind_kph'],
                regional_comparison.loc[continent, 'pressure_mb'] / 10  # Scale for visibility
            ],
            theta=['Temperature', 'Humidity', 'Wind Speed', 'Pressure (x10)'],
            fill='toself',
            name=continent
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title="Regional Climate Characteristics Comparison",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<div class="main-header">🌍 ClimateScope Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Semi-Real-Time Weather Analysis & Interactive Visualizations")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.stop()
    
    # Sidebar filters
    st.sidebar.header("🎛️ Interactive Filters")
    
    # Continent filter
    continents = ['All'] + sorted(df['continent'].dropna().unique().tolist())
    selected_continent = st.sidebar.selectbox("Select Continent", continents)
    
    # Temperature range filter
    temp_range = st.sidebar.slider(
        "Temperature Range (°C)",
        float(df['temperature_celsius'].min()),
        float(df['temperature_celsius'].max()),
        (float(df['temperature_celsius'].min()), float(df['temperature_celsius'].max()))
    )
    
    # Humidity range filter
    humidity_range = st.sidebar.slider(
        "Humidity Range (%)",
        int(df['humidity'].min()),
        int(df['humidity'].max()),
        (int(df['humidity'].min()), int(df['humidity'].max()))
    )
    
    # Weather condition filter
    conditions = ['All'] + sorted(df['condition_text'].unique().tolist())
    selected_condition = st.sidebar.selectbox("Weather Condition", conditions)
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_continent != 'All':
        filtered_df = filtered_df[filtered_df['continent'] == selected_continent]
    
    filtered_df = filtered_df[
        (filtered_df['temperature_celsius'] >= temp_range[0]) &
        (filtered_df['temperature_celsius'] <= temp_range[1]) &
        (filtered_df['humidity'] >= humidity_range[0]) &
        (filtered_df['humidity'] <= humidity_range[1])
    ]
    
    if selected_condition != 'All':
        filtered_df = filtered_df[filtered_df['condition_text'] == selected_condition]
    
    # Show filtered data info
    st.sidebar.markdown(f"**Filtered Data**: {len(filtered_df):,} locations")
    
    # Dashboard sections
    if len(filtered_df) > 0:
        create_global_overview(filtered_df)
        st.markdown("---")
        
        create_regional_analysis(filtered_df)
        st.markdown("---")
        
        create_weather_patterns(filtered_df)
        st.markdown("---")
        
        create_air_quality_monitor(filtered_df)
        st.markdown("---")
        
        create_climate_insights(filtered_df)
    else:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
    
    # Footer
    st.markdown("---")

if __name__ == "__main__":
    main()
