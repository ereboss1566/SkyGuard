import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the enhanced storm features data
file_path = 'data/enhanced/enhanced_storm_features.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Select key numeric columns for correlation analysis (excluding location data)
key_features = [
    'temperature_celsius', 'dew_point', 'pressure_mb', 'wind_speed', 'wind_gust',
    'humidity', 'cloud', 'visibility_km', 'uv_index', 'precip_mm',
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
    'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10',
    'reflectivity_max', 'reflectivity_mean', 'brightness_temp_min'
]

print("\nKey features for analysis:")
print(key_features)

# Filter to only include rows with at least some key features
df_filtered = df.dropna(subset=key_features, how='all')
print(f"\nDataset shape after filtering: {df_filtered.shape}")

# Calculate correlation matrix for key features only
key_features_available = [col for col in key_features if col in df_filtered.columns and df_filtered[col].notna().sum() > 0]
print(f"\nAvailable key features: {key_features_available}")

if len(key_features_available) > 1:
    corr_matrix = df_filtered[key_features_available].corr()
    
    # Create visualizations
    plt.figure(figsize=(20, 15))
    
    # 1. Correlation heatmap
    plt.subplot(2, 3, 1)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Storm Features')
    
    # 2. Temperature vs Pressure (if both available)
    if 'temperature_celsius' in key_features_available and 'pressure_mb' in key_features_available:
        plt.subplot(2, 3, 2)
        # Remove rows with NaN values for these specific columns
        temp_press_data = df_filtered[['temperature_celsius', 'pressure_mb']].dropna()
        plt.scatter(temp_press_data['temperature_celsius'], temp_press_data['pressure_mb'], alpha=0.6)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Pressure (mb)')
        plt.title('Temperature vs Pressure')
    
    # 3. Humidity vs Visibility (if both available)
    if 'humidity' in key_features_available and 'visibility_km' in key_features_available:
        plt.subplot(2, 3, 3)
        humid_vis_data = df_filtered[['humidity', 'visibility_km']].dropna()
        plt.scatter(humid_vis_data['humidity'], humid_vis_data['visibility_km'], alpha=0.6)
        plt.xlabel('Humidity (%)')
        plt.ylabel('Visibility (km)')
        plt.title('Humidity vs Visibility')
    
    # 4. Wind Speed vs Wind Gust (if both available)
    if 'wind_speed' in key_features_available and 'wind_gust' in key_features_available:
        plt.subplot(2, 3, 4)
        wind_data = df_filtered[['wind_speed', 'wind_gust']].dropna()
        plt.scatter(wind_data['wind_speed'], wind_data['wind_gust'], alpha=0.6)
        plt.xlabel('Wind Speed')
        plt.ylabel('Wind Gust')
        plt.title('Wind Speed vs Wind Gust')
    
    # 5. Distribution of temperature
    if 'temperature_celsius' in key_features_available:
        plt.subplot(2, 3, 5)
        temp_data = df_filtered['temperature_celsius'].dropna()
        plt.hist(temp_data, bins=20, alpha=0.7)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Frequency')
        plt.title('Temperature Distribution')
    
    # 6. Time series of temperature (last 50 data points)
    if 'temperature_celsius' in key_features_available:
        plt.subplot(2, 3, 6)
        sample_df = df_filtered[['timestamp', 'temperature_celsius']].dropna().tail(50)
        plt.plot(sample_df['timestamp'], sample_df['temperature_celsius'])
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title('Temperature Time Series (Last 50 Points)')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/storm_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Display key correlations (if available)
    print("\nKey Correlations:")
    if 'temperature_celsius' in key_features_available and 'pressure_mb' in key_features_available:
        temp_press_corr = corr_matrix.loc['temperature_celsius', 'pressure_mb']
        print(f"- Temperature and Pressure: {temp_press_corr:.3f}")
        
    if 'wind_speed' in key_features_available and 'wind_gust' in key_features_available:
        wind_corr = corr_matrix.loc['wind_speed', 'wind_gust']
        print(f"- Wind Speed and Wind Gust: {wind_corr:.3f}")
        
    if 'humidity' in key_features_available and 'visibility_km' in key_features_available:
        humid_vis_corr = corr_matrix.loc['humidity', 'visibility_km']
        print(f"- Humidity and Visibility: {humid_vis_corr:.3f}")
        
    if 'reflectivity_max' in key_features_available and 'brightness_temp_min' in key_features_available:
        ref_br_corr = corr_matrix.loc['reflectivity_max', 'brightness_temp_min']
        print(f"- Reflectivity and Brightness Temperature: {ref_br_corr:.3f}")
        
    # Check for any obvious patterns in the data
    print("\nStatistical Summary of Key Features:")
    print(df_filtered[key_features_available].describe())
else:
    print("Not enough data available for analysis")

# Analyze data completeness
print("\nData Completeness Analysis:")
completeness = df_filtered[key_features_available].notna().sum() / len(df_filtered) * 100
print(completeness.sort_values(ascending=False))