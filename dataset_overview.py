import pandas as pd

# Load the enhanced features dataset
df = pd.read_csv("data/enhanced/enhanced_storm_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Display basic information
print("=== ENHANCED STORM FEATURES DATASET ===")
print(f"Shape: {df.shape}")
print(f"Time range: {df.index.min()} to {df.index.max()}")
print(f"Total time span: {df.index.max() - df.index.min()}")

# Display column names grouped by source
print("\n=== COLUMN GROUPS ===")

# Radar features (from our original data)
radar_cols = [col for col in df.columns if 'reflectivity' in col]
print(f"Radar features ({len(radar_cols)}): {radar_cols}")

# Satellite features (from our original data)
satellite_cols = [col for col in df.columns if 'brightness' in col or 'motion' in col]
print(f"Satellite features ({len(satellite_cols)}): {satellite_cols}")

# METAR features (from our original data)
metar_cols = [col for col in df.columns if any(x in col for x in ['temperature', 'dew_point', 'pressure', 'wind', 'visibility', 'rain']) and 'weather' not in col and 'fahrenheit' not in col and 'mph' not in col and 'kph' not in col]
print(f"METAR features ({len(metar_cols)}): {metar_cols}")

# Weather data features (from historical data)
weather_cols = [col for col in df.columns if any(x in col for x in ['temperature', 'wind', 'pressure', 'precip', 'humidity', 'cloud', 'visibility', 'uv', 'gust']) and ('_weather' in col or '_kph' in col or '_mb' in col or '_mm' in col or '_in' in col)]
print(f"Historical weather features ({len(weather_cols)}): {weather_cols}")

# Air quality features (from historical data)
air_cols = [col for col in df.columns if 'air_quality' in col]
print(f"Air quality features ({len(air_cols)}): {air_cols}")

# Location features (from historical data)
loc_cols = [col for col in df.columns if any(x in col for x in ['country', 'location', 'region', 'latitude', 'longitude', 'timezone', 'last_updated']) and ('_loc' in col or col in ['country', 'region', 'latitude', 'longitude', 'timezone', 'last_updated'])]
print(f"Location features ({len(loc_cols)}): {loc_cols}")

# Show some basic statistics
print("\n=== BASIC STATISTICS (numeric columns only) ===")
print(df.describe())

# Check for missing data
print("\n=== MISSING DATA COUNT (top 10 columns with most missing data) ===")
missing_data = df.isnull().sum().sort_values(ascending=False)
print(missing_data.head(10))