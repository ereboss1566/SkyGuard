import pandas as pd
import os

# Create directory for enhanced features if it doesn't exist
os.makedirs('data/enhanced', exist_ok=True)

# Load the previously merged storm features
storm_features_df = pd.read_csv("storm_features.csv")
storm_features_df['timestamp'] = pd.to_datetime(storm_features_df['timestamp'])
storm_features_df.set_index('timestamp', inplace=True)

# Load historical weather data
weather_df = pd.read_csv("data/historical/csv/Weather data.csv")
# Convert epoch time to datetime
weather_df['timestamp'] = pd.to_datetime(weather_df['last_updated_epoch'], unit='s')
weather_df.set_index('timestamp', inplace=True)

# Load air quality data
air_quality_df = pd.read_csv("data/historical/csv/Air quality information.csv")
# Convert epoch time to datetime
air_quality_df['timestamp'] = pd.to_datetime(air_quality_df['last_updated_epoch'], unit='s')
air_quality_df.set_index('timestamp', inplace=True)

# Load location data
location_df = pd.read_csv("data/historical/csv/Location information.csv")
# Convert epoch time to datetime
location_df['timestamp'] = pd.to_datetime(location_df['last_updated_epoch'], unit='s')
location_df.set_index('timestamp', inplace=True)

# Resample historical data to 5-minute intervals
weather_resampled = weather_df.resample("5min").ffill()
air_quality_resampled = air_quality_df.resample("5min").ffill()
location_resampled = location_df.resample("5min").ffill()

# Merge all datasets
enhanced_features_df = storm_features_df \
    .join(weather_resampled, how="outer", rsuffix='_weather') \
    .join(air_quality_resampled, how="outer", rsuffix='_air') \
    .join(location_resampled, how="outer", rsuffix='_loc')

# Final cleanup - sort by index, forward-fill, and drop rows with all NaN values
enhanced_features_df = enhanced_features_df.sort_index().ffill().dropna(how='all')

# Display sample
print("Enhanced features DataFrame (first 5 rows):")
print(enhanced_features_df.head())

# Save to CSV
enhanced_features_df.to_csv("data/enhanced/enhanced_storm_features.csv")
print("\nEnhanced merged data saved to 'data/enhanced/enhanced_storm_features.csv'")

# Display some basic info about the merged dataset
print(f"\nDataset shape: {enhanced_features_df.shape}")
print(f"Time range: {enhanced_features_df.index.min()} to {enhanced_features_df.index.max()}")
print(f"Total columns: {len(enhanced_features_df.columns)}")