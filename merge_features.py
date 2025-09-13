import pandas as pd

# Load the datasets
radar_features_df = pd.read_csv("radar_features.csv")
satellite_features_df = pd.read_csv("satellite_features.csv")
vaah_features_df = pd.read_csv("vaah_metar.csv")

# Convert timestamp columns to datetime and set as index
radar_features_df['timestamp'] = pd.to_datetime(radar_features_df['timestamp'])
radar_features_df.set_index('timestamp', inplace=True)

satellite_features_df['timestamp'] = pd.to_datetime(satellite_features_df['timestamp'])
satellite_features_df.set_index('timestamp', inplace=True)

# For METAR data, the timestamp column is named 'time'
vaah_features_df['time'] = pd.to_datetime(vaah_features_df['time'])
vaah_features_df.set_index('time', inplace=True)
# Rename the index to 'timestamp' for consistency
vaah_features_df.index.name = 'timestamp'

# Resample each to 5-min intervals and forward-fill missing values
# Using '5min' instead of '5T' to avoid deprecation warning
radar_resampled = radar_features_df.resample("5min").ffill()
satellite_resampled = satellite_features_df.resample("5min").ffill()
metar_resampled = vaah_features_df.resample("5min").ffill()

# Merge all datasets
storm_features_df = radar_resampled \
    .join(satellite_resampled, how="outer") \
    .join(metar_resampled, how="outer")

# Final cleanup - sort by index, forward-fill, and drop rows with all NaN values
storm_features_df = storm_features_df.sort_index().ffill().dropna(how='all')

# Display sample
print("Storm features DataFrame (first 5 rows):")
print(storm_features_df.head())

# Save to CSV
storm_features_df.to_csv("storm_features.csv")
print("\\nMerged data saved to 'storm_features.csv'")

# Display some basic info about the merged dataset
print(f"\\nDataset shape: {storm_features_df.shape}")
print(f"Time range: {storm_features_df.index.min()} to {storm_features_df.index.max()}")
print(f"Columns: {list(storm_features_df.columns)}")