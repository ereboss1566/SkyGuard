import pandas as pd
import numpy as np

# Load the enhanced storm features data
file_path = 'data/enhanced/enhanced_storm_features.csv'
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Dataset Shape:", df.shape)
print("Columns:", list(df.columns))

# Select key numeric features for modeling from all data sources
key_features = [
    # Radar features
    'reflectivity_max', 'reflectivity_mean',
    
    # Satellite features
    'brightness_temp_min', 'motion_vector_x', 'motion_vector_y',
    
    # METAR (weather station) features
    'temperature', 'dew_point', 'pressure', 'wind_speed', 'wind_gust',
    'wind_direction', 'visibility', 'rain',
    
    # Enhanced weather features
    'temperature_celsius', 'pressure_mb', 'wind_kph', 'humidity', 
    'cloud', 'visibility_km', 'uv_index', 'precip_mm',
    
    # Air quality features
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
    'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
    'air_quality_PM2.5', 'air_quality_PM10'
]

# Check which features are actually available
available_features = []
missing_features = []

for feature in key_features:
    if feature in df.columns:
        available_features.append(feature)
        print(f"+ {feature}")
    else:
        missing_features.append(feature)
        print(f"- {feature}")

print(f"\nAvailable features: {len(available_features)}")
print(f"Missing features: {len(missing_features)}")

# Check for data issues
X = df[available_features]
print(f"\nX shape: {X.shape}")
print(f"X columns: {list(X.columns)}")

# Check for completely missing columns
missing_data_summary = X.isnull().sum()
print("\nMissing data summary:")
print(missing_data_summary)

# Check for columns with all NaN values
all_nan_columns = missing_data_summary[missing_data_summary == len(X)].index.tolist()
print(f"\nColumns with all NaN values: {all_nan_columns}")

# Remove columns with all NaN values
if len(all_nan_columns) > 0:
    X = X.drop(columns=all_nan_columns)
    print(f"Removed {len(all_nan_columns)} columns with all NaN values")
    print(f"New X shape: {X.shape}")

print(f"\nFinal columns: {list(X.columns)}")