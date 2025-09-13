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
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Select numeric columns for correlation analysis
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumeric columns for analysis:")
print(numeric_columns)

# Calculate correlation matrix
corr_matrix = df[numeric_columns].corr()

# Create visualizations
plt.figure(figsize=(20, 16))

# 1. Correlation heatmap
plt.subplot(2, 3, 1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Storm Features')

# 2. Temperature vs Pressure
plt.subplot(2, 3, 2)
plt.scatter(df['temperature'], df['pressure'], alpha=0.5)
plt.xlabel('Temperature (°C)')
plt.ylabel('Pressure (mb)')
plt.title('Temperature vs Pressure')

# 3. Wind Speed vs Wind Gust
plt.subplot(2, 3, 3)
plt.scatter(df['wind_speed'], df['wind_gust'], alpha=0.5)
plt.xlabel('Wind Speed')
plt.ylabel('Wind Gust')
plt.title('Wind Speed vs Wind Gust')

# 4. Reflectivity vs Brightness Temperature
plt.subplot(2, 3, 4)
plt.scatter(df['reflectivity_max'], df['brightness_temp_min'], alpha=0.5)
plt.xlabel('Max Reflectivity')
plt.ylabel('Min Brightness Temperature')
plt.title('Reflectivity vs Brightness Temperature')

# 5. Humidity vs Visibility
plt.subplot(2, 3, 5)
plt.scatter(df['humidity'], df['visibility'], alpha=0.5)
plt.xlabel('Humidity (%)')
plt.ylabel('Visibility')
plt.title('Humidity vs Visibility')

# 6. Time series of key variables
plt.subplot(2, 3, 6)
sample_df = df.tail(100)  # Last 100 data points for visualization
plt.plot(sample_df['timestamp'], sample_df['temperature'], label='Temperature')
plt.plot(sample_df['timestamp'], sample_df['pressure'], label='Pressure')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Time Series of Temperature and Pressure')
plt.legend()

plt.tight_layout()
plt.savefig('outputs/storm_features_analysis.png')
plt.show()

# Display key correlations
print("\nKey Correlations:")
print("- Temperature and Pressure:", corr_matrix.loc['temperature', 'pressure'])
print("- Wind Speed and Wind Gust:", corr_matrix.loc['wind_speed', 'wind_gust'])
print("- Humidity and Visibility:", corr_matrix.loc['humidity', 'visibility'])
print("- Reflectivity and Brightness Temperature:", corr_matrix.loc['reflectivity_max', 'brightness_temp_min'])
print("- Temperature and Humidity:", corr_matrix.loc['temperature', 'humidity'])

# Check for any obvious patterns in the data
print("\nStatistical Summary:")
print(df[numeric_columns].describe())