import pandas as pd
import plotly.express as px
import numpy as np

# Load the storm features dataset
storm_df = pd.read_csv('storm_features.csv')

# Display the first few rows to understand the data
print("First few rows of the dataset:")
print(storm_df.head())

# Check the columns in the dataset
print("\nColumns in the dataset:")
print(storm_df.columns.tolist())

# Check if humidity column exists, if not calculate it
if 'humidity' not in storm_df.columns:
    # Calculate humidity from temperature and dew point
    # Using the simplified formula: RH = 100 * (EXP((17.625 * DP)/(243.04 + DP)) / EXP((17.625 * T)/(243.04 + T)))
    # Where DP is dew point and T is temperature in Celsius
    storm_df['humidity'] = 100 * (np.exp((17.625 * storm_df['dew_point']) / (243.04 + storm_df['dew_point'])) / 
                                  np.exp((17.625 * storm_df['temperature']) / (243.04 + storm_df['temperature'])))
    print("\nCalculated humidity column")

# Check for missing values in the required columns
required_columns = ['reflectivity_max', 'brightness_temp_min', 'pressure', 'wind_gust', 'humidity']
print("\nMissing values in required columns:")
for col in required_columns:
    if col in storm_df.columns:
        missing_count = storm_df[col].isnull().sum()
        print(f"{col}: {missing_count}")
    else:
        print(f"{col}: Column not found")

# Remove rows with missing values in the required columns for visualization
storm_df_clean = storm_df.dropna(subset=required_columns)
print(f"\nDataset size after removing rows with missing values: {len(storm_df_clean)}")

# Create histograms for the specified columns
hist_columns = ['reflectivity_max', 'brightness_temp_min', 'pressure', 'wind_gust', 'humidity']
for col in hist_columns:
    if col in storm_df_clean.columns:
        fig = px.histogram(storm_df_clean, x=col, nbins=40, 
                          title=f'Histogram of {col.replace("_", " ").title()}')
        fig.show()
    else:
        print(f"Column {col} not found in the dataset")

# Create scatter plots
scatter_pairs = [
    ('reflectivity_max', 'wind_gust'),
    ('brightness_temp_min', 'wind_gust'),
    ('humidity', 'wind_gust')
]

for x_col, y_col in scatter_pairs:
    # Check if both columns exist
    if x_col in storm_df_clean.columns and y_col in storm_df_clean.columns:
        fig = px.scatter(storm_df_clean, x=x_col, y=y_col, opacity=0.6,
                        hover_data=['timestamp'],
                        title=f'{x_col.replace("_", " ").title()} vs {y_col.replace("_", " ").title()}')
        fig.show()
    else:
        missing_cols = [col for col in [x_col, y_col] if col not in storm_df_clean.columns]
        print(f"Missing columns for scatter plot: {missing_cols}")

print("\nVisualizations completed!")