# SkyGuard Feature Engineering Summary

## Data Sources

1. **Radar Data** (`radar_features.csv`)
   - Reflectivity measurements from weather radar
   - Key features: `reflectivity_max`, `reflectivity_mean`

2. **Satellite Data** (`satellite_features.csv`)
   - Satellite imagery features for weather monitoring
   - Key features: `brightness_temp_min`, `motion_vector_x`, `motion_vector_y`

3. **METAR Data** (`vaah_metar.csv`)
   - Aviation weather reports from VAH (Vasai)
   - Key features: `temperature`, `dew_point`, `pressure`, `wind_speed`, `wind_gust`, 
     `wind_direction`, `visibility`, `rain`

4. **Historical Weather Data** (`data/historical/csv/Weather data.csv`)
   - Comprehensive weather measurements from various Indian locations
   - Key features: `temperature_celsius`, `wind_kph`, `pressure_mb`, `humidity`, etc.

5. **Air Quality Data** (`data/historical/csv/Air quality information.csv`)
   - Air pollution measurements
   - Key features: `air_quality_PM2.5`, `air_quality_Ozone`, `air_quality_Carbon_Monoxide`, etc.

6. **Location Data** (`data/historical/csv/Location information.csv`)
   - Geographic information for weather stations
   - Key features: `country`, `region`, `latitude`, `longitude`, `timezone`

## Processed Datasets

### 1. Basic Merged Features (`storm_features.csv`)
- Combined radar, satellite, and METAR data
- Resampled to 5-minute intervals
- Shape: 18 rows × 13 columns
- Time range: 2025-09-13 09:30:00 to 2025-09-13 17:45:00

### 2. Enhanced Features (`data/enhanced/enhanced_storm_features.csv`)
- Combined all available data sources
- Resampled to 5-minute intervals
- Shape: 98 rows × 50 columns
- Time range: 2023-08-25 21:50:00 to 2025-09-13 17:45:00

## Feature Engineering Scripts

1. `merge_features.py` - Merges basic radar, satellite, and METAR data
2. `merge_enhanced_features.py` - Merges all available data sources
3. `convert_excel_to_csv.py` - Converts original Excel files to CSV format

## Next Steps

1. Perform exploratory data analysis on the enhanced dataset
2. Extract relevant features for storm prediction
3. Handle missing data and outliers
4. Create train/validation/test splits
5. Begin model development and training