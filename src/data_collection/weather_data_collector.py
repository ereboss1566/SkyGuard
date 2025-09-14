"""
Real-time Data Collection Pipeline for SkyGuard Storm Prediction System
"""
import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataCollector:
    """Base class for weather data collection"""
    
    def __init__(self):
        self.data = {}
        self.last_updated = None
    
    def collect_data(self) -> Dict:
        """Collect data from the source"""
        raise NotImplementedError("Subclasses must implement collect_data method")
    
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """Preprocess raw data into the format expected by our model"""
        raise NotImplementedError("Subclasses must implement preprocess_data method")

class OpenWeatherMapCollector(WeatherDataCollector):
    """Collect data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    def collect_data(self, lat: float = 22.3072, lon: float = 73.1812) -> Dict:
        """Collect current weather data for a location"""
        try:
            # Current weather data
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.last_updated = datetime.now()
            
            logger.info(f"Collected OpenWeatherMap data at {self.last_updated}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error collecting OpenWeatherMap data: {e}")
            return {}
    
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """Convert OpenWeatherMap data to our model's format"""
        if not raw_data:
            return pd.DataFrame()
        
        # Extract relevant features
        processed_data = {
            # Basic weather features
            'temperature': [raw_data.get('main', {}).get('temp')],
            'temperature_celsius': [raw_data.get('main', {}).get('temp')],
            'dew_point': [raw_data.get('main', {}).get('dew_point', np.nan)],  # May not be available
            'pressure': [raw_data.get('main', {}).get('pressure')],
            'pressure_mb': [raw_data.get('main', {}).get('pressure')],
            'humidity': [raw_data.get('main', {}).get('humidity')],
            'wind_speed': [raw_data.get('wind', {}).get('speed')],
            'wind_kph': [raw_data.get('wind', {}).get('speed', 0) * 3.6],  # Convert m/s to km/h
            'wind_direction': [raw_data.get('wind', {}).get('deg')],
            'visibility': [raw_data.get('visibility', np.nan)],  # In meters
            'visibility_km': [raw_data.get('visibility', np.nan) / 1000 if raw_data.get('visibility') else np.nan],
            'rain': [raw_data.get('rain', {}).get('1h', 0) if raw_data.get('rain') else 0],
            'precip_mm': [raw_data.get('rain', {}).get('1h', 0) if raw_data.get('rain') else 0],
            'cloud': [raw_data.get('clouds', {}).get('all')],
            
            # UV index (might need separate API call)
            'uv_index': [np.nan],
            
            # Features that will need approximation or external sources
            'reflectivity_max': [np.nan],
            'reflectivity_mean': [np.nan],
            'brightness_temp_min': [np.nan],
            'motion_vector_x': [np.nan],
            'motion_vector_y': [np.nan],
            'air_quality_Carbon_Monoxide': [np.nan],
            'air_quality_Ozone': [np.nan],
            'air_quality_Nitrogen_dioxide': [np.nan],
            'air_quality_Sulphur_dioxide': [np.nan],
            'air_quality_PM2.5': [np.nan],
            'air_quality_PM10': [np.nan],
        }
        
        return pd.DataFrame(processed_data)

class WeatherGovCollector(WeatherDataCollector):
    """Collect data from Weather.gov API (US only)"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.weather.gov"
    
    def collect_data(self, lat: float = 39.7392, lon: float = -104.9903) -> Dict:
        """Collect current weather data for a location"""
        try:
            # Get grid point for the location
            url = f"{self.base_url}/points/{lat},{lon}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            grid_data = response.json()
            grid_id = grid_data['properties']['gridId']
            grid_x = grid_data['properties']['gridX']
            grid_y = grid_data['properties']['gridY']
            
            # Get current observations
            url = f"{self.base_url}/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.last_updated = datetime.now()
            
            logger.info(f"Collected Weather.gov data at {self.last_updated}")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error collecting Weather.gov data: {e}")
            return {}
    
    def preprocess_data(self, raw_data: Dict) -> pd.DataFrame:
        """Convert Weather.gov data to our model's format"""
        if not raw_data:
            return pd.DataFrame()
        
        # Get current period data (first period in the forecast)
        periods = raw_data.get('properties', {}).get('periods', [])
        if not periods:
            return pd.DataFrame()
        
        current_period = periods[0]
        
        # Extract relevant features
        processed_data = {
            # Basic weather features
            'temperature': [current_period.get('temperature')],
            'temperature_celsius': [(current_period.get('temperature', 0) - 32) * 5/9],  # Convert F to C
            'dew_point': [np.nan],  # Not directly available
            'pressure': [np.nan],  # Not directly available
            'pressure_mb': [np.nan],
            'humidity': [current_period.get('relativeHumidity', {}).get('value')],
            'wind_speed': [self._parse_wind_speed(current_period.get('windSpeed', '0 mph'))],
            'wind_kph': [self._parse_wind_speed(current_period.get('windSpeed', '0 mph')) * 1.60934],  # Convert mph to km/h
            'wind_direction': [self._parse_wind_direction(current_period.get('windDirection', 'N'))],
            'visibility': [np.nan],  # Not directly available
            'visibility_km': [np.nan],
            'rain': [np.nan],  # Not directly available
            'precip_mm': [current_period.get('probabilityOfPrecipitation', {}).get('value', 0)],
            'cloud': [np.nan],  # Not directly available
            
            # UV index
            'uv_index': [np.nan],
            
            # Features that will need approximation or external sources
            'reflectivity_max': [np.nan],
            'reflectivity_mean': [np.nan],
            'brightness_temp_min': [np.nan],
            'motion_vector_x': [np.nan],
            'motion_vector_y': [np.nan],
            'air_quality_Carbon_Monoxide': [np.nan],
            'air_quality_Ozone': [np.nan],
            'air_quality_Nitrogen_dioxide': [np.nan],
            'air_quality_Sulphur_dioxide': [np.nan],
            'air_quality_PM2.5': [np.nan],
            'air_quality_PM10': [np.nan],
        }
        
        return pd.DataFrame(processed_data)
    
    def _parse_wind_speed(self, wind_speed_str: str) -> float:
        """Parse wind speed from string (e.g., '10 mph')"""
        try:
            return float(wind_speed_str.split()[0])
        except (ValueError, IndexError):
            return 0.0
    
    def _parse_wind_direction(self, wind_dir_str: str) -> float:
        """Convert wind direction string to degrees"""
        directions = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        return float(directions.get(wind_dir_str, 0))

class DataFusionEngine:
    """Combine data from multiple sources"""
    
    def __init__(self):
        self.collectors = []
    
    def add_collector(self, collector: WeatherDataCollector):
        """Add a data collector to the fusion engine"""
        self.collectors.append(collector)
    
    def collect_all_data(self) -> List[pd.DataFrame]:
        """Collect data from all sources"""
        data_frames = []
        
        for collector in self.collectors:
            try:
                raw_data = collector.collect_data()
                processed_data = collector.preprocess_data(raw_data)
                if not processed_data.empty:
                    data_frames.append(processed_data)
            except Exception as e:
                logger.error(f"Error collecting data from {type(collector).__name__}: {e}")
        
        return data_frames
    
    def fuse_data(self, data_frames: List[pd.DataFrame]) -> pd.DataFrame:
        """Fuse data from multiple sources using weighted averaging or best available"""
        if not data_frames:
            return pd.DataFrame()
        
        if len(data_frames) == 1:
            return data_frames[0]
        
        # For now, we'll use the first (presumably highest quality) data source
        # In a more sophisticated implementation, we would fuse the data
        return data_frames[0].copy()
    
    def impute_missing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing features using domain knowledge or derived calculations"""
        df = data.copy()
        
        # Derive some features from others where possible
        # Convert temperature to dew point approximation (simplified)
        if 'temperature' in df.columns and 'humidity' in df.columns and 'dew_point' in df.columns:
            # Simple approximation: dew point = temperature - ((100 - humidity) / 5)
            mask = df['dew_point'].isna()
            df.loc[mask, 'dew_point'] = df.loc[mask, 'temperature'] - ((100 - df.loc[mask, 'humidity']) / 5)
        
        # Ensure we have all required columns, even if NaN
        required_columns = [
            'reflectivity_max', 'reflectivity_mean', 'brightness_temp_min', 
            'motion_vector_x', 'motion_vector_y', 'temperature', 'dew_point', 
            'pressure', 'wind_speed', 'wind_direction', 'visibility', 'rain', 
            'temperature_celsius', 'pressure_mb', 'wind_kph', 'humidity', 
            'cloud', 'visibility_km', 'uv_index', 'precip_mm', 
            'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
            'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
            'air_quality_PM2.5', 'air_quality_PM10'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        return df[required_columns]  # Return in the order expected by our model

def main():
    """Main function to demonstrate the data collection pipeline"""
    print("=== SkyGuard Real-time Data Collection Pipeline ===")
    
    # Initialize fusion engine
    fusion_engine = DataFusionEngine()
    
    # Add collectors (you would need to provide actual API keys)
    # For demonstration, we'll create collectors without API keys
    try:
        # Note: You would need to provide actual API keys for real usage
        openweather_collector = OpenWeatherMapCollector("YOUR_OPENWEATHER_API_KEY")
        fusion_engine.add_collector(openweather_collector)
    except Exception as e:
        logger.warning(f"Could not initialize OpenWeatherMap collector: {e}")
    
    try:
        weathergov_collector = WeatherGovCollector()
        fusion_engine.add_collector(weathergov_collector)
    except Exception as e:
        logger.warning(f"Could not initialize Weather.gov collector: {e}")
    
    print(f"Initialized {len(fusion_engine.collectors)} data collectors")
    
    # Collect data (this would normally be done in a loop)
    print("\nCollecting data from all sources...")
    data_frames = fusion_engine.collect_all_data()
    
    if data_frames:
        print(f"Successfully collected data from {len(data_frames)} sources")
        
        # Fuse data
        fused_data = fusion_engine.fuse_data(data_frames)
        print(f"Fused data shape: {fused_data.shape}")
        
        # Impute missing features
        final_data = fusion_engine.impute_missing_features(fused_data)
        print(f"Final data shape: {final_data.shape}")
        
        # Display sample of the data
        print("\nSample of processed data:")
        print(final_data.head())
        
        # Save to CSV for further processing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/realtime/weather_data_{timestamp}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        final_data.to_csv(filename, index=False)
        print(f"\nData saved to {filename}")
    else:
        print("No data collected from any sources")

if __name__ == "__main__":
    main()