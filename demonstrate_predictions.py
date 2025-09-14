import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        # Load the best performing model (Random Forest)
        model = joblib.load('models/optimized/randomforest_optimized_model.pkl')
        scaler = joblib.load('models/optimized/scaler.pkl')
        imputer = joblib.load('models/optimized/imputer.pkl')
        print("Model and preprocessors loaded successfully!")
        return model, scaler, imputer
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Make sure you've run the training script first.")
        return None, None, None

def load_sample_historical_data():
    """Load a sample of historical data for demonstration"""
    try:
        df = pd.read_csv('data/enhanced/enhanced_storm_features.csv')
        timestamps = pd.to_datetime(df['timestamp'])
        # Select only the features the model expects
        expected_features = [
            'reflectivity_max', 'reflectivity_mean', 'brightness_temp_min', 
            'motion_vector_x', 'motion_vector_y', 'temperature', 'dew_point', 
            'pressure', 'wind_speed', 'wind_direction', 'visibility', 'rain', 
            'temperature_celsius', 'pressure_mb', 'wind_kph', 'humidity', 
            'cloud', 'visibility_km', 'uv_index', 'precip_mm', 
            'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
            'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
            'air_quality_PM2.5', 'air_quality_PM10'
        ]
        # Filter to only the features we need
        available_features = [col for col in expected_features if col in df.columns]
        df_filtered = df[available_features].copy()
        return df_filtered, timestamps
    except FileNotFoundError:
        print("Historical data file not found.")
        return None, None

def predict_storm(model, scaler, imputer, input_data):
    """Make a storm prediction using the trained model"""
    # Handle missing values
    input_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)
    
    # Scale the features
    input_scaled = scaler.transform(input_imputed)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0]

def main():
    print("=== SkyGuard Storm Prediction System ===")
    print("Loading model and preprocessors...")
    
    # Load model and preprocessors
    model, scaler, imputer = load_model_and_preprocessors()
    
    if model is None:
        return
    
    # Load historical data for demonstration
    print("\nLoading historical data for demonstration...")
    df_data, timestamps = load_sample_historical_data()
    
    if df_data is None:
        print("Could not load historical data for demonstration.")
        return
    
    print(f"Loaded {len(df_data)} historical data points")
    
    # Demonstrate predictions on a few sample data points
    print("\n" + "="*60)
    print("DEMONSTRATION: Storm Predictions on Historical Data")
    print("="*60)
    
    # Select a few sample points to demonstrate predictions
    sample_indices = [0, 10, 20, 30, 40]  # Just a few sample indices
    
    storm_count = 0
    for i in sample_indices:
        if i < len(df_data):
            # Get a single row of data
            sample_row = df_data.iloc[[i]]  # Double brackets to keep it as DataFrame
            timestamp = timestamps.iloc[i]
            
            # Make prediction
            prediction, probability = predict_storm(model, scaler, imputer, sample_row)
            
            # Display results
            print(f"\nPrediction for {timestamp}:")
            if prediction == 1:
                print("  ALERT: Thunderstorm/Gale force winds predicted!")
                print(f"    Confidence: {probability[1]*100:.2f}%")
                storm_count += 1
            else:
                print("  CLEAR: No storm predicted")
                print(f"    Confidence: {probability[0]*100:.2f}%")
    
    print(f"\nSummary: {storm_count} out of {len(sample_indices)} samples predicted to have storms")
    
    # Show how to make a custom prediction
    print("\n" + "="*60)
    print("CUSTOM PREDICTION EXAMPLE")
    print("="*60)
    
    # Create sample data representing current conditions that would likely cause a storm
    storm_conditions = {
        'reflectivity_max': [300.0],  # Very high reflectivity
        'reflectivity_mean': [250.0],  # High average reflectivity
        'brightness_temp_min': [200.0],  # Very cold cloud tops
        'motion_vector_x': [-1.0],  # Fast cloud movement
        'motion_vector_y': [0.5],  # Fast cloud movement
        'temperature': [30.0],  # High temperature
        'dew_point': [28.0],  # High dew point (very humid)
        'pressure': [990.0],  # Low pressure
        'wind_speed': [40.0],  # High wind speed
        'wind_direction': [180],  # Wind direction
        'visibility': [2.0],  # Poor visibility
        'rain': [15.0],  # Heavy rain
        'temperature_celsius': [30.0],  # High temperature
        'pressure_mb': [990.0],  # Low pressure
        'wind_kph': [40.0],  # High wind speed
        'humidity': [95.0],  # Very high humidity
        'cloud': [95.0],  # Nearly complete cloud cover
        'visibility_km': [2.0],  # Poor visibility
        'uv_index': [1.0],  # Low UV (cloudy)
        'precip_mm': [15.0],  # Heavy precipitation
        'air_quality_Carbon_Monoxide': [1.0],  # Higher pollution
        'air_quality_Ozone': [80.0],  # Higher ozone
        'air_quality_Nitrogen_dioxide': [40.0],  # Higher NO2
        'air_quality_Sulphur_dioxide': [20.0],  # Higher SO2
        'air_quality_PM2.5': [50.0],  # Higher PM2.5
        'air_quality_PM10': [70.0],  # Higher PM10
    }
    
    storm_data = pd.DataFrame(storm_conditions)
    prediction, probability = predict_storm(model, scaler, imputer, storm_data)
    
    print("Custom storm-like conditions prediction:")
    if prediction == 1:
        print("  ALERT: Thunderstorm/Gale force winds predicted!")
        print(f"    Confidence: {probability[1]*100:.2f}%")
    else:
        print("  CLEAR: No storm predicted")
        print(f"    Confidence: {probability[0]*100:.2f}%")
    
    # Show how to make a custom prediction for clear conditions
    clear_conditions = {
        'reflectivity_max': [0.0],  # No reflectivity
        'reflectivity_mean': [0.0],  # No reflectivity
        'brightness_temp_min': [280.0],  # Warm cloud tops
        'motion_vector_x': [0.0],  # No cloud movement
        'motion_vector_y': [0.0],  # No cloud movement
        'temperature': [25.0],  # Moderate temperature
        'dew_point': [15.0],  # Moderate dew point
        'pressure': [1020.0],  # High pressure
        'wind_speed': [10.0],  # Light wind
        'wind_direction': [90],  # Wind direction
        'visibility': [10.0],  # Good visibility
        'rain': [0.0],  # No rain
        'temperature_celsius': [25.0],  # Moderate temperature
        'pressure_mb': [1020.0],  # High pressure
        'wind_kph': [10.0],  # Light wind
        'humidity': [50.0],  # Moderate humidity
        'cloud': [20.0],  # Mostly clear
        'visibility_km': [10.0],  # Good visibility
        'uv_index': [7.0],  # High UV (clear)
        'precip_mm': [0.0],  # No precipitation
        'air_quality_Carbon_Monoxide': [0.5],  # Lower pollution
        'air_quality_Ozone': [30.0],  # Lower ozone
        'air_quality_Nitrogen_dioxide': [15.0],  # Lower NO2
        'air_quality_Sulphur_dioxide': [5.0],  # Lower SO2
        'air_quality_PM2.5': [10.0],  # Lower PM2.5
        'air_quality_PM10': [20.0],  # Lower PM10
    }
    
    clear_data = pd.DataFrame(clear_conditions)
    prediction, probability = predict_storm(model, scaler, imputer, clear_data)
    
    print("\nCustom clear conditions prediction:")
    if prediction == 1:
        print("  ALERT: Thunderstorm/Gale force winds predicted!")
        print(f"    Confidence: {probability[1]*100:.2f}%")
    else:
        print("  CLEAR: No storm predicted")
        print(f"    Confidence: {probability[0]*100:.2f}%")

if __name__ == "__main__":
    main()