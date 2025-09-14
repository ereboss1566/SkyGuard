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

def prepare_sample_data():
    """Create sample data for prediction (in a real scenario, this would come from live data sources)"""
    # This is just example data - in reality, you would get this from weather APIs or sensors
    # These are the actual features used in training (26 features)
    sample_data = {
        # Radar features
        'reflectivity_max': [254.0],  # High reflectivity indicates precipitation
        'reflectivity_mean': [196.1],  # Average reflectivity
        
        # Satellite features
        'brightness_temp_min': [220.0],  # Low brightness temperature indicates cold cloud tops
        'motion_vector_x': [-0.5],  # Cloud movement vector
        'motion_vector_y': [0.3],  # Cloud movement vector
        
        # METAR (weather station) features
        'temperature': [25.0],  # Temperature in Celsius
        'dew_point': [22.0],  # Dew point temperature
        'pressure': [1005.0],  # Atmospheric pressure in hPa
        'wind_speed': [25.0],  # Wind speed in km/h
        'wind_direction': [180],  # Wind direction in degrees
        'visibility': [5.0],  # Visibility in km
        'rain': [5.0],  # Rainfall amount
        
        # Enhanced weather features
        'temperature_celsius': [25.0],  # Temperature in Celsius
        'pressure_mb': [1005.0],  # Pressure in millibars
        'wind_kph': [25.0],  # Wind speed in km/h
        'humidity': [85.0],  # Relative humidity percentage
        'cloud': [80.0],  # Cloud cover percentage
        'visibility_km': [5.0],  # Visibility in km
        'uv_index': [3.0],  # UV index
        'precip_mm': [5.0],  # Precipitation in mm
        
        # Air quality features
        'air_quality_Carbon_Monoxide': [0.8],  # CO levels
        'air_quality_Ozone': [60.0],  # O3 levels
        'air_quality_Nitrogen_dioxide': [30.0],  # NO2 levels
        'air_quality_Sulphur_dioxide': [15.0],  # SO2 levels
        'air_quality_PM2.5': [25.0],  # PM2.5 levels
        'air_quality_PM10': [40.0],  # PM10 levels
    }
    
    return pd.DataFrame(sample_data)

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
    
    # Prepare sample data (in a real system, this would come from live weather data)
    print("\nPreparing sample weather data...")
    sample_data = prepare_sample_data()
    
    # Display the input features
    print("\nInput weather features:")
    for col in sample_data.columns:
        print(f"  {col}: {sample_data[col].iloc[0]}")
    
    # Make prediction
    print("\nMaking storm prediction...")
    prediction, probability = predict_storm(model, scaler, imputer, sample_data)
    
    # Display results
    print("\n" + "="*50)
    print("STORM PREDICTION RESULTS")
    print("="*50)
    
    if prediction == 1:
        print("ALERT: Thunderstorm/Gale force winds predicted!")
        print(f"   Confidence: {probability[1]*100:.2f}%")
    else:
        print("CLEAR: No storm predicted")
        print(f"   Confidence: {probability[0]*100:.2f}%")
    
    print(f"\nPrediction probabilities:")
    print(f"  No storm: {probability[0]*100:.2f}%")
    print(f"  Storm: {probability[1]*100:.2f}%")
    
    # Next prediction time (in a real system, this would be based on data availability)
    next_prediction = datetime.now() + timedelta(hours=1)
    print(f"\nNext prediction will be available at: {next_prediction.strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()