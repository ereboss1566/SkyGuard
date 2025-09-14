"""
Script to make a real prediction using current weather data and suggest verification websites
"""
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json

def get_current_weather_data():
    """Get current weather data using a free weather API"""
    print("=== Getting Current Weather Data ===")
    
    # Using sample data for Ahmedabad, India
    lat, lon = 23.0225, 72.5714  # Ahmedabad coordinates
    
    # Sample current weather data for Ahmedabad (typical conditions)
    sample_data = {
        'temperature': 35.0,  # Celsius
        'humidity': 65.0,     # Percentage
        'pressure': 1010.0,   # hPa
        'wind_speed': 12.0,   # km/h
        'wind_direction': 270, # degrees (west)
        'visibility': 10.0,    # km
        'cloud': 40.0,        # Percentage
        'precip_mm': 0.0,     # mm in last hour
        'uv_index': 8.0,      # UV index
        'dew_point': 24.0,    # Celsius (calculated approximation)
        'temperature_celsius': 35.0,
        'pressure_mb': 1010.0,
        'wind_kph': 12.0,
        'visibility_km': 10.0,
        # Features that will need approximation or are not directly available
        'reflectivity_max': 50.0,  # Low reflectivity (clear conditions)
        'reflectivity_mean': 30.0,
        'brightness_temp_min': 270.0,  # Approximation
        'motion_vector_x': 0.1,  # Approximation
        'motion_vector_y': 0.0,  # Approximation
        'rain': 0.0,
        'air_quality_Carbon_Monoxide': 1.2,  # Approximation for urban area
        'air_quality_Ozone': 50.0,  # Approximation
        'air_quality_Nitrogen_dioxide': 35.0,  # Approximation
        'air_quality_Sulphur_dioxide': 15.0,  # Approximation
        'air_quality_PM2.5': 45.0,  # Approximation for urban area
        'air_quality_PM10': 65.0,  # Approximation for urban area
    }
    
    print(f"Using sample weather data for Ahmedabad (coordinates: {lat}, {lon})")
    print("Current conditions:")
    print(f"  Temperature: {sample_data['temperature']}°C")
    print(f"  Humidity: {sample_data['humidity']}%")
    print(f"  Pressure: {sample_data['pressure']} hPa")
    print(f"  Wind Speed: {sample_data['wind_speed']} km/h")
    print(f"  Precipitation: {sample_data['precip_mm']} mm")
    print(f"  Cloud Cover: {sample_data['cloud']}%")
    
    return sample_data

def make_storm_prediction(weather_data):
    """Make a storm prediction using the trained model"""
    try:
        # Import the prediction service
        from src.prediction.storm_prediction_service import StormPredictionService, AlertingSystem
        
        # Initialize services
        prediction_service = StormPredictionService()
        alerting_system = AlertingSystem(storm_threshold=0.6)
        
        # Convert to DataFrame
        weather_df = pd.DataFrame([weather_data])
        
        # Make prediction
        result = prediction_service.predict_storm_probability(weather_df)
        
        # Evaluate alert
        alert = alerting_system.evaluate_alert(result)
        
        return result, alert
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None

def get_verification_websites():
    """Return a list of websites where you can verify weather conditions"""
    websites = [
        {
            "name": "Weather.com (The Weather Channel)",
            "url": "https://weather.com",
            "description": "Comprehensive weather information including radar, satellite, and severe weather alerts"
        },
        {
            "name": "AccuWeather",
            "url": "https://www.accuweather.com",
            "description": "Detailed weather forecasts and current conditions with radar and satellite imagery"
        },
        {
            "name": "Weather Underground",
            "url": "https://www.wunderground.com",
            "description": "Hyperlocal weather data, radar, and community weather reports"
        },
        {
            "name": "NOAA National Weather Service",
            "url": "https://www.weather.gov",
            "description": "Official US weather service with radar, satellite, and severe weather alerts"
        },
        {
            "name": "Windy.com",
            "url": "https://www.windy.com",
            "description": "Interactive weather map with wind, waves, radar, and satellite data"
        },
        {
            "name": "Intellicast",
            "url": "https://www.intellicast.com",
            "description": "Detailed weather radar and satellite imagery"
        },
        {
            "name": "India Meteorological Department",
            "url": "https://www.imd.gov.in",
            "description": "Official weather service for India with current conditions and forecasts"
        }
    ]
    
    return websites

def main():
    """Main function to make a prediction and provide verification resources"""
    print("=== SkyGuard Storm Prediction for Current Conditions ===\n")
    
    # Get current weather data
    weather_data = get_current_weather_data()
    
    # Make prediction
    print("\n=== Making Storm Prediction ===")
    result, alert = make_storm_prediction(weather_data)
    
    if result and alert:
        print(f"\nPrediction Results:")
        print(f"  Prediction: {'STORM' if result['prediction'] == 1 else 'NO STORM'}")
        print(f"  Probability of No Storm: {result['probability_no_storm']*100:.2f}%")
        print(f"  Probability of Storm: {result['probability_storm']*100:.2f}%")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        print(f"  Model Used: {result['model_used']}")
        
        print(f"\nAlert Level: {alert['alert_level']}")
        print(f"Alert Message: {alert['message']}")
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\nPrediction made at: {current_time}")
        
        # Provide verification websites
        print("\n" + "="*60)
        print("VERIFY CURRENT CONDITIONS ON THESE WEBSITES:")
        print("="*60)
        
        websites = get_verification_websites()
        for i, site in enumerate(websites, 1):
            print(f"\n{i}. {site['name']}")
            print(f"   URL: {site['url']}")
            print(f"   Description: {site['description']}")
        
        print("\n" + "="*60)
        print("RECOMMENDED VERIFICATION APPROACH:")
        print("="*60)
        print("1. Check current radar imagery for precipitation patterns")
        print("2. Verify wind speed and direction")
        print("3. Confirm pressure trends")
        print("4. Check for any active weather alerts or warnings")
        print("5. Compare cloud cover and visibility reports")
        print("6. Look for any reported thunderstorms or severe weather")
        
        # Additional information
        print("\n" + "="*60)
        print("PREDICTION INTERPRETATION:")
        print("="*60)
        if result['prediction'] == 1:
            print("ALERT: The model indicates a high probability of thunderstorms or gale force winds.")
            print("   Recommended actions:")
            print("   - Monitor official weather alerts")
            print("   - Secure outdoor equipment")
            print("   - Plan indoor activities if possible")
        else:
            print("CLEAR: The model indicates low probability of severe weather.")
            print("   However, always monitor official weather services for updates.")
        
        print("Note: This prediction is based on the model trained on historical data.")
        print("The actual weather conditions should be verified through official sources.")
        
    else:
        print("Failed to make prediction. Please check the system setup.")

if __name__ == "__main__":
    main()