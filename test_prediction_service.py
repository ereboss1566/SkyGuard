"""
Test script for SkyGuard real-time components
"""
import pandas as pd
import numpy as np

def test_prediction_service():
    """Test the prediction service with sample data"""
    try:
        # Import the prediction service
        from src.prediction.storm_prediction_service import StormPredictionService, AlertingSystem
        
        # Initialize services
        prediction_service = StormPredictionService()
        alerting_system = AlertingSystem(storm_threshold=0.6)
        
        print("=== Testing SkyGuard Prediction Service ===")
        
        # Test with storm-like conditions
        print("\n1. Testing with storm-like conditions...")
        storm_data = {
            'reflectivity_max': [300.0],
            'reflectivity_mean': [250.0],
            'brightness_temp_min': [200.0],
            'motion_vector_x': [-1.0],
            'motion_vector_y': [0.5],
            'temperature': [30.0],
            'dew_point': [28.0],
            'pressure': [990.0],
            'wind_speed': [40.0],
            'wind_direction': [180],
            'visibility': [2.0],
            'rain': [15.0],
            'temperature_celsius': [30.0],
            'pressure_mb': [990.0],
            'wind_kph': [40.0],
            'humidity': [95.0],
            'cloud': [95.0],
            'visibility_km': [2.0],
            'uv_index': [1.0],
            'precip_mm': [15.0],
            'air_quality_Carbon_Monoxide': [1.0],
            'air_quality_Ozone': [80.0],
            'air_quality_Nitrogen_dioxide': [40.0],
            'air_quality_Sulphur_dioxide': [20.0],
            'air_quality_PM2.5': [50.0],
            'air_quality_PM10': [70.0],
        }
        
        storm_df = pd.DataFrame(storm_data)
        storm_result = prediction_service.predict_storm_probability(storm_df)
        
        print(f"   Prediction: {'STORM' if storm_result['prediction'] == 1 else 'NO STORM'}")
        print(f"   Probability - No Storm: {storm_result['probability_no_storm']*100:.2f}%")
        print(f"   Probability - Storm: {storm_result['probability_storm']*100:.2f}%")
        print(f"   Confidence: {storm_result['confidence']*100:.2f}%")
        
        # Test alerting
        storm_alert = alerting_system.evaluate_alert(storm_result)
        print(f"   Alert Level: {storm_alert['alert_level']}")
        print(f"   Alert Message: {storm_alert['message']}")
        
        # Test with clear conditions
        print("\n2. Testing with clear conditions...")
        clear_data = {
            'reflectivity_max': [0.0],
            'reflectivity_mean': [0.0],
            'brightness_temp_min': [280.0],
            'motion_vector_x': [0.0],
            'motion_vector_y': [0.0],
            'temperature': [25.0],
            'dew_point': [15.0],
            'pressure': [1020.0],
            'wind_speed': [10.0],
            'wind_direction': [90],
            'visibility': [10.0],
            'rain': [0.0],
            'temperature_celsius': [25.0],
            'pressure_mb': [1020.0],
            'wind_kph': [10.0],
            'humidity': [50.0],
            'cloud': [20.0],
            'visibility_km': [10.0],
            'uv_index': [7.0],
            'precip_mm': [0.0],
            'air_quality_Carbon_Monoxide': [0.5],
            'air_quality_Ozone': [30.0],
            'air_quality_Nitrogen_dioxide': [15.0],
            'air_quality_Sulphur_dioxide': [5.0],
            'air_quality_PM2.5': [10.0],
            'air_quality_PM10': [20.0],
        }
        
        clear_df = pd.DataFrame(clear_data)
        clear_result = prediction_service.predict_storm_probability(clear_df)
        
        print(f"   Prediction: {'STORM' if clear_result['prediction'] == 1 else 'NO STORM'}")
        print(f"   Probability - No Storm: {clear_result['probability_no_storm']*100:.2f}%")
        print(f"   Probability - Storm: {clear_result['probability_storm']*100:.2f}%")
        print(f"   Confidence: {clear_result['confidence']*100:.2f}%")
        
        # Test alerting
        clear_alert = alerting_system.evaluate_alert(clear_result)
        print(f"   Alert Level: {clear_alert['alert_level']}")
        print(f"   Alert Message: {clear_alert['message']}")
        
        print("\n=== Test Completed Successfully ===")
        return True
        
    except Exception as e:
        print(f"Error in test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_prediction_service()