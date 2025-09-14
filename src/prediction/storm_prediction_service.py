"""
Real-time Prediction Service for SkyGuard Storm Prediction System
"""
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StormPredictionService:
    """Real-time storm prediction service"""
    
    def __init__(self, model_path: str = "models/optimized"):
        """Initialize the prediction service with trained models"""
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.imputer = None
        self.last_prediction_time = None
        self.load_models()
    
    def load_models(self):
        """Load the trained models and preprocessing objects"""
        try:
            # Load the Random Forest model (best performing)
            model_file = os.path.join(self.model_path, "randomforest_optimized_model.pkl")
            self.model = joblib.load(model_file)
            
            # Load preprocessing objects
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            self.scaler = joblib.load(scaler_file)
            
            imputer_file = os.path.join(self.model_path, "imputer.pkl")
            self.imputer = joblib.load(imputer_file)
            
            logger.info("Successfully loaded models and preprocessing objects")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def preprocess_input_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess input data to match training format"""
        # Ensure we have the right columns in the right order
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
        
        # Select only the features we need and in the right order
        try:
            processed_data = input_data[expected_features].copy()
        except KeyError as e:
            missing_cols = [col for col in expected_features if col not in input_data.columns]
            logger.warning(f"Missing columns in input data: {missing_cols}")
            
            # Create missing columns with NaN values
            processed_data = input_data.copy()
            for col in missing_cols:
                processed_data[col] = np.nan
            
            # Reorder to match expected features
            processed_data = processed_data[expected_features]
        
        return processed_data
    
    def predict_storm_probability(self, weather_data: pd.DataFrame) -> Dict:
        """Make storm prediction using the trained model"""
        try:
            # Preprocess the input data
            processed_data = self.preprocess_input_data(weather_data)
            
            # Handle missing values using the trained imputer
            imputed_data = pd.DataFrame(
                self.imputer.transform(processed_data),
                columns=processed_data.columns
            )
            
            # Scale the features using the trained scaler
            scaled_data = self.scaler.transform(imputed_data)
            
            # Make prediction
            prediction = self.model.predict(scaled_data)
            probability = self.model.predict_proba(scaled_data)
            
            # Store prediction time
            self.last_prediction_time = datetime.now()
            
            # Return results
            result = {
                'prediction': int(prediction[0]),
                'probability_no_storm': float(probability[0][0]),
                'probability_storm': float(probability[0][1]),
                'confidence': float(max(probability[0])),
                'timestamp': self.last_prediction_time.isoformat(),
                'model_used': 'Random Forest (Optimized)'
            }
            
            logger.info(f"Prediction made at {self.last_prediction_time}")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
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
        
        importance_df = pd.DataFrame({
            'feature': expected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

class AlertingSystem:
    """Simple alerting system based on prediction probabilities"""
    
    def __init__(self, storm_threshold: float = 0.7):
        """Initialize alerting system with threshold"""
        self.storm_threshold = storm_threshold
        self.alert_history = []
    
    def evaluate_alert(self, prediction_result: Dict) -> Dict:
        """Evaluate if an alert should be triggered"""
        storm_prob = prediction_result['probability_storm']
        timestamp = prediction_result['timestamp']
        
        alert_level = "NONE"
        alert_message = "No significant weather events predicted"
        
        if storm_prob >= self.storm_threshold:
            alert_level = "STORM_WARNING"
            alert_message = f"STORM WARNING: {storm_prob*100:.1f}% probability of severe weather"
        elif storm_prob >= self.storm_threshold * 0.7:
            alert_level = "STORM_WATCH"
            alert_message = f"STORM WATCH: {storm_prob*100:.1f}% probability of severe weather"
        
        alert_info = {
            'alert_level': alert_level,
            'message': alert_message,
            'probability': storm_prob,
            'timestamp': timestamp
        }
        
        # Store alert in history
        self.alert_history.append(alert_info)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        return alert_info
    
    def get_alert_history(self, limit: int = 10) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if self.alert_history else []

def main():
    """Main function to demonstrate the prediction service"""
    print("=== SkyGuard Real-time Prediction Service ===")
    
    try:
        # Initialize prediction service
        prediction_service = StormPredictionService()
        alerting_system = AlertingSystem(storm_threshold=0.6)
        
        print("Prediction service initialized successfully")
        
        # Get feature importance
        print("\nTop 10 Feature Importances:")
        feature_importance = prediction_service.get_feature_importance()
        print(feature_importance.head(10))
        
        # For demonstration, let's create some sample weather data
        # In a real system, this would come from the data collection pipeline
        sample_data = {
            # High probability storm conditions
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
        
        weather_df = pd.DataFrame(sample_data)
        
        print("\nMaking prediction with sample storm-like conditions...")
        prediction_result = prediction_service.predict_storm_probability(weather_df)
        
        print("\nPrediction Results:")
        print(f"  Prediction: {'STORM' if prediction_result['prediction'] == 1 else 'NO STORM'}")
        print(f"  Probability of No Storm: {prediction_result['probability_no_storm']*100:.2f}%")
        print(f"  Probability of Storm: {prediction_result['probability_storm']*100:.2f}%")
        print(f"  Confidence: {prediction_result['confidence']*100:.2f}%")
        print(f"  Model Used: {prediction_result['model_used']}")
        
        # Evaluate alert
        alert_info = alerting_system.evaluate_alert(prediction_result)
        print(f"\nAlert Level: {alert_info['alert_level']}")
        print(f"Alert Message: {alert_info['message']}")
        
        # Now let's test with clear conditions
        clear_data = {
            # Clear weather conditions
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
        
        clear_df = pd.DataFrame(clear_data)
        
        print("\n" + "="*50)
        print("Making prediction with clear conditions...")
        clear_prediction_result = prediction_service.predict_storm_probability(clear_df)
        
        print("\nPrediction Results:")
        print(f"  Prediction: {'STORM' if clear_prediction_result['prediction'] == 1 else 'NO STORM'}")
        print(f"  Probability of No Storm: {clear_prediction_result['probability_no_storm']*100:.2f}%")
        print(f"  Probability of Storm: {clear_prediction_result['probability_storm']*100:.2f}%")
        print(f"  Confidence: {clear_prediction_result['confidence']*100:.2f}%")
        print(f"  Model Used: {clear_prediction_result['model_used']}")
        
        # Evaluate alert for clear conditions
        clear_alert_info = alerting_system.evaluate_alert(clear_prediction_result)
        print(f"\nAlert Level: {clear_alert_info['alert_level']}")
        print(f"Alert Message: {clear_alert_info['message']}")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()