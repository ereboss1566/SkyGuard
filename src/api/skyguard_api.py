"""
REST API for SkyGuard Storm Prediction System
"""
import os
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SkyGuard Storm Prediction API",
    description="Real-time storm prediction API for airfield weather monitoring",
    version="1.0.0"
)

# Global variables for model and preprocessors
model = None
scaler = None
imputer = None

# Pydantic models for request/response
class WeatherData(BaseModel):
    """Weather data input model"""
    reflectivity_max: Optional[float] = 0.0
    reflectivity_mean: Optional[float] = 0.0
    brightness_temp_min: Optional[float] = 250.0
    motion_vector_x: Optional[float] = 0.0
    motion_vector_y: Optional[float] = 0.0
    temperature: Optional[float] = 25.0
    dew_point: Optional[float] = 15.0
    pressure: Optional[float] = 1013.0
    wind_speed: Optional[float] = 5.0
    wind_direction: Optional[float] = 0.0
    visibility: Optional[float] = 10.0
    rain: Optional[float] = 0.0
    temperature_celsius: Optional[float] = 25.0
    pressure_mb: Optional[float] = 1013.0
    wind_kph: Optional[float] = 5.0 * 1.60934
    humidity: Optional[float] = 50.0
    cloud: Optional[float] = 20.0
    visibility_km: Optional[float] = 10.0
    uv_index: Optional[float] = 5.0
    precip_mm: Optional[float] = 0.0
    air_quality_Carbon_Monoxide: Optional[float] = 0.5
    air_quality_Ozone: Optional[float] = 30.0
    air_quality_Nitrogen_dioxide: Optional[float] = 15.0
    air_quality_Sulphur_dioxide: Optional[float] = 5.0
    air_quality_PM2_5: Optional[float] = 10.0
    air_quality_PM10: Optional[float] = 20.0

class PredictionResult(BaseModel):
    """Prediction result model"""
    prediction: int
    probability_no_storm: float
    probability_storm: float
    confidence: float
    timestamp: str
    model_used: str

class AlertInfo(BaseModel):
    """Alert information model"""
    alert_level: str
    message: str
    probability: float
    timestamp: str

def load_models():
    """Load trained models and preprocessing objects"""
    global model, scaler, imputer
    
    try:
        # Load the Random Forest model (best performing)
        model_file = "models/optimized/randomforest_optimized_model.pkl"
        model = joblib.load(model_file)
        
        # Load preprocessing objects
        scaler_file = "models/optimized/scaler.pkl"
        scaler = joblib.load(scaler_file)
        
        imputer_file = "models/optimized/imputer.pkl"
        imputer = joblib.load(imputer_file)
        
        logger.info("Successfully loaded models and preprocessing objects")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

def preprocess_input_data(weather_data: WeatherData) -> pd.DataFrame:
    """Preprocess input data to match training format"""
    # Convert to DataFrame
    data_dict = weather_data.dict()
    # Fix the column name for PM2.5
    data_dict['air_quality_PM2.5'] = data_dict.pop('air_quality_PM2_5', 0.0)
    
    df = pd.DataFrame([data_dict])
    
    # Map incoming feature names to expected feature names (matching the trained model)
    feature_mapping = {
        'reflectivity_max': 'reflectivity_max_x',
        'reflectivity_mean': 'reflectivity_mean_x',
        'brightness_temp_min': 'brightness_temp_min_x',
        'motion_vector_x': 'motion_vector_x_x',
        'motion_vector_y': 'motion_vector_y_x'
    }
    
    # Rename columns
    df = df.rename(columns=feature_mapping)
    
    # Ensure we have the right columns in the right order (matching the trained model)
    expected_features = [
        'reflectivity_max_x', 'reflectivity_mean_x', 'brightness_temp_min_x', 
        'motion_vector_x_x', 'motion_vector_y_x', 'temperature', 'dew_point', 
        'pressure', 'wind_speed', 'wind_direction', 'visibility', 'rain', 
        'temperature_celsius', 'pressure_mb', 'wind_kph', 'humidity', 
        'cloud', 'visibility_km', 'uv_index', 'precip_mm', 
        'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
        'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
        'air_quality_PM2.5', 'air_quality_PM10'
    ]
    
    # Ensure all expected columns are present
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0
    
    # Select only the features we need and in the right order
    processed_data = df[expected_features].copy()
    
    return processed_data

def predict_storm_probability(weather_data: pd.DataFrame) -> dict:
    """Make storm prediction using the trained model"""
    global model, scaler, imputer
    
    try:
        # Handle missing values using the trained imputer
        imputed_data = pd.DataFrame(
            imputer.transform(weather_data),
            columns=weather_data.columns
        )
        
        # Scale the features using the trained scaler
        scaled_data = scaler.transform(imputed_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probability = model.predict_proba(scaled_data)
        
        # Return results
        result = {
            'prediction': int(prediction[0]),
            'probability_no_storm': float(probability[0][0]),
            'probability_storm': float(probability[0][1]),
            'confidence': float(max(probability[0])),
            'timestamp': datetime.now().isoformat(),
            'model_used': 'Random Forest (Optimized)'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise

def evaluate_alert(probability_storm: float, storm_threshold: float = 0.6) -> dict:
    """Evaluate if an alert should be triggered"""
    timestamp = datetime.now().isoformat()
    
    alert_level = "NONE"
    alert_message = "No significant weather events predicted"
    
    if probability_storm >= storm_threshold:
        alert_level = "STORM_WARNING"
        alert_message = f"STORM WARNING: {probability_storm*100:.1f}% probability of severe weather"
    elif probability_storm >= storm_threshold * 0.7:
        alert_level = "STORM_WATCH"
        alert_message = f"STORM WATCH: {probability_storm*100:.1f}% probability of severe weather"
    
    return {
        'alert_level': alert_level,
        'message': alert_message,
        'probability': probability_storm,
        'timestamp': timestamp
    }

@app.on_event("startup")
async def startup_event():
    """Load models when the app starts"""
    logger.info("Starting up SkyGuard API...")
    load_models()
    logger.info("SkyGuard API ready")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SkyGuard Storm Prediction API", 
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResult)
async def predict_storm(weather_data: WeatherData):
    """Make a storm prediction based on weather data"""
    try:
        # Preprocess the input data
        processed_data = preprocess_input_data(weather_data)
        
        # Make prediction
        result = predict_storm_probability(processed_data)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_with_alert", response_model=dict)
async def predict_storm_with_alert(weather_data: WeatherData, storm_threshold: float = 0.6):
    """Make a storm prediction and evaluate alert level"""
    try:
        # Preprocess the input data
        processed_data = preprocess_input_data(weather_data)
        
        # Make prediction
        prediction_result = predict_storm_probability(processed_data)
        
        # Evaluate alert
        alert_info = evaluate_alert(prediction_result['probability_storm'], storm_threshold)
        
        # Combine results
        result = {
            "prediction": prediction_result,
            "alert": alert_info
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in prediction with alert endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/feature_importance")
async def get_feature_importance(top_n: int = 10):
    """Get feature importance from the trained model"""
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        expected_features = [
            'reflectivity_max_x', 'reflectivity_mean_x', 'brightness_temp_min_x', 
            'motion_vector_x_x', 'motion_vector_y_x', 'temperature', 'dew_point', 
            'pressure', 'wind_speed', 'wind_direction', 'visibility', 'rain', 
            'temperature_celsius', 'pressure_mb', 'wind_kph', 'humidity', 
            'cloud', 'visibility_km', 'uv_index', 'precip_mm', 
            'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
            'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
            'air_quality_PM2.5', 'air_quality_PM10'
        ]
        
        importance_df = pd.DataFrame({
            'feature': expected_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Return top N features
        top_features = importance_df.head(top_n).to_dict('records')
        
        return {
            "feature_importance": top_features,
            "total_features": len(expected_features)
        }
        
    except Exception as e:
        logger.error(f"Error in feature importance endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)