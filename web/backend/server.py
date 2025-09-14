# Simple backend server to serve the SkyGuard API
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os
import configparser

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for development
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Global variables for model components
model = None
scaler = None
imputer = None

def load_model_components():
    """Load the trained model and preprocessing objects"""
    global model, scaler, imputer
    
    try:
        # Define paths to model components
        model_path = os.path.join('..', '..', 'models', 'optimized', 'randomforest_optimized_model.pkl')
        scaler_path = os.path.join('..', '..', 'models', 'optimized', 'scaler.pkl')
        imputer_path = os.path.join('..', '..', 'models', 'optimized', 'imputer.pkl')
        
        # Try current directory first
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'optimized', 'randomforest_optimized_model.pkl')
            scaler_path = os.path.join('models', 'optimized', 'scaler.pkl')
            imputer_path = os.path.join('models', 'optimized', 'imputer.pkl')
        
        # Try absolute path
        if not os.path.exists(model_path):
            base_path = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_path, '..', '..', 'models', 'optimized', 'randomforest_optimized_model.pkl')
            scaler_path = os.path.join(base_path, '..', '..', 'models', 'optimized', 'scaler.pkl')
            imputer_path = os.path.join(base_path, '..', '..', 'models', 'optimized', 'imputer.pkl')
        
        print(f"Looking for model at: {model_path}")
        print(f"Looking for scaler at: {scaler_path}")
        print(f"Looking for imputer at: {imputer_path}")
        
        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("✓ Model loaded successfully")
        else:
            print(f"✗ Model file not found at {model_path}")
            
        # Load scaler
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✓ Scaler loaded successfully")
        else:
            print(f"✗ Scaler file not found at {scaler_path}")
            
        # Load imputer
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
            print("✓ Imputer loaded successfully")
        else:
            print(f"✗ Imputer file not found at {imputer_path}")
            
    except Exception as e:
        print(f"Error loading model components: {e}")
        import traceback
        traceback.print_exc()

# Load model components when server starts
print("Initializing SkyGuard Backend Server...")
print("=" * 50)
load_model_components()
print("=" * 50)

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "SkyGuard Storm Prediction API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "imputer_loaded": imputer is not None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_status = "loaded" if model is not None else "not loaded"
    scaler_status = "loaded" if scaler is not None else "not loaded"
    imputer_status = "loaded" if imputer is not None else "not loaded"
    
    return jsonify({
        "status": "healthy" if (model and scaler and imputer) else "degraded",
        "components": {
            "model": model_status,
            "scaler": scaler_status,
            "imputer": imputer_status
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make storm prediction based on weather data"""
    try:
        # Get weather data from request
        weather_data = request.get_json()
        
        if not weather_data:
            return jsonify({
                "success": False,
                "error": "No weather data provided"
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([weather_data])
        
        # Expected features in the right order
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
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0.0  # Default value for missing features
        
        # Select only the features we need and in the right order
        processed_data = df[expected_features].copy()
        
        # Handle missing values using the trained imputer
        if imputer:
            imputed_data = pd.DataFrame(
                imputer.transform(processed_data),
                columns=processed_data.columns
            )
        else:
            imputed_data = processed_data.fillna(0)  # Fallback if no imputer
        
        # Scale the features using the trained scaler
        if scaler:
            scaled_data = scaler.transform(imputed_data)
        else:
            scaled_data = imputed_data.values  # Fallback if no scaler
        
        # Make prediction
        if model:
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)
        else:
            # Fallback prediction for demo
            prediction = [0]  # No storm
            probability = [[0.8, 0.2]]  # 80% no storm, 20% storm
        
        # Return results
        result = {
            "success": True,
            "prediction": {
                "storm": bool(prediction[0]),
                "probability_no_storm": float(probability[0][0]),
                "probability_storm": float(probability[0][1]),
                "confidence": float(max(probability[0]))
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction error: {str(e)}"
        }), 500

@app.route('/predict_with_alert', methods=['POST'])
def predict_with_alert():
    """Make storm prediction with alert level"""
    try:
        # Get data from request
        data = request.get_json()
        weather_data = data.get('weather_data', {})
        storm_threshold = data.get('storm_threshold', 0.6)
        
        # Make prediction (reuse the predict function logic)
        df = pd.DataFrame([weather_data])
        
        # Expected features in the right order
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
        
        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = 0.0  # Default value for missing features
        
        # Select only the features we need and in the right order
        processed_data = df[expected_features].copy()
        
        # Handle missing values using the trained imputer
        if imputer:
            imputed_data = pd.DataFrame(
                imputer.transform(processed_data),
                columns=processed_data.columns
            )
        else:
            imputed_data = processed_data.fillna(0)  # Fallback if no imputer
        
        # Scale the features using the trained scaler
        if scaler:
            scaled_data = scaler.transform(imputed_data)
        else:
            scaled_data = imputed_data.values  # Fallback if no scaler
        
        # Make prediction
        if model:
            prediction = model.predict(scaled_data)
            probability = model.predict_proba(scaled_data)
        else:
            # Fallback prediction for demo
            prediction = [0]  # No storm
            probability = [[0.8, 0.2]]  # 80% no storm, 20% storm
        
        # Determine alert level
        storm_prob = probability[0][1]
        alert_level = "NONE"
        alert_message = "No significant weather events predicted"
        
        if storm_prob >= storm_threshold:
            alert_level = "STORM_WARNING"
            alert_message = f"STORM WARNING: {storm_prob*100:.1f}% probability of severe weather"
        elif storm_prob >= storm_threshold * 0.7:
            alert_level = "STORM_WATCH"
            alert_message = f"STORM WATCH: {storm_prob*100:.1f}% probability of severe weather"
        
        # Return results
        result = {
            "success": True,
            "prediction": {
                "storm": bool(prediction[0]),
                "probability_no_storm": float(probability[0][0]),
                "probability_storm": float(probability[0][1]),
                "confidence": float(max(probability[0]))
            },
            "alert": {
                "alert_level": alert_level,
                "message": alert_message,
                "probability": float(storm_prob)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Prediction with alert error: {str(e)}"
        }), 500

@app.route('/feature_importance')
def feature_importance():
    """Get feature importance from the trained model"""
    try:
        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded"
            }), 500
        
        # Expected features
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
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': expected_features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Return top 10 features
        top_features = importance_df.head(10).to_dict('records')
        
        return jsonify({
            "success": True,
            "feature_importance": top_features,
            "total_features": len(expected_features)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Feature importance error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("Starting SkyGuard Backend Server...")
    print("Access at: http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    app.run(host='0.0.0.0', port=8000, debug=True)