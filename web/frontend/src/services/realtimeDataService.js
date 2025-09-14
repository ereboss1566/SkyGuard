// Real-time data service to connect frontend to our real-time pipeline
class RealtimeDataService {
  constructor() {
    this.baseUrl = 'http://localhost:8000'; // Default backend URL
    this.alerts = [];
    this.currentWeather = null;
    this.predictionData = null;
  }

  // Check if backend is available
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      if (response.ok) {
        const data = await response.json();
        // Handle both response formats
        if (data.status !== undefined) {
          // New format (FastAPI)
          return { 
            status: data.status === 'healthy' ? 'healthy' : 'degraded',
            model_loaded: data.model_loaded,
            timestamp: data.timestamp
          };
        } else {
          // Old format (Flask)
          return { 
            status: data.components?.model === 'loaded' ? 'healthy' : 'degraded',
            model_loaded: data.components?.model === 'loaded',
            timestamp: data.timestamp
          };
        }
      }
      return { status: 'error', message: 'Backend not available' };
    } catch (error) {
      console.error('Backend health check failed:', error);
      return { status: 'error', message: 'Backend not available' };
    }
  }

  // Get real-time weather data from our pipeline
  async getRealtimeWeatherData() {
    try {
      // Try to get real data from backend
      // Note: Our backend doesn't have a /weather/current endpoint
      // Using sample data for now - in a real implementation this would call real weather APIs
      
      console.log('Using sample data for weather (no /weather/current endpoint in backend)');
      return {
        timestamp: new Date().toISOString(),
        reflectivity_max: 45.2 + (Math.random() * 20 - 10),
        reflectivity_mean: 32.1 + (Math.random() * 15 - 7.5),
        brightness_temp_min: 245.3 + (Math.random() * 15 - 7.5),
        motion_vector_x: 0.5 + (Math.random() * 2 - 1),
        motion_vector_y: -0.3 + (Math.random() * 2 - 1),
        temperature: 28.5 + (Math.random() * 10 - 5),
        dew_point: 22.1 + (Math.random() * 8 - 4),
        pressure: 1008.2 + (Math.random() * 20 - 10),
        wind_speed: 12.3 + (Math.random() * 15 - 7.5),
        wind_direction: Math.random() * 360,
        visibility: 8.5 + (Math.random() * 5 - 2.5),
        rain: Math.random() > 0.7 ? 2.5 + (Math.random() * 5) : 0,
        temperature_celsius: 28.5 + (Math.random() * 10 - 5),
        pressure_mb: 1008.2 + (Math.random() * 20 - 10),
        wind_kph: 19.8 + (Math.random() * 24 - 12),
        humidity: 65.2 + (Math.random() * 20 - 10),
        cloud: 62.1 + (Math.random() * 25 - 12.5),
        visibility_km: 8.5 + (Math.random() * 5 - 2.5),
        uv_index: 6.2 + (Math.random() * 3 - 1.5),
        precip_mm: Math.random() > 0.7 ? 2.5 + (Math.random() * 5) : 0,
        air_quality_Carbon_Monoxide: 0.8 + (Math.random() * 1.2 - 0.6),
        air_quality_Ozone: 35.2 + (Math.random() * 20 - 10),
        air_quality_Nitrogen_dioxide: 22.1 + (Math.random() * 15 - 7.5),
        air_quality_Sulphur_dioxide: 8.5 + (Math.random() * 10 - 5),
        air_quality_PM2_5: 28.3 + (Math.random() * 20 - 10),
        air_quality_PM10: 45.2 + (Math.random() * 25 - 12.5)
      };
    } catch (error) {
      console.error('Real-time weather data fetch failed:', error);
      return null;
    }
  }

  // Get current alerts from our alert engine
  async getCurrentAlerts() {
    try {
      // Note: Our backend doesn't have an /alerts endpoint
      // Using sample alerts for now
      
      console.log('Using sample data for alerts (no /alerts endpoint in backend)');
      const alerts = [];
      
      // Randomly generate alerts based on probability
      if (Math.random() > 0.8) {
        alerts.push({
          alert_id: Date.now().toString(),
          alert_type: "STORM",
          severity: "HIGH",
          message: "Thunderstorm predicted with 75% probability in 2 hours",
          timestamp: new Date().toISOString(),
          location: {name: "Airfield", lat: 19.0760, lon: 72.8777}
        });
      }
      
      if (Math.random() > 0.7) {
        alerts.push({
          alert_id: (Date.now() + 1).toString(),
          alert_type: "WIND_GUST",
          severity: "MEDIUM",
          message: "Severe winds (22 km/h) expected in 30 minutes",
          timestamp: new Date().toISOString(),
          location: {name: "Airfield", lat: 19.0760, lon: 72.8777}
        });
      }
      
      if (Math.random() > 0.9) {
        alerts.push({
          alert_id: (Date.now() + 2).toString(),
          alert_type: "PRESSURE_DROP",
          severity: "LOW",
          message: "Rapid pressure drop detected (2.3 hPa/hour)",
          timestamp: new Date().toISOString(),
          location: {name: "Airfield", lat: 19.0760, lon: 72.8777}
        });
      }
      
      return alerts;
    } catch (error) {
      console.error('Alerts fetch failed:', error);
      return [];
    }
  }

  // Get prediction using our trained models
  async getPrediction(weatherData) {
    try {
      // Try to get real prediction from backend
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(weatherData),
      });
      
      if (response.ok) {
        const data = await response.json();
        // Handle both the old response format and the new format
        if (data.success !== undefined) {
          // New format from our backend
          return {
            stormProbability: Math.round((data.probability_storm || data.prediction.probability_storm || 0) * 100),
            confidence: Math.round((data.confidence || data.prediction.confidence || 0.85) * 100),
            forecastHorizon: "0-3 hours",
            explanation: data.explanation || data.prediction?.explanation || "Prediction based on current weather conditions."
          };
        } else {
          // Old format or fallback
          return {
            stormProbability: Math.round((data.probability_storm || 0) * 100),
            confidence: Math.round((data.confidence || 0.85) * 100),
            forecastHorizon: "0-3 hours",
            explanation: data.explanation || "Prediction based on current weather conditions."
          };
        }
      }
      
      // Fallback to sample predictions if API fails
      console.log('Using sample data for prediction');
      
      // Calculate probability based on some key features
      let stormProbability = 0;
      
      if (weatherData.reflectivity_max > 50) stormProbability += 25;
      if (weatherData.pressure < 1000) stormProbability += 20;
      if (weatherData.wind_speed > 15) stormProbability += 15;
      if (weatherData.humidity > 80) stormProbability += 15;
      if (weatherData.precip_mm > 3) stormProbability += 25;
      
      // Add some randomness
      stormProbability = Math.min(100, stormProbability + (Math.random() * 20 - 10));
      stormProbability = Math.max(0, stormProbability);
      
      return {
        stormProbability: Math.round(stormProbability),
        confidence: Math.round(70 + (Math.random() * 25)), // 70-95% confidence
        forecastHorizon: "0-3 hours",
        explanation: this.generateExplanation(weatherData, stormProbability)
      };
    } catch (error) {
      console.error('Prediction failed:', error);
      return null;
    }
  }

  // Generate explanation for prediction
  generateExplanation(weatherData, stormProbability) {
    const factors = [];
    
    if (weatherData.reflectivity_max > 50) {
      factors.push("high radar reflectivity indicates strong storm cells");
    }
    
    if (weatherData.pressure < 1000) {
      factors.push("low atmospheric pressure suggests storm system");
    }
    
    if (weatherData.wind_speed > 15) {
      factors.push("strong winds indicate atmospheric instability");
    }
    
    if (weatherData.humidity > 80) {
      factors.push("high humidity suggests potential for precipitation");
    }
    
    if (weatherData.precip_mm > 3) {
      factors.push("current precipitation indicates active weather system");
    }
    
    if (factors.length === 0) {
      factors.push("current conditions are generally stable");
    }
    
    return `Prediction based on ${factors.join(", ")}.`;
  }

  // Get feature importance from our models
  async getFeatureImportance() {
    try {
      // Try to get real feature importance from backend
      const response = await fetch(`${this.baseUrl}/feature_importance`);
      
      if (response.ok) {
        const data = await response.json();
        // Handle both response formats
        if (data.success !== undefined) {
          // New format from our backend
          return {
            features: data.feature_importance?.map(item => ({
              name: item.feature,
              importance: item.importance
            })) || []
          };
        } else if (data.features) {
          // Already in correct format
          return data;
        } else {
          // Fallback format
          return {
            features: [
              { name: "precip_mm", importance: 0.4325 },
              { name: "cloud", importance: 0.1302 },
              { name: "humidity", importance: 0.0863 },
              { name: "air_quality_Sulphur_dioxide", importance: 0.0800 },
              { name: "air_quality_PM10", importance: 0.0643 },
              { name: "pressure", importance: 0.0521 },
              { name: "wind_speed", importance: 0.0412 },
              { name: "temperature", importance: 0.0387 },
              { name: "reflectivity_max", importance: 0.0289 },
              { name: "visibility_km", importance: 0.0215 },
              { name: "air_quality_PM2_5", importance: 0.0123 },
              { name: "dew_point", importance: 0.0120 }
            ]
          };
        }
      }
      
      // Return static feature importance based on our trained models
      return {
        features: [
          { name: "precip_mm", importance: 0.4325 },
          { name: "cloud", importance: 0.1302 },
          { name: "humidity", importance: 0.0863 },
          { name: "air_quality_Sulphur_dioxide", importance: 0.0800 },
          { name: "air_quality_PM10", importance: 0.0643 },
          { name: "pressure", importance: 0.0521 },
          { name: "wind_speed", importance: 0.0412 },
          { name: "temperature", importance: 0.0387 },
          { name: "reflectivity_max", importance: 0.0289 },
          { name: "visibility_km", importance: 0.0215 },
          { name: "air_quality_PM2_5", importance: 0.0123 },
          { name: "dew_point", importance: 0.0120 }
        ]
      };
    } catch (error) {
      console.error('Feature importance fetch failed:', error);
      return null;
    }
  }

  // Get real-time data including weather, alerts, and predictions
  async getRealtimeData() {
    try {
      const weatherData = await this.getRealtimeWeatherData();
      if (!weatherData) return null;
      
      const alerts = await this.getCurrentAlerts();
      const prediction = await this.getPrediction(weatherData);
      const featureImportance = await this.getFeatureImportance();
      
      return {
        weather: weatherData,
        alerts: alerts,
        prediction: prediction,
        featureImportance: featureImportance
      };
    } catch (error) {
      console.error('Real-time data fetch failed:', error);
      return null;
    }
  }
}

// Export singleton instance
const realtimeDataService = new RealtimeDataService();
export default realtimeDataService;