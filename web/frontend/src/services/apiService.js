// API service to connect frontend to backend
class ApiService {
  constructor() {
    this.baseUrl = 'http://localhost:8000'; // Default backend URL
  }

  // Check if backend is available
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      return await response.json();
    } catch (error) {
      console.error('Backend health check failed:', error);
      return { status: 'error', message: 'Backend not available' };
    }
  }

  // Make storm prediction
  async predictStorm(weatherData) {
    try {
      const response = await fetch(`${this.baseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(weatherData),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Prediction API call failed:', error);
      return {
        success: false,
        error: 'Failed to get prediction from backend'
      };
    }
  }

  // Get prediction with alert
  async predictStormWithAlert(weatherData, threshold = 0.6) {
    try {
      const response = await fetch(`${this.baseUrl}/predict_with_alert`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          weather_data: weatherData,
          storm_threshold: threshold
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Prediction with alert API call failed:', error);
      return {
        success: false,
        error: 'Failed to get prediction with alert from backend'
      };
    }
  }

  // Get feature importance
  async getFeatureImportance() {
    try {
      const response = await fetch(`${this.baseUrl}/feature_importance`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('Feature importance API call failed:', error);
      return {
        success: false,
        error: 'Failed to get feature importance from backend'
      };
    }
  }

  // Generate realistic weather data for a location (in a real implementation, this would call weather APIs)
  generateRealisticWeatherData(lat, lon, locationName) {
    // This generates realistic weather data based on location characteristics
    // In a production system, this would call real weather APIs
    
    // Base values that vary by location
    const baseTemp = this.calculateBaseTemperature(lat, lon);
    const baseHumidity = this.calculateBaseHumidity(lat, lon);
    const basePressure = 1013.25; // Standard atmospheric pressure
    
    // Add some randomness
    const temperature = baseTemp + (Math.random() * 10 - 5);
    const humidity = Math.min(100, Math.max(20, baseHumidity + (Math.random() * 20 - 10)));
    const pressure = basePressure + (Math.random() * 20 - 10);
    const windSpeed = Math.random() * 25;
    const windDirection = Math.random() * 360;
    const cloudCover = Math.random() * 100;
    const precipitation = Math.random() * 10;
    
    return {
      temperature: parseFloat(temperature.toFixed(1)),
      humidity: parseFloat(humidity.toFixed(1)),
      pressure: parseFloat(pressure.toFixed(1)),
      wind_speed: parseFloat(windSpeed.toFixed(1)),
      wind_direction: parseFloat(windDirection.toFixed(0)),
      cloud_cover: parseFloat(cloudCover.toFixed(1)),
      precipitation: parseFloat(precipitation.toFixed(1)),
      // Additional features our model expects
      temperature_celsius: parseFloat(temperature.toFixed(1)),
      pressure_mb: parseFloat(pressure.toFixed(1)),
      wind_kph: parseFloat((windSpeed * 1.60934).toFixed(1)),
      visibility_km: parseFloat((10 + Math.random() * 15).toFixed(1)),
      uv_index: parseFloat((Math.random() * 10).toFixed(1)),
      air_quality_Carbon_Monoxide: parseFloat((0.5 + Math.random() * 1.5).toFixed(1)),
      air_quality_Ozone: parseFloat((30 + Math.random() * 40).toFixed(1)),
      air_quality_Nitrogen_dioxide: parseFloat((20 + Math.random() * 30).toFixed(1)),
      air_quality_Sulphur_dioxide: parseFloat((5 + Math.random() * 20).toFixed(1)),
      air_quality_PM2_5: parseFloat((20 + Math.random() * 40).toFixed(1)),
      air_quality_PM10: parseFloat((30 + Math.random() * 50).toFixed(1)),
      reflectivity_max: parseFloat((50 + Math.random() * 200).toFixed(1)),
      reflectivity_mean: parseFloat((30 + Math.random() * 150).toFixed(1)),
      brightness_temp_min: parseFloat((200 + Math.random() * 100).toFixed(1)),
      motion_vector_x: parseFloat((Math.random() * 2 - 1).toFixed(2)),
      motion_vector_y: parseFloat((Math.random() * 2 - 1).toFixed(2)),
      rain: parseFloat(precipitation.toFixed(1)),
      visibility: parseFloat((10 + Math.random() * 15).toFixed(1)),
      dew_point: parseFloat((temperature - (10 + Math.random() * 15)).toFixed(1)),
      cloud: parseFloat(cloudCover.toFixed(1)),
      precip_mm: parseFloat(precipitation.toFixed(1)),
      wind_speed_kph: parseFloat((windSpeed * 1.60934).toFixed(1))
    };
  }

  // Calculate base temperature based on latitude and longitude
  calculateBaseTemperature(lat, lon) {
    // Simplified temperature calculation based on latitude
    // Equator (0°) is hottest, poles are coldest
    const latFactor = Math.abs(lat) / 90;
    const baseTemp = 35 - (latFactor * 40); // Range from -5°C to 35°C
    
    // Adjust for season (simplified)
    const month = new Date().getMonth();
    const seasonalAdjustment = month >= 3 && month <= 8 ? 5 : -5; // Summer vs winter
    
    return baseTemp + seasonalAdjustment;
  }

  // Calculate base humidity based on location
  calculateBaseHumidity(lat, lon) {
    // Coastal areas tend to be more humid
    const isCoastal = Math.abs(lat) < 30 && (Math.abs(lon) > 60 && Math.abs(lon) < 90);
    const baseHumidity = isCoastal ? 70 : 50;
    
    // Add some randomness
    return baseHumidity + (Math.random() * 20 - 10);
  }

  // Simulate getting real weather data (in a real implementation, this would call weather APIs)
  async getRealWeatherData(lat, lon, locationName) {
    // This is a placeholder - in reality, you would call real weather APIs
    console.log(`Fetching weather data for lat: ${lat}, lon: ${lon}`);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Return sample data (this would come from real APIs in production)
    return this.generateRealisticWeatherData(lat, lon, locationName);
  }
}

// Export singleton instance
const apiService = new ApiService();
export default apiService;