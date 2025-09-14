import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import CitySearch from './CitySearch';
import locationService from '../services/locationService';
import apiService from '../services/apiService';
import './Dashboard.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = ({ 
  currentLocation, 
  savedLocations, 
  onLocationChange, 
  onAddLocation, 
  onRemoveLocation 
}) => {
  const [weatherData, setWeatherData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [connectionError, setConnectionError] = useState(null);

  // Mock data for charts (will be replaced with real data)
  const probabilityChartData = predictionData ? {
    labels: ['No Storm', 'Storm'],
    datasets: [
      {
        data: [
          Math.round((1 - predictionData.stormProbability/100) * 100),
          predictionData.stormProbability
        ],
        backgroundColor: ['#27ae60', '#e74c3c'],
        borderColor: ['#219653', '#c0392b'],
        borderWidth: 1
      }
    ]
  } : {
    labels: ['No Storm', 'Storm'],
    datasets: [
      {
        data: [42, 58],
        backgroundColor: ['#27ae60', '#e74c3c'],
        borderColor: ['#219653', '#c0392b'],
        borderWidth: 1
      }
    ]
  };

  const weatherTrendData = weatherData ? {
    labels: ['Now', '-30m', '-1h', '-2h', '-3h'],
    datasets: [
      {
        label: 'Temperature (°C)',
        data: [
          weatherData.temperature || 32.5,
          (weatherData.temperature || 32.5) - 1.2,
          (weatherData.temperature || 32.5) - 2.8,
          (weatherData.temperature || 32.5) - 3.3,
          (weatherData.temperature || 32.5) - 4.1
        ],
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.2)',
        tension: 0.1
      },
      {
        label: 'Humidity (%)',
        data: [
          weatherData.humidity || 72,
          (weatherData.humidity || 72) + 3,
          (weatherData.humidity || 72) + 6,
          (weatherData.humidity || 72) + 10,
          (weatherData.humidity || 72) + 13
        ],
        borderColor: '#9b59b6',
        backgroundColor: 'rgba(155, 89, 182, 0.2)',
        tension: 0.1
      }
    ]
  } : {
    labels: ['Now', '-30m', '-1h', '-2h', '-3h'],
    datasets: [
      {
        label: 'Temperature (°C)',
        data: [32.5, 31.8, 30.2, 29.7, 28.9],
        borderColor: '#3498db',
        backgroundColor: 'rgba(52, 152, 219, 0.2)',
        tension: 0.1
      },
      {
        label: 'Humidity (%)',
        data: [72, 75, 78, 82, 85],
        borderColor: '#9b59b6',
        backgroundColor: 'rgba(155, 89, 182, 0.2)',
        tension: 0.1
      }
    ]
  };

  const riskLevelData = predictionData ? {
    labels: ['Safe', 'Caution', 'Danger'],
    datasets: [
      {
        data: [
          predictionData.stormProbability < 30 ? 70 : 30,
          predictionData.stormProbability >= 30 && predictionData.stormProbability < 70 ? 40 : 20,
          predictionData.stormProbability >= 70 ? 50 : 10
        ],
        backgroundColor: [
          predictionData.stormProbability < 30 ? '#27ae60' : '#f39c12',
          predictionData.stormProbability >= 30 && predictionData.stormProbability < 70 ? '#f39c12' : '#27ae60',
          predictionData.stormProbability >= 70 ? '#e74c3c' : '#27ae60'
        ],
        borderColor: [
          predictionData.stormProbability < 30 ? '#219653' : '#e67e22',
          predictionData.stormProbability >= 30 && predictionData.stormProbability < 70 ? '#e67e22' : '#219653',
          predictionData.stormProbability >= 70 ? '#c0392b' : '#219653'
        ],
        borderWidth: 1
      }
    ]
  } : {
    labels: ['Safe', 'Caution', 'Danger'],
    datasets: [
      {
        data: [60, 30, 10],
        backgroundColor: ['#27ae60', '#f39c12', '#e74c3c'],
        borderColor: ['#219653', '#e67e22', '#c0392b'],
        borderWidth: 1
      }
    ]
  };

  // Check backend status on component mount
  useEffect(() => {
    const checkBackend = async () => {
      setBackendStatus('checking');
      const health = await apiService.checkHealth();
      if (health.status === 'healthy') {
        setBackendStatus('connected');
        setConnectionError(null);
      } else {
        setBackendStatus('disconnected');
        setConnectionError(health.message || 'Backend not available');
      }
    };
    
    checkBackend();
    
    // Check periodically
    const interval = setInterval(checkBackend, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch weather data and make prediction when location changes
  useEffect(() => {
    if (currentLocation && backendStatus === 'connected') {
      fetchWeatherAndPredict();
    }
  }, [currentLocation, backendStatus]);

  const fetchWeatherAndPredict = async () => {
    setLoading(true);
    setConnectionError(null);
    
    try {
      // Get real weather data
      const weather = await apiService.getRealWeatherData(
        currentLocation.lat, 
        currentLocation.lon,
        currentLocation.name
      );
      
      setWeatherData(weather);
      
      // Make prediction
      const prediction = await apiService.predictStormWithAlert(weather);
      
      if (prediction.success) {
        setPredictionData({
          stormProbability: Math.round(prediction.prediction.probability_storm * 100),
          confidence: Math.round(prediction.prediction.confidence * 100),
          forecastHorizon: "0-3 hours"
        });
        
        // Create alerts based on prediction
        if (prediction.alert) {
          const newAlert = {
            id: Date.now(),
            type: prediction.alert.alert_level,
            message: prediction.alert.message,
            severity: prediction.alert.alert_level === 'STORM_WARNING' ? 'HIGH' : 
                      prediction.alert.alert_level === 'STORM_WATCH' ? 'MEDIUM' : 'LOW',
            timestamp: new Date(),
            location: currentLocation.name,
            probability: prediction.alert.probability
          };
          
          setAlerts([newAlert]);
        }
      } else {
        // Use sample data if API fails
        setPredictionData({
          stormProbability: 58,
          confidence: 87,
          forecastHorizon: "0-3 hours"
        });
        
        setAlerts([
          {
            id: 1,
            type: "STORM_WARNING",
            message: "High probability of thunderstorms in 2 hours",
            severity: "HIGH",
            timestamp: new Date(Date.now() - 3600000),
            location: currentLocation.name
          }
        ]);
      }
    } catch (error) {
      console.error('Error fetching weather and predicting:', error);
      setConnectionError('Failed to fetch data or make prediction');
      
      // Use sample data if API fails
      setWeatherData({
        temperature: 32.5,
        humidity: 72,
        pressure: 1008.2,
        wind_speed: 15.3,
        wind_direction: 180,
        precipitation: 2.1,
        cloud_cover: 65,
        visibility_km: 10.0,
        uv_index: 7.0,
        air_quality_Carbon_Monoxide: 0.8,
        air_quality_Ozone: 40.0,
        air_quality_Nitrogen_dioxide: 25.0,
        air_quality_Sulphur_dioxide: 10.0,
        air_quality_PM2_5: 30.0,
        air_quality_PM10: 50.0,
        reflectivity_max: 150.0,
        reflectivity_mean: 120.0,
        brightness_temp_min: 250.0,
        motion_vector_x: 0.2,
        motion_vector_y: -0.1,
        rain: 2.0,
        visibility: 10.0,
        dew_point: 25.0,
        temperature_celsius: 32.5,
        pressure_mb: 1008.2,
        wind_kph: 24.6,
        humidity: 72.0,
        cloud: 65.0,
        precip_mm: 2.1
      });
      
      setPredictionData({
        stormProbability: 58,
        confidence: 87,
        forecastHorizon: "0-3 hours"
      });
      
      setAlerts([
        {
          id: 1,
          type: "STORM_WARNING",
          message: "High probability of thunderstorms in 2 hours",
          severity: "HIGH",
          timestamp: new Date(Date.now() - 3600000),
          location: currentLocation.name
        }
      ]);
    } finally {
      setLoading(false);
    }
  };

  // Handle location selection from search
  const handleLocationSelect = (location) => {
    onLocationChange(location);
  };

  // Save current location
  const saveCurrentLocation = () => {
    if (!savedLocations.find(loc => loc.name === currentLocation.name)) {
      onAddLocation(currentLocation);
    }
  };

  // Load saved location
  const loadSavedLocation = (location) => {
    onLocationChange(location);
  };

  // Remove saved location
  const removeSavedLocation = (locationId) => {
    onRemoveLocation(locationId);
  };

  // Refresh data
  const refreshData = () => {
    fetchWeatherAndPredict();
  };

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1>Weather Prediction Dashboard</h1>
        <div className="dashboard-controls">
          <span className={`status-indicator ${backendStatus === 'connected' ? 'running' : backendStatus === 'disconnected' ? 'stopped' : 'warning'}`}></span>
          <span>Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}</span>
          <button className="btn btn-primary" onClick={refreshData} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh Data'}
          </button>
        </div>
      </div>

      {/* Connection Error Message */}
      {connectionError && (
        <div className="card">
          <div className="alert alert-danger">
            <strong>Connection Error:</strong> {connectionError}
            <br />
            <small>Please make sure the backend server is running at http://localhost:8000</small>
          </div>
        </div>
      )}

      {/* City Search Section */}
      <div className="card">
        <div className="card-header">
          <h2>Location Search</h2>
        </div>
        <div className="location-search">
          <CitySearch 
            onLocationSelect={handleLocationSelect}
            currentLocation={currentLocation}
          />
          
          <div className="location-actions">
            <button onClick={saveCurrentLocation} className="btn btn-success">
              Save Current Location
            </button>
          </div>
          
          {savedLocations.length > 0 && (
            <div className="saved-locations">
              <h4>Saved Locations:</h4>
              <div className="locations-list">
                {savedLocations.map((location) => (
                  <div key={location.id} className="location-item">
                    <span 
                      className="location-name"
                      onClick={() => loadSavedLocation(location)}
                    >
                      {location.name}
                    </span>
                    {!location.isDefault && (
                      <button 
                        className="btn btn-danger btn-small"
                        onClick={() => removeSavedLocation(location.id)}
                      >
                        Remove
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {loading && (
        <div className="card">
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Fetching weather data and making predictions...</p>
          </div>
        </div>
      )}

      {weatherData && predictionData && (
        <>
          <div className="dashboard-grid">
            {/* Weather Map */}
            <div className="card dashboard-map">
              <div className="card-header">
                <h2>Weather Map - {currentLocation.name}</h2>
              </div>
              <MapContainer center={[currentLocation.lat, currentLocation.lon]} zoom={10} style={{ height: '400px', width: '100%' }}>
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <Marker position={[currentLocation.lat, currentLocation.lon]}>
                  <Popup>
                    <strong>{currentLocation.name}</strong><br />
                    Current Location<br />
                    Temp: {weatherData.temperature?.toFixed(1) || 'N/A'}°C
                  </Popup>
                </Marker>
                <Circle 
                  center={[currentLocation.lat, currentLocation.lon]} 
                  radius={5000} 
                  color="#e74c3c" 
                  fillColor="#e74c3c" 
                  fillOpacity={0.3}
                >
                  <Popup>
                    <strong>Alert Zone</strong><br />
                    Radius: 5km<br />
                    Storm Probability: {predictionData.stormProbability}%
                  </Popup>
                </Circle>
              </MapContainer>
            </div>

            {/* Current Weather */}
            <div className="card dashboard-weather">
              <div className="card-header">
                <h2>Current Weather Conditions - {currentLocation.name}</h2>
              </div>
              <div className="weather-grid">
                <div className="weather-item">
                  <span className="weather-label">Temperature</span>
                  <span className="weather-value">{weatherData.temperature?.toFixed(1) || 'N/A'}°C</span>
                </div>
                <div className="weather-item">
                  <span className="weather-label">Humidity</span>
                  <span className="weather-value">{weatherData.humidity?.toFixed(0) || 'N/A'}%</span>
                </div>
                <div className="weather-item">
                  <span className="weather-label">Pressure</span>
                  <span className="weather-value">{weatherData.pressure?.toFixed(1) || 'N/A'} hPa</span>
                </div>
                <div className="weather-item">
                  <span className="weather-label">Wind Speed</span>
                  <span className="weather-value">{weatherData.wind_speed?.toFixed(1) || 'N/A'} km/h</span>
                </div>
                <div className="weather-item">
                  <span className="weather-label">Wind Direction</span>
                  <span className="weather-value">{weatherData.wind_direction?.toFixed(0) || 'N/A'}°</span>
                </div>
                <div className="weather-item">
                  <span className="weather-label">Precipitation</span>
                  <span className="weather-value">{weatherData.precip_mm?.toFixed(1) || 'N/A'} mm</span>
                </div>
                <div className="weather-item">
                  <span className="weather-label">Cloud Cover</span>
                  <span className="weather-value">{weatherData.cloud?.toFixed(0) || 'N/A'}%</span>
                </div>
              </div>
            </div>

            {/* Prediction Results */}
            <div className="card dashboard-prediction">
              <div className="card-header">
                <h2>Prediction Results - {currentLocation.name}</h2>
              </div>
              <div className="prediction-content">
                <div className="prediction-main">
                  <h3>Storm Probability: {predictionData.stormProbability}%</h3>
                  <div className="risk-indicator">
                    <span className={`risk-level ${predictionData.stormProbability > 70 ? 'high' : predictionData.stormProbability > 40 ? 'medium' : 'low'}`}>
                      {predictionData.stormProbability > 70 ? 'HIGH RISK' : predictionData.stormProbability > 40 ? 'MODERATE RISK' : 'LOW RISK'}
                    </span>
                  </div>
                  <p>Confidence: {predictionData.confidence}%</p>
                  <p>Forecast Horizon: {predictionData.forecastHorizon}</p>
                </div>
                <div className="prediction-chart">
                  <Doughnut 
                    data={probabilityChartData} 
                    options={{ 
                      responsive: true, 
                      plugins: {
                        legend: {
                          position: 'bottom'
                        }
                      }
                    }} 
                  />
                </div>
              </div>
            </div>

            {/* Alerts */}
            <div className="card dashboard-alerts">
              <div className="card-header">
                <h2>Recent Alerts - {currentLocation.name}</h2>
              </div>
              <div className="alerts-list">
                {alerts.length > 0 ? (
                  alerts.map(alert => (
                    <div key={alert.id} className={`alert alert-${alert.severity.toLowerCase()}`}>
                      <strong>{alert.type.replace('_', ' ')}</strong>: {alert.message}
                      <br />
                      <small>{alert.timestamp.toLocaleString()}</small>
                    </div>
                  ))
                ) : (
                  <div className="no-alerts">
                    <p>No active alerts at this time</p>
                  </div>
                )}
              </div>
            </div>

            {/* Weather Trends */}
            <div className="card dashboard-trends">
              <div className="card-header">
                <h2>Weather Trends - {currentLocation.name}</h2>
              </div>
              <Line 
                data={weatherTrendData} 
                options={{ 
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'top'
                    }
                  }
                }} 
              />
            </div>

            {/* Risk Distribution */}
            <div className="card dashboard-risk">
              <div className="card-header">
                <h2>Risk Distribution - {currentLocation.name}</h2>
              </div>
              <Doughnut 
                data={riskLevelData} 
                options={{ 
                  responsive: true,
                  plugins: {
                    legend: {
                      position: 'bottom'
                    }
                  }
                }} 
              />
            </div>
          </div>

          {/* Explainability Section */}
          <div className="card dashboard-explainability">
            <div className="card-header">
              <h2>Why This Prediction? - {currentLocation.name}</h2>
            </div>
            <div className="explainability-content">
              <p>The model predicts a <strong>{predictionData.stormProbability}%</strong> probability of thunderstorms based on current weather conditions.</p>
              <ul>
                <li>High humidity levels ({weatherData.humidity?.toFixed(0) || 'N/A'}%) indicating atmospheric instability</li>
                <li>Significant cloud cover ({weatherData.cloud?.toFixed(0) || 'N/A'}%) with developing convective patterns</li>
                <li>Moderate wind speeds ({weatherData.wind_speed?.toFixed(1) || 'N/A'} km/h) with directional shear</li>
                <li>Recent precipitation ({weatherData.precip_mm?.toFixed(1) || 'N/A'}mm) suggesting active weather systems</li>
                <li>Pressure trends showing gradual changes</li>
              </ul>
              <p><strong>Recommendation:</strong> Monitor radar closely for developing storm cells. Prepare ground crews for possible equipment securing.</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;