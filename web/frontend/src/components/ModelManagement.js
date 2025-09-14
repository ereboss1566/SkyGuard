import React, { useState, useEffect } from 'react';
import apiService from '../services/apiService';
import './ModelManagement.css';

const ModelManagement = ({ currentLocation }) => {
  const [models, setModels] = useState([
    {
      id: 1,
      name: `Random Forest - Storm Prediction - ${currentLocation.name}`,
      type: 'nowcasting',
      accuracy: 98.5,
      lastTrained: new Date(Date.now() - 86400000),
      status: 'deployed',
      version: 'v2.1.3'
    },
    {
      id: 2,
      name: `LSTM - Wind Gust Prediction - ${currentLocation.name}`,
      type: 'nowcasting',
      accuracy: 92.3,
      lastTrained: new Date(Date.now() - 172800000),
      status: 'deployed',
      version: 'v1.5.2'
    },
    {
      id: 3,
      name: `CNN - Radar Pattern Analysis - ${currentLocation.name}`,
      type: 'nowcasting',
      accuracy: 89.7,
      lastTrained: new Date(Date.now() - 259200000),
      status: 'deployed',
      version: 'v1.2.4'
    },
    {
      id: 4,
      name: `Ensemble Model - 24hr Forecast - ${currentLocation.name}`,
      type: 'medium-term',
      accuracy: 85.2,
      lastTrained: new Date(Date.now() - 604800000),
      status: 'deployed',
      version: 'v3.0.1'
    }
  ]);

  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [backendStatus, setBackendStatus] = useState('checking');
  const [featureImportance, setFeatureImportance] = useState([]);

  // Check backend status and fetch feature importance
  useEffect(() => {
    const initialize = async () => {
      // Check backend status
      setBackendStatus('checking');
      const health = await apiService.checkHealth();
      setBackendStatus(health.status === 'healthy' ? 'connected' : 'disconnected');
      
      // Fetch feature importance
      if (health.status === 'healthy') {
        const importance = await apiService.getFeatureImportance();
        if (importance.success) {
          setFeatureImportance(importance.feature_importance);
        }
      }
    };
    
    initialize();
  }, []);

  const startTraining = async () => {
    setTrainingStatus('training');
    // Simulate training process
    setTimeout(() => {
      setTrainingStatus('completed');
      // Update model accuracy
      setModels(models.map(model => 
        model.id === selectedModel.id 
          ? { ...model, accuracy: Math.min(99.9, model.accuracy + Math.random() * 2), lastTrained: new Date() } 
          : model
      ));
      setTimeout(() => setTrainingStatus('idle'), 3000);
    }, 3000);
  };

  const deployModel = (id) => {
    setModels(models.map(model => 
      model.id === id 
        ? { ...model, status: 'deployed' } 
        : model.status === 'deployed' ? { ...model, status: 'archived' } : model
    ));
  };

  // Update models when location changes
  useEffect(() => {
    setModels(prevModels => 
      prevModels.map(model => ({
        ...model,
        name: model.name.replace(/- [^-]+$/, `- ${currentLocation.name}`)
      }))
    );
    
    setSelectedModel(prevSelected => ({
      ...prevSelected,
      name: prevSelected.name.replace(/- [^-]+$/, `- ${currentLocation.name}`)
    }));
  }, [currentLocation]);

  return (
    <div className="model-management">
      <div className="card">
        <div className="card-header">
          <h2>Model Management - {currentLocation.name}</h2>
          <div className="backend-status">
            <span className={`status-indicator ${backendStatus === 'connected' ? 'running' : backendStatus === 'disconnected' ? 'stopped' : 'warning'}`}></span>
            <span>Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}</span>
          </div>
        </div>
        <div className="models-grid">
          {models.map(model => (
            <div 
              key={model.id} 
              className={`model-card ${selectedModel.id === model.id ? 'selected' : ''}`}
              onClick={() => setSelectedModel(model)}
            >
              <h3>{model.name}</h3>
              <div className="model-info">
                <span className={`model-type ${model.type}`}>
                  {model.type === 'nowcasting' ? 'Nowcasting (0-3 hrs)' : 'Medium-term (24 hrs)'}
                </span>
                <span className={`model-status ${model.status}`}>
                  {model.status.toUpperCase()}
                </span>
              </div>
              <div className="model-metrics">
                <p>Accuracy: <strong>{model.accuracy}%</strong></p>
                <p>Version: {model.version}</p>
                <p>Last Trained: {model.lastTrained.toLocaleDateString()}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Model Details</h2>
        </div>
        {selectedModel && (
          <div className="model-details">
            <h3>{selectedModel.name}</h3>
            <div className="model-details-grid">
              <div className="detail-item">
                <label>Type:</label>
                <span>{selectedModel.type === 'nowcasting' ? 'Nowcasting (0-3 hrs)' : 'Medium-term (24 hrs)'}</span>
              </div>
              <div className="detail-item">
                <label>Accuracy:</label>
                <span>{selectedModel.accuracy}%</span>
              </div>
              <div className="detail-item">
                <label>Version:</label>
                <span>{selectedModel.version}</span>
              </div>
              <div className="detail-item">
                <label>Status:</label>
                <span className={`model-status ${selectedModel.status}`}>
                  {selectedModel.status.toUpperCase()}
                </span>
              </div>
              <div className="detail-item">
                <label>Last Trained:</label>
                <span>{selectedModel.lastTrained.toLocaleString()}</span>
              </div>
              <div className="detail-item">
                <label>Features Used:</label>
                <span>
                  {selectedModel.id === 1 ? '26 weather features including precipitation, humidity, pressure' : 
                   selectedModel.id === 2 ? 'Time-series wind data, pressure trends' :
                   selectedModel.id === 3 ? 'Radar reflectivity, satellite imagery' :
                   'Synoptic-scale data, historical patterns'}
                </span>
              </div>
            </div>
            
            <div className="model-actions">
              <button 
                className="btn btn-primary"
                onClick={startTraining}
                disabled={trainingStatus === 'training'}
              >
                {trainingStatus === 'training' ? 'Training in Progress...' : 'Retrain Model'}
              </button>
              <button 
                className="btn btn-success"
                onClick={() => deployModel(selectedModel.id)}
                disabled={selectedModel.status === 'deployed'}
              >
                Deploy Model
              </button>
              <button className="btn btn-danger">
                Archive Model
              </button>
            </div>
            
            {trainingStatus === 'training' && (
              <div className="training-progress">
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
                <p>Training model with latest data...</p>
              </div>
            )}
            
            {trainingStatus === 'completed' && (
              <div className="alert alert-success">
                Model training completed successfully! Accuracy improved to {selectedModel.accuracy.toFixed(1)}%
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Feature Importance - {currentLocation.name}</h2>
        </div>
        <div className="feature-importance">
          {featureImportance.length > 0 ? (
            featureImportance.map((feature, index) => (
              <div key={index} className="feature-bar">
                <span className="feature-name">{feature.feature}</span>
                <div className="bar-container">
                  <div 
                    className="bar" 
                    style={{width: `${feature.importance * 100}%`}}
                  ></div>
                </div>
                <span className="feature-value">{(feature.importance * 100).toFixed(1)}%</span>
              </div>
            ))
          ) : (
            // Fallback to static data if API fails
            <>
              <div className="feature-bar">
                <span className="feature-name">Precipitation (precip_mm)</span>
                <div className="bar-container">
                  <div className="bar" style={{width: '43%'}}></div>
                </div>
                <span className="feature-value">43.3%</span>
              </div>
              <div className="feature-bar">
                <span className="feature-name">Cloud cover</span>
                <div className="bar-container">
                  <div className="bar" style={{width: '13%'}}></div>
                </div>
                <span className="feature-value">13.0%</span>
              </div>
              <div className="feature-bar">
                <span className="feature-name">Humidity</span>
                <div className="bar-container">
                  <div className="bar" style={{width: '8.6%'}}></div>
                </div>
                <span className="feature-value">8.6%</span>
              </div>
              <div className="feature-bar">
                <span className="feature-name">Air Quality (SO2)</span>
                <div className="bar-container">
                  <div className="bar" style={{width: '8.0%'}}></div>
                </div>
                <span className="feature-value">8.0%</span>
              </div>
              <div className="feature-bar">
                <span className="feature-name">Air Quality (PM10)</span>
                <div className="bar-container">
                  <div className="bar" style={{width: '6.4%'}}></div>
                </div>
                <span className="feature-value">6.4%</span>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelManagement;