import React, { useState, useEffect } from 'react';
import apiService from '../services/apiService';
import './AlertSystem.css';

const AlertSystem = ({ currentLocation }) => {
  const [alerts, setAlerts] = useState([
    {
      id: 1,
      type: 'STORM_WARNING',
      severity: 'HIGH',
      message: 'Thunderstorm predicted with 80% probability in 2 hours',
      location: currentLocation.name,
      timestamp: new Date(Date.now() - 3600000),
      status: 'active',
      acknowledged: false,
      explanation: 'Triggered by radar echoes + pressure drop in north sector'
    },
    {
      id: 2,
      type: 'WIND_GUST',
      severity: 'MEDIUM',
      message: 'Severe winds (65 km/h) expected in 30 minutes',
      location: currentLocation.name,
      timestamp: new Date(Date.now() - 7200000),
      status: 'active',
      acknowledged: true,
      explanation: 'Wind shear detected in upper atmosphere layers'
    },
    {
      id: 3,
      type: 'STORM_WATCH',
      severity: 'LOW',
      message: 'Possible thunderstorm development in 4-6 hours',
      location: currentLocation.name,
      timestamp: new Date(Date.now() - 10800000),
      status: 'resolved',
      acknowledged: true,
      explanation: 'Cloud formation patterns indicate potential development'
    }
  ]);

  const [alertSettings, setAlertSettings] = useState({
    stormThreshold: 60,
    windThreshold: 50,
    enableAudio: true,
    enableEmail: false,
    enableSMS: false
  });

  const [newAlert, setNewAlert] = useState({
    type: 'STORM_WARNING',
    severity: 'MEDIUM',
    message: '',
    location: currentLocation.name
  });

  const [backendStatus, setBackendStatus] = useState('checking');

  // Check backend status
  useEffect(() => {
    const checkBackend = async () => {
      setBackendStatus('checking');
      const health = await apiService.checkHealth();
      setBackendStatus(health.status === 'healthy' ? 'connected' : 'disconnected');
    };
    
    checkBackend();
  }, []);

  const acknowledgeAlert = (id) => {
    setAlerts(alerts.map(alert => 
      alert.id === id ? { ...alert, acknowledged: true } : alert
    ));
  };

  const resolveAlert = (id) => {
    setAlerts(alerts.map(alert => 
      alert.id === id ? { ...alert, status: 'resolved' } : alert
    ));
  };

  const addAlert = () => {
    if (newAlert.message) {
      const alert = {
        id: alerts.length + 1,
        ...newAlert,
        timestamp: new Date(),
        status: 'active',
        acknowledged: false,
        explanation: 'Manually created alert'
      };
      setAlerts([alert, ...alerts]);
      setNewAlert({
        type: 'STORM_WARNING',
        severity: 'MEDIUM',
        message: '',
        location: currentLocation.name
      });
    }
  };

  const updateSettings = (field, value) => {
    setAlertSettings({ ...alertSettings, [field]: value });
  };

  // Simulate new alerts
  useEffect(() => {
    const interval = setInterval(() => {
      // Randomly add new alerts for demo
      if (Math.random() > 0.95) {
        const severities = ['LOW', 'MEDIUM', 'HIGH'];
        const types = ['STORM_WATCH', 'WIND_GUST', 'PRESSURE_DROP'];
        const messages = [
          'Radar detecting developing storm cells',
          'Wind gusts increasing in northern sector',
          'Pressure dropping rapidly',
          'Cloud cover increasing significantly'
        ];
        
        const newAlert = {
          id: alerts.length + 1,
          type: types[Math.floor(Math.random() * types.length)],
          severity: severities[Math.floor(Math.random() * severities.length)],
          message: messages[Math.floor(Math.random() * messages.length)],
          location: currentLocation.name,
          timestamp: new Date(),
          status: 'active',
          acknowledged: false,
          explanation: 'Automatically detected by system'
        };
        
        setAlerts(prevAlerts => [newAlert, ...prevAlerts]);
      }
    }, 10000);
    
    return () => clearInterval(interval);
  }, [alerts.length, currentLocation.name]);

  // Update new alert location when current location changes
  useEffect(() => {
    setNewAlert(prev => ({
      ...prev,
      location: currentLocation.name
    }));
    
    // Update existing alerts location (in a real app, you might filter instead)
    setAlerts(prevAlerts => 
      prevAlerts.map(alert => ({
        ...alert,
        location: currentLocation.name
      }))
    );
  }, [currentLocation]);

  return (
    <div className="alert-system">
      <div className="card">
        <div className="card-header">
          <h2>Active Alerts - {currentLocation.name}</h2>
          <div className="backend-status">
            <span className={`status-indicator ${backendStatus === 'connected' ? 'running' : backendStatus === 'disconnected' ? 'stopped' : 'warning'}`}></span>
            <span>Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}</span>
          </div>
        </div>
        <div className="alerts-list">
          {alerts.filter(alert => alert.status === 'active').map(alert => (
            <div key={alert.id} className={`alert-card ${alert.severity.toLowerCase()} ${alert.acknowledged ? 'acknowledged' : ''}`}>
              <div className="alert-header">
                <span className={`alert-type ${alert.type}`}>{alert.type.replace('_', ' ')}</span>
                <span className={`alert-severity ${alert.severity}`}>{alert.severity}</span>
              </div>
              <div className="alert-content">
                <p className="alert-message">{alert.message}</p>
                <p className="alert-location">{alert.location}</p>
                <p className="alert-time">{alert.timestamp.toLocaleString()}</p>
                <p className="alert-explanation"><strong>Reason:</strong> {alert.explanation}</p>
              </div>
              <div className="alert-actions">
                {!alert.acknowledged && (
                  <button 
                    className="btn btn-primary"
                    onClick={() => acknowledgeAlert(alert.id)}
                  >
                    Acknowledge
                  </button>
                )}
                <button 
                  className="btn btn-danger"
                  onClick={() => resolveAlert(alert.id)}
                >
                  Resolve
                </button>
              </div>
            </div>
          ))}
          
          {alerts.filter(alert => alert.status === 'active').length === 0 && (
            <div className="no-alerts">
              <p>No active alerts at this time for {currentLocation.name}</p>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Alert History - {currentLocation.name}</h2>
        </div>
        <div className="alerts-history">
          <table className="alerts-table">
            <thead>
              <tr>
                <th>Type</th>
                <th>Severity</th>
                <th>Message</th>
                <th>Location</th>
                <th>Time</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {alerts.map(alert => (
                <tr key={alert.id} className={alert.status === 'resolved' ? 'resolved' : ''}>
                  <td>{alert.type.replace('_', ' ')}</td>
                  <td>
                    <span className={`severity-badge ${alert.severity}`}>
                      {alert.severity}
                    </span>
                  </td>
                  <td>{alert.message}</td>
                  <td>{alert.location}</td>
                  <td>{alert.timestamp.toLocaleTimeString()}</td>
                  <td>
                    <span className={`status-badge ${alert.status}`}>
                      {alert.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Create Manual Alert</h2>
        </div>
        <div className="manual-alert-form">
          <div className="form-group">
            <label htmlFor="alertType">Alert Type</label>
            <select
              id="alertType"
              value={newAlert.type}
              onChange={(e) => setNewAlert({...newAlert, type: e.target.value})}
            >
              <option value="STORM_WARNING">Storm Warning</option>
              <option value="STORM_WATCH">Storm Watch</option>
              <option value="WIND_GUST">Wind Gust</option>
              <option value="PRESSURE_DROP">Pressure Drop</option>
              <option value="VISIBILITY">Visibility Issue</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="alertSeverity">Severity</label>
            <select
              id="alertSeverity"
              value={newAlert.severity}
              onChange={(e) => setNewAlert({...newAlert, severity: e.target.value})}
            >
              <option value="LOW">Low</option>
              <option value="MEDIUM">Medium</option>
              <option value="HIGH">High</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="alertMessage">Message</label>
            <textarea
              id="alertMessage"
              value={newAlert.message}
              onChange={(e) => setNewAlert({...newAlert, message: e.target.value})}
              placeholder="Enter alert message"
              rows="3"
            />
          </div>
          <div className="form-group">
            <label htmlFor="alertLocation">Location</label>
            <input
              type="text"
              id="alertLocation"
              value={newAlert.location}
              onChange={(e) => setNewAlert({...newAlert, location: e.target.value})}
              placeholder="Enter location"
            />
          </div>
          <button className="btn btn-success" onClick={addAlert}>
            Create Alert
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Alert Settings</h2>
        </div>
        <div className="alert-settings">
          <div className="setting-group">
            <label>Storm Probability Threshold (%)</label>
            <input
              type="range"
              min="30"
              max="90"
              value={alertSettings.stormThreshold}
              onChange={(e) => updateSettings('stormThreshold', parseInt(e.target.value))}
            />
            <span>{alertSettings.stormThreshold}%</span>
          </div>
          <div className="setting-group">
            <label>Wind Gust Threshold (km/h)</label>
            <input
              type="range"
              min="30"
              max="100"
              value={alertSettings.windThreshold}
              onChange={(e) => updateSettings('windThreshold', parseInt(e.target.value))}
            />
            <span>{alertSettings.windThreshold} km/h</span>
          </div>
          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={alertSettings.enableAudio}
                onChange={(e) => updateSettings('enableAudio', e.target.checked)}
              />
              Enable Audio Alerts
            </label>
          </div>
          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={alertSettings.enableEmail}
                onChange={(e) => updateSettings('enableEmail', e.target.checked)}
              />
              Enable Email Notifications
            </label>
          </div>
          <div className="setting-group">
            <label>
              <input
                type="checkbox"
                checked={alertSettings.enableSMS}
                onChange={(e) => updateSettings('enableSMS', e.target.checked)}
              />
              Enable SMS Notifications
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertSystem;