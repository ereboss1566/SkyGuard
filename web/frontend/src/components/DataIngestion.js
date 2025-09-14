import React, { useState, useEffect } from 'react';
import apiService from '../services/apiService';
import './DataIngestion.css';

const DataIngestion = ({ currentLocation }) => {
  const [dataSources, setDataSources] = useState([
    {
      id: 1,
      name: `Radar Data (DWR) - ${currentLocation.name}`,
      type: 'radar',
      status: 'active',
      lastUpdate: new Date(Date.now() - 300000),
      frequency: '5 minutes'
    },
    {
      id: 2,
      name: `Satellite Imagery - ${currentLocation.name}`,
      type: 'satellite',
      status: 'active',
      lastUpdate: new Date(Date.now() - 600000),
      frequency: '10 minutes'
    },
    {
      id: 3,
      name: `Weather Stations (AWS) - ${currentLocation.name}`,
      type: 'aws',
      status: 'active',
      lastUpdate: new Date(Date.now() - 180000),
      frequency: '3 minutes'
    },
    {
      id: 4,
      name: `Historical Records - ${currentLocation.name}`,
      type: 'historical',
      status: 'active',
      lastUpdate: new Date(Date.now() - 86400000),
      frequency: 'daily'
    }
  ]);

  const [newSource, setNewSource] = useState({
    name: '',
    type: 'radar',
    frequency: '5 minutes'
  });

  const [backendStatus, setBackendStatus] = useState('checking');
  const [pipelineStatus, setPipelineStatus] = useState({
    ingestion: 'active',
    cleaning: 'active',
    synchronization: 'active',
    storage: 'pending'
  });

  // Check backend status
  useEffect(() => {
    const checkBackend = async () => {
      setBackendStatus('checking');
      const health = await apiService.checkHealth();
      setBackendStatus(health.status === 'healthy' ? 'connected' : 'disconnected');
    };
    
    checkBackend();
  }, []);

  const addDataSource = () => {
    if (newSource.name) {
      const source = {
        id: dataSources.length + 1,
        name: `${newSource.name} - ${currentLocation.name}`,
        type: newSource.type,
        status: 'pending',
        lastUpdate: new Date(),
        frequency: newSource.frequency
      };
      setDataSources([...dataSources, source]);
      setNewSource({ name: '', type: 'radar', frequency: '5 minutes' });
    }
  };

  const removeDataSource = (id) => {
    setDataSources(dataSources.filter(source => source.id !== id));
  };

  const toggleStatus = (id) => {
    setDataSources(dataSources.map(source => 
      source.id === id 
        ? { ...source, status: source.status === 'active' ? 'inactive' : 'active' } 
        : source
    ));
  };

  // Update data sources when location changes
  useEffect(() => {
    setDataSources(prevSources => 
      prevSources.map(source => ({
        ...source,
        name: source.name.replace(/- [^-]+$/, `- ${currentLocation.name}`)
      }))
    );
  }, [currentLocation]);

  return (
    <div className="data-ingestion">
      <div className="card">
        <div className="card-header">
          <h2>Data Sources Management - {currentLocation.name}</h2>
          <div className="backend-status">
            <span className={`status-indicator ${backendStatus === 'connected' ? 'running' : backendStatus === 'disconnected' ? 'stopped' : 'warning'}`}></span>
            <span>Backend: {backendStatus === 'connected' ? 'Connected' : backendStatus === 'disconnected' ? 'Disconnected' : 'Checking...'}</span>
          </div>
        </div>
        <div className="data-sources-list">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Last Update</th>
                <th>Frequency</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {dataSources.map(source => (
                <tr key={source.id}>
                  <td>{source.name}</td>
                  <td>
                    <span className={`source-type ${source.type}`}>
                      {source.type.toUpperCase()}
                    </span>
                  </td>
                  <td>
                    <span className={`status-indicator ${source.status}`}>
                      {source.status.toUpperCase()}
                    </span>
                  </td>
                  <td>{source.lastUpdate.toLocaleTimeString()}</td>
                  <td>{source.frequency}</td>
                  <td>
                    <button 
                      className="btn btn-primary"
                      onClick={() => toggleStatus(source.id)}
                    >
                      {source.status === 'active' ? 'Stop' : 'Start'}
                    </button>
                    <button 
                      className="btn btn-danger"
                      onClick={() => removeDataSource(source.id)}
                    >
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Add New Data Source</h2>
        </div>
        <div className="add-source-form">
          <div className="form-group">
            <label htmlFor="sourceName">Source Name</label>
            <input
              type="text"
              id="sourceName"
              value={newSource.name}
              onChange={(e) => setNewSource({...newSource, name: e.target.value})}
              placeholder="Enter data source name"
            />
          </div>
          <div className="form-group">
            <label htmlFor="sourceType">Source Type</label>
            <select
              id="sourceType"
              value={newSource.type}
              onChange={(e) => setNewSource({...newSource, type: e.target.value})}
            >
              <option value="radar">Radar Data (DWR)</option>
              <option value="satellite">Satellite Imagery</option>
              <option value="aws">Weather Stations (AWS)</option>
              <option value="historical">Historical Records</option>
              <option value="air-quality">Air Quality Data</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="sourceFrequency">Update Frequency</label>
            <select
              id="sourceFrequency"
              value={newSource.frequency}
              onChange={(e) => setNewSource({...newSource, frequency: e.target.value})}
            >
              <option value="1 minute">1 minute</option>
              <option value="3 minutes">3 minutes</option>
              <option value="5 minutes">5 minutes</option>
              <option value="10 minutes">10 minutes</option>
              <option value="30 minutes">30 minutes</option>
              <option value="hourly">Hourly</option>
              <option value="daily">Daily</option>
            </select>
          </div>
          <button className="btn btn-success" onClick={addDataSource}>
            Add Data Source
          </button>
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <h2>Data Processing Pipeline - {currentLocation.name}</h2>
        </div>
        <div className="pipeline-status">
          <div className={`pipeline-step ${pipelineStatus.ingestion === 'active' ? 'active' : ''}`}>
            <h3>Data Ingestion</h3>
            <p>Collecting data from all sources</p>
            <span className={`status-indicator ${pipelineStatus.ingestion}`}></span>
          </div>
          <div className={`pipeline-step ${pipelineStatus.cleaning === 'active' ? 'active' : ''}`}>
            <h3>Data Cleaning</h3>
            <p>Imputing missing values, filtering noise</p>
            <span className={`status-indicator ${pipelineStatus.cleaning}`}></span>
          </div>
          <div className={`pipeline-step ${pipelineStatus.synchronization === 'active' ? 'active' : ''}`}>
            <h3>Data Synchronization</h3>
            <p>Aligning timestamps across sources</p>
            <span className={`status-indicator ${pipelineStatus.synchronization}`}></span>
          </div>
          <div className={`pipeline-step ${pipelineStatus.storage === 'active' ? 'active' : ''}`}>
            <h3>Data Storage</h3>
            <p>Storing in time-series database</p>
            <span className={`status-indicator ${pipelineStatus.storage}`}></span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataIngestion;