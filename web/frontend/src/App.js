import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import DataIngestion from './components/DataIngestion';
import ModelManagement from './components/ModelManagement';
import AlertSystem from './components/AlertSystem';
import './App.css';

// Simple state management for locations
const useAppState = () => {
  const [currentLocation, setCurrentLocation] = useState(() => {
    // Load from localStorage or use default
    const saved = localStorage.getItem('skyguard_current_location');
    return saved ? JSON.parse(saved) : {
      id: 1,
      name: 'Mumbai',
      lat: 19.0760,
      lon: 72.8777,
      country: 'India'
    };
  });

  const [savedLocations, setSavedLocations] = useState(() => {
    // Load from localStorage or use defaults
    const saved = localStorage.getItem('skyguard_locations');
    return saved ? JSON.parse(saved) : [
      { id: 1, name: 'Mumbai', lat: 19.0760, lon: 72.8777, country: 'India' },
      { id: 2, name: 'Delhi', lat: 28.7041, lon: 77.1025, country: 'India' },
      { id: 3, name: 'Bangalore', lat: 12.9716, lon: 77.5946, country: 'India' }
    ];
  });

  // Save to localStorage whenever state changes
  useEffect(() => {
    localStorage.setItem('skyguard_current_location', JSON.stringify(currentLocation));
  }, [currentLocation]);

  useEffect(() => {
    localStorage.setItem('skyguard_locations', JSON.stringify(savedLocations));
  }, [savedLocations]);

  const updateCurrentLocation = (location) => {
    setCurrentLocation(location);
  };

  const addLocation = (location) => {
    // Check if location already exists
    const exists = savedLocations.find(loc => loc.name === location.name);
    if (!exists) {
      const newLocations = [...savedLocations, location];
      setSavedLocations(newLocations);
    }
    setCurrentLocation(location);
  };

  const removeLocation = (locationId) => {
    // Don't remove if it's the current location and only one left
    if (savedLocations.length <= 1) return;
    
    const newLocations = savedLocations.filter(loc => loc.id !== locationId);
    setSavedLocations(newLocations);
    
    // If we removed the current location, switch to first available
    if (currentLocation.id === locationId) {
      setCurrentLocation(newLocations[0]);
    }
  };

  return {
    currentLocation,
    savedLocations,
    updateCurrentLocation,
    addLocation,
    removeLocation
  };
};

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const appState = useAppState();

  const renderActiveTab = () => {
    const tabProps = {
      currentLocation: appState.currentLocation,
      savedLocations: appState.savedLocations,
      onLocationChange: appState.updateCurrentLocation,
      onAddLocation: appState.addLocation,
      onRemoveLocation: appState.removeLocation
    };

    switch (activeTab) {
      case 'dashboard':
        return <Dashboard {...tabProps} />;
      case 'data':
        return <DataIngestion {...tabProps} />;
      case 'models':
        return <ModelManagement {...tabProps} />;
      case 'alerts':
        return <AlertSystem {...tabProps} />;
      default:
        return <Dashboard {...tabProps} />;
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>SkyGuard - AI/ML Weather Prediction System</h1>
        <nav className="app-nav">
          <button 
            className={activeTab === 'dashboard' ? 'active' : ''}
            onClick={() => setActiveTab('dashboard')}
          >
            Dashboard
          </button>
          <button 
            className={activeTab === 'data' ? 'active' : ''}
            onClick={() => setActiveTab('data')}
          >
            Data Ingestion
          </button>
          <button 
            className={activeTab === 'models' ? 'active' : ''}
            onClick={() => setActiveTab('models')}
          >
            Models
          </button>
          <button 
            className={activeTab === 'alerts' ? 'active' : ''}
            onClick={() => setActiveTab('alerts')}
          >
            Alerts
          </button>
        </nav>
      </header>
      
      <main className="app-main">
        {renderActiveTab()}
      </main>
      
      <footer className="app-footer">
        <p>SkyGuard AI/ML Weather Prediction System for Airfields &copy; 2025</p>
      </footer>
    </div>
  );
}

export default App;