// Location state management service
class LocationState {
  constructor() {
    // Initialize with default locations
    this.locations = [
      { id: 1, name: 'Mumbai', lat: 19.0760, lon: 72.8777, country: 'India', isDefault: true },
      { id: 2, name: 'Delhi', lat: 28.7041, lon: 77.1025, country: 'India', isDefault: true },
      { id: 3, name: 'Bangalore', lat: 12.9716, lon: 77.5946, country: 'India', isDefault: true }
    ];
    
    // Load saved locations from localStorage if available
    this.loadFromStorage();
    
    // Current selected location
    this.currentLocation = this.locations[0];
    
    // Callbacks for state changes
    this.subscribers = [];
  }

  // Subscribe to location changes
  subscribe(callback) {
    this.subscribers.push(callback);
    // Return unsubscribe function
    return () => {
      const index = this.subscribers.indexOf(callback);
      if (index > -1) {
        this.subscribers.splice(index, 1);
      }
    };
  }

  // Notify all subscribers of state changes
  notifySubscribers() {
    this.subscribers.forEach(callback => callback(this.getCurrentLocation()));
    this.saveToStorage();
  }

  // Load locations from localStorage
  loadFromStorage() {
    try {
      const savedLocations = localStorage.getItem('skyguard_locations');
      if (savedLocations) {
        const parsedLocations = JSON.parse(savedLocations);
        // Merge with default locations, preserving defaults
        const mergedLocations = [...this.locations];
        parsedLocations.forEach(savedLoc => {
          if (!mergedLocations.find(loc => loc.name === savedLoc.name)) {
            mergedLocations.push(savedLoc);
          }
        });
        this.locations = mergedLocations;
      }
    } catch (error) {
      console.warn('Failed to load locations from storage:', error);
    }
  }

  // Save locations to localStorage
  saveToStorage() {
    try {
      localStorage.setItem('skyguard_locations', JSON.stringify(this.locations));
      localStorage.setItem('skyguard_current_location', JSON.stringify(this.currentLocation));
    } catch (error) {
      console.warn('Failed to save locations to storage:', error);
    }
  }

  // Get all locations
  getLocations() {
    return [...this.locations];
  }

  // Get current location
  getCurrentLocation() {
    return { ...this.currentLocation };
  }

  // Set current location
  setCurrentLocation(location) {
    this.currentLocation = { ...location };
    this.notifySubscribers();
  }

  // Add new location
  addLocation(location) {
    // Check if location already exists
    const existingLocation = this.locations.find(loc => loc.name === location.name);
    if (existingLocation) {
      // If it exists, just set it as current
      this.setCurrentLocation(existingLocation);
      return existingLocation;
    }
    
    // Add new location with unique ID
    const newLocation = {
      id: Date.now(), // Simple unique ID
      ...location,
      isDefault: false
    };
    
    this.locations.push(newLocation);
    this.setCurrentLocation(newLocation);
    this.notifySubscribers();
    return newLocation;
  }

  // Remove location (can't remove default locations)
  removeLocation(locationId) {
    const locationToRemove = this.locations.find(loc => loc.id === locationId);
    if (locationToRemove && !locationToRemove.isDefault) {
      this.locations = this.locations.filter(loc => loc.id !== locationId);
      
      // If we removed the current location, set to first available
      if (this.currentLocation.id === locationId) {
        this.currentLocation = this.locations[0];
      }
      
      this.notifySubscribers();
      return true;
    }
    return false;
  }

  // Search for locations
  searchLocations(query) {
    const normalizedQuery = query.toLowerCase().trim();
    return this.locations.filter(location => 
      location.name.toLowerCase().includes(normalizedQuery) ||
      location.country.toLowerCase().includes(normalizedQuery)
    );
  }

  // Initialize from storage on app start
  initialize() {
    try {
      const savedCurrentLocation = localStorage.getItem('skyguard_current_location');
      if (savedCurrentLocation) {
        this.currentLocation = JSON.parse(savedCurrentLocation);
      }
    } catch (error) {
      console.warn('Failed to load current location from storage:', error);
    }
  }
}

// Export singleton instance
const locationState = new LocationState();
export default locationState;