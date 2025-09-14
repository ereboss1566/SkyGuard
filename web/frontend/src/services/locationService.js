// Location service for handling geocoding and location management
class LocationService {
  constructor() {
    // Predefined list of major cities for demo purposes
    this.cities = [
      { name: 'Mumbai', lat: 19.0760, lon: 72.8777, country: 'India' },
      { name: 'Delhi', lat: 28.7041, lon: 77.1025, country: 'India' },
      { name: 'Bangalore', lat: 12.9716, lon: 77.5946, country: 'India' },
      { name: 'Chennai', lat: 13.0827, lon: 80.2707, country: 'India' },
      { name: 'Kolkata', lat: 22.5726, lon: 88.3639, country: 'India' },
      { name: 'Ahmedabad', lat: 23.0225, lon: 72.5714, country: 'India' },
      { name: 'Hyderabad', lat: 17.3850, lon: 78.4867, country: 'India' },
      { name: 'Pune', lat: 18.5204, lon: 73.8567, country: 'India' },
      { name: 'Jaipur', lat: 26.9124, lon: 75.7873, country: 'India' },
      { name: 'Lucknow', lat: 26.8467, lon: 80.9462, country: 'India' },
      { name: 'Kanpur', lat: 26.4499, lon: 80.3319, country: 'India' },
      { name: 'Nagpur', lat: 21.1458, lon: 79.0882, country: 'India' },
      { name: 'Indore', lat: 22.7196, lon: 75.8577, country: 'India' },
      { name: 'Thane', lat: 19.2183, lon: 72.9781, country: 'India' },
      { name: 'Bhopal', lat: 23.2599, lon: 77.4126, country: 'India' },
      { name: 'Visakhapatnam', lat: 17.6868, lon: 83.2185, country: 'India' },
      { name: 'Pimpri-Chinchwad', lat: 18.6275, lon: 73.8141, country: 'India' },
      { name: 'Patna', lat: 25.5941, lon: 85.1376, country: 'India' },
      { name: 'Vadodara', lat: 22.3072, lon: 73.1812, country: 'India' },
      { name: 'Ghaziabad', lat: 28.6692, lon: 77.4538, country: 'India' }
    ];
  }

  // Search for a city by name
  searchCity(query) {
    const normalizedQuery = query.toLowerCase().trim();
    return this.cities.filter(city => 
      city.name.toLowerCase().includes(normalizedQuery) ||
      city.country.toLowerCase().includes(normalizedQuery)
    );
  }

  // Get exact match for a city
  getCity(query) {
    const normalizedQuery = query.toLowerCase().trim();
    return this.cities.find(city => 
      city.name.toLowerCase() === normalizedQuery
    );
  }

  // Get coordinates for a city
  getCoordinates(cityName) {
    const city = this.getCity(cityName);
    return city ? { lat: city.lat, lon: city.lon } : null;
  }

  // Get all cities
  getAllCities() {
    return this.cities;
  }

  // Add a new city (for user saved locations)
  addCity(name, lat, lon, country = 'India') {
    const newCity = { name, lat, lon, country };
    this.cities.push(newCity);
    return newCity;
  }

  // In a real implementation, this would call a geocoding API
  async geocodeAddress(address) {
    // This is a mock implementation
    // In production, you would use:
    // - Google Geocoding API
    // - OpenCage Geocoding API
    // - MapBox Geocoding API
    // - OpenStreetMap Nominatim
    
    console.log(`Geocoding address: ${address}`);
    
    // For demo, return a random city from our list
    const randomIndex = Math.floor(Math.random() * this.cities.length);
    return this.cities[randomIndex];
  }
}

// Export a singleton instance
const locationService = new LocationService();
export default locationService;