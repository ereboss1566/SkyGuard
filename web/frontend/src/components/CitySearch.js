import React, { useState, useEffect, useRef } from 'react';
import locationService from '../services/locationService';
import './CitySearch.css';

const CitySearch = ({ onLocationSelect, currentLocation }) => {
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const searchRef = useRef(null);

  // Handle input change with debouncing
  useEffect(() => {
    if (query.length > 1) {
      const timeoutId = setTimeout(() => {
        searchCities(query);
      }, 300);
      
      return () => clearTimeout(timeoutId);
    } else {
      setSuggestions([]);
      setShowSuggestions(false);
    }
  }, [query]);

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const searchCities = (searchQuery) => {
    setIsSearching(true);
    try {
      const results = locationService.searchCity(searchQuery);
      setSuggestions(results.slice(0, 8)); // Limit to 8 suggestions
      setShowSuggestions(results.length > 0);
    } catch (error) {
      console.error('Search error:', error);
      setSuggestions([]);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSelectCity = (city) => {
    setQuery(city.name);
    setShowSuggestions(false);
    onLocationSelect(city);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      const city = locationService.getCity(query);
      if (city) {
        handleSelectCity(city);
      } else if (suggestions.length > 0) {
        // Select the first suggestion if exact match not found
        handleSelectCity(suggestions[0]);
      } else {
        // Try geocoding
        handleGeocodeAddress(query);
      }
    }
  };

  const handleGeocodeAddress = async (address) => {
    try {
      setIsSearching(true);
      // In a real implementation, this would call a geocoding API
      const geocodedCity = await locationService.geocodeAddress(address);
      if (geocodedCity) {
        handleSelectCity(geocodedCity);
      } else {
        alert('Location not found. Please try a different search term.');
      }
    } catch (error) {
      console.error('Geocoding error:', error);
      alert('Error finding location. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <div className="city-search" ref={searchRef}>
      <form onSubmit={handleSubmit} className="search-form">
        <div className="search-input-container">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onFocus={() => query.length > 1 && setShowSuggestions(suggestions.length > 0)}
            placeholder="Search for a city (e.g., Mumbai, Delhi, Bangalore)"
            className="search-input"
            autoComplete="off"
          />
          <button type="submit" className="search-button" disabled={isSearching}>
            {isSearching ? (
              <span className="search-spinner">...</span>
            ) : (
              'Search'
            )}
          </button>
        </div>
      </form>

      {showSuggestions && suggestions.length > 0 && (
        <div className="suggestions-dropdown">
          <ul className="suggestions-list">
            {suggestions.map((city, index) => (
              <li 
                key={`${city.name}-${index}`}
                className="suggestion-item"
                onClick={() => handleSelectCity(city)}
              >
                <div className="suggestion-name">{city.name}</div>
                <div className="suggestion-details">{city.country}</div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {currentLocation && (
        <div className="current-location-display">
          <h4>Current Location</h4>
          <div className="location-info">
            <span className="location-name">{currentLocation.name}</span>
            <span className="location-coordinates">
              {currentLocation.lat.toFixed(4)}°N, {currentLocation.lon.toFixed(4)}°E
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default CitySearch;