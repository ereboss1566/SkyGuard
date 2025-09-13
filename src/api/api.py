"""SkyGuard API Module

This module provides a RESTful API for accessing weather data and alerts.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Depends, Query, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SkyGuard API",
    description="API for accessing weather data and alerts",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class Location(BaseModel):
    """Location model."""
    name: str = Field(..., description="Location name")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    country: Optional[str] = Field(None, description="Country")
    state: Optional[str] = Field(None, description="State or province")


class WeatherData(BaseModel):
    """Weather data model."""
    location: Location
    timestamp: datetime
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    wind_direction: float = Field(..., description="Wind direction in degrees")
    precipitation: float = Field(..., description="Precipitation in mm")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    cloud_cover: float = Field(..., description="Cloud cover percentage")
    visibility: float = Field(..., description="Visibility in km")
    weather_condition: str = Field(..., description="Weather condition description")


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Alert types."""
    STORM = "storm"
    FLOOD = "flood"
    TORNADO = "tornado"
    HURRICANE = "hurricane"
    HEATWAVE = "heatwave"
    COLDWAVE = "coldwave"
    WILDFIRE = "wildfire"
    OTHER = "other"


class Alert(BaseModel):
    """Alert model."""
    alert_id: Optional[str] = Field(None, description="Alert ID")
    alert_type: AlertType
    severity: AlertSeverity
    message: str = Field(..., description="Alert message")
    location: Location
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")


class AlertCreate(BaseModel):
    """Model for creating alerts."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str = Field(..., description="Alert message")
    location: Location


class ForecastData(BaseModel):
    """Weather forecast model."""
    location: Location
    timestamp: datetime
    forecast_time: datetime
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    precipitation: float
    pressure: float
    cloud_cover: float
    weather_condition: str


# Mock data storage (in a real app, this would be a database)
weather_data = []
alerts = []
forecasts = []


# API routes
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {"message": "Welcome to SkyGuard API"}


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/weather/current", response_model=List[WeatherData], tags=["Weather"])
async def get_current_weather(
    location_name: Optional[str] = Query(None, description="Filter by location name"),
    lat: Optional[float] = Query(None, description="Filter by latitude"),
    lon: Optional[float] = Query(None, description="Filter by longitude"),
):
    """Get current weather data."""
    # In a real app, this would query a database or external API
    filtered_data = weather_data
    
    if location_name:
        filtered_data = [d for d in filtered_data if location_name.lower() in d.location.name.lower()]
    
    if lat is not None and lon is not None:
        # Find locations within 0.1 degrees (rough approximation)
        filtered_data = [
            d for d in filtered_data 
            if abs(d.location.lat - lat) < 0.1 and abs(d.location.lon - lon) < 0.1
        ]
    
    return filtered_data


@app.get("/weather/forecast", response_model=List[ForecastData], tags=["Weather"])
async def get_weather_forecast(
    location_name: Optional[str] = Query(None, description="Filter by location name"),
    lat: Optional[float] = Query(None, description="Filter by latitude"),
    lon: Optional[float] = Query(None, description="Filter by longitude"),
    days: int = Query(3, description="Number of forecast days", ge=1, le=7),
):
    """Get weather forecast."""
    # In a real app, this would query a database or external API
    filtered_data = forecasts
    
    if location_name:
        filtered_data = [d for d in filtered_data if location_name.lower() in d.location.name.lower()]
    
    if lat is not None and lon is not None:
        # Find locations within 0.1 degrees (rough approximation)
        filtered_data = [
            d for d in filtered_data 
            if abs(d.location.lat - lat) < 0.1 and abs(d.location.lon - lon) < 0.1
        ]
    
    # Filter by forecast days
    current_date = datetime.now().date()
    filtered_data = [
        d for d in filtered_data 
        if (d.forecast_time.date() - current_date).days <= days
    ]
    
    return filtered_data


@app.get("/alerts", response_model=List[Alert], tags=["Alerts"])
async def get_alerts(
    alert_type: Optional[AlertType] = Query(None, description="Filter by alert type"),
    severity: Optional[AlertSeverity] = Query(None, description="Filter by severity"),
    location_name: Optional[str] = Query(None, description="Filter by location name"),
):
    """Get alerts."""
    filtered_alerts = alerts
    
    if alert_type:
        filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]
    
    if severity:
        filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
    
    if location_name:
        filtered_alerts = [a for a in filtered_alerts if location_name.lower() in a.location.name.lower()]
    
    return filtered_alerts


@app.get("/alerts/{alert_id}", response_model=Alert, tags=["Alerts"])
async def get_alert(alert_id: str = Path(..., description="Alert ID")):
    """Get alert by ID."""
    for alert in alerts:
        if alert.alert_id == alert_id:
            return alert
    
    raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found")


@app.post("/alerts", response_model=Alert, status_code=201, tags=["Alerts"])
async def create_alert(alert_data: AlertCreate = Body(...)):
    """Create a new alert."""
    # Generate alert ID
    alert_id = f"{alert_data.alert_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create alert
    alert = Alert(
        alert_id=alert_id,
        **alert_data.dict()
    )
    
    # In a real app, this would save to a database
    alerts.append(alert)
    
    return alert


@app.delete("/alerts/{alert_id}", status_code=204, tags=["Alerts"])
async def delete_alert(alert_id: str = Path(..., description="Alert ID")):
    """Delete an alert."""
    global alerts
    original_count = len(alerts)
    alerts = [a for a in alerts if a.alert_id != alert_id]
    
    if len(alerts) == original_count:
        raise HTTPException(status_code=404, detail=f"Alert with ID {alert_id} not found")
    
    return None


# Add some sample data for testing
def add_sample_data():
    """Add sample data for testing."""
    # Sample locations
    locations = [
        Location(name="New York", lat=40.7128, lon=-74.0060, country="USA", state="NY"),
        Location(name="Los Angeles", lat=34.0522, lon=-118.2437, country="USA", state="CA"),
        Location(name="Chicago", lat=41.8781, lon=-87.6298, country="USA", state="IL"),
    ]
    
    # Sample weather data
    for location in locations:
        weather_data.append(
            WeatherData(
                location=location,
                timestamp=datetime.now(),
                temperature=25.5,
                humidity=65.0,
                wind_speed=5.2,
                wind_direction=180.0,
                precipitation=0.0,
                pressure=1013.2,
                cloud_cover=25.0,
                visibility=10.0,
                weather_condition="Partly cloudy"
            )
        )
    
    # Sample alerts
    alerts.append(
        Alert(
            alert_id="storm_20230601120000",
            alert_type=AlertType.STORM,
            severity=AlertSeverity.HIGH,
            message="Severe thunderstorm warning",
            location=locations[0],
            timestamp=datetime.now()
        )
    )
    
    alerts.append(
        Alert(
            alert_id="heatwave_20230601130000",
            alert_type=AlertType.HEATWAVE,
            severity=AlertSeverity.MEDIUM,
            message="Heat advisory in effect",
            location=locations[1],
            timestamp=datetime.now()
        )
    )
    
    # Sample forecasts
    for location in locations:
        for days_ahead in range(1, 6):  # 5-day forecast
            forecasts.append(
                ForecastData(
                    location=location,
                    timestamp=datetime.now(),
                    forecast_time=datetime.now().replace(hour=12, minute=0, second=0, microsecond=0),
                    temperature=25.0 + days_ahead,
                    humidity=65.0 - days_ahead,
                    wind_speed=5.0 + (days_ahead * 0.5),
                    wind_direction=180.0,
                    precipitation=days_ahead * 0.5 if days_ahead > 2 else 0.0,
                    pressure=1013.0 - days_ahead,
                    cloud_cover=25.0 + (days_ahead * 10),
                    weather_condition="Sunny" if days_ahead < 3 else "Partly cloudy"
                )
            )


# Add sample data on startup
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    add_sample_data()
    logger.info("Added sample data")


# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)