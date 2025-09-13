"""SkyGuard Dashboard Module

This module provides a web-based dashboard for visualizing weather data and alerts.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import json

# Initialize the Dash app
app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    title="SkyGuard Dashboard"
)
server = app.server

# Load configuration
def load_config():
    """Load dashboard configuration."""
    config_path = os.environ.get("SKYGUARD_CONFIG", "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {
        "api_url": "http://localhost:8000",
        "refresh_interval": 300,  # seconds
        "default_location": "New York"
    }


config = load_config()

# Generate mock data if needed
def generate_mock_weather_data():
    """Generate mock weather data for testing."""
    # Create date range for the past week
    dates = pd.date_range(end=datetime.now(), periods=168, freq="H")  # hourly data for 7 days
    
    # Generate random weather data
    data = {
        "timestamp": dates,
        "temperature": np.random.normal(25, 5, len(dates)),
        "humidity": np.random.normal(60, 10, len(dates)),
        "wind_speed": np.random.normal(15, 5, len(dates)),
        "precipitation": np.random.exponential(0.5, len(dates)),
        "pressure": np.random.normal(1013, 5, len(dates))
    }
    
    # Add some weather events
    events = []
    for i in range(5):
        event_time = dates[np.random.randint(0, len(dates))]
        event_type = np.random.choice(["Storm", "Heavy Rain", "High Winds", "Heat Wave"])
        events.append({
            "time": event_time,
            "type": event_type,
            "severity": np.random.choice(["Low", "Medium", "High", "Critical"])
        })
    
    return pd.DataFrame(data), events


# Fetch data from API
def fetch_weather_data(location="New York"):
    """Fetch weather data from the API."""
    try:
        response = requests.get(
            f"{config['api_url']}/weather/current",
            params={"location_name": location}
        )
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Error fetching weather data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching weather data: {str(e)}")
        return None


def fetch_forecast_data(location="New York", days=5):
    """Fetch forecast data from the API."""
    try:
        response = requests.get(
            f"{config['api_url']}/weather/forecast",
            params={"location_name": location, "days": days}
        )
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        else:
            print(f"Error fetching forecast data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception fetching forecast data: {str(e)}")
        return None


def fetch_alerts():
    """Fetch alerts from the API."""
    try:
        response = requests.get(f"{config['api_url']}/alerts")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching alerts: {response.status_code}")
            return []
    except Exception as e:
        print(f"Exception fetching alerts: {str(e)}")
        return []


# Use mock data if API is not available
try:
    current_weather = fetch_weather_data(config["default_location"])
    forecast_data = fetch_forecast_data(config["default_location"])
    alerts = fetch_alerts()
    
    if current_weather is None or forecast_data is None:
        weather_data, weather_events = generate_mock_weather_data()
    else:
        # Process API data
        weather_data = current_weather
        weather_events = alerts
        
except Exception:
    weather_data, weather_events = generate_mock_weather_data()


# Define the dashboard layout
app.layout = html.Div(
    className="dashboard-container",
    children=[
        # Header
        html.Div(
            className="header",
            children=[
                html.H1("SkyGuard Weather Dashboard"),
                html.Div(
                    className="header-controls",
                    children=[
                        html.Div(
                            className="location-selector",
                            children=[
                                html.Label("Location:"),
                                dcc.Dropdown(
                                    id="location-dropdown",
                                    options=[
                                        {"label": "New York", "value": "New York"},
                                        {"label": "Los Angeles", "value": "Los Angeles"},
                                        {"label": "Chicago", "value": "Chicago"},
                                        {"label": "Miami", "value": "Miami"},
                                        {"label": "Seattle", "value": "Seattle"}
                                    ],
                                    value=config["default_location"],
                                    clearable=False
                                )
                            ]
                        ),
                        html.Div(
                            className="time-range-selector",
                            children=[
                                html.Label("Time Range:"),
                                dcc.Dropdown(
                                    id="time-range-dropdown",
                                    options=[
                                        {"label": "Last 24 Hours", "value": "24h"},
                                        {"label": "Last 3 Days", "value": "3d"},
                                        {"label": "Last Week", "value": "1w"},
                                        {"label": "Last Month", "value": "1m"}
                                    ],
                                    value="24h",
                                    clearable=False
                                )
                            ]
                        ),
                        html.Button("Refresh Data", id="refresh-button", className="refresh-button")
                    ]
                )
            ]
        ),
        
        # Main content
        html.Div(
            className="dashboard-content",
            children=[
                # Left column - Current conditions and alerts
                html.Div(
                    className="left-column",
                    children=[
                        # Current conditions card
                        html.Div(
                            className="card current-conditions",
                            children=[
                                html.H2("Current Conditions"),
                                html.Div(id="current-conditions-content")
                            ]
                        ),
                        
                        # Alerts card
                        html.Div(
                            className="card alerts",
                            children=[
                                html.H2("Weather Alerts"),
                                html.Div(id="alerts-content")
                            ]
                        )
                    ]
                ),
                
                # Right column - Charts
                html.Div(
                    className="right-column",
                    children=[
                        # Temperature chart
                        html.Div(
                            className="card chart",
                            children=[
                                html.H2("Temperature Trend"),
                                dcc.Graph(id="temperature-chart")
                            ]
                        ),
                        
                        # Precipitation chart
                        html.Div(
                            className="card chart",
                            children=[
                                html.H2("Precipitation"),
                                dcc.Graph(id="precipitation-chart")
                            ]
                        ),
                        
                        # Wind and pressure chart
                        html.Div(
                            className="card chart",
                            children=[
                                html.H2("Wind Speed & Pressure"),
                                dcc.Graph(id="wind-pressure-chart")
                            ]
                        )
                    ]
                )
            ]
        ),
        
        # Footer
        html.Div(
            className="footer",
            children=[
                html.P(f"SkyGuard Dashboard v0.1.0 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
                dcc.Interval(
                    id="refresh-interval",
                    interval=config["refresh_interval"] * 1000,  # milliseconds
                    n_intervals=0
                )
            ]
        )
    ]
)


# Callback to update current conditions
@app.callback(
    Output("current-conditions-content", "children"),
    [Input("refresh-interval", "n_intervals"),
     Input("refresh-button", "n_clicks"),
     Input("location-dropdown", "value")]
)
def update_current_conditions(n_intervals, n_clicks, location):
    """Update current conditions display."""
    # In a real app, this would fetch the latest data
    # For now, we'll use our mock data or API data
    
    try:
        current = fetch_weather_data(location)
        if current is None or len(current) == 0:
            # Use mock data
            temp = weather_data["temperature"].iloc[-1]
            humidity = weather_data["humidity"].iloc[-1]
            wind = weather_data["wind_speed"].iloc[-1]
            pressure = weather_data["pressure"].iloc[-1]
            condition = "Partly Cloudy"
        else:
            # Use API data
            temp = current["temperature"].iloc[0]
            humidity = current["humidity"].iloc[0]
            wind = current["wind_speed"].iloc[0]
            pressure = current["pressure"].iloc[0]
            condition = current["weather_condition"].iloc[0]
    except Exception:
        # Fallback to mock data
        temp = weather_data["temperature"].iloc[-1]
        humidity = weather_data["humidity"].iloc[-1]
        wind = weather_data["wind_speed"].iloc[-1]
        pressure = weather_data["pressure"].iloc[-1]
        condition = "Partly Cloudy"
    
    return html.Div(
        className="current-conditions-grid",
        children=[
            html.Div(
                className="condition-item",
                children=[
                    html.H3(f"{temp:.1f}°C"),
                    html.P("Temperature")
                ]
            ),
            html.Div(
                className="condition-item",
                children=[
                    html.H3(f"{humidity:.1f}%"),
                    html.P("Humidity")
                ]
            ),
            html.Div(
                className="condition-item",
                children=[
                    html.H3(f"{wind:.1f} m/s"),
                    html.P("Wind Speed")
                ]
            ),
            html.Div(
                className="condition-item",
                children=[
                    html.H3(f"{pressure:.1f} hPa"),
                    html.P("Pressure")
                ]
            ),
            html.Div(
                className="condition-item condition-status",
                children=[
                    html.H3(condition),
                    html.P("Condition")
                ]
            ),
            html.Div(
                className="condition-item condition-time",
                children=[
                    html.H3(datetime.now().strftime("%H:%M")),
                    html.P(datetime.now().strftime("%Y-%m-%d"))
                ]
            )
        ]
    )


# Callback to update alerts
@app.callback(
    Output("alerts-content", "children"),
    [Input("refresh-interval", "n_intervals"),
     Input("refresh-button", "n_clicks")]
)
def update_alerts(n_intervals, n_clicks):
    """Update alerts display."""
    # In a real app, this would fetch the latest alerts
    # For now, we'll use our mock data or API data
    
    try:
        alerts = fetch_alerts()
        if not alerts:
            # Use mock data
            alerts = weather_events
    except Exception:
        # Fallback to mock data
        alerts = weather_events
    
    if not alerts:
        return html.Div(
            className="no-alerts",
            children=[
                html.P("No active weather alerts.")
            ]
        )
    
    alert_items = []
    for alert in alerts[:5]:  # Show up to 5 alerts
        # Extract alert details based on data structure
        if isinstance(alert, dict):
            if "severity" in alert and "type" in alert and "time" in alert:
                # Mock data format
                severity = alert["severity"]
                alert_type = alert["type"]
                message = f"{alert_type} alert"
                time = alert["time"]
                if isinstance(time, str):
                    time_str = time
                else:
                    time_str = time.strftime("%Y-%m-%d %H:%M")
            else:
                # API data format
                severity = alert.get("severity", "Unknown")
                alert_type = alert.get("alert_type", "Unknown")
                message = alert.get("message", f"{alert_type} alert")
                time_str = alert.get("timestamp", datetime.now().isoformat())
                if isinstance(time_str, datetime):
                    time_str = time_str.strftime("%Y-%m-%d %H:%M")
        else:
            # Fallback
            severity = "Unknown"
            alert_type = "Unknown"
            message = "Weather alert"
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Determine severity class for styling
        severity_class = f"severity-{severity.lower() if isinstance(severity, str) else 'unknown'}"
        
        alert_items.append(
            html.Div(
                className=f"alert-item {severity_class}",
                children=[
                    html.Div(
                        className="alert-severity",
                        children=[
                            html.Span(severity.upper() if isinstance(severity, str) else "UNKNOWN")
                        ]
                    ),
                    html.Div(
                        className="alert-details",
                        children=[
                            html.H4(alert_type),
                            html.P(message),
                            html.Span(time_str, className="alert-time")
                        ]
                    )
                ]
            )
        )
    
    return html.Div(
        className="alerts-list",
        children=alert_items
    )


# Callback to update temperature chart
@app.callback(
    Output("temperature-chart", "figure"),
    [Input("refresh-interval", "n_intervals"),
     Input("refresh-button", "n_clicks"),
     Input("location-dropdown", "value"),
     Input("time-range-dropdown", "value")]
)
def update_temperature_chart(n_intervals, n_clicks, location, time_range):
    """Update temperature chart."""
    # Filter data based on time range
    end_time = datetime.now()
    
    if time_range == "24h":
        start_time = end_time - timedelta(hours=24)
    elif time_range == "3d":
        start_time = end_time - timedelta(days=3)
    elif time_range == "1w":
        start_time = end_time - timedelta(weeks=1)
    elif time_range == "1m":
        start_time = end_time - timedelta(days=30)
    else:
        start_time = end_time - timedelta(hours=24)
    
    # In a real app, this would fetch data for the specific location and time range
    # For now, we'll filter our mock data
    try:
        filtered_data = weather_data[
            (weather_data["timestamp"] >= start_time) & 
            (weather_data["timestamp"] <= end_time)
        ]
    except Exception:
        # If there's an error with the data, create a simple placeholder
        dates = pd.date_range(start=start_time, end=end_time, freq="H")
        filtered_data = pd.DataFrame({
            "timestamp": dates,
            "temperature": np.random.normal(25, 5, len(dates))
        })
    
    # Create the figure
    fig = px.line(
        filtered_data, 
        x="timestamp", 
        y="temperature",
        labels={"timestamp": "Time", "temperature": "Temperature (°C)"},
        title=f"Temperature in {location}"
    )
    
    # Add forecast data if available
    try:
        forecast = fetch_forecast_data(location)
        if forecast is not None and len(forecast) > 0:
            # Process forecast data
            forecast_x = forecast["forecast_time"]
            forecast_y = forecast["temperature"]
            
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode="lines+markers",
                    line=dict(dash="dash", color="rgba(255, 165, 0, 0.8)"),
                    name="Forecast"
                )
            )
    except Exception as e:
        print(f"Error adding forecast data: {str(e)}")
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# Callback to update precipitation chart
@app.callback(
    Output("precipitation-chart", "figure"),
    [Input("refresh-interval", "n_intervals"),
     Input("refresh-button", "n_clicks"),
     Input("location-dropdown", "value"),
     Input("time-range-dropdown", "value")]
)
def update_precipitation_chart(n_intervals, n_clicks, location, time_range):
    """Update precipitation chart."""
    # Filter data based on time range
    end_time = datetime.now()
    
    if time_range == "24h":
        start_time = end_time - timedelta(hours=24)
    elif time_range == "3d":
        start_time = end_time - timedelta(days=3)
    elif time_range == "1w":
        start_time = end_time - timedelta(weeks=1)
    elif time_range == "1m":
        start_time = end_time - timedelta(days=30)
    else:
        start_time = end_time - timedelta(hours=24)
    
    # In a real app, this would fetch data for the specific location and time range
    # For now, we'll filter our mock data
    try:
        filtered_data = weather_data[
            (weather_data["timestamp"] >= start_time) & 
            (weather_data["timestamp"] <= end_time)
        ]
    except Exception:
        # If there's an error with the data, create a simple placeholder
        dates = pd.date_range(start=start_time, end=end_time, freq="H")
        filtered_data = pd.DataFrame({
            "timestamp": dates,
            "precipitation": np.random.exponential(0.5, len(dates))
        })
    
    # Create the figure
    fig = px.bar(
        filtered_data, 
        x="timestamp", 
        y="precipitation",
        labels={"timestamp": "Time", "precipitation": "Precipitation (mm)"},
        title=f"Precipitation in {location}"
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Precipitation (mm)",
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest"
    )
    
    # Update bar colors
    fig.update_traces(marker_color="rgba(0, 119, 190, 0.7)")
    
    return fig


# Callback to update wind and pressure chart
@app.callback(
    Output("wind-pressure-chart", "figure"),
    [Input("refresh-interval", "n_intervals"),
     Input("refresh-button", "n_clicks"),
     Input("location-dropdown", "value"),
     Input("time-range-dropdown", "value")]
)
def update_wind_pressure_chart(n_intervals, n_clicks, location, time_range):
    """Update wind and pressure chart."""
    # Filter data based on time range
    end_time = datetime.now()
    
    if time_range == "24h":
        start_time = end_time - timedelta(hours=24)
    elif time_range == "3d":
        start_time = end_time - timedelta(days=3)
    elif time_range == "1w":
        start_time = end_time - timedelta(weeks=1)
    elif time_range == "1m":
        start_time = end_time - timedelta(days=30)
    else:
        start_time = end_time - timedelta(hours=24)
    
    # In a real app, this would fetch data for the specific location and time range
    # For now, we'll filter our mock data
    try:
        filtered_data = weather_data[
            (weather_data["timestamp"] >= start_time) & 
            (weather_data["timestamp"] <= end_time)
        ]
    except Exception:
        # If there's an error with the data, create a simple placeholder
        dates = pd.date_range(start=start_time, end=end_time, freq="H")
        filtered_data = pd.DataFrame({
            "timestamp": dates,
            "wind_speed": np.random.normal(15, 5, len(dates)),
            "pressure": np.random.normal(1013, 5, len(dates))
        })
    
    # Create the figure with two y-axes
    fig = go.Figure()
    
    # Add wind speed trace
    fig.add_trace(
        go.Scatter(
            x=filtered_data["timestamp"],
            y=filtered_data["wind_speed"],
            name="Wind Speed",
            line=dict(color="rgba(255, 87, 51, 0.8)", width=2)
        )
    )
    
    # Add pressure trace on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=filtered_data["timestamp"],
            y=filtered_data["pressure"],
            name="Pressure",
            line=dict(color="rgba(75, 192, 192, 0.8)", width=2),
            yaxis="y2"
        )
    )
    
    # Customize layout with two y-axes
    fig.update_layout(
        title=f"Wind Speed & Pressure in {location}",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="Wind Speed (m/s)",
            titlefont=dict(color="rgba(255, 87, 51, 0.8)"),
            tickfont=dict(color="rgba(255, 87, 51, 0.8)")
        ),
        yaxis2=dict(
            title="Pressure (hPa)",
            titlefont=dict(color="rgba(75, 192, 192, 0.8)"),
            tickfont=dict(color="rgba(75, 192, 192, 0.8)"),
            anchor="x",
            overlaying="y",
            side="right"
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)