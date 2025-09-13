# 📥 METAR Data Ingestion for Ahmedabad (VAAH)

"""
Fetch METAR weather reports for Ahmedabad airfield (ICAO: VAAH) using the aviationweather.gov API.
Parse the latest 7 days of hourly METAR data and extract key weather parameters:
- Temperature
- Dew Point
- Pressure (QNH)
- Wind Speed
- Wind Gust
- Wind Direction
- Visibility
- Rain Indicator

Store the data in a pandas DataFrame called 'vaah_metar_df' with a datetime index.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import re

# Define parameters
icao_code = "VAAH"
end_time = datetime.now()
start_time = end_time - timedelta(days=7)

# Format timestamps
start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
end_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

# API endpoint for plain text format
url = f"https://aviationweather.gov/api/data/metar?ids={icao_code}&start={start_str}&end={end_str}"

# Fetch data
response = requests.get(url)

# Check if response is valid
if response.status_code == 200 and response.text:
    # Split the response into individual METAR reports
    metar_reports = response.text.strip().split('\n\n')
    
    # Parse data into a list of dictionaries
    parsed_data = []
    
    for report in metar_reports:
        if not report.strip():
            continue
            
        # Extract observation time (DDHHMMZ format)
        time_match = re.search(r'(\d{6})Z', report)
        if time_match:
            # Convert DDHHMM to datetime
            day_hour_min = time_match.group(1)
            day = int(day_hour_min[:2])
            hour = int(day_hour_min[2:4])
            minute = int(day_hour_min[4:6])
            
            # Create datetime (approximate date based on current date)
            obs_time = datetime(end_time.year, end_time.month, day, hour, minute)
        else:
            obs_time = None
            
        # Extract temperature and dew point (TT/DD format)
        temp_match = re.search(r'(\d{2})/(\d{2})', report)
        if temp_match:
            temperature = int(temp_match.group(1))
            dew_point = int(temp_match.group(2))
        else:
            temperature = None
            dew_point = None
            
        # Extract pressure (QXXXX format)
        pressure_match = re.search(r'Q(\d{4})', report)
        if pressure_match:
            pressure = int(pressure_match.group(1))
        else:
            pressure = None
            
        # Extract wind information (dddff(f)(gxx)KT format)
        wind_match = re.search(r'(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT', report)
        if wind_match:
            wind_direction = wind_match.group(1)
            wind_speed = int(wind_match.group(2))
            wind_gust = int(wind_match.group(3)) if wind_match.group(3) else None
        else:
            wind_direction = None
            wind_speed = None
            wind_gust = None
            
        # Extract visibility (can be in meters or miles)
        visibility_match = re.search(r'(\d{4})', report)
        if visibility_match:
            visibility = int(visibility_match.group(1))
        else:
            visibility = None
            
        # Check for rain (RA in weather codes)
        rain = 1 if 'RA' in report else 0
        
        # Create parsed entry
        parsed_entry = {
            'time': obs_time,
            'temperature': temperature,
            'dew_point': dew_point,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'wind_gust': wind_gust,
            'wind_direction': wind_direction,
            'visibility': visibility,
            'rain': rain
        }
        parsed_data.append(parsed_entry)

    # Create DataFrame
    vaah_metar_df = pd.DataFrame(parsed_data)

    # Set time as index
    vaah_metar_df.set_index('time', inplace=True)

    # Save to CSV
    vaah_metar_df.to_csv('vaah_metar.csv')

    # Display sample
    print("DataFrame head:")
    print(vaah_metar_df.head())
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")