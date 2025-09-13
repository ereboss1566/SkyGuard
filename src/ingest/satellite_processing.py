# ☁️ Download MODIS IR Imagery from NASA Worldview Snapshots

"""
Use NASA Worldview Snapshots API to download MODIS IR imagery over Gujarat.
Bounding box: Bottom Left (21.0, 71.0), Top Right (24.0, 74.0)
Date range: Last 24 hours
Layer: MODIS_Terra_Brightness_Temp_IR
Resolution: 0.05 degrees
Format: PNG

Save images to 'data/satellite/' folder with timestamped filenames.
"""

import requests
import os
from datetime import datetime, timedelta
from urllib.parse import urlencode

# Create satellite directory if it doesn't exist
sat_dir = "data/satellite/"
os.makedirs(sat_dir, exist_ok=True)

# Define bounding box over Gujarat
bbox = {
    "BBOX": "21.0,71.0,24.0,74.0",  # lat/lon
    "CRS": "EPSG:4326"
}

# Define time range (last 24 hours)
end_time = datetime.utcnow()
start_time = end_time - timedelta(hours=24)

# Generate hourly timestamps
timestamps = [start_time + timedelta(hours=i) for i in range(25)]

# NASA Worldview API endpoint
base_url = "https://wvs.earthdata.nasa.gov/api/v1/snapshot"

# Layer to download
layer = "MODIS_Terra_Brightness_Temp_Band31"

# Download loop
print("Downloading MODIS IR imagery from NASA Worldview...")
for ts in timestamps:
    try:
        # Format timestamp for API
        time_str = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Create filename with timestamp
        filename = ts.strftime("%Y%m%d_%H%M") + ".png"
        filepath = os.path.join(sat_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(filepath):
            print(f"File {filename} already exists, skipping...")
            continue
        
        # Prepare API parameters
        params = {
            "REQUEST": "GetSnapshot",
            "LAYERS": layer,
            "BBOX": bbox["BBOX"],
            "CRS": bbox["CRS"],
            "FORMAT": "image/png",
            "WIDTH": 600,
            "HEIGHT": 600,
            "TIME": time_str
        }
        
        # Construct full URL
        url = base_url + "?" + urlencode(params)
        
        # Download image
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            # Save image
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename} (Status code: {response.status_code})")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error while downloading {ts.strftime('%Y%m%d_%H%M')}: {e}")
    except Exception as e:
        print(f"Error processing {ts.strftime('%Y%m%d_%H%M')}: {e}")

print("Finished downloading satellite images.")

# Process the satellite images
import cv2
import pandas as pd
import numpy as np

print("Processing satellite images...")
records = []

# Process each image in the directory
for fname in sorted(os.listdir(sat_dir)):
    if fname.endswith(".png"):
        try:
            ts_str = fname.split(".")[0]
            timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M")
            img = cv2.imread(os.path.join(sat_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                brightness_temp_min = img.min()
                # Simulate motion vector (placeholder)
                motion_vector_x = np.random.uniform(-1, 1)
                motion_vector_y = np.random.uniform(-1, 1)
                records.append([timestamp, brightness_temp_min, motion_vector_x, motion_vector_y])
            else:
                print(f"Could not read image: {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

if records:
    satellite_features_df = pd.DataFrame(records, columns=["timestamp", "brightness_temp_min", "motion_vector_x", "motion_vector_y"])
    satellite_features_df.set_index("timestamp", inplace=True)
    print("\nSatellite features DataFrame:")
    print(satellite_features_df.head())
    
    # Save DataFrame to CSV
    satellite_features_df.to_csv("satellite_features.csv")
    print("\nSaved satellite features to satellite_features.csv")
else:
    print("No valid satellite images found.")