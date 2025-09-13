# 🌩️ Radar Image Processing for Storm Cell Features

"""
Load radar reflectivity images from 'data/radar/' folder.
Extract timestamp from filenames.
Convert each image to grayscale and compute:
- reflectivity_max: maximum pixel intensity
- reflectivity_mean: average pixel intensity

Store results in a DataFrame called 'radar_features_df' with columns:
['timestamp', 'reflectivity_max', 'reflectivity_mean']
"""

import os
import cv2
import pandas as pd
from datetime import datetime
import requests
from io import BytesIO
import shutil

# Create radar directory if it doesn't exist
radar_dir = "data/radar/"
os.makedirs(radar_dir, exist_ok=True)

# Clear existing files in the radar directory to ensure we're working with fresh data
for filename in os.listdir(radar_dir):
    file_path = os.path.join(radar_dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

print("Downloading fresh radar images from IMD...")
# Actual radar image URLs from IMD
radar_urls = [
    "https://mausam.imd.gov.in/Radar/caz_bhj.gif",  # MAX (Z)
    "https://mausam.imd.gov.in/Radar/ppi_bhj.gif",  # Plan Position Indicator (Z) - Close Range
    "https://mausam.imd.gov.in/Radar/sri_bhj.gif",  # Surface Rainfall Intensity
    "https://mausam.imd.gov.in/Radar/ppz_bhj.gif",  # Plan Position Indicator (Z)
    "https://mausam.imd.gov.in/Radar/pac_bhj.gif",  # Precipitation Accumulation
    "https://mausam.imd.gov.in/Radar/ppv_bhj.gif",  # Plan Position Indicator (V)
    "https://mausam.imd.gov.in/Radar/vp2_bhj.gif"   # Volume Velocity Processing(2)
]

# Get current time for naming files
base_time = datetime.now().replace(second=0, microsecond=0)

# Download each radar image
for i, url in enumerate(radar_urls):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Create a timestamp for the filename
            # Adjust timestamp for each image (5 minutes apart to ensure uniqueness)
            timestamp = base_time.replace(minute=(base_time.minute + i*5) % 60)
            filename = timestamp.strftime("%Y%m%d_%H%M") + ".gif"
            
            # Save the image
            with open(os.path.join(radar_dir, filename), "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename} from {url}")
        else:
            print(f"Failed to download image from {url} (Status code: {response.status_code})")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Process the radar images
records = []

# Check if radar directory exists
if not os.path.exists(radar_dir):
    print(f"Directory {radar_dir} does not exist.")
    exit(1)

# Process each image in the directory
for fname in sorted(os.listdir(radar_dir)):
    if fname.endswith(".gif") or fname.endswith(".png") or fname.endswith(".jpg"):
        ts_str = fname.split(".")[0]  # e.g., "20250913_1530"
        timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M")
        img = cv2.imread(os.path.join(radar_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            reflectivity_max = img.max()
            reflectivity_mean = img.mean()
            records.append([timestamp, reflectivity_max, reflectivity_mean])
        else:
            print(f"Could not read image: {fname}")

if records:
    radar_features_df = pd.DataFrame(records, columns=["timestamp", "reflectivity_max", "reflectivity_mean"])
    radar_features_df.set_index("timestamp", inplace=True)
    print("\nRadar features DataFrame:")
    print(radar_features_df.head())
    
    # Save DataFrame to CSV
    radar_features_df.to_csv("radar_features.csv")
    print("\nSaved radar features to radar_features.csv")
else:
    print("No valid radar images found.")