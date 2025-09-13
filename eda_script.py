import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load ----------
csv_path = "data/enhanced/enhanced_storm_features.csv"  # adjust path for our data structure
df = pd.read_csv(csv_path, parse_dates=["timestamp"])

# ---------- Clean & coerce ----------
numeric_cols = [
    "reflectivity_max","reflectivity_mean","brightness_temp_min",
    "motion_vector_x","motion_vector_y",
    "temperature","dew_point","pressure","wind_speed","wind_gust","wind_direction",
    "visibility","rain","humidity","air_quality_PM2.5","air_quality_PM10",
    "temperature_celsius","pressure_mb","wind_kph","precip_mm","uv_index","gust_kph"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Make a binary rain flag if 'rain' exists but is not strictly 0/1
if "rain" in df.columns:
    # Treat any positive precip or nonzero rain indicator as 1
    df["rain_flag"] = (df["rain"].fillna(0).astype(float) > 0).astype(int)

# Sort by time and set index
df = df.sort_values("timestamp").set_index("timestamp")

# Create output dir
os.makedirs("outputs/eda", exist_ok=True)

# ---------- Histograms ----------
hist_cols = [c for c in ["reflectivity_max","brightness_temp_min","pressure","wind_speed","wind_gust","humidity","visibility"] if c in df.columns]
if hist_cols:
    plt.figure(figsize=(14, 8))
    df[hist_cols].hist(bins=40, figsize=(14, 8), layout=(len(hist_cols)//3 + 1, 3), edgecolor="black", grid=False)
    plt.tight_layout()
    plt.savefig("outputs/eda/histograms_core.png", dpi=200)
    plt.close()

# ---------- Scatter plots (relationships to gust risk) ----------
pairs = []
if "reflectivity_max" in df.columns and "wind_gust" in df.columns:
    pairs.append(("reflectivity_max","wind_gust"))
if "brightness_temp_min" in df.columns and "wind_gust" in df.columns:
    pairs.append(("brightness_temp_min","wind_gust"))
if "humidity" in df.columns and "wind_gust" in df.columns:
    pairs.append(("humidity","wind_gust"))

for x, y in pairs:
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x=x, y=y, alpha=0.4, s=20)
    sns.regplot(data=df, x=x, y=y, scatter=False, color="red", ci=None)
    plt.title(f"{x} vs {y}")
    plt.tight_layout()
    plt.savefig(f"outputs/eda/scatter_{x}_vs_{y}.png", dpi=200)
    plt.close()

# ---------- Time series views ----------
ts_cols = [c for c in ["reflectivity_max","brightness_temp_min","pressure","wind_gust"] if c in df.columns]
if ts_cols:
    plt.figure(figsize=(14,6))
    df[ts_cols].rolling("30min").mean().plot(ax=plt.gca())
    plt.title("30-min rolling mean of key signals")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig("outputs/eda/timeseries_rolling30min.png", dpi=200)
    plt.close()

# ---------- Correlation heatmap ----------
corr_cols = [c for c in [
    "reflectivity_max","reflectivity_mean","brightness_temp_min",
    "motion_vector_x","motion_vector_y",
    "pressure","wind_speed","wind_gust","humidity","visibility"
] if c in df.columns]

# Add some additional columns that might be available in our enhanced dataset
additional_corr_cols = [c for c in [
    "temperature","dew_point","precip_mm","uv_index","gust_kph"
] if c in df.columns]

corr_cols.extend(additional_corr_cols)

corr_df = df[corr_cols].dropna().copy()
if not corr_df.empty:
    plt.figure(figsize=(12,10))
    corr = corr_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, fmt='.2f')
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig("outputs/eda/corr_heatmap.png", dpi=200)
    plt.close()

print("EDA complete. Saved figures to outputs/eda/")