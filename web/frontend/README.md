# SkyGuard Frontend - AI/ML Weather Prediction System

A web-based frontend for the SkyGuard AI/ML weather prediction system designed for airfield operations.

## Features

### 1. Real-time Dashboard
- Interactive weather map with storm cell visualization
- Current weather conditions display
- Storm probability predictions with confidence scores
- Risk level indicators (Safe, Caution, Danger)
- Weather trend charts

### 2. Data Ingestion Management
- Monitor and manage multiple weather data sources
- Radar data, satellite imagery, weather stations, historical records
- Pipeline status monitoring
- Add/remove data sources dynamically

### 3. Model Management
- View and manage deployed ML models
- Retrain models with latest data
- Deploy new model versions
- Feature importance visualization
- Model accuracy tracking

### 4. Alert System
- Real-time alert monitoring
- Manual alert creation
- Alert acknowledgment and resolution
- Alert history tracking
- Customizable alert thresholds
- Notification settings (audio, email, SMS)

## Technical Stack

- **Frontend Framework**: React.js
- **Mapping**: Leaflet.js with React-Leaflet
- **Charts**: Chart.js with React-Chartjs-2
- **Build Tool**: Webpack
- **Styling**: CSS3 with responsive design
- **State Management**: React built-in hooks

## Installation

1. Navigate to the frontend directory:
```bash
cd web/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Build for production:
```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── Dashboard.js          # Main dashboard with maps and predictions
│   ├── DataIngestion.js      # Data source management
│   ├── ModelManagement.js    # ML model management
│   └── AlertSystem.js        # Alert monitoring and management
├── App.js                    # Main application component
├── index.js                  # Entry point
└── styles/                   # CSS files
```

## Requirements Addressed

This frontend addresses all requirements from Problem Statement 5:

1. **Data Ingestion & Processing**
   - Visualizes multiple data sources (radar, satellite, weather stations)
   - Shows pipeline status and data flow

2. **Weather Prediction Models**
   - Displays nowcasting (0-3 hrs) and medium-term (24 hrs) predictions
   - Shows model accuracy and feature importance

3. **Alerts & Explainability**
   - Real-time alert system with severity levels
   - Explanation of why alerts were issued
   - Manual alert creation capability

4. **Dashboard Integration**
   - Real-time maps with radar overlays
   - Color-coded risk levels
   - Animated storm movement visualization
   - Plain language explanations

## Browser Support

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Responsive design for desktop and tablet devices

## Development

To extend the frontend:
1. Add new components in the `src/components/` directory
2. Import and register new components in `App.js`
3. Add corresponding CSS files in the component directories

## License

MIT License