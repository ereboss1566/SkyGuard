# SkyGuard Real-time Storm Prediction System

A real-time weather monitoring and storm prediction system based on machine learning models.

## System Architecture

```
SkyGuard System
├── Data Collection Layer
│   ├── OpenWeatherMap API
│   ├── Weather.gov API
│   └── Data Fusion Engine
├── Prediction Service
│   ├── Preprocessing Pipeline
│   ├── Storm Prediction Model
│   └── Alerting System
├── API Layer
│   └── RESTful API Endpoints
├── Web Frontend
│   ├── Real-time Dashboard
│   ├── Data Ingestion Management
│   ├── Model Management
│   └── Alert System
├── Monitoring & Dashboard
│   ├── Performance Tracking
│   └── Model Drift Detection
└── Storage
    ├── Real-time Weather Data
    ├── Prediction Results
    └── System Logs
```

## Setup

### Prerequisites
- Python 3.8+
- Node.js 14+
- npm 6+

### Installation

1. Install backend dependencies:
```bash
cd web/backend
pip install -r requirements.txt
```

2. Install frontend dependencies:
```bash
cd ../frontend
npm install
```

### Configuration

Edit `config.ini` to configure:
- API keys for weather services
- Update intervals
- Alerting thresholds
- Logging settings
- Data retention policies

## Running the Complete System

### Windows
Double-click `start_system.bat` or run:
```bash
start_system.bat
```

### Linux/Mac
Make the script executable and run:
```bash
chmod +x start_system.sh
./start_system.sh
```

### Manual Startup
1. Start the backend server:
```bash
cd web/backend
python server.py
```

2. In a new terminal, start the frontend:
```bash
cd web/frontend
npm start
```

## API Endpoints

- `GET /` - System information
- `GET /health` - Health check
- `POST /predict` - Make storm prediction
- `POST /predict_with_alert` - Make prediction with alert evaluation
- `GET /feature_importance` - Get model feature importance

## Web Frontend Features

### Dashboard
- Real-time weather prediction dashboard with interactive maps
- Location-based predictions for any city
- Storm probability with confidence scores
- Risk level indicators (Safe, Caution, Danger)

### Data Ingestion Management
- Monitor and manage multiple weather data sources
- Pipeline status monitoring
- Add/remove data sources dynamically

### Model Management
- View and manage deployed ML models
- Retrain models with latest data
- Feature importance visualization

### Alert System
- Real-time alert monitoring with severity levels
- Manual alert creation
- Customizable alert thresholds
- Notification settings

## Features

- Real-time weather data collection from multiple sources
- Storm prediction using trained ML models
- Alerting system with configurable thresholds
- RESTful API for integration
- Web-based user interface
- Performance monitoring and dashboard
- Model drift detection
- Data retention and cleanup
- Location-based predictions

## Directory Structure

```
skyguard/
├── config.ini              # System configuration
├── requirements_realtime.txt # Python dependencies
├── skyguard_orchestrator.py # Main system orchestrator
├── data/                   # Data storage
│   ├── realtime/           # Real-time weather data
│   └── predictions/        # Prediction results
├── logs/                   # System logs
├── models/                 # ML models
│   └── optimized/          # Optimized production models
├── outputs/                # Generated reports and visualizations
├── src/                    # Source code
│   ├── api/               # REST API service
│   ├── data_collection/   # Weather data collectors
│   ├── prediction/        # Prediction service
│   └── monitoring/        # System monitoring
├── web/                    # Web frontend and backend
│   ├── backend/           # Flask backend API
│   │   ├── server.py      # Main backend server
│   │   └── requirements.txt # Backend dependencies
│   └── frontend/          # React-based dashboard
│       ├── src/           # React source code
│       ├── public/        # Static assets
│       └── package.json    # Frontend dependencies
├── start_system.bat       # Windows startup script
├── start_system.sh         # Linux/Mac startup script
└── tests/                 # Unit tests
```

## Next Steps for Full Production Deployment

1. **Real Weather Data Integration**:
   - Register for OpenWeatherMap, WeatherAPI, or government weather service APIs
   - Implement actual API calls in the data collection modules
   - Add authentication and rate limiting for API usage

2. **Enhanced Prediction Models**:
   - Implement LSTM for wind gust prediction
   - Add CNN for radar/satellite pattern analysis
   - Create synoptic-scale data integration for medium-term forecasting

3. **Database Integration**:
   - Add PostgreSQL or MongoDB for persistent data storage
   - Implement time-series database for weather data
   - Add data archiving and cleanup policies

4. **Advanced Features**:
   - Implement continuous model retraining pipeline
   - Add A/B testing for model comparisons
   - Create ensemble models combining multiple approaches
   - Add data drift detection mechanisms

5. **Deployment**:
   - Containerize with Docker for easy deployment
   - Set up CI/CD pipeline for automated deployments
   - Add monitoring and alerting for system health
   - Implement load balancing for high availability