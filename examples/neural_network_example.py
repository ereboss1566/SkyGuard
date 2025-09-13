import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the weather model
from src.models.weather_model import SevereWeatherClassifier, WeatherForecaster

# Set random seed for reproducibility
np.random.seed(42)

# Generate some sample data for classification
def generate_classification_data(n_samples=1000, n_features=10):
    """Generate synthetic weather data for classification."""
    X = np.random.randn(n_samples, n_features)
    
    # Create a simple rule: if the sum of the first 3 features is > 0, it's a severe weather event
    y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
    
    # Add some noise
    noise_idx = np.random.choice(range(n_samples), size=int(n_samples * 0.1), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    return X, y

# Generate some sample data for regression
def generate_regression_data(n_samples=1000, n_features=10):
    """Generate synthetic weather data for regression."""
    X = np.random.randn(n_samples, n_features)
    
    # Create a simple rule: temperature is a function of the first 3 features
    y = 15 + 5 * X[:, 0] - 3 * X[:, 1] + 2 * X[:, 2] + np.random.randn(n_samples) * 2
    
    return X, y

# Main function to demonstrate the models
def main():
    print("SkyGuard Neural Network Model Example")
    print("-" * 40)
    
    # Classification example
    print("\nClassification Example (Severe Weather Prediction)")
    X_class, y_class = generate_classification_data(n_samples=1000, n_features=10)
    
    # Create a neural network classifier with dropout
    nn_params = {
        'input_dim': X_class.shape[1],
        'hidden_units': [64, 32],
        'dropout_rate': 0.3,  # Dropout rate of 30%
        'output_dim': 2,      # Binary classification
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    classifier = SevereWeatherClassifier(model_type='neural_network', model_params=nn_params)
    
    # Train the model
    print("\nTraining neural network classifier...")
    metrics = classifier.train(X_class, y_class, test_size=0.2)
    
    # Print metrics
    print("\nClassification Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    X_test = X_class[-10:]
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    print("\nSample Predictions (Classification):")
    for i in range(5):
        print(f"Sample {i+1}: Predicted Class = {y_pred[i]}, Probability = {y_proba[i]}")
    
    # Regression example
    print("\n" + "-" * 40)
    print("\nRegression Example (Temperature Forecasting)")
    X_reg, y_reg = generate_regression_data(n_samples=1000, n_features=10)
    
    # Create a neural network regressor with dropout
    nn_params = {
        'input_dim': X_reg.shape[1],
        'hidden_units': [64, 32],
        'dropout_rate': 0.3,  # Dropout rate of 30%
        'epochs': 50,
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    forecaster = WeatherForecaster(model_type='neural_network', model_params=nn_params)
    
    # Train the model
    print("\nTraining neural network regressor...")
    metrics = forecaster.train(X_reg, y_reg, test_size=0.2)
    
    # Print metrics
    print("\nRegression Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions
    X_test = X_reg[-10:]
    y_pred = forecaster.predict(X_test)
    
    print("\nSample Predictions (Regression):")
    for i in range(5):
        print(f"Sample {i+1}: Actual = {y_reg[-10+i]:.2f}, Predicted = {y_pred[i]:.2f}")
    
    # Compare with traditional models
    print("\n" + "-" * 40)
    print("\nComparing with Traditional Models")
    
    # Traditional classifier
    trad_classifier = SevereWeatherClassifier(model_type='random_forest')
    trad_metrics = trad_classifier.train(X_class, y_class, test_size=0.2)
    
    print("\nRandom Forest Classifier Metrics:")
    for metric, value in trad_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Traditional regressor
    trad_forecaster = WeatherForecaster(model_type='gradient_boosting')
    trad_metrics = trad_forecaster.train(X_reg, y_reg, test_size=0.2)
    
    print("\nGradient Boosting Regressor Metrics:")
    for metric, value in trad_metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()