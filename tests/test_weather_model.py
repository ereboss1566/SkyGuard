"""Tests for the weather model module."""

import unittest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.weather_model import (
    BaseWeatherModel,
    SevereWeatherClassifier,
    WeatherForecaster,
    create_weather_model
)


class TestBaseWeatherModel(unittest.TestCase):
    """Tests for the BaseWeatherModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = BaseWeatherModel()
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNone(self.model.model)
        self.assertFalse(self.model.is_trained)
    
    def test_not_implemented_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        
        with self.assertRaises(NotImplementedError):
            self.model.train(X, y)
        
        with self.assertRaises(NotImplementedError):
            self.model.predict(X)
        
        with self.assertRaises(NotImplementedError):
            self.model.evaluate(X, y)
    
    def test_save_load_not_implemented(self):
        """Test that save and load methods raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.model.save("model.pkl")
        
        with self.assertRaises(NotImplementedError):
            self.model.load("model.pkl")


class TestSevereWeatherClassifier(unittest.TestCase):
    """Tests for the SevereWeatherClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SevereWeatherClassifier()
        
        # Create sample data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)  # Binary classification
        
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 2, 20)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model.is_trained)
    
    def test_train(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        """Test model prediction."""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(isinstance(p, (int, np.integer)) for p in predictions))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        self.model.train(self.X_train, self.y_train)
        probabilities = self.model.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))  # Binary classification
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        
        for metric_name, value in metrics.items():
            self.assertIsInstance(value, float)
            self.assertTrue(0 <= value <= 1)
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning."""
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5]
        }
        
        best_params = self.model.tune_hyperparameters(
            self.X_train, self.y_train, param_grid, cv=2
        )
        
        self.assertIsInstance(best_params, dict)
        self.assertIn('n_estimators', best_params)
        self.assertIn('max_depth', best_params)


class TestWeatherForecaster(unittest.TestCase):
    """Tests for the WeatherForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = WeatherForecaster()
        
        # Create sample time series data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.X_train = np.random.rand(90, 5)  # Features
        self.y_train = np.random.rand(90)  # Target variable (e.g., temperature)
        
        self.X_test = np.random.rand(10, 5)
        self.y_test = np.random.rand(10)
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model.is_trained)
    
    def test_train(self):
        """Test model training."""
        self.model.train(self.X_train, self.y_train)
        self.assertTrue(self.model.is_trained)
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        """Test model prediction."""
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in predictions))
    
    def test_evaluate(self):
        """Test model evaluation."""
        self.model.train(self.X_train, self.y_train)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        for metric_name, value in metrics.items():
            self.assertIsInstance(value, float)
    
    def test_forecast_future(self):
        """Test forecasting future values."""
        self.model.train(self.X_train, self.y_train)
        
        # Create future feature data
        future_X = np.random.rand(5, 5)
        
        forecast = self.model.forecast_future(future_X)
        
        self.assertEqual(len(forecast), len(future_X))
        self.assertTrue(all(isinstance(p, (float, np.floating)) for p in forecast))


class TestModelFactory(unittest.TestCase):
    """Tests for the model factory function."""
    
    def test_create_classifier(self):
        """Test creating a classifier model."""
        model = create_weather_model(model_type="classifier")
        self.assertIsInstance(model, SevereWeatherClassifier)
    
    def test_create_forecaster(self):
        """Test creating a forecaster model."""
        model = create_weather_model(model_type="forecaster")
        self.assertIsInstance(model, WeatherForecaster)
    
    def test_invalid_model_type(self):
        """Test creating a model with an invalid type."""
        with self.assertRaises(ValueError):
            create_weather_model(model_type="invalid_type")


if __name__ == '__main__':
    unittest.main()