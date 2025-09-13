"""
Weather Prediction Models for SkyGuard

This module contains machine learning models for predicting severe weather events.
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class BaseWeatherModel:
    """Base class for all weather prediction models."""
    
    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the weather model.
        
        Args:
            model_type: Type of model to use
            model_params: Parameters for the model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()
        self.feature_importance = None
        self.is_neural_network = False
        
    def _create_model(self) -> BaseEstimator:
        """Create the underlying model based on model_type.
        
        Returns:
            Initialized model instance
        """
        raise NotImplementedError("Subclasses must implement _create_model method")
    
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError("Subclasses must implement train method")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        if self.is_neural_network:
            # For neural networks, return class predictions
            predictions = self.model.predict(X)
            return np.argmax(predictions, axis=1)
        else:
            return self.model.predict(X)
    
    def save(self, model_path: str) -> None:
        """Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
    def load(self, model_path: str) -> None:
        """Load the model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model = joblib.load(model_path)
        
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance if available.
        
        Returns:
            DataFrame with feature importances or None
        """
        return self.feature_importance


class SevereWeatherClassifier(BaseWeatherModel):
    """Classifier for predicting severe weather events."""
    
    def __init__(self, model_type: str = 'random_forest', 
                 model_params: Optional[Dict[str, Any]] = None):
        """Initialize the severe weather classifier.
        
        Args:
            model_type: Type of classifier ('random_forest', 'gradient_boosting', 'neural_network', etc.)
            model_params: Parameters for the classifier
        """
        super().__init__(model_type, model_params)
        
    def _create_model(self) -> BaseEstimator:
        """Create the classifier based on model_type.
        
        Returns:
            Initialized classifier
        """
        if self.model_type == 'random_forest':
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            return RandomForestClassifier(**params)
        elif self.model_type == 'neural_network':
            self.is_neural_network = True
            # Extract neural network parameters
            input_dim = self.model_params.get('input_dim', 10)
            hidden_units = self.model_params.get('hidden_units', [64, 32])
            dropout_rate = self.model_params.get('dropout_rate', 0.3)
            output_dim = self.model_params.get('output_dim', 2)  # Binary classification by default
            
            # Create neural network model
            model = Sequential()
            model.add(Dense(hidden_units[0], activation='relu', input_dim=input_dim))
            model.add(Dropout(dropout_rate))  # Add dropout layer for regularization
            
            # Add additional hidden layers with dropout
            for units in hidden_units[1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(dropout_rate))
            
            # Output layer
            model.add(Dense(output_dim, activation='softmax'))
            
            # Compile model
            model.compile(
                optimizer=self.model_params.get('optimizer', 'adam'),
                loss=self.model_params.get('loss', 'sparse_categorical_crossentropy'),
                metrics=['accuracy']
            )
            
            return model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, 
              tune_hyperparams: bool = False) -> Dict[str, float]:
        """Train the classifier on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector (binary labels)
            test_size: Proportion of data to use for testing
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training and validation metrics
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if tune_hyperparams and not self.is_neural_network:
            self._tune_hyperparameters(X_train, y_train)
        
        # Train the model
        if self.is_neural_network:
            # Neural network training with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            epochs = self.model_params.get('epochs', 100)
            batch_size = self.model_params.get('batch_size', 32)
            validation_split = self.model_params.get('validation_split', 0.2)
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate on test set
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            y_pred_proba = self.model.predict(X_test)
            
            # For neural networks, we need to ensure the predictions are properly formatted
            # Check if output is one-hot encoded or multi-dimensional
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred = np.argmax(y_pred_proba, axis=1)
            else:
                # For binary classification with single output neuron
                y_pred = (y_pred_proba > 0.5).astype(int)
                if len(y_pred.shape) > 1:
                    y_pred = y_pred.flatten()
            
            # Ensure y_test is in the right format for comparison
            if len(y_test.shape) > 1:
                y_test_comp = np.argmax(y_test, axis=1)
            else:
                y_test_comp = y_test
                
            metrics = {
                'accuracy': test_accuracy,  # Use the accuracy from model.evaluate
                'precision': precision_score(y_test_comp, y_pred, average='weighted'),
                'recall': recall_score(y_test_comp, y_pred, average='weighted'),
                'f1': f1_score(y_test_comp, y_pred, average='weighted'),
                'loss': test_loss
            }
        else:
            # Traditional ML model training
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Store feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature_importance': self.model.feature_importances_
                })
        
        return metrics
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        if self.model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
                param_grid=param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        if self.is_neural_network:
            return self.model.predict(X)
        else:
            return self.model.predict_proba(X)


class WeatherForecaster(BaseWeatherModel):
    """Regressor for forecasting weather variables."""
    
    def __init__(self, model_type: str = 'gradient_boosting', 
                 model_params: Optional[Dict[str, Any]] = None):
        """Initialize the weather forecaster.
        
        Args:
            model_type: Type of regressor ('gradient_boosting', 'neural_network', etc.)
            model_params: Parameters for the regressor
        """
        super().__init__(model_type, model_params)
        
    def _create_model(self) -> BaseEstimator:
        """Create the regressor based on model_type.
        
        Returns:
            Initialized regressor
        """
        if self.model_type == 'gradient_boosting':
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            }
            # Update with user-provided parameters
            params.update(self.model_params)
            return GradientBoostingRegressor(**params)
        elif self.model_type == 'neural_network':
            self.is_neural_network = True
            # Extract neural network parameters
            input_dim = self.model_params.get('input_dim', 10)
            hidden_units = self.model_params.get('hidden_units', [64, 32])
            dropout_rate = self.model_params.get('dropout_rate', 0.3)
            
            # Create neural network model
            model = Sequential()
            model.add(Dense(hidden_units[0], activation='relu', input_dim=input_dim))
            model.add(Dropout(dropout_rate))  # Add dropout layer for regularization
            
            # Add additional hidden layers with dropout
            for units in hidden_units[1:]:
                model.add(Dense(units, activation='relu'))
                model.add(Dropout(dropout_rate))
            
            # Output layer (single unit for regression)
            model.add(Dense(1))
            
            # Compile model
            model.compile(
                optimizer=self.model_params.get('optimizer', 'adam'),
                loss=self.model_params.get('loss', 'mse'),
                metrics=['mae']
            )
            
            return model
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, 
              tune_hyperparams: bool = False) -> Dict[str, float]:
        """Train the regressor on the provided data.
        
        Args:
            X: Feature matrix
            y: Target vector (continuous values)
            test_size: Proportion of data to use for testing
            tune_hyperparams: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training and validation metrics
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if tune_hyperparams and not self.is_neural_network:
            self._tune_hyperparameters(X_train, y_train)
        
        # Train the model
        if self.is_neural_network:
            # Reshape y for neural network if needed
            if len(y_train.shape) == 1:
                y_train_reshaped = y_train.reshape(-1, 1)
                y_test_reshaped = y_test.reshape(-1, 1)
            else:
                y_train_reshaped = y_train
                y_test_reshaped = y_test
            
            # Neural network training with early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            epochs = self.model_params.get('epochs', 100)
            batch_size = self.model_params.get('batch_size', 32)
            validation_split = self.model_params.get('validation_split', 0.2)
            
            history = self.model.fit(
                X_train, y_train_reshaped,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate R² manually for neural network
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'mae': np.mean(np.abs(y_test - y_pred))
            }
        else:
            # Traditional ML model training
            self.model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'r2': self.model.score(X_test, y_test)
            }
            
            # Store feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.DataFrame({
                    'feature_importance': self.model.feature_importances_
                })
        
        return metrics
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        if self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                estimator=GradientBoostingRegressor(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X, y)
            
            # Update model with best parameters
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")


def create_weather_model(model_purpose: str, **kwargs) -> BaseWeatherModel:
    """Factory function to create appropriate weather model.
    
    Args:
        model_purpose: Purpose of the model ('classification' or 'regression')
        **kwargs: Additional arguments for specific model types
        
    Returns:
        Weather model instance
    """
    if model_purpose.lower() == 'classification':
        return SevereWeatherClassifier(**kwargs)
    elif model_purpose.lower() == 'regression':
        return WeatherForecaster(**kwargs)
    else:
        raise ValueError(f"Unknown model purpose: {model_purpose}")