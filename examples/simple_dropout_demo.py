import sys
import os
import numpy as np

# Add the project root to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the weather model
from src.models.weather_model import SevereWeatherClassifier

# Set random seed for reproducibility
np.random.seed(42)

def main():
    print("SkyGuard Simple Dropout Layer Demo")
    print("-" * 40)
    
    # Generate some simple data for binary classification
    n_samples = 1000
    n_features = 10
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule for binary classification
    
    print(f"Generated dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Create a neural network classifier with dropout
    dropout_rate = 0.3  # 30% dropout
    
    nn_params = {
        'input_dim': X.shape[1],
        'hidden_units': [64, 32],
        'dropout_rate': dropout_rate,
        'output_dim': 2,
        'epochs': 10,  # Small number of epochs for quick demonstration
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    print(f"\nTraining neural network with {dropout_rate:.1f} dropout rate...")
    model = SevereWeatherClassifier(model_type='neural_network', model_params=nn_params)
    
    # Split data manually for demonstration
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model directly with TensorFlow/Keras
    print("\nTraining with TensorFlow/Keras directly:")
    history = model.model.fit(
        X_train, y_train,
        epochs=nn_params['epochs'],
        batch_size=nn_params['batch_size'],
        validation_split=nn_params['validation_split'],
        verbose=1
    )
    
    # Evaluate the model
    print("\nEvaluating model:")
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Make predictions
    print("\nMaking predictions:")
    y_pred_proba = model.model.predict(X_test[:5])
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    print("\nSample predictions:")
    for i in range(5):
        print(f"Sample {i+1}: True class = {y_test[i]}, Predicted class = {y_pred[i]}, Probabilities = {y_pred_proba[i]}")
    
    print("\nDropout successfully demonstrated!")

if __name__ == "__main__":
    main()