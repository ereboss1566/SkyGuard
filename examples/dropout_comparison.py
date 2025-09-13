import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the project root to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the weather model
from src.models.weather_model import SevereWeatherClassifier

# Set random seed for reproducibility
np.random.seed(42)

def generate_data(n_samples=1000, n_features=10):
    """Generate synthetic data for binary classification."""
    X = np.random.randn(n_samples, n_features)
    # Create a simple rule for binary classification
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

def train_and_evaluate_model(dropout_rate, X_train, X_test, y_train, y_test):
    """Train a neural network model with the specified dropout rate and evaluate it."""
    print(f"\nTraining model with dropout rate: {dropout_rate}")
    
    nn_params = {
        'input_dim': X_train.shape[1],
        'hidden_units': [64, 32],
        'dropout_rate': dropout_rate,
        'output_dim': 2,  # Binary classification
        'epochs': 15,     # Reduced for faster execution
        'batch_size': 32,
        'validation_split': 0.2
    }
    
    model = SevereWeatherClassifier(model_type='neural_network', model_params=nn_params)
    
    # Train the model directly with TensorFlow/Keras
    history = model.model.fit(
        X_train, y_train,
        epochs=nn_params['epochs'],
        batch_size=nn_params['batch_size'],
        validation_split=nn_params['validation_split'],
        verbose=1
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred_proba = model.model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Return metrics
    return {
        'accuracy': test_accuracy,
        'loss': test_loss,
        'val_accuracy': history.history['val_accuracy'][-1],
        'val_loss': history.history['val_loss'][-1],
        'history': history.history
    }

def main():
    # Generate data
    X, y = generate_data(n_samples=1000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Generated dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Dropout rates to compare
    dropout_rates = [0.0, 0.2, 0.5]
    
    # Store results
    results = {}
    
    # Train and evaluate models with different dropout rates
    for rate in dropout_rates:
        results[rate] = train_and_evaluate_model(rate, X_train, X_test, y_train, y_test)
    
    # Print results summary
    print("\nResults Summary:")
    print("-" * 40)
    print(f"{'Dropout Rate':15} {'Accuracy':10} {'Val Accuracy':15} {'Loss':10} {'Val Loss':10}")
    print("-" * 60)
    
    for rate, metrics in results.items():
        print(f"{rate:15.1f} {metrics['accuracy']:10.4f} {metrics['val_accuracy']:15.4f} "
              f"{metrics['loss']:10.4f} {metrics['val_loss']:10.4f}")
    
    # Plot training history for each dropout rate
    plt.figure(figsize=(12, 8))
    
    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    for rate, metrics in results.items():
        plt.plot(metrics['history']['accuracy'], label=f'Train (dropout={rate})')
        plt.plot(metrics['history']['val_accuracy'], linestyle='--', label=f'Val (dropout={rate})')
    
    plt.title('Model Accuracy with Different Dropout Rates')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    for rate, metrics in results.items():
        plt.plot(metrics['history']['loss'], label=f'Train (dropout={rate})')
        plt.plot(metrics['history']['val_loss'], linestyle='--', label=f'Val (dropout={rate})')
    
    plt.title('Model Loss with Different Dropout Rates')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'dropout_comparison_results.png'))
    print("\nPlots saved to examples directory.")

if __name__ == "__main__":
    main()