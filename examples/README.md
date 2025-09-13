# SkyGuard Examples

This directory contains example scripts demonstrating how to use the SkyGuard weather prediction models.

## Neural Network with Dropout Layers

The SkyGuard model now supports neural network models with dropout layers for both classification and regression tasks. Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

### Available Examples

1. **neural_network_example.py** - Demonstrates how to create and train neural network models with dropout for both classification and regression tasks.

2. **dropout_comparison.py** - Compares the performance of models with different dropout rates (0.0, 0.1, 0.3, 0.5, 0.7) and generates plots to visualize the impact of dropout on model accuracy and F1 score.

## Running the Examples

To run the examples, navigate to the SkyGuard project root directory and execute:

```bash
python examples/neural_network_example.py
```

or

```bash
python examples/dropout_comparison.py
```

## Using Dropout in Your Models

To use dropout in your own models, specify the `neural_network` model type and set the `dropout_rate` parameter when creating a classifier or forecaster:

```python
from src.models.weather_model import SevereWeatherClassifier

# Create a neural network classifier with dropout
model_params = {
    'input_dim': X.shape[1],  # Number of input features
    'hidden_units': [64, 32],  # Size of hidden layers
    'dropout_rate': 0.3,       # 30% dropout rate
    'output_dim': 2,           # Binary classification
    'epochs': 100,             # Number of training epochs
    'batch_size': 32,          # Batch size for training
    'validation_split': 0.2    # Validation data proportion
}

model = SevereWeatherClassifier(model_type='neural_network', model_params=model_params)
```

## Choosing the Right Dropout Rate

The optimal dropout rate depends on your specific dataset and model architecture:

- **No dropout (0.0)**: May lead to overfitting on complex datasets
- **Low dropout (0.1-0.3)**: Good starting point for most applications
- **Medium dropout (0.3-0.5)**: Useful for models with many parameters or limited training data
- **High dropout (0.5-0.7)**: May help with severe overfitting but could impair learning

Use the `dropout_comparison.py` script as a template to find the optimal dropout rate for your specific use case.