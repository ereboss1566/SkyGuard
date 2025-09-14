import joblib
import pandas as pd

# Load the model and check what features it expects
model = joblib.load('models/optimized/randomforest_optimized_model.pkl')
scaler = joblib.load('models/optimized/scaler.pkl')
imputer = joblib.load('models/optimized/imputer.pkl')

print("Model type:", type(model))
print("Scaler type:", type(scaler))
print("Imputer type:", type(imputer))

# Check the features the scaler was trained on
if hasattr(scaler, 'feature_names_in_'):
    print("\nFeatures expected by scaler:")
    for i, feature in enumerate(scaler.feature_names_in_):
        print(f"  {i+1}. {feature}")
    print(f"\nTotal features: {len(scaler.feature_names_in_)}")
else:
    print("\nScaler does not have feature names recorded")
    
# Check the features the imputer was trained on
if hasattr(imputer, 'feature_names_in_'):
    print("\nFeatures expected by imputer:")
    for i, feature in enumerate(imputer.feature_names_in_):
        print(f"  {i+1}. {feature}")
    print(f"\nTotal features: {len(imputer.feature_names_in_)}")
else:
    print("\nImputer does not have feature names recorded")