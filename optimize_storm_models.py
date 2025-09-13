import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Create directory for optimized models if it doesn't exist
os.makedirs('models/optimized', exist_ok=True)

# Load the enhanced storm features data
file_path = 'data/enhanced/enhanced_storm_features.csv'
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Dataset Shape:", df.shape)

# Select key numeric features for modeling
key_features = [
    # Radar features
    'reflectivity_max', 'reflectivity_mean',
    
    # Satellite features
    'brightness_temp_min', 'motion_vector_x', 'motion_vector_y',
    
    # METAR (weather station) features
    'temperature', 'dew_point', 'pressure', 'wind_speed', 'wind_direction', 
    'visibility', 'rain',
    
    # Enhanced weather features
    'temperature_celsius', 'pressure_mb', 'wind_kph', 'humidity', 
    'cloud', 'visibility_km', 'uv_index', 'precip_mm',
    
    # Air quality features
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 
    'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 
    'air_quality_PM2.5', 'air_quality_PM10'
]

# Filter to available features in the dataset
available_features = [col for col in key_features if col in df.columns]

# Prepare the data
X = df[available_features]

# Remove columns with all NaN values
missing_data_summary = X.isnull().sum()
all_nan_columns = missing_data_summary[missing_data_summary == len(X)].index.tolist()

if len(all_nan_columns) > 0:
    print(f"\nRemoving {len(all_nan_columns)} columns with all NaN values:")
    for col in all_nan_columns:
        print(f"  - {col}")
    X = X.drop(columns=all_nan_columns)

# Create target variable based on storm indicators
storm_conditions = (
    (df['reflectivity_max'] > df['reflectivity_max'].quantile(0.8)) | 
    (df['pressure'] < df['pressure'].quantile(0.2)) | 
    (df['wind_speed'] > df['wind_speed'].quantile(0.8)) | 
    (df['precip_mm'] > df['precip_mm'].quantile(0.8))
)

# If no storm conditions are met, we'll use a simpler rule
if storm_conditions.sum() == 0:
    # Use a combination of wind speed, pressure, and humidity
    storm_conditions = (
        (df['wind_speed'] > df['wind_speed'].quantile(0.7)) | 
        (df['pressure'] < df['pressure'].quantile(0.3)) | 
        (df['humidity'] > df['humidity'].quantile(0.8))
    )

df['storm_indicator'] = storm_conditions.astype(int)
y = df['storm_indicator']

print(f"\nTarget distribution:")
print(y.value_counts())
print(f"Storm percentage: {y.mean()*100:.2f}%")

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Define models and hyperparameter grids for optimization
models_params = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'ExtraTrees': {
        'model': ExtraTreesClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }
}

# Perform hyperparameter optimization
optimized_models = {}
results = {}

print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

for model_name, model_info in models_params.items():
    print(f"\nOptimizing {model_name}...")
    
    # Use scaled data for models that benefit from it
    if model_name in ['GradientBoosting']:  # These models can benefit from scaling
        X_train_data = X_train_scaled
        X_test_data = X_test_scaled
    else:
        X_train_data = X_train
        X_test_data = X_test
    
    # Perform grid search
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=3,  # Using 3-fold CV due to limited data
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_data, y_train)
    
    # Store best model
    optimized_models[model_name] = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test_data)
    y_pred_proba = grid_search.predict_proba(X_test_data)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_accuracy': accuracy,
        'test_roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"Best CV Score (ROC AUC): {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {roc_auc:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

# Compare optimized models
print("\n" + "="*60)
print("OPTIMIZED MODEL COMPARISON")
print("="*60)

comparison_data = []
for model_name, result in results.items():
    comparison_data.append({
        'Model': model_name,
        'CV ROC AUC': result['best_score'],
        'Test Accuracy': result['test_accuracy'],
        'Test ROC AUC': result['test_roc_auc']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('Test ROC AUC', ascending=False)

print(comparison_df)

# Visualize model comparison
plt.figure(figsize=(12, 6))
x = np.arange(len(comparison_df))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width/2, comparison_df['Test Accuracy'], width, label='Test Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, comparison_df['Test ROC AUC'], width, label='Test ROC AUC', alpha=0.8)

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Optimized Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model'])
ax.legend()

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/optimized_model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Select best model based on ROC AUC
best_model_name = comparison_df.iloc[0]['Model']
best_model = optimized_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best Model Parameters: {results[best_model_name]['best_params']}")

# Feature importance for the best model (if available)
if hasattr(best_model, 'feature_importances_'):
    print(f"\nTop 10 Feature Importances for {best_model_name}:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
    plt.title(f'Top 10 Feature Importances ({best_model_name})')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(f'outputs/{best_model_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save optimized models and preprocessing objects
print("\n" + "="*60)
print("SAVING OPTIMIZED MODELS")
print("="*60)

# Save optimized models
for name, model in optimized_models.items():
    joblib.dump(model, f'models/optimized/{name.lower()}_optimized_model.pkl')
    print(f"Saved {name} optimized model")

# Save preprocessing objects
joblib.dump(scaler, 'models/optimized/scaler.pkl')
joblib.dump(imputer, 'models/optimized/imputer.pkl')
print("Saved preprocessing objects (scaler, imputer)")

# Summary
print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION SUMMARY")
print("="*60)
print(f"• Optimized 3 models: Random Forest, Gradient Boosting, Extra Trees")
print(f"• Used {len(X.columns)} features from multiple data sources")
print(f"• Best performing model: {best_model_name}")
print(f"  - CV ROC AUC: {results[best_model_name]['best_score']:.4f}")
print(f"  - Test Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
print(f"  - Test ROC AUC: {results[best_model_name]['test_roc_auc']:.4f}")
print(f"• Models saved to 'models/optimized/' directory")