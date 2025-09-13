import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the enhanced storm features data
file_path = 'data/enhanced/enhanced_storm_features.csv'
df = pd.read_csv(file_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("Dataset Shape:", df.shape)

# Select key numeric features for modeling from all data sources
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
print(f"\nAvailable features for modeling ({len(available_features)} total):")
for i, feature in enumerate(available_features, 1):
    print(f"{i:2d}. {feature}")

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
# We'll create a synthetic target based on combinations of high reflectivity, 
# low pressure, high wind speed, and high precipitation
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

# Initialize multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Train and evaluate individual models
model_results = {}
trained_models = {}

print("\n" + "="*60)
print("TRAINING INDIVIDUAL MODELS")
print("="*60)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model (use scaled data for SVM, KNN, and Neural Network)
    if name in ['SVM', 'K-Nearest Neighbors', 'Neural Network']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Store trained model
    trained_models[name] = model
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.0
    
    model_results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

# Create ensemble predictions using voting classifier
print("\n" + "="*60)
print("CREATING ENSEMBLE MODEL")
print("="*60)

# Create a voting classifier with selected best models
best_models = [
    ('rf', trained_models['Random Forest']),
    ('gb', trained_models['Gradient Boosting']),
    ('et', trained_models['Extra Trees']),
    ('lr', trained_models['Logistic Regression'])
]

voting_clf = VotingClassifier(estimators=best_models, voting='soft')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_proba = voting_clf.predict_proba(X_test)[:, 1]

voting_accuracy = accuracy_score(y_test, voting_pred)
try:
    voting_roc_auc = roc_auc_score(y_test, voting_proba)
except:
    voting_roc_auc = 0.0

print(f"Voting Ensemble - Accuracy: {voting_accuracy:.4f}, ROC AUC: {voting_roc_auc:.4f}")

# Create weighted ensemble (weighted by ROC AUC scores)
print("\nCreating weighted ensemble...")
roc_auc_scores = []
model_names = []
for name in models.keys():
    if model_results[name]['roc_auc'] > 0:
        roc_auc_scores.append(model_results[name]['roc_auc'])
        model_names.append(name)

if len(roc_auc_scores) > 0:
    weights = np.array(roc_auc_scores) / sum(roc_auc_scores)
    
    # Weighted averaging of probabilities (only for models with valid ROC AUC)
    weighted_proba = np.average([
        model_results[name]['probabilities'] for name in model_names
    ], axis=0, weights=weights)
    
    weighted_pred = (weighted_proba >= 0.5).astype(int)
    weighted_accuracy = accuracy_score(y_test, weighted_pred)
    try:
        weighted_roc_auc = roc_auc_score(y_test, weighted_proba)
    except:
        weighted_roc_auc = 0.0
    
    print(f"Weighted Ensemble - Accuracy: {weighted_accuracy:.4f}, ROC AUC: {weighted_roc_auc:.4f}")
else:
    # Fallback to simple averaging if no valid ROC AUC scores
    weighted_proba = np.mean([
        model_results[name]['probabilities'] for name in models.keys()
    ], axis=0)
    
    weighted_pred = (weighted_proba >= 0.5).astype(int)
    weighted_accuracy = accuracy_score(y_test, weighted_pred)
    try:
        weighted_roc_auc = roc_auc_score(y_test, weighted_proba)
    except:
        weighted_roc_auc = 0.0
    
    print(f"Weighted Ensemble (simple average) - Accuracy: {weighted_accuracy:.4f}, ROC AUC: {weighted_roc_auc:.4f}")

# Compile all results
all_results = {
    'Voting Ensemble': {'accuracy': voting_accuracy, 'roc_auc': voting_roc_auc},
    'Weighted Ensemble': {'accuracy': weighted_accuracy, 'roc_auc': weighted_roc_auc}
}

for name in models.keys():
    all_results[name] = {
        'accuracy': model_results[name]['accuracy'],
        'roc_auc': model_results[name]['roc_auc']
    }

# Create results dataframe
results_df = pd.DataFrame(all_results).T
results_df = results_df.sort_values('roc_auc', ascending=False)

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)
print(results_df)

# Visualize model performance comparison
plt.figure(figsize=(12, 8))
x = np.arange(len(results_df))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width/2, results_df['accuracy'], width, label='Accuracy', alpha=0.8)
bars2 = ax.bar(x + width/2, results_df['roc_auc'], width, label='ROC AUC', alpha=0.8)

ax.set_xlabel('Models')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(results_df.index, rotation=45, ha='right')
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
plt.savefig('outputs/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance from Random Forest (as an example)
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*60)

rf_model = trained_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15))

# Visualize top feature importances
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature', palette='viridis')
plt.title('Top 15 Feature Importances (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrix for best model
best_model_name = results_df.index[0]
best_model_pred = None
if best_model_name in model_results:
    best_model_pred = model_results[best_model_name]['predictions']
elif best_model_name == 'Voting Ensemble':
    best_model_pred = voting_pred
else:  # Weighted Ensemble
    best_model_pred = weighted_pred

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, best_model_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the trained models and preprocessing objects
import joblib

# Save individual models
for name, model in trained_models.items():
    joblib.dump(model, f'models/{name.lower().replace(" ", "_").replace("-", "_")}_model.pkl')

# Save ensemble models
joblib.dump(voting_clf, 'models/voting_ensemble_model.pkl')

# Save scaler and imputer
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(imputer, 'models/imputer.pkl')

print("\n" + "="*60)
print("MODELS AND PREPROCESSING OBJECTS SAVED")
print("="*60)
print("Models saved to 'models/' directory:")
for name in trained_models.keys():
    print(f"  - {name.lower().replace(' ', '_').replace('-', '_')}_model.pkl")
print("  - voting_ensemble_model.pkl")
print("Preprocessing objects:")
print("  - scaler.pkl")
print("  - imputer.pkl")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"• Trained {len(models)} individual models")
print(f"• Created 2 ensemble models (Voting and Weighted)")
print(f"• Used {len(X.columns)} features from multiple data sources:")
print("  - Radar data (reflectivity)")
print("  - Satellite data (brightness temperature, motion vectors)")
print("  - Weather station data (METAR)")
print("  - Enhanced weather features")
print("  - Air quality data")
print(f"• Best performing model: {best_model_name}")
print(f"  - Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
print(f"  - ROC AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f}")