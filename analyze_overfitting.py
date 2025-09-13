import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

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

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# 1. Learning Curve Analysis
print("\n" + "="*60)
print("LEARNING CURVE ANALYSIS")
print("="*60)

# Create learning curves for Random Forest (as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, 
    cv=3,  # Using 3-fold CV due to limited data
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='roc_auc'
)

# Calculate mean and std for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curves
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('ROC AUC Score')
plt.title('Learning Curve - Random Forest')
plt.legend(loc='best')
plt.grid(True)

# Analyze overfitting/underfitting from learning curve
final_train_score = train_mean[-1]
final_val_score = val_mean[-1]
score_diff = final_train_score - final_val_score

print(f"Final Training Score: {final_train_score:.4f}")
print(f"Final Validation Score: {final_val_score:.4f}")
print(f"Score Difference: {score_diff:.4f}")

if score_diff > 0.1:
    print("[WARNING] Potential OVERFITTING detected (large gap between training and validation)")
elif final_val_score < 0.7:
    print("[WARNING] Potential UNDERFITTING detected (low validation score)")
else:
    print("[OK] Model appears to be well-fitted")

# 2. Validation Curve Analysis
print("\n" + "="*60)
print("VALIDATION CURVE ANALYSIS")
print("="*60)

# Analyze the effect of n_estimators on performance
param_range = [10, 25, 50, 100, 150, 200]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), 
    X_train, y_train, 
    param_name='n_estimators', 
    param_range=param_range,
    cv=3, 
    scoring='roc_auc',
    n_jobs=-1
)

# Calculate mean and std for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot validation curves
plt.subplot(2, 2, 2)
plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Number of Estimators')
plt.ylabel('ROC AUC Score')
plt.title('Validation Curve - n_estimators')
plt.legend(loc='best')
plt.grid(True)

# Find optimal parameter value
optimal_idx = np.argmax(val_mean)
optimal_param = param_range[optimal_idx]
print(f"Optimal n_estimators: {optimal_param}")
print(f"Best Validation Score: {val_mean[optimal_idx]:.4f}")

# 3. Cross-validation analysis
print("\n" + "="*60)
print("CROSS-VALIDATION ANALYSIS")
print("="*60)

from sklearn.model_selection import cross_val_score

# Perform cross-validation with different metrics
cv_scores_accuracy = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
cv_scores_roc = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')

print(f"CV Accuracy: {cv_scores_accuracy.mean():.4f} (+/- {cv_scores_accuracy.std() * 2:.4f})")
print(f"CV ROC AUC: {cv_scores_roc.mean():.4f} (+/- {cv_scores_roc.std() * 2:.4f})")

# Check for overfitting with CV
model.fit(X_train, y_train)
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
train_roc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])

test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
test_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Training ROC AUC: {train_roc:.4f}")
print(f"Test ROC AUC: {test_roc:.4f}")

acc_diff = train_accuracy - test_accuracy
roc_diff = train_roc - test_roc

print(f"\nAccuracy Difference (Train-Test): {acc_diff:.4f}")
print(f"ROC AUC Difference (Train-Test): {roc_diff:.4f}")

if acc_diff > 0.1 or roc_diff > 0.1:
    print("[WARNING] OVERFITTING detected - model performs significantly better on training data")
elif test_accuracy < 0.7:
    print("[WARNING] UNDERFITTING detected - model performs poorly on both training and test data")
else:
    print("[OK] Model appears to be well-generalized")

# 4. Feature importance stability analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE STABILITY")
print("="*60)

# Train multiple models with different random states and check feature importance consistency
feature_importances = []
for i in range(5):
    model_temp = RandomForestClassifier(n_estimators=100, random_state=i)
    model_temp.fit(X_train, y_train)
    feature_importances.append(model_temp.feature_importances_)

# Calculate mean and std of feature importances
fi_mean = np.mean(feature_importances, axis=0)
fi_std = np.std(feature_importances, axis=0)

# Create DataFrame for analysis
fi_df = pd.DataFrame({
    'feature': X.columns,
    'importance_mean': fi_mean,
    'importance_std': fi_std
}).sort_values('importance_mean', ascending=False)

print("Top 10 Features with Importance Stability:")
print(fi_df.head(10))

# Check if top features are stable
top_features_stable = fi_df.head(5)
unstable_features = top_features_stable[top_features_stable['importance_std'] > 0.05]

if len(unstable_features) > 0:
    print(f"\n[WARNING] Unstable features detected (high std):")
    for _, row in unstable_features.iterrows():
        print(f"  - {row['feature']}: {row['importance_mean']:.4f} +/- {row['importance_std']:.4f}")
else:
    print("\n[OK] Top features appear stable across different model initializations")

# 5. Visualization of overfitting analysis
plt.subplot(2, 2, 3)
# Scatter plot of training vs test predictions
plt.scatter(train_roc, test_roc, s=100, alpha=0.7)
plt.plot([0, 1], [0, 1], 'r--', lw=2)
plt.xlabel('Training ROC AUC')
plt.ylabel('Test ROC AUC')
plt.title('Training vs Test Performance')
plt.grid(True)

# Add perfect fit line
if train_roc > test_roc:
    plt.text(0.05, 0.95, 'Overfitting', fontsize=12, verticalalignment='top')
elif test_roc > train_roc:
    plt.text(0.05, 0.95, 'Underfitting', fontsize=12, verticalalignment='top')
else:
    plt.text(0.05, 0.95, 'Well-fitted', fontsize=12, verticalalignment='top')

# 6. Model complexity vs performance
plt.subplot(2, 2, 4)
# Show feature importance stability
top_features = fi_df.head(10)
plt.barh(range(len(top_features)), top_features['importance_mean'], 
         xerr=top_features['importance_std'], alpha=0.7)
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances with Uncertainty')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('outputs/overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary Report
print("\n" + "="*60)
print("OVERFITTING/UNDERFITTING ANALYSIS SUMMARY")
print("="*60)
print(f"Dataset Size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"Class Distribution: {y.mean()*100:.1f}% positive class")

# Overall assessment
if score_diff > 0.1 or acc_diff > 0.1 or roc_diff > 0.1:
    overall_assessment = "OVERFITTING"
    recommendation = "Reduce model complexity, add regularization, or get more data"
elif test_accuracy < 0.7 or final_val_score < 0.7:
    overall_assessment = "UNDERFITTING"
    recommendation = "Increase model complexity or add more relevant features"
else:
    overall_assessment = "WELL-FITTED"
    recommendation = "Model is well-balanced"

print(f"\nOverall Assessment: {overall_assessment}")
print(f"Recommendation: {recommendation}")

print(f"\nKey Metrics:")
print(f"  - CV ROC AUC: {cv_scores_roc.mean():.4f} +/- {cv_scores_roc.std()*2:.4f}")
print(f"  - Test ROC AUC: {test_roc:.4f}")
print(f"  - Train-Test Gap: {roc_diff:.4f}")

# Save analysis results
fi_df.to_csv('outputs/feature_importance_stability.csv', index=False)
print(f"\nFeature importance stability saved to 'outputs/feature_importance_stability.csv'")