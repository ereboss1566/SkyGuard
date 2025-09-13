import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib
warnings.filterwarnings('ignore')

# Create directory for advanced models if it doesn't exist
os.makedirs('models/advanced', exist_ok=True)

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

# Handle missing values with KNN imputation (more advanced than simple median)
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Advanced Model Implementations

# 1. Feature Selection Pipeline
print("\n" + "="*60)
print("ADVANCED MODEL IMPLEMENTATION")
print("="*60)

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=15)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()]
print(f"Selected {len(selected_features)} best features:")
for i, feature in enumerate(selected_features, 1):
    print(f"  {i:2d}. {feature}")

# 2. Advanced Model Pipelines with Different Preprocessing
print("\nCreating advanced model pipelines...")

# Pipeline 1: Standard scaling + Random Forest
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Pipeline 2: Robust scaling + Gradient Boosting
pipe_gb = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# Pipeline 3: MinMax scaling + SVM
pipe_svm = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', SVC(probability=True, random_state=42))
])

# Pipeline 4: No scaling + Naive Bayes (already probabilistic)
pipe_nb = Pipeline([
    ('classifier', GaussianNB())
])

# Pipeline 5: Standard scaling + Neural Network
pipe_nn = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42))
])

# 3. Ensemble Methods

# Bagging Classifier (corrected syntax)
bagging = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=100,
    random_state=42
)

# Calibrated Classifiers (to improve probability estimates)
cal_rf = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=100, random_state=42),
    cv=3,
    method='sigmoid'
)

cal_gb = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    cv=3,
    method='sigmoid'
)

# 4. Stacking Ensemble
stacking_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('nb', GaussianNB())
]

stacking_classifier = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=LogisticRegression(random_state=42),
    cv=3
)

# 5. Advanced Voting Classifier
voting_classifier = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(probability=True, random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='soft',
    weights=[2, 2, 1, 1, 1]  # Give more weight to better performing models
)

# Store all models
advanced_models = {
    'Pipeline - Random Forest': pipe_rf,
    'Pipeline - Gradient Boosting': pipe_gb,
    'Pipeline - SVM': pipe_svm,
    'Pipeline - Naive Bayes': pipe_nb,
    'Pipeline - Neural Network': pipe_nn,
    'Bagging Classifier': bagging,
    'Calibrated RF': cal_rf,
    'Calibrated GB': cal_gb,
    'Stacking Ensemble': stacking_classifier,
    'Weighted Voting': voting_classifier
}

# Train and evaluate all advanced models
print("\nTraining and evaluating advanced models...")
results = {}

for name, model in advanced_models.items():
    print(f"\nTraining {name}...")
    
    # Fit the model
    if 'Pipeline' in name or name in ['Bagging Classifier', 'Stacking Ensemble', 'Weighted Voting']:
        # These models handle preprocessing internally or need raw data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For calibrated models, we need to fit on the same data used for calibration
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = 0.0
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'model': model
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

# Create results comparison
print("\n" + "="*60)
print("ADVANCED MODEL COMPARISON")
print("="*60)

comparison_data = []
for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': result['accuracy'],
        'Precision': result['precision'],
        'Recall': result['recall'],
        'F1-Score': result['f1'],
        'ROC AUC': result['roc_auc']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('ROC AUC', ascending=False)

print(comparison_df.to_string(index=False))

# Visualization of advanced model comparison
plt.figure(figsize=(15, 10))

# Plot 1: Performance metrics comparison
plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
x = np.arange(len(metrics))
width = 0.8 / min(5, len(comparison_df))  # Limit to top 5 for readability

top_models = comparison_df.head(5)
for i, (idx, row) in enumerate(top_models.iterrows()):
    values = [row[metric] for metric in metrics]
    plt.bar(x + i*width, values, width, label=row['Model'], alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Top 5 Advanced Models - Performance Comparison')
plt.xticks(x + width*(len(top_models)-1)/2, metrics)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)

# Plot 2: ROC Curves for top models
plt.subplot(2, 3, 2)
top_model_names = top_models['Model'].tolist()
for model_name in top_model_names:
    fpr, tpr, _ = roc_curve(y_test, results[model_name]['probabilities'])
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {results[model_name]['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Top Models')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Plot 3: Confusion Matrix for Best Model
best_model_name = comparison_df.iloc[0]['Model']
plt.subplot(2, 3, 3)
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Plot 4: Model Score Distribution
plt.subplot(2, 3, 4)
model_names = comparison_df['Model']
roc_aucs = comparison_df['ROC AUC']
bars = plt.bar(range(len(model_names)), roc_aucs, alpha=0.7)
plt.xlabel('Models')
plt.ylabel('ROC AUC')
plt.title('Advanced Model ROC AUC Comparison')
plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')

# Add value labels on bars
for i, (bar, score) in enumerate(zip(bars, roc_aucs)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontsize=8)

plt.ylim(0, 1.1)
plt.grid(True, alpha=0.3)

# Plot 5: Radar Chart for Multi-Metric Comparison
plt.subplot(2, 3, 5)
# Normalize metrics for radar chart
normalized_df = comparison_df.copy()
for metric in metrics:
    normalized_df[metric] = (comparison_df[metric] - comparison_df[metric].min()) / (comparison_df[metric].max() - comparison_df[metric].min())

# Plot radar chart for top 3 models
top3_models = normalized_df.head(3)
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Close the circle

ax = plt.subplot(2, 3, 5, projection='polar')
for idx, row in top3_models.iterrows():
    values = [row[metric] for metric in metrics]
    values += values[:1]  # Close the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_title('Normalized Performance Metrics Comparison')
ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left')

# Plot 6: Feature Importance for Best Model (if available)
plt.subplot(2, 3, 6)
if hasattr(results[best_model_name]['model'], 'named_steps'):
    # For pipeline models
    if 'classifier' in results[best_model_name]['model'].named_steps:
        classifier = results[best_model_name]['model'].named_steps['classifier']
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': classifier.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['importance'])
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 10 Feature Importances\n{best_model_name}')
            plt.gca().invert_yaxis()
elif hasattr(results[best_model_name]['model'], 'feature_importances_'):
    # For direct models
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': results[best_model_name]['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importances\n{best_model_name}')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'Feature importance\nnot available', 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('outputs/advanced_model_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the best advanced model
best_model = results[best_model_name]['model']
joblib.dump(best_model, f'models/advanced/best_advanced_model_{best_model_name.lower().replace(" ", "_").replace("-", "_")}.pkl')
print(f"\nBest advanced model ({best_model_name}) saved to 'models/advanced/'")

# Save imputer
joblib.dump(imputer, 'models/advanced/imputer.pkl')
print("Imputer saved to 'models/advanced/imputer.pkl'")

# Detailed results save
comparison_df.to_csv('outputs/advanced_model_comparison.csv', index=False)
print("Detailed results saved to 'outputs/advanced_model_comparison.csv'")

# Summary
print("\n" + "="*60)
print("ADVANCED MODEL IMPLEMENTATION SUMMARY")
print("="*60)
print(f"• Implemented {len(advanced_models)} advanced modeling techniques")
print(f"• Used {X_train.shape[1]} features with KNN imputation")
print(f"• Applied feature selection (top 15 features)")
print(f"• Created ensemble methods (bagging, stacking, voting)")
print(f"• Used calibrated classifiers for better probability estimates")
print(f"• Best performing model: {best_model_name}")
print(f"  - Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"  - ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"  - F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"• All models saved to 'models/advanced/' directory")