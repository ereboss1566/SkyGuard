import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow is available. LSTM model will be implemented.")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow is not available. LSTM model cannot be implemented.")

if TENSORFLOW_AVAILABLE:
    # Create directory for LSTM models if it doesn't exist
    os.makedirs('models/lstm', exist_ok=True)
    
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
    
    # Create sequences for LSTM (using past 5 time steps to predict current)
    def create_sequences(X, y, sequence_length=5):
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y.iloc[i])
        return np.array(X_seq), np.array(y_seq)
    
    # Create sequences
    sequence_length = 5
    X_seq, y_seq = create_sequences(X_imputed, y, sequence_length)
    
    print(f"\nSequence data shape: {X_seq.shape}")
    print(f"Sequence labels shape: {y_seq.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
    )
    
    # Scale the features
    scaler = MinMaxScaler()
    # Reshape for scaling (combine all time steps)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
    
    # Fit scaler on training data and transform both
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D for LSTM
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    X_test_scaled = X_test_scaled.reshape(X_test.shape)
    
    print(f"\nTraining set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print("\nLSTM Model Summary:")
    model.summary()
    
    # Train the model
    print("\nTraining LSTM model...")
    history = model.fit(
        X_train_scaled, y_train,
        batch_size=16,
        epochs=50,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )
    
    # Evaluate the model
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    except:
        roc_auc = 0.0
    
    print(f"\nLSTM Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('outputs/lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save the LSTM model
    model.save('models/lstm/storm_lstm_model.h5')
    print("\nLSTM model saved to 'models/lstm/storm_lstm_model.h5'")
    
    # Save preprocessing objects
    import joblib
    joblib.dump(scaler, 'models/lstm/scaler.pkl')
    joblib.dump(imputer, 'models/lstm/imputer.pkl')
    print("Preprocessing objects saved to 'models/lstm/'")
    
    print("\n" + "="*60)
    print("LSTM MODEL SUMMARY")
    print("="*60)
    print(f"• Used {X_seq.shape[2]} features from multiple data sources")
    print(f"• Sequence length: {sequence_length} time steps")
    print(f"• Training samples: {X_train_scaled.shape[0]}")
    print(f"• Test samples: {X_test_scaled.shape[0]}")
    print(f"• Model accuracy: {accuracy:.4f}")
    print(f"• Model ROC AUC: {roc_auc:.4f}")
    print(f"• Model saved to 'models/lstm/storm_lstm_model.h5'")
else:
    print("\nSkipping LSTM implementation due to missing TensorFlow dependency.")
    print("To implement LSTM model, please install TensorFlow:")
    print("pip install tensorflow")