"""
Simplified test script for load forecasting models.
This script uses minimal dependencies to demonstrate the performance of different models.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, LSTM, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Silence TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_model(model_type, input_shape, output_dim):
    """Build a model based on the specified type."""
    model = Sequential()
    
    if model_type == 'gru':
        # Original GRU model
        model.add(GRU(256, activation='relu', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(128, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(GRU(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    elif model_type == 'lstm':
        # LSTM model
        model.add(LSTM(256, activation='relu', return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Output layer
    model.add(Dense(output_dim))
    
    # Compile
    model.compile(optimizer='adam', loss='mse')
    
    return model

def process_data(data_path, test_size=0.2):
    """Process data for model training and evaluation."""
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Process datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
    
    # Convert weekday to numerical
    if 'weekday' in df.columns:
        weekday_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['weekday_num'] = df['weekday'].map(weekday_map)
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Create time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Define target columns
    targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
    
    # Create lag features
    for target in targets:
        df[f'{target}_lag_1'] = df[target].shift(1)
        df[f'{target}_lag_24'] = df[target].shift(24)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Define features
    categorical_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'weekday_num']
    numerical_features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    
    # Add lag features
    for target in targets:
        numerical_features.extend([f'{target}_lag_1', f'{target}_lag_24'])
    
    # Filter features that exist in the dataframe
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    # Combine all features
    features = numerical_features + categorical_features
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    
    # Create X and y
    X_train = train_df[features].values
    X_test = test_df[features].values
    
    y_train = np.zeros((len(train_df), len(targets)))
    y_test = np.zeros((len(test_df), len(targets)))
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train = feature_scaler.fit_transform(X_train)
    X_test = feature_scaler.transform(X_test)
    
    # Reshape for RNN
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    # Scale targets and save scalers
    scalers_y = {}
    for i, target in enumerate(targets):
        scaler = StandardScaler()
        y_train[:, i] = scaler.fit_transform(train_df[[target]]).ravel()
        y_test[:, i] = scaler.transform(test_df[[target]]).ravel()
        scalers_y[target] = scaler
    
    data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'targets': targets,
        'features': features,
        'scalers_y': scalers_y,
        'feature_scaler': feature_scaler,
        'train_df': train_df,
        'test_df': test_df
    }
    
    return data

def evaluate_model(model, data):
    """Evaluate model and return performance metrics."""
    # Make predictions
    y_pred = model.predict(data['X_test'])
    
    # Inverse transform predictions and actual values
    y_test_inv = np.zeros_like(data['y_test'])
    y_pred_inv = np.zeros_like(y_pred)
    
    for i, target in enumerate(data['targets']):
        y_test_inv[:, i] = data['scalers_y'][target].inverse_transform(data['y_test'][:, i].reshape(-1, 1)).ravel()
        y_pred_inv[:, i] = data['scalers_y'][target].inverse_transform(y_pred[:, i].reshape(-1, 1)).ravel()
    
    # Calculate metrics
    metrics = {}
    
    for i, target in enumerate(data['targets']):
        # RMSE
        rmse = np.sqrt(np.mean(np.square(y_test_inv[:, i] - y_pred_inv[:, i])))
        
        # MAPE
        mask = y_test_inv[:, i] != 0
        mape = np.mean(np.abs((y_test_inv[:, i][mask] - y_pred_inv[:, i][mask]) / y_test_inv[:, i][mask])) * 100
        
        # R-squared
        ss_res = np.sum(np.square(y_test_inv[:, i] - y_pred_inv[:, i]))
        ss_tot = np.sum(np.square(y_test_inv[:, i] - np.mean(y_test_inv[:, i])))
        r2 = 1 - (ss_res / ss_tot)
        
        metrics[target] = {
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    # Calculate average metrics
    avg_rmse = np.mean([metrics[target]['RMSE'] for target in data['targets']])
    avg_mape = np.mean([metrics[target]['MAPE'] for target in data['targets']])
    avg_r2 = np.mean([metrics[target]['R2'] for target in data['targets']])
    
    metrics['Average'] = {
        'RMSE': avg_rmse,
        'MAPE': avg_mape,
        'R2': avg_r2
    }
    
    return metrics, y_test_inv, y_pred_inv

def train_and_evaluate(model_type, data_path, epochs=15):
    """Train and evaluate a model."""
    print(f"\n{'='*50}")
    print(f" Testing {model_type.upper()} Model")
    print(f"{'='*50}\n")
    
    # Process data
    data = process_data(data_path)
    
    # Build model
    input_shape = (data['X_train'].shape[1], data['X_train'].shape[2])
    output_dim = data['y_train'].shape[1]
    model = build_model(model_type, input_shape, output_dim)
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print(f"\nTraining {model_type.upper()} model...")
    history = model.fit(
        data['X_train'], data['y_train'],
        epochs=epochs,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # Evaluate
    print(f"\nEvaluating {model_type.upper()} model...")
    metrics, y_test_inv, y_pred_inv = evaluate_model(model, data)
    
    # Print metrics
    print("\nPerformance Metrics:")
    
    for target, target_metrics in metrics.items():
        print(f"\n{target}:")
        for metric, value in target_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return model, metrics

def find_data_path():
    """Find the data file."""
    possible_paths = [
        "data/final_data.csv", 
        "../final_data.csv", 
        "C:/Users/ansha/OneDrive/Desktop/SIH/final_data.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    """Main function."""
    # Find data file
    data_path = find_data_path()
    
    if data_path is None:
        print("Error: Could not find final_data.csv")
        sys.exit(1)
    
    # Models to test
    model_types = ['gru', 'lstm']
    
    results = {}
    
    # Test each model
    for model_type in model_types:
        try:
            model, metrics = train_and_evaluate(model_type, data_path, epochs=15)
            results[model_type] = metrics
        except Exception as e:
            print(f"Error with {model_type} model: {e}")
    
    # Compare results
    print("\n" + "="*50)
    print(" Model Comparison Summary")
    print("="*50 + "\n")
    
    print(f"{'Model':<10} {'Avg RMSE':<15} {'Avg MAPE (%)':<15} {'Avg R²':<15}")
    print("-" * 55)
    
    for model_type, metrics in results.items():
        avg_metrics = metrics['Average']
        print(f"{model_type.upper():<10} {avg_metrics['RMSE']:<15.2f} {avg_metrics['MAPE']:<15.2f} {avg_metrics['R2']:<15.4f}")
    
    # Identify best model
    best_model = min(results.items(), key=lambda x: x[1]['Average']['MAPE'])[0]
    print(f"\nBest model based on Average MAPE: {best_model.upper()}")

if __name__ == "__main__":
    main()
