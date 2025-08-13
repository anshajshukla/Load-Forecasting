import os
import numpy as np
import pandas as pd
# Set matplotlib to use non-interactive backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf

# Set TensorFlow memory growth to avoid using all GPU memory
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU memory growth enabled")
except:
    print("No GPU available or GPU memory configuration failed")

def prepare_prediction_data(df, feature_scaler, targets, features):
    """
    Prepare data for making predictions.
    
    Args:
        df: DataFrame with input features
        feature_scaler: Scaler for features
        targets: List of target names
        features: List of feature names
    
    Returns:
        X: Prepared features for prediction
    """
    # Extract features
    X = df[features].values
    
    # Scale features
    X = feature_scaler.transform(X)
    
    # Reshape for GRU (samples, time steps, features)
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    return X

def predict_load(model, X, scalers_y, targets, model_type='gru'):
    """
    Make load predictions.
    
    Args:
        model: Trained model
        X: Prepared input features
        scalers_y: Dictionary of target scalers
        targets: List of target names
        model_type: Type of model (e.g., 'gru', 'lstm')
    
    Returns:
        predictions: DataFrame with predictions
    """
    # Make predictions
    if model_type == 'gru':
        y_pred = model.model.predict(X)
    elif model_type == 'lstm':
        y_pred = model.model.predict(X, batch_size=1)
    else:
        raise ValueError("Unsupported model type")
    
    # Inverse transform predictions
    y_pred_inv = np.zeros_like(y_pred)
    for i, target in enumerate(targets):
        y_pred_inv[:, i] = scalers_y[target].inverse_transform(y_pred[:, i].reshape(-1, 1)).ravel()
    
    # Create DataFrame with predictions
    predictions = pd.DataFrame(y_pred_inv, columns=targets)
    
    return predictions

def predict_future(model, last_data, feature_scaler, scalers_y, targets, features, steps=24, save_dir='../models/saved_models', model_type='gru'):
    """
    Predict future load values.
    
    Args:
        model: Trained model
        last_data: Last available data point
        feature_scaler: Scaler for features
        scalers_y: Dictionary of target scalers
        targets: List of target names
        features: List of feature names
        steps: Number of future steps to predict
        save_dir: Directory to save prediction results
        model_type: Type of model (e.g., 'gru', 'lstm')
    
    Returns:
        future_predictions: DataFrame with future predictions
    """
    print(f"Generating {steps} future predictions using {model_type.upper()} model...")
    
    # Make a copy of the last data
    current_data = last_data.copy()
    
    # Initialize results storage
    future_predictions = []
    timestamps = []
    
    # Get the last timestamp
    last_timestamp = current_data.index[-1]
    
    # Predict step by step
    for i in range(steps):
        # Prepare data for prediction
        X = prepare_prediction_data(current_data, feature_scaler, targets, features)
        
        # Make prediction
        pred = predict_load(model, X, scalers_y, targets, model_type)
        
        # Add prediction to results
        future_predictions.append(pred.iloc[-1])
        
        # Calculate next timestamp
        next_timestamp = last_timestamp + timedelta(hours=2 if '2:00' in current_data.index.strftime('%H:%M').values else 1)
        timestamps.append(next_timestamp)
        
        # Update current data for next prediction
        # This would include updating time features and lag values based on predictions
        # This is a simplified version and would need to be adapted to the actual features
        new_row = current_data.iloc[-1].copy()
        for j, target in enumerate(targets):
            new_row[target] = pred.iloc[-1][target]
            new_row[f'{target}_lag_1'] = new_row[target]
            # Update lag_24 if we have enough predictions
            if i >= 23:
                new_row[f'{target}_lag_24'] = future_predictions[i-23][target]
        
        # Update time features
        new_row['hour'] = next_timestamp.hour
        new_row['day_of_week'] = next_timestamp.dayofweek
        new_row['month'] = next_timestamp.month
        new_row['is_weekend'] = 1 if next_timestamp.dayofweek >= 5 else 0
        
        # Add the updated row to current_data
        current_data = current_data.append(pd.Series(new_row, name=next_timestamp))
    
    # Combine all predictions into a DataFrame
    future_df = pd.DataFrame(future_predictions, index=timestamps)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for filename
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save predictions to CSV with model type in the filename
    predictions_path = os.path.join(save_dir, f'{model_type}_future_predictions.csv')
    future_df.to_csv(predictions_path)
    print(f"Future predictions saved to {predictions_path}")
    
    # Create plot with enhanced styling
    plt.figure(figsize=(15, 10))
    
    # Define colors for different targets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, target in enumerate(targets):
        plt.plot(future_df.index, future_df[target], marker='o', linewidth=2, 
                 label=f'{target} Forecast', color=colors[i % len(colors)])
    
    plt.title(f'Load Forecast for Next {steps} Hours using {model_type.upper()} Model', fontsize=16)
    plt.ylabel('Load (MW)', fontsize=14)
    plt.xlabel('Datetime', fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    
    # Add timestamp
    plt.figtext(0.01, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                fontsize=8, color='gray')
    
    # Save the plot with model type in the filename
    plt.savefig(os.path.join(figures_dir, f'{model_type}_future_predictions.png'), dpi=300)
    plt.close()
    print(f"Forecast plot saved to {os.path.join(figures_dir, f'{model_type}_future_predictions.png')}")

def save_predictions(predictions, save_dir):
    """
    Save predictions to file and generate visualizations.
    
    Args:
        predictions: DataFrame with predictions
        save_dir: Directory to save predictions
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save to CSV
    csv_path = os.path.join(save_dir, f'predictions_{timestamp}.csv')
    predictions.to_csv(csv_path)
    print(f"Predictions saved to {csv_path}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot each target
    for column in predictions.columns:
        plt.plot(predictions.index, predictions[column], marker='o', label=column)
    
    plt.title('Load Forecasts by Region')
    plt.xlabel('Time')
    plt.ylabel('Load')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'forecast_plot_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Forecast plot saved to {plot_path}")
