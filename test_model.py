# Set non-interactive backend first
import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Silence TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import our modules
from src.model_factory import ModelFactory
from src.data_processor import DataProcessor

def print_section(title):
    """Print a section header for better output readability."""
    print("\n" + "="*50)
    print(f" {title}")
    print("="*50 + "\n")

def test_model(model_type='gru', data_path=None, epochs=20, test_size=0.2):
    """Test a specific model architecture and report accuracy."""
    
    # Find the data file if not specified
    if data_path is None:
        for path in ["data/final_data.csv", "../final_data.csv", "C:/Users/ansha/OneDrive/Desktop/SIH/final_data.csv"]:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError("Could not find final_data.csv. Please specify a valid data_path.")
    
    print_section(f"Testing {model_type.upper()} Model")
    print(f"Data path: {data_path}")
    
    # Process data
    print("Loading and processing data...")
    data_processor = DataProcessor(data_path, test_size=test_size)
    X_train, X_test, y_train, y_test = data_processor.process_data()
    
    # Create model
    print(f"\nCreating {model_type.upper()} model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_dim = y_train.shape[1]
    model = ModelFactory.create_model(model_type, input_shape, output_dim)
    model.compile_model()
    
    # Train model (simplified training)
    print(f"\nTraining {model_type.upper()} model with {epochs} epochs...")
    
    # Configure callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train
    history = model.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        validation_split=0.15,
        callbacks=[early_stopping, lr_reducer],
        verbose=1
    )
    
    # Evaluate on test data
    print(f"\nEvaluating {model_type.upper()} model...")
    test_loss = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss (MSE): {test_loss:.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_test_inv = np.zeros_like(y_test)
    y_pred_inv = np.zeros_like(y_pred)
    
    targets = data_processor.targets
    
    for i, target in enumerate(targets):
        y_test_inv[:, i] = data_processor.scalers_y[target].inverse_transform(y_test[:, i].reshape(-1, 1)).ravel()
        y_pred_inv[:, i] = data_processor.scalers_y[target].inverse_transform(y_pred[:, i].reshape(-1, 1)).ravel()
    
    # Calculate and print metrics for each target
    print("\nPerformance Metrics:")
    all_mape = []
    
    for i, target in enumerate(targets):
        # Calculate RMSE
        rmse = np.sqrt(np.mean(np.square(y_test_inv[:, i] - y_pred_inv[:, i])))
        
        # Calculate MAPE
        mask = y_test_inv[:, i] != 0
        mape = np.mean(np.abs((y_test_inv[:, i][mask] - y_pred_inv[:, i][mask]) / y_test_inv[:, i][mask])) * 100
        all_mape.append(mape)
        
        # Calculate R²
        ss_res = np.sum(np.square(y_test_inv[:, i] - y_pred_inv[:, i]))
        ss_tot = np.sum(np.square(y_test_inv[:, i] - np.mean(y_test_inv[:, i])))
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"  {target}:")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    R²:   {r2:.4f}")
    
    print(f"\nAverage MAPE across all regions: {np.mean(all_mape):.2f}%")
    
    # Create and save a simple plot for one target
    plt.figure(figsize=(12, 6))
    sample_size = min(100, len(y_test_inv))
    sample_indices = np.random.choice(len(y_test_inv), sample_size, replace=False)
    
    # Plot for DELHI (first target)
    plt.plot(y_test_inv[sample_indices, 0], 'b-', label='Actual')
    plt.plot(y_pred_inv[sample_indices, 0], 'r-', label='Predicted')
    plt.title(f'{model_type.upper()} Model - {targets[0]} Load Forecast')
    plt.ylabel('Load (MW)')
    plt.xlabel('Sample')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    os.makedirs('models/saved_models/figures', exist_ok=True)
    plt.savefig(f'models/saved_models/figures/{model_type}_test_plot.png')
    plt.close()
    
    print(f"\nTest plot saved to models/saved_models/figures/{model_type}_test_plot.png")
    
    return {
        'model_type': model_type,
        'mse': test_loss,
        'mape': np.mean(all_mape),
        'model': model
    }

def compare_models(data_path=None, epochs=20, models=None):
    """Compare different model architectures."""
    
    if models is None:
        models = ['gru', 'enhanced_gru', 'lstm', 'bilstm']
    
    print_section("Comparing Model Architectures")
    
    results = {}
    
    for model_type in models:
        try:
            result = test_model(model_type, data_path, epochs)
            results[model_type] = result
        except Exception as e:
            print(f"Error testing {model_type}: {e}")
    
    # Print summary of results
    print_section("Model Comparison Summary")
    
    print(f"{'Model Type':<15} {'MSE':<10} {'MAPE (%)':<10}")
    print("-" * 35)
    
    for model_type, result in results.items():
        print(f"{model_type.upper():<15} {result['mse']:<10.4f} {result['mape']:<10.2f}")
    
    # Identify best model
    best_model = min(results.items(), key=lambda x: x[1]['mape'])[0]
    print(f"\nBest model based on MAPE: {best_model.upper()}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test load forecasting models")
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "enhanced_gru", "lstm", "bilstm", "cnn_lstm"], 
                      help="Model type to test")
    parser.add_argument("--compare", action="store_true", help="Compare multiple models")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.data_path, args.epochs)
    else:
        test_model(args.model, args.data_path, args.epochs)
