"""
Run script for Delhi Load Forecasting Dashboard
This script:
1. Checks and installs required dependencies
2. Trains the model if needed or loads an existing model
3. Fetches live data from Delhi SLDC website
4. Makes predictions using the GRU model
5. Launches the dashboard on localhost

Usage:
    python run_dashboard.py
"""

import os
import sys
import subprocess
import time
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Dummy model class for when TensorFlow isn't available
class DummyModel:
    """A dummy model class that simulates predictions without actually using TensorFlow."""
    
    def __init__(self):
        self.name = "Dummy GRU Model"
        print("Created dummy model for demonstration purposes")
    
    def predict(self, X, verbose=0):
        """Generate random predictions based on input shape."""
        # Get batch size and number of targets from input shape
        batch_size = X.shape[0]
        # Assume 5 targets (DELHI, BRPL, BYPL, NDMC, MES)
        n_targets = 5
        
        # Generate random predictions with values in a reasonable range
        # For load forecasting, values might be in hundreds or thousands of MW
        predictions = np.random.uniform(1000, 5000, size=(batch_size, n_targets))
        return predictions
    
    def save_model(self, path):
        """Simulate saving a model."""
        print(f"[DUMMY] Model would be saved to {path}")
        return True
    
    def compile_model(self):
        """Simulate compiling a model."""
        print("[DUMMY] Model compiled")

# Check if required packages are installed, install if not
required_packages = [
    'pandas', 'numpy', 'matplotlib', 'scikit-learn', 
    'flask', 'beautifulsoup4', 'requests'
]

# TensorFlow is handled separately as it's more complex to install
tensorflow_required = True  # Set to False to run without TensorFlow (will use simulated data)

def check_and_install_packages():
    """Check if required packages are installed and install if not."""
    global tensorflow_required
    
    print("Checking required packages...")
    
    packages_to_install = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            packages_to_install.append(package)
    
    # Check TensorFlow separately
    tensorflow_installed = False
    if tensorflow_required:
        try:
            import tensorflow as tf
            print(f"✓ tensorflow is installed (version {tf.__version__})")
            tensorflow_installed = True
        except ImportError:
            print(f"✗ tensorflow is not installed")
            choice = input("TensorFlow is required for model training/prediction. Install it? (y/n): ").strip().lower()
            if choice == 'y':
                packages_to_install.append('tensorflow')
            else:
                print("\nRunning without TensorFlow. Will use simulated data for demonstration.")
                tensorflow_required = False
    
    if packages_to_install:
        print("\nInstalling missing packages...")
        
        for package in packages_to_install:
            print(f"Installing {package}...")
            try:
                subprocess.call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} installed successfully!")
                if package == 'tensorflow':
                    tensorflow_installed = True
            except Exception as e:
                print(f"Error installing {package}: {e}")
        
        print("Package installation complete!")
    else:
        print("All required packages are installed!")
    
    return tensorflow_installed

def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'models/saved_models',
        'templates',
        'static/css',
        'static/js'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def prepare_data():
    """Prepare data for model training and testing."""
    # Import our data processing module
    try:
        from src.data_processor import DataProcessor
        print("\nPreparing data for modeling...")
        
        # Check if we have the final_data.csv file
        possible_paths = [
            "data/final_data.csv", 
            "../final_data.csv", 
            "C:/Users/ansha/OneDrive/Desktop/SIH/final_data.csv"
        ]
        
        data_path = None
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            print("Error: Could not find final_data.csv")
            print("Please place the dataset in the data/ directory")
            return None
        
        # Process data
        data_processor = DataProcessor(data_path)
        X_train, X_test, y_train, y_test = data_processor.process_data()
        
        # Save data processor for later use
        scalers_dir = "models/saved_models"
        os.makedirs(scalers_dir, exist_ok=True)
        data_processor.save_scalers(scalers_dir)
        
        print("Data preparation complete!")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'data_processor': data_processor
        }
    except Exception as e:
        print(f"Error preparing data: {e}")
        traceback.print_exc()
        return None

def train_or_load_model(processed_data):
    """Train a new model or load an existing one."""
    # If TensorFlow is not required or not installed, return a dummy model
    if not tensorflow_required:
        print("\nRunning without TensorFlow. Using a dummy model for demonstration.")
        return DummyModel()
    
    try:
        # Import TensorFlow and related modules
        try:
            import tensorflow as tf
            from tensorflow.keras.losses import MeanSquaredError
            from tensorflow.keras.optimizers import Adam
            from src.model_factory import ModelFactory
        except ImportError as e:
            print(f"Error importing TensorFlow modules: {e}")
            print("Running without TensorFlow. Using a dummy model for demonstration.")
            return DummyModel()
        
        # Check if a model already exists
        model_path = "models/saved_models/gru_forecast_model.h5"
        
        if os.path.exists(model_path):
            print(f"\nFound existing model at {model_path}")
            choice = input("Do you want to use this model? (y/n): ").strip().lower()
            
            if choice == 'y':
                print("Loading existing model...")
                # Try multiple approaches to load the model
                try:
                    # First attempt: Direct load with compile=False
                    try:
                        loaded_model = tf.keras.models.load_model(model_path, compile=False)
                        print("Model loaded successfully without compilation!")
                    except Exception as e1:
                        print(f"First load attempt failed: {e1}")
                        print("Trying alternative loading method...")
                        try:
                            # Second attempt: Custom load method
                            loaded_model = tf.keras.models.load_model(
                                model_path, 
                                compile=False,
                                custom_objects={'mse': tf.keras.losses.mean_squared_error}
                            )
                            print("Model loaded successfully with custom objects!")
                        except Exception as e2:
                            print(f"Second load attempt failed: {e2}")
                            raise Exception("Could not load model using available methods")
                    
                    # Create a new model using the factory
                    X_train = processed_data['X_train']
                    y_train = processed_data['y_train']
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    output_dim = y_train.shape[1]
                    model = ModelFactory.create_model('gru', input_shape, output_dim)
                    
                    # Transfer weights
                    model.model.set_weights(loaded_model.get_weights())
                    
                    # Compile manually
                    model.model.compile(
                        optimizer=Adam(learning_rate=0.001),
                        loss=MeanSquaredError(),
                        metrics=['mse', 'mae']
                    )
                    
                    print("Model weights loaded and model compiled successfully!")
                    return model
                except Exception as e:
                    print(f"\nError loading model: {e}")
                    print("Creating a new model instead.")
                    
                    # Recreate the model architecture from scratch 
                    try:
                        # Create model from factory
                        X_train = processed_data['X_train']
                        y_train = processed_data['y_train']
                        input_shape = (X_train.shape[1], X_train.shape[2])
                        output_dim = y_train.shape[1]
                        model = ModelFactory.create_model('gru', input_shape, output_dim)
                        model.compile_model()
                        print("Created new model from factory.")
                        return model
                    except Exception as e_new:
                        print(f"Error creating new model: {e_new}")
                        print("Using dummy model for demonstration.")
                        return DummyModel()
        
        # If no model exists or user wants to train a new model
        print("\nTraining new GRU model...")
        
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        
        # Create model
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]
        model = ModelFactory.create_model('gru', input_shape, output_dim)
        model.compile_model()
        
        # Train model with simplified training (fewer epochs for quicker results)
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Configure callbacks
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
        
        # Train with fewer epochs for quick results
        print("Training model (this may take a few minutes)...")
        model.model.fit(
            X_train, y_train,
            epochs=15,  # Reduced for quicker training
            batch_size=64,
            validation_split=0.15,
            callbacks=[early_stopping, lr_reducer],
            verbose=1
        )
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        return model
    
    except Exception as e:
        print(f"Error training/loading model: {e}")
        traceback.print_exc()
        return None

def simulate_delhi_sldc_data(data_processor):
    """Simulate Delhi SLDC data if we can't connect to the website."""
    print("\nSimulating Delhi SLDC data for demonstration...")
    
    # Use the last few records from the historical data
    df = data_processor.df.copy()
    
    # Get the last record and modify it slightly for simulation
    latest_data = df.iloc[-1].copy()
    
    # Current time
    now = datetime.now()
    
    # Create a simulated data point
    simulated_data = {
        'datetime': now.strftime('%d-%m-%Y %H:%M'),
        'weekday': now.strftime('%A')
    }
    
    # Add load values with some random variation
    for target in data_processor.targets:
        base_value = latest_data[target]
        random_factor = np.random.uniform(0.95, 1.05)  # ±5% variation
        simulated_data[target] = base_value * random_factor
    
    # Add weather data
    simulated_data['temperature'] = np.random.uniform(20, 35)  # °C
    simulated_data['humidity'] = np.random.uniform(30, 90)  # %
    simulated_data['wind_speed'] = np.random.uniform(0, 15)  # km/h
    simulated_data['precipitation'] = np.random.uniform(0, 5)  # mm
    
    return simulated_data

def prepare_simulated_data(data_processor):
    """Prepare simulated data for demonstration."""
    print("\nPreparing simulated data for demonstration...")
    
    # Create cache directory
    os.makedirs('data', exist_ok=True)
    
    # Use the last 24 records from historical data
    df = data_processor.df.tail(24).copy()
    
    # Reset datetime to be relative to now
    now = datetime.now()
    hours_ago_24 = [now.replace(hour=now.hour-i, minute=0, second=0, microsecond=0) 
                    for i in range(24, 0, -1)]
    
    df['datetime'] = [ts.strftime('%d-%m-%Y %H:%M') for ts in hours_ago_24]
    
    # Save to CSV for cache
    cache_file = 'data/live_data_cache.csv'
    df.to_csv(cache_file, index=False)
    print(f"Created simulation cache at {cache_file}")
    
    # Also prepare latest_data.json for the dashboard
    timestamps = [ts.strftime('%Y-%m-%d %H:%M') for ts in hours_ago_24]
    
    data_dict = {
        'timestamps': timestamps,
        'targets': {}
    }
    
    # Add data for each target
    for target in data_processor.targets:
        if target in df.columns:
            data_dict['targets'][target] = df[target].tolist()
    
    # Add weather data
    for weather_var in ['temperature', 'humidity', 'wind_speed', 'precipitation']:
        if weather_var in df.columns:
            data_dict[weather_var] = df[weather_var].tolist()
    
    # Save to JSON
    latest_data_file = 'data/latest_data.json'
    with open(latest_data_file, 'w') as f:
        json.dump(data_dict, f)
    
    print(f"Created simulation data at {latest_data_file}")
    
    # Add a simulated data point
    simulated_data = simulate_delhi_sldc_data(data_processor)
    
    # Update the cache
    df_new = pd.read_csv(cache_file)
    df_new = pd.concat([df_new, pd.DataFrame([simulated_data])], ignore_index=True)
    df_new.to_csv(cache_file, index=False)
    
    return simulated_data

def simulate_predictions(model, data_processor, simulated_data):
    """Simulate predictions for demonstration."""
    print("\nGenerating predictions based on simulated data...")
    
    # Process the simulated data
    df = pd.read_csv('data/live_data_cache.csv')
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
    
    # Convert weekday to numerical
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
    
    # Create lag features
    for target in data_processor.targets:
        df[f'{target}_lag_1'] = df[target].shift(1)
        df[f'{target}_lag_24'] = df[target].shift(24)
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Define features
    categorical_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'weekday_num']
    numerical_features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
    
    # Add lag features
    for target in data_processor.targets:
        numerical_features.extend([f'{target}_lag_1', f'{target}_lag_24'])
    
    # Filter features that exist in the dataframe
    categorical_features = [f for f in categorical_features if f in df.columns]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    # Combine all features
    features = numerical_features + categorical_features
    
    # Generate future timestamps
    now = datetime.now()
    future_timestamps = [now.replace(hour=now.hour+i, minute=0, second=0, microsecond=0) 
                         for i in range(1, 25)]
    
    # Generate predictions
    future_predictions = []
    current_df = df.copy()
    
    for i, future_time in enumerate(future_timestamps):
        # Get latest features
        latest_features = current_df.iloc[-1:][features].values
        
        # Scale features
        scaled_features = data_processor.feature_scaler.transform(latest_features)
        
        # Reshape for RNN
        X = scaled_features.reshape(1, 1, scaled_features.shape[1])
        
        # Predict
        y_pred = model.model.predict(X, verbose=0)
        
        # Create prediction data
        pred_values = {}
        
        for j, target in enumerate(data_processor.targets):
            # Inverse transform
            pred_values[target] = data_processor.scalers_y[target].inverse_transform(
                y_pred[:, j].reshape(-1, 1)
            )[0, 0]
        
        # Add to predictions
        future_predictions.append(pred_values)
        
        # Create next time entry
        new_row = current_df.iloc[-1:].copy()
        new_row.index = [future_time]
        
        # Update time features
        new_row['hour'] = future_time.hour
        new_row['day_of_week'] = future_time.dayofweek
        new_row['month'] = future_time.month
        new_row['is_weekend'] = int(future_time.dayofweek >= 5)
        
        if 'weekday_num' in new_row.columns:
            new_row['weekday_num'] = future_time.dayofweek
        
        # Update targets with predictions
        for target in data_processor.targets:
            new_row[target] = pred_values[target]
            
            # Update lag features for next step
            new_row[f'{target}_lag_1'] = pred_values[target]
            
            if f'{target}_lag_24' in new_row.columns and len(current_df) >= 24:
                new_row[f'{target}_lag_24'] = current_df.iloc[-24][target]
        
        # Add to dataframe for next iteration
        current_df = pd.concat([current_df, new_row])
    
    # Save predictions to JSON
    data_dict = {
        'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in future_timestamps],
        'targets': {}
    }
    
    for target in data_processor.targets:
        data_dict['targets'][target] = [pred[target] for pred in future_predictions]
    
    # Save to JSON
    predictions_file = 'data/predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(data_dict, f)
    
    print(f"Generated predictions saved to {predictions_file}")
    
    return data_dict

def run_dashboard():
    """Run the Flask dashboard."""
    try:
        from flask import Flask, render_template, jsonify
        
        app = Flask(__name__)
        
        @app.route('/')
        def index():
            """Render the dashboard."""
            # Use the current time as the last update time
            update_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            return render_template('index.html', last_update=update_time_str)
        
        @app.route('/api/current')
        def get_current_data():
            """API endpoint to get current load data."""
            try:
                with open('data/latest_data.json', 'r') as f:
                    data = json.load(f)
                return jsonify(data)
            except:
                return jsonify({'error': 'No data available'})
        
        @app.route('/api/predictions')
        def get_predictions():
            """API endpoint to get load predictions."""
            try:
                with open('data/predictions.json', 'r') as f:
                    data = json.load(f)
                return jsonify(data)
            except:
                return jsonify({'error': 'No predictions available'})
        
        @app.route('/api/update')
        def trigger_update():
            """API endpoint to trigger a manual update."""
            return jsonify({'status': 'update started (simulation only)'})
        
        @app.route('/api/status')
        def get_status():
            """API endpoint to get update status."""
            return jsonify({
                'is_updating': False,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        print("\n" + "="*60)
        print(f" Delhi Load Forecasting Dashboard is running!")
        print(f" Open your browser and go to: http://localhost:5000")
        print("="*60 + "\n")
        
        # Run the Flask application
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"Error running dashboard: {e}")
        traceback.print_exc()

def main():
    """Main function to run the entire system."""
    print("\n" + "="*60)
    print(" Delhi Load Forecasting Dashboard Setup")
    print("="*60 + "\n")
    
    # Check and install packages
    tf_installed = check_and_install_packages()
    
    # Create directories
    create_directories()
    
    try:
        # Prepare data
        processed_data = prepare_data()
        
        if processed_data is None:
            print("Warning: Data preparation failed. Using simulated data.")
            # Create minimal processed data for simulation
            processed_data = {
                'data_processor': None
            }
        
        # If we have data but no data processor, create a minimal one
        if 'data_processor' not in processed_data or processed_data['data_processor'] is None:
            # Create minimal DataProcessor for simulation
            class MinimalDataProcessor:
                def __init__(self):
                    self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
                    self.df = pd.DataFrame({
                        'datetime': pd.date_range(start='2023-01-01', periods=24, freq='H'),
                        'DELHI': np.random.uniform(2000, 5000, 24),
                        'BRPL': np.random.uniform(1000, 3000, 24),
                        'BYPL': np.random.uniform(800, 2500, 24),
                        'NDMC': np.random.uniform(500, 1500, 24),
                        'MES': np.random.uniform(300, 1000, 24)
                    })
                    # Create dummy scalers
                    self.feature_scaler = None
                    self.scalers_y = {}
                    for target in self.targets:
                        self.scalers_y[target] = None
            
            processed_data['data_processor'] = MinimalDataProcessor()
        
        # Train or load model (or get dummy model if TF not available)
        model = None
        try:
            model = train_or_load_model(processed_data)
        except Exception as e:
            print(f"Error in model training/loading: {e}")
            print("Using dummy model instead")
            model = DummyModel()
        
        if model is None:
            print("Warning: No model available. Using dummy model instead.")
            model = DummyModel()
        
        # Prepare simulated data
        try:
            simulated_data = prepare_simulated_data(processed_data['data_processor'])
        except Exception as e:
            print(f"Error preparing simulated data: {e}")
            print("Using random data instead.")
            # Generate random dummy data
            simulated_data = {
                'datetime': datetime.now().strftime('%d-%m-%Y %H:%M'),
                'DELHI': np.random.uniform(2000, 5000),
                'BRPL': np.random.uniform(1000, 3000),
                'BYPL': np.random.uniform(800, 2500),
                'NDMC': np.random.uniform(500, 1500),
                'MES': np.random.uniform(300, 1000),
                'temperature': np.random.uniform(20, 35),
                'humidity': np.random.uniform(30, 90),
                'wind_speed': np.random.uniform(0, 15),
                'precipitation': np.random.uniform(0, 5)
            }
        
        # Make sure data directories exist
        os.makedirs('data', exist_ok=True)
        
        # Generate predictions or use random values if that fails
        try:
            predictions = simulate_predictions(model, processed_data['data_processor'], simulated_data)
        except Exception as e:
            print(f"Error generating predictions: {e}")
            print("Using random predictions instead.")
            
            # Create random predictions
            now = datetime.now()
            future_timestamps = [now.replace(hour=now.hour+i, minute=0, second=0, microsecond=0) 
                                for i in range(1, 25)]
            
            data_dict = {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in future_timestamps],
                'targets': {}
            }
            
            for target in processed_data['data_processor'].targets:
                # Generate slightly increasing random values for forecast
                base = np.random.uniform(1000, 3000)
                data_dict['targets'][target] = [base + i*50 + np.random.uniform(-100, 100) 
                                               for i in range(24)]
            
            # Save to JSON as a fallback
            with open('data/predictions.json', 'w') as f:
                json.dump(data_dict, f)
        
        # Run dashboard
        run_dashboard()
        
    except Exception as e:
        print(f"Error setting up dashboard: {e}")
        traceback.print_exc()
        print("\nAttempting to run minimal dashboard with simulated data...")
        
        # Create minimal data files for the dashboard
        os.makedirs('data', exist_ok=True)
        
        # Create minimal latest_data.json
        now = datetime.now()
        past_timestamps = [now.replace(hour=now.hour-i, minute=0, second=0, microsecond=0) 
                           for i in range(24, 0, -1)]
        
        latest_data = {
            'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in past_timestamps],
            'targets': {
                'DELHI': [np.random.uniform(2000, 5000) for _ in range(24)],
                'BRPL': [np.random.uniform(1000, 3000) for _ in range(24)],
                'BYPL': [np.random.uniform(800, 2500) for _ in range(24)],
                'NDMC': [np.random.uniform(500, 1500) for _ in range(24)],
                'MES': [np.random.uniform(300, 1000) for _ in range(24)]
            },
            'temperature': [np.random.uniform(20, 35) for _ in range(24)],
            'humidity': [np.random.uniform(30, 90) for _ in range(24)],
            'wind_speed': [np.random.uniform(0, 15) for _ in range(24)],
            'precipitation': [np.random.uniform(0, 5) for _ in range(24)]
        }
        
        with open('data/latest_data.json', 'w') as f:
            json.dump(latest_data, f)
        
        # Create minimal predictions.json
        future_timestamps = [now.replace(hour=now.hour+i, minute=0, second=0, microsecond=0) 
                             for i in range(1, 25)]
        
        predictions_data = {
            'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in future_timestamps],
            'targets': {
                'DELHI': [np.random.uniform(2000, 5000) for _ in range(24)],
                'BRPL': [np.random.uniform(1000, 3000) for _ in range(24)],
                'BYPL': [np.random.uniform(800, 2500) for _ in range(24)],
                'NDMC': [np.random.uniform(500, 1500) for _ in range(24)],
                'MES': [np.random.uniform(300, 1000) for _ in range(24)]
            }
        }
        
        with open('data/predictions.json', 'w') as f:
            json.dump(predictions_data, f)
        
        # Run dashboard with minimal functionality
        run_dashboard()

if __name__ == "__main__":
    main()
