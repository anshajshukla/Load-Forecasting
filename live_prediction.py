"""
Script to make live predictions using trained model and display results
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json
import datetime
from tensorflow.keras.models import load_model
from fetch_live_data import DelhiSLDCDataFetcher

class LivePredictor:
    """Class to make live predictions using trained model."""
    
    def __init__(self, model_path='models/saved_models/gru_forecast_model.h5',
                 scalers_dir='models/saved_models',
                 data_fetcher=None,
                 prediction_steps=24):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            scalers_dir: Directory containing scalers
            data_fetcher: Instance of DelhiSLDCDataFetcher
            prediction_steps: Number of future steps to predict
        """
        self.model_path = model_path
        self.scalers_dir = scalers_dir
        self.prediction_steps = prediction_steps
        self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        
        # Create data fetcher if not provided
        if data_fetcher is None:
            self.data_fetcher = DelhiSLDCDataFetcher(
                model_path=model_path,
                scalers_dir=scalers_dir
            )
        else:
            self.data_fetcher = data_fetcher
        
        # Load model and scalers
        self.model = self.load_model()
        self.scalers_y = self.data_fetcher.load_target_scalers()
    
    def load_model(self):
        """
        Load the trained model.
        
        Returns:
            model: Loaded model
        """
        try:
            if os.path.exists(self.model_path):
                model = load_model(self.model_path)
                print(f"Model loaded from {self.model_path}")
                return model
            else:
                print(f"Model file not found: {self.model_path}")
                return None
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict_current_load(self):
        """
        Make prediction for current load.
        
        Returns:
            dict: Dictionary containing prediction results
        """
        if self.model is None:
            print("No model loaded, cannot make predictions")
            return None
        
        # Get processed data
        processed_data = self.data_fetcher.prepare_data_for_prediction()
        
        if processed_data is None:
            print("No data available for prediction")
            return None
        
        try:
            # Make prediction
            X = processed_data['X']
            y_pred = self.model.predict(X)
            
            # Inverse transform prediction
            y_pred_inv = {}
            
            for i, target in enumerate(self.targets):
                if self.scalers_y and target in self.scalers_y:
                    y_pred_inv[target] = self.scalers_y[target].inverse_transform(
                        y_pred[:, i].reshape(-1, 1)
                    )[0, 0]
                else:
                    # If scaler not available, use the prediction as is
                    y_pred_inv[target] = y_pred[0, i]
            
            # Get actual values if available
            latest_data = processed_data['latest_data']
            actual_values = {}
            
            for target in self.targets:
                if target in latest_data.columns:
                    actual_values[target] = latest_data[target].values[0]
                else:
                    actual_values[target] = None
            
            # Calculate accuracy if actual values are available
            accuracy = {}
            
            for target in self.targets:
                if actual_values[target] is not None and actual_values[target] > 0:
                    # MAPE
                    error = abs(y_pred_inv[target] - actual_values[target]) / actual_values[target] * 100
                    accuracy[target] = 100 - error  # Convert error to accuracy
                else:
                    accuracy[target] = None
            
            # Results
            results = {
                'timestamp': latest_data.index[0].strftime('%Y-%m-%d %H:%M'),
                'predictions': y_pred_inv,
                'actual_values': actual_values,
                'accuracy': accuracy
            }
            
            # Print results
            print("\nCurrent Load Prediction Results:")
            print(f"Timestamp: {results['timestamp']}")
            
            for target in self.targets:
                pred = results['predictions'][target]
                actual = results['actual_values'][target]
                acc = results['accuracy'][target]
                
                print(f"\n{target}:")
                print(f"  Predicted: {pred:.2f} MW")
                
                if actual is not None:
                    print(f"  Actual: {actual:.2f} MW")
                    
                    if acc is not None:
                        print(f"  Accuracy: {acc:.2f}%")
                        
                        if acc < 80:
                            print("  Warning: Low prediction accuracy!")
            
            return results
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_future_load(self):
        """
        Make predictions for future hours.
        
        Returns:
            dict: Dictionary containing future predictions
        """
        if self.model is None:
            print("No model loaded, cannot make predictions")
            return None
        
        try:
            # Get processed data
            processed_data = self.data_fetcher.prepare_data_for_prediction()
            
            if processed_data is None:
                print("No data available for prediction")
                return None
            
            # Initialize variables
            df = processed_data['df'].copy()
            features = processed_data['features']
            future_predictions = []
            timestamps = []
            
            # Current time as reference
            last_time = df.index[-1]
            
            # Make predictions for the next hours
            for i in range(self.prediction_steps):
                # Calculate next timestamp
                next_time = last_time + datetime.timedelta(hours=1)
                timestamps.append(next_time)
                
                # Prepare features for prediction
                X = df.iloc[-1:][features].values
                
                # Scale features
                feature_scaler = self.data_fetcher.load_feature_scaler()
                
                if feature_scaler is not None:
                    try:
                        # If it's a scikit-learn pipeline
                        X_scaled = feature_scaler.transform(df.iloc[-1:][features])
                    except:
                        # If it's a simple scaler
                        X_scaled = feature_scaler.transform(X)
                else:
                    # If no scaler, use normalized data
                    X_scaled = X
                
                # Reshape for RNN (1, timesteps, features)
                X_scaled = X_scaled.reshape(1, 1, X_scaled.shape[1])
                
                # Make prediction
                y_pred = self.model.predict(X_scaled)
                
                # Create a new row for the prediction
                new_row = df.iloc[-1:].copy()
                new_row.index = [next_time]
                
                # Update time features
                new_row['hour'] = next_time.hour
                new_row['day_of_week'] = next_time.dayofweek
                new_row['month'] = next_time.month
                new_row['is_weekend'] = int(next_time.dayofweek >= 5)
                
                if 'weekday_num' in new_row.columns:
                    new_row['weekday_num'] = next_time.dayofweek
                
                # Update prediction results
                for i, target in enumerate(self.targets):
                    if self.scalers_y and target in self.scalers_y:
                        # Inverse transform prediction
                        value = self.scalers_y[target].inverse_transform(
                            y_pred[:, i].reshape(-1, 1)
                        )[0, 0]
                    else:
                        # If scaler not available, use the prediction as is
                        value = y_pred[0, i]
                    
                    # Update the predicted value
                    new_row[target] = value
                    
                    # Update lag features for next step
                    new_row[f'{target}_lag_1'] = value
                    
                    if f'{target}_lag_24' in new_row.columns and len(df) >= 24:
                        new_row[f'{target}_lag_24'] = df.iloc[-24][target]
                
                # Append new row to the dataframe
                df = pd.concat([df, new_row])
                
                # Store prediction
                pred_values = {}
                for i, target in enumerate(self.targets):
                    if self.scalers_y and target in self.scalers_y:
                        pred_values[target] = self.scalers_y[target].inverse_transform(
                            y_pred[:, i].reshape(-1, 1)
                        )[0, 0]
                    else:
                        pred_values[target] = y_pred[0, i]
                
                future_predictions.append(pred_values)
                
                # Update last time
                last_time = next_time
            
            # Prepare results
            future_df = pd.DataFrame(future_predictions, index=timestamps)
            
            # Print future predictions
            print("\nFuture Load Predictions:")
            
            for target in self.targets:
                print(f"\n{target}:")
                for i, (idx, row) in enumerate(future_df.iterrows()):
                    print(f"  {idx.strftime('%Y-%m-%d %H:%M')}: {row[target]:.2f} MW")
                    if i >= 4:  # Show only the first 5 predictions
                        print(f"  ... and {len(future_df) - 5} more hours")
                        break
            
            # Save predictions for dashboard
            self.save_predictions(future_df)
            
            return {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in timestamps],
                'predictions': future_predictions
            }
            
        except Exception as e:
            print(f"Error making future predictions: {e}")
            return None
    
    def save_predictions(self, future_df, output_file='data/predictions.json'):
        """
        Save predictions to a JSON file for the dashboard.
        
        Args:
            future_df: DataFrame containing predictions
            output_file: Path to save the JSON file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Prepare data for JSON
            data_dict = {
                'timestamps': [ts.strftime('%Y-%m-%d %H:%M') for ts in future_df.index],
                'targets': {}
            }
            
            # Add data for each target
            for target in self.targets:
                data_dict['targets'][target] = future_df[target].tolist()
            
            # Save to JSON
            with open(output_file, 'w') as f:
                json.dump(data_dict, f)
            
            print(f"\nPredictions saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def run_prediction_cycle(self):
        """Run a complete prediction cycle."""
        # Fetch live data
        live_data = self.data_fetcher.fetch_live_data()
        
        if live_data:
            # Update cache
            self.data_fetcher.update_cache(live_data)
            
            # Export latest data for dashboard
            self.data_fetcher.export_latest_data()
            
            # Make current load prediction
            current_results = self.predict_current_load()
            
            # Make future load predictions
            future_results = self.predict_future_load()
            
            return {
                'current': current_results,
                'future': future_results
            }
        else:
            print("No live data available, cannot make predictions")
            return None

# Example usage
if __name__ == "__main__":
    predictor = LivePredictor()
    results = predictor.run_prediction_cycle()
    
    if results:
        print("\nPrediction cycle completed successfully")
    else:
        print("\nFailed to complete prediction cycle")
