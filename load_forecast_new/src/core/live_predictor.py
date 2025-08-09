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
        """Load the trained model."""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            return load_model(self.model_path)
        else:
            print(f"Model not found at {self.model_path}")
            return None
    
    def make_predictions(self, input_data):
        """
        Make predictions using the loaded model.
        
        Args:
            input_data: Preprocessed input data for prediction
            
        Returns:
            Scaled predictions
        """
        if self.model is None:
            print("Model not loaded. Cannot make predictions.")
            return None
        
        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            print(f"Error making predictions: {e}")
            return None
    
    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform predictions to original scale.
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Unscaled predictions
        """
        unscaled_predictions = {}
        
        for i, target in enumerate(self.targets):
            if target in self.scalers_y:
                # Reshape for inverse transform
                pred_reshaped = predictions[:, i].reshape(-1, 1)
                unscaled = self.scalers_y[target].inverse_transform(pred_reshaped)
                unscaled_predictions[target] = unscaled.flatten()
            else:
                print(f"Scaler for {target} not found")
                unscaled_predictions[target] = predictions[:, i]
        
        return unscaled_predictions
    
    def run_prediction_cycle(self):
        """
        Run complete prediction cycle: fetch data, preprocess, predict, save results.
        
        Returns:
            Dictionary with prediction results
        """
        print(f"\n{'='*50}")
        print(f"Starting prediction cycle at {datetime.datetime.now()}")
        print(f"{'='*50}")
        
        try:
            # Step 1: Fetch and preprocess data
            print("Step 1: Fetching and preprocessing live data...")
            input_data = self.data_fetcher.fetch_and_preprocess_live_data()
            
            if input_data is None:
                print("Failed to fetch and preprocess data")
                return None
            
            print(f"Input data shape: {input_data.shape}")
            
            # Step 2: Make predictions
            print("Step 2: Making predictions...")
            predictions = self.make_predictions(input_data)
            
            if predictions is None:
                print("Failed to make predictions")
                return None
            
            print(f"Raw predictions shape: {predictions.shape}")
            
            # Step 3: Inverse transform predictions
            print("Step 3: Inverse transforming predictions...")
            unscaled_predictions = self.inverse_transform_predictions(predictions)
            
            # Step 4: Create timestamps for predictions
            now = datetime.datetime.now()
            prediction_timestamps = [
                (now + datetime.timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S')
                for i in range(self.prediction_steps)
            ]
            
            # Step 5: Format results
            results = {
                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_horizon': f"{self.prediction_steps} hours",
                'predictions': {}
            }
            
            for target in self.targets:
                if target in unscaled_predictions:
                    results['predictions'][target] = {
                        'values': unscaled_predictions[target].tolist(),
                        'timestamps': prediction_timestamps,
                        'unit': 'MW'
                    }
            
            # Step 6: Save results
            self.save_predictions(results)
            
            # Step 7: Display summary
            self.display_prediction_summary(results)
            
            print(f"\n{'='*50}")
            print("Prediction cycle completed successfully!")
            print(f"{'='*50}\n")
            
            return results
            
        except Exception as e:
            print(f"Error in prediction cycle: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_predictions(self, results):
        """
        Save prediction results to JSON file.
        
        Args:
            results: Dictionary containing prediction results
        """
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Save to JSON file
            output_file = 'data/predictions.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Predictions saved to {output_file}")
            
        except Exception as e:
            print(f"Error saving predictions: {e}")
    
    def display_prediction_summary(self, results):
        """
        Display a summary of predictions.
        
        Args:
            results: Dictionary containing prediction results
        """
        print(f"\n📊 PREDICTION SUMMARY")
        print(f"{'='*40}")
        print(f"Timestamp: {results['timestamp']}")
        print(f"Horizon: {results['prediction_horizon']}")
        
        for target, data in results['predictions'].items():
            values = data['values']
            if values:
                avg_load = np.mean(values)
                max_load = np.max(values)
                min_load = np.min(values)
                
                print(f"\n🏢 {target}:")
                print(f"  Average: {avg_load:.2f} MW")
                print(f"  Peak: {max_load:.2f} MW")
                print(f"  Minimum: {min_load:.2f} MW")
                
                # Show next few predictions
                print(f"  Next 6 hours: {', '.join([f'{v:.1f}' for v in values[:6]])} MW")
        
        # Calculate total Delhi load (first prediction)
        delhi_predictions = results['predictions'].get('DELHI', {}).get('values', [])
        if delhi_predictions:
            next_hour_total = delhi_predictions[0]
            print(f"\n🌆 Next Hour Total Delhi Load: {next_hour_total:.2f} MW")


def main():
    """Main function to run live predictions."""
    print("🚀 Starting Live Delhi Load Forecasting System")
    print("=" * 60)
    
    # Initialize predictor
    predictor = LivePredictor()
    
    # Run prediction cycle
    results = predictor.run_prediction_cycle()
    
    if results:
        print("✅ Live prediction completed successfully!")
    else:
        print("❌ Live prediction failed!")


if __name__ == "__main__":
    main()
