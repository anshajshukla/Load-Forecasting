"""
Main Forecasting Engine
Core forecasting system that integrates data processing, model management, and prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from .data_processor import DataProcessor
from .model_manager import ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """
    Advanced forecasting engine for load prediction with SIH 2024 enhancements
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ForecastingEngine
        
        Args:
            config: Configuration dictionary for forecasting parameters
        """
        self.config = config or self._get_default_config()
        self.data_processor = DataProcessor(self.config.get('data_processing', {}))
        self.model_manager = ModelManager(self.config.get('model_management', {}))
        self.is_trained = False
        self.forecast_horizon = self.config['forecast_horizon']
        self.prediction_cache = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for forecasting engine"""
        return {
            'forecast_horizon': 24,  # hours
            'model_types': ['gru', 'lstm'],
            'ensemble_method': 'weighted_average',
            'confidence_intervals': True,
            'real_time_updates': True,
            'cache_predictions': True,
            'target_column': 'load_demand',
            'data_processing': {
                'sequence_length': 24,
                'feature_engineering': True,
                'outlier_detection': True
            },
            'model_management': {
                'epochs': 50,
                'batch_size': 32,
                'early_stopping_patience': 10
            }
        }
    
    def train_forecasting_models(self, data_path: str, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Train all forecasting models on the provided data
        
        Args:
            data_path: Path to the training data
            target_column: Name of the target column (optional)
            
        Returns:
            Training results summary
        """
        logger.info("🚀 Starting forecasting model training pipeline...")
        
        target_col = target_column or self.config['target_column']
        
        try:
            # Step 1: Process data
            logger.info("📊 Processing training data...")
            processed_data = self.data_processor.process_data(data_path)
            
            if target_col not in processed_data.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            # Step 2: Prepare sequences
            logger.info("🔄 Preparing sequences for training...")
            X, y = self.model_manager.prepare_sequences(processed_data, target_col)
            
            # Step 3: Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.model_manager.split_data(X, y)
            
            # Step 4: Train models
            training_results = {}
            for model_type in self.config['model_types']:
                logger.info(f"🏋️ Training {model_type.upper()} model...")
                
                model = self.model_manager.train_model(
                    model_type, X_train, y_train, X_val, y_val
                )
                
                # Evaluate model
                metrics = self.model_manager.evaluate_model(model_type, X_test, y_test)
                training_results[model_type] = {
                    'model': model,
                    'metrics': metrics,
                    'training_samples': len(X_train),
                    'validation_samples': len(X_val),
                    'test_samples': len(X_test)
                }
                
                # Save model
                save_path = self.model_manager.save_model(model_type)
                training_results[model_type]['save_path'] = save_path
            
            self.is_trained = True
            
            # Create training summary
            summary = {
                'training_completed': True,
                'models_trained': list(training_results.keys()),
                'best_model': self._find_best_model(training_results),
                'data_info': self.data_processor.get_feature_info(),
                'training_results': training_results
            }
            
            logger.info("✅ Model training pipeline completed successfully!")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error in training pipeline: {e}")
            raise
    
    def _find_best_model(self, training_results: Dict) -> str:
        """
        Find the best performing model based on validation metrics
        
        Args:
            training_results: Dictionary of training results
            
        Returns:
            Name of the best model
        """
        best_model = None
        best_rmse = float('inf')
        
        for model_name, results in training_results.items():
            rmse = results['metrics'].get('rmse', float('inf'))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
        
        return best_model
    
    def generate_forecast(self, input_data: pd.DataFrame, 
                         forecast_steps: Optional[int] = None,
                         model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate load forecast for specified time horizon
        
        Args:
            input_data: Recent data for making predictions
            forecast_steps: Number of steps to forecast (optional)
            model_name: Specific model to use (optional, uses best if None)
            
        Returns:
            Dictionary containing forecast results
        """
        if not self.is_trained:
            raise RuntimeError("Models must be trained before generating forecasts")
        
        steps = forecast_steps or self.forecast_horizon
        logger.info(f"🔮 Generating {steps}-step forecast...")
        
        try:
            # Process input data
            processed_input = self._prepare_input_data(input_data)
            
            # Generate predictions for each model
            predictions = {}
            confidence_intervals = {}
            
            models_to_use = [model_name] if model_name else list(self.model_manager.models.keys())
            
            for model in models_to_use:
                if model in self.model_manager.models:
                    pred, conf_int = self._predict_with_model(processed_input, model, steps)
                    predictions[model] = pred
                    confidence_intervals[model] = conf_int
            
            # Create ensemble prediction if multiple models
            if len(predictions) > 1:
                ensemble_pred = self._create_ensemble_prediction(predictions)
                predictions['ensemble'] = ensemble_pred
            
            # Generate forecast timestamps
            last_timestamp = input_data.index[-1] if hasattr(input_data, 'index') else datetime.now()
            forecast_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(steps)]
            
            # Create forecast result
            forecast_result = {
                'timestamps': forecast_timestamps,
                'predictions': predictions,
                'confidence_intervals': confidence_intervals,
                'forecast_horizon': steps,
                'generated_at': datetime.now(),
                'input_data_shape': processed_input.shape,
                'models_used': list(predictions.keys())
            }
            
            # Cache results if enabled
            if self.config['cache_predictions']:
                cache_key = f"forecast_{last_timestamp}_{steps}"
                self.prediction_cache[cache_key] = forecast_result
            
            logger.info(f"✅ Forecast generated successfully for {steps} steps")
            return forecast_result
            
        except Exception as e:
            logger.error(f"❌ Error generating forecast: {e}")
            raise
    
    def _prepare_input_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare input data for prediction
        
        Args:
            data: Input DataFrame
            
        Returns:
            Processed input array
        """
        # Apply same preprocessing as training data
        if hasattr(self.data_processor, 'scalers') and self.data_processor.scalers:
            # Apply saved scalers
            for col, scaler in self.data_processor.scalers.items():
                if col in data.columns:
                    data[col] = scaler.transform(data[[col]])
        
        # Get sequence length from config
        seq_len = self.config['data_processing']['sequence_length']
        
        # Take last sequence_length rows
        if len(data) >= seq_len:
            sequence_data = data.iloc[-seq_len:].values
        else:
            # Pad with zeros if insufficient data
            padding_needed = seq_len - len(data)
            padding = np.zeros((padding_needed, data.shape[1]))
            sequence_data = np.vstack([padding, data.values])
        
        # Reshape for model input (1, seq_len, features)
        return sequence_data.reshape(1, seq_len, -1)
    
    def _predict_with_model(self, input_data: np.ndarray, model_name: str, 
                           steps: int) -> Tuple[List[float], List[Tuple[float, float]]]:
        """
        Generate predictions with a specific model
        
        Args:
            input_data: Preprocessed input data
            model_name: Name of the model to use
            steps: Number of prediction steps
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        predictions = []
        confidence_intervals = []
        
        current_input = input_data.copy()
        
        for step in range(steps):
            # Make prediction
            pred = self.model_manager.predict(model_name, current_input)[0]
            predictions.append(float(pred))
            
            # Calculate confidence interval (simplified approach)
            # In practice, you might use prediction intervals from the model
            std_dev = np.std(predictions) if len(predictions) > 1 else abs(pred * 0.1)
            conf_int = (pred - 1.96 * std_dev, pred + 1.96 * std_dev)
            confidence_intervals.append(conf_int)
            
            # Update input for next prediction (sliding window)
            # This is a simplified approach; you might want to include
            # actual feature engineering for multi-step forecasting
            if steps > 1:
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred  # Update last timestep with prediction
        
        return predictions, confidence_intervals
    
    def _create_ensemble_prediction(self, predictions: Dict[str, List[float]]) -> List[float]:
        """
        Create ensemble prediction from multiple models
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Ensemble predictions
        """
        if self.config['ensemble_method'] == 'simple_average':
            # Simple average
            pred_arrays = [np.array(pred) for pred in predictions.values()]
            ensemble = np.mean(pred_arrays, axis=0).tolist()
            
        elif self.config['ensemble_method'] == 'weighted_average':
            # Weighted average based on model performance
            weights = self._get_model_weights(list(predictions.keys()))
            weighted_preds = []
            
            for i in range(len(next(iter(predictions.values())))):
                weighted_sum = sum(weights[model] * predictions[model][i] 
                                 for model in predictions.keys())
                weighted_preds.append(weighted_sum)
            
            ensemble = weighted_preds
        
        else:
            # Default to simple average
            pred_arrays = [np.array(pred) for pred in predictions.values()]
            ensemble = np.mean(pred_arrays, axis=0).tolist()
        
        return ensemble
    
    def _get_model_weights(self, model_names: List[str]) -> Dict[str, float]:
        """
        Calculate weights for ensemble based on model performance
        
        Args:
            model_names: List of model names
            
        Returns:
            Dictionary of model weights
        """
        weights = {}
        total_weight = 0
        
        for model in model_names:
            if model in self.model_manager.metrics:
                # Use inverse of RMSE as weight (lower RMSE = higher weight)
                rmse = self.model_manager.metrics[model].get('rmse', 1.0)
                weight = 1.0 / (rmse + 1e-8)  # Add small value to avoid division by zero
            else:
                weight = 1.0  # Default weight
            
            weights[model] = weight
            total_weight += weight
        
        # Normalize weights to sum to 1
        for model in weights:
            weights[model] /= total_weight
        
        return weights
    
    def evaluate_forecast_accuracy(self, actual_data: pd.DataFrame, 
                                  forecast_result: Dict) -> Dict[str, Any]:
        """
        Evaluate forecast accuracy against actual data
        
        Args:
            actual_data: DataFrame with actual values
            forecast_result: Result from generate_forecast
            
        Returns:
            Dictionary with accuracy metrics
        """
        logger.info("📊 Evaluating forecast accuracy...")
        
        evaluation_results = {}
        target_col = self.config['target_column']
        
        if target_col not in actual_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in actual data")
        
        # Align timestamps
        forecast_timestamps = forecast_result['timestamps']
        actual_values = []
        
        for timestamp in forecast_timestamps:
            if timestamp in actual_data.index:
                actual_values.append(actual_data.loc[timestamp, target_col])
            else:
                actual_values.append(np.nan)
        
        # Remove NaN values
        valid_indices = ~np.isnan(actual_values)
        actual_values = np.array(actual_values)[valid_indices]
        
        # Evaluate each model
        for model_name, predictions in forecast_result['predictions'].items():
            model_predictions = np.array(predictions)[valid_indices]
            
            if len(actual_values) > 0:
                metrics = self.model_manager.calculate_metrics(actual_values, model_predictions)
                evaluation_results[model_name] = metrics
        
        logger.info("✅ Forecast accuracy evaluation completed")
        return evaluation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and information
        
        Returns:
            Dictionary with system status
        """
        return {
            'is_trained': self.is_trained,
            'available_models': list(self.model_manager.models.keys()),
            'forecast_horizon': self.forecast_horizon,
            'cached_predictions': len(self.prediction_cache),
            'data_processor_info': self.data_processor.get_feature_info(),
            'model_summary': self.model_manager.get_model_summary(),
            'config': self.config
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize forecasting engine
    engine = ForecastingEngine()
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    sample_data = pd.DataFrame({
        'timestamp': dates,
        'load_demand': np.random.normal(1000, 200, 1000) + 100 * np.sin(np.arange(1000) * 2 * np.pi / 24),
        'temperature': np.random.normal(25, 5, 1000),
        'humidity': np.random.normal(60, 10, 1000)
    })
    sample_data.set_index('timestamp', inplace=True)
    
    # Save sample data
    sample_data.to_csv('sample_training_data.csv')
    
    print("🧪 Testing Forecasting Engine...")
    
    # Train models
    training_results = engine.train_forecasting_models('sample_training_data.csv')
    print(f"✅ Training completed. Best model: {training_results['best_model']}")
    
    # Generate forecast
    recent_data = sample_data.tail(50)  # Use last 50 hours for forecasting
    forecast = engine.generate_forecast(recent_data, forecast_steps=12)
    print(f"✅ Forecast generated for {len(forecast['predictions']['ensemble'])} steps")
    
    # Get system status
    status = engine.get_system_status()
    print(f"📊 System Status: {status['available_models']} models available")
    
    print("🏆 Forecasting Engine test completed successfully!")
