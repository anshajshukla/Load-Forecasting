"""
Model Manager Module
Handles model training, validation, and management for the load forecasting system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Advanced model manager for training, validating, and managing ML models
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ModelManager
        
        Args:
            config: Configuration dictionary for model parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.model_history = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for model management"""
        return {
            'models_dir': 'models/saved_models',
            'validation_split': 0.2,
            'test_split': 0.1,
            'sequence_length': 24,  # hours
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            'metrics': ['mae', 'mse', 'rmse', 'mape', 'r2'],
            'model_types': ['gru', 'lstm', 'ensemble'],
            'save_best_only': True
        }
    
    def prepare_sequences(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series modeling
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X, y) sequences
        """
        logger.info(f"📝 Preparing sequences with length {self.config['sequence_length']}...")
        
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Get feature columns (exclude target)
        feature_columns = [col for col in data.columns if col != target_column]
        
        X, y = [], []
        seq_len = self.config['sequence_length']
        
        for i in range(seq_len, len(data)):
            # Features sequence
            X.append(data[feature_columns].iloc[i-seq_len:i].values)
            # Target value
            y.append(data[target_column].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"✅ Sequences prepared. X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature sequences
            y: Target values
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("🔄 Splitting data into train/val/test sets...")
        
        n_samples = len(X)
        test_size = int(n_samples * self.config['test_split'])
        val_size = int(n_samples * self.config['validation_split'])
        train_size = n_samples - test_size - val_size
        
        # Time series split (maintaining chronological order)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"✅ Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_gru_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build GRU model for time series forecasting
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            
        Returns:
            Compiled GRU model
        """
        logger.info("🏗️ Building GRU model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.GRU(128, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"✅ GRU model built with {model.count_params():,} parameters")
        return model
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build LSTM model for time series forecasting
        
        Args:
            input_shape: Shape of input sequences (timesteps, features)
            
        Returns:
            Compiled LSTM model
        """
        logger.info("🏗️ Building LSTM model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"✅ LSTM model built with {model.count_params():,} parameters")
        return model
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """
        Train a deep learning model
        
        Args:
            model_name: Name of the model ('gru' or 'lstm')
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Trained model
        """
        logger.info(f"🚀 Training {model_name.upper()} model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        if model_name.lower() == 'gru':
            model = self.build_gru_model(input_shape)
        elif model_name.lower() == 'lstm':
            model = self.build_lstm_model(input_shape)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Store model and history
        self.models[model_name] = model
        self.model_history[model_name] = history.history
        
        logger.info(f"✅ {model_name.upper()} model training completed")
        return model
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Mean Absolute Percentage Error
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # R-squared
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Custom metrics for load forecasting
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        metrics['mean_error'] = np.mean(y_true - y_pred)
        
        return metrics
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test set
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"📊 Evaluating {model_name} model...")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        y_pred = model.predict(X_test).flatten()
        
        metrics = self.calculate_metrics(y_test, y_pred)
        self.metrics[model_name] = metrics
        
        logger.info(f"✅ {model_name} evaluation completed:")
        for metric_name, value in metrics.items():
            logger.info(f"   {metric_name.upper()}: {value:.4f}")
        
        return metrics
    
    def save_model(self, model_name: str, save_path: Optional[str] = None) -> str:
        """
        Save trained model to disk
        
        Args:
            model_name: Name of the model to save
            save_path: Custom save path (optional)
            
        Returns:
            Path where model was saved
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Create save directory
        models_dir = Path(self.config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        if save_path is None:
            save_path = models_dir / f"{model_name}_forecast_model.h5"
        
        # Save model
        self.models[model_name].save(save_path)
        
        # Save metrics
        metrics_path = save_path.with_suffix('.json')
        if model_name in self.metrics:
            import json
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics[model_name], f, indent=2)
        
        logger.info(f"✅ Model {model_name} saved to {save_path}")
        return str(save_path)
    
    def load_model(self, model_name: str, model_path: str) -> tf.keras.Model:
        """
        Load trained model from disk
        
        Args:
            model_name: Name to assign to the loaded model
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            model = tf.keras.models.load_model(model_path)
            self.models[model_name] = model
            
            # Load metrics if available
            metrics_path = Path(model_path).with_suffix('.json')
            if metrics_path.exists():
                import json
                with open(metrics_path, 'r') as f:
                    self.metrics[model_name] = json.load(f)
            
            logger.info(f"✅ Model {model_name} loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model
        
        Args:
            model_name: Name of the model to use
            X: Input features
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        return predictions.flatten()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of all trained models
        
        Returns:
            Dictionary with model information
        """
        summary = {
            'total_models': len(self.models),
            'model_names': list(self.models.keys()),
            'best_model': None,
            'metrics_comparison': {}
        }
        
        # Find best model based on validation RMSE
        if self.metrics:
            best_rmse = float('inf')
            for model_name, metrics in self.metrics.items():
                summary['metrics_comparison'][model_name] = metrics
                if 'rmse' in metrics and metrics['rmse'] < best_rmse:
                    best_rmse = metrics['rmse']
                    summary['best_model'] = model_name
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    # Initialize model manager
    manager = ModelManager()
    
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    sequence_length = 24
    
    # Generate sample time series data
    X = np.random.randn(n_samples, sequence_length, n_features)
    y = np.random.randn(n_samples)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = manager.split_data(X, y)
    
    # Train GRU model
    gru_model = manager.train_model('gru', X_train, y_train, X_val, y_val)
    
    # Evaluate model
    gru_metrics = manager.evaluate_model('gru', X_test, y_test)
    
    # Get model summary
    summary = manager.get_model_summary()
    print("📊 Model Summary:")
    print(f"   Total models: {summary['total_models']}")
    print(f"   Best model: {summary['best_model']}")
    
    print("\n🏆 Model completed successfully!")
