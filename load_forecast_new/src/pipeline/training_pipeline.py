"""
Model Training Pipeline for Delhi SLDC Load Forecasting.
Handles data preprocessing, model training, validation, and performance tracking.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import joblib

# Import our database manager
from database.db_manager import DatabaseManager, create_database_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration class for model training."""
    sequence_length: int = 24  # Hours to look back
    prediction_horizon: int = 24  # Hours to predict ahead
    validation_split: float = 0.2
    test_split: float = 0.1
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    gru_units: List[int] = None
    dropout_rate: float = 0.2
    patience: int = 15
    
    def __post_init__(self):
        if self.gru_units is None:
            self.gru_units = [128, 64, 32]

class ModelTrainingPipeline:
    """Comprehensive model training pipeline with validation and monitoring."""
    
    def __init__(self, db_manager: DatabaseManager = None, config: TrainingConfig = None):
        """
        Initialize the model training pipeline.
        
        Args:
            db_manager: Database manager instance
            config: Training configuration
        """
        self.db_manager = db_manager or create_database_manager()
        self.config = config or TrainingConfig()
        self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        
        # Create output directories
        self.models_dir = Path('data_pipeline/models/trained_models')
        self.scalers_dir = Path('data_pipeline/models/scalers')
        self.metrics_dir = Path('data_pipeline/models/metrics')
        self.plots_dir = Path('data_pipeline/models/plots')
        
        for dir_path in [self.models_dir, self.scalers_dir, self.metrics_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize scalers for each target
        self.target_scalers = {}
        self.feature_scaler = None
        
        # Training session tracking
        self.current_session_id = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical data from database."""
        try:
            query = """
            SELECT datetime, DELHI, BRPL, BYPL, NDMC, MES, 
                   temperature, humidity, wind_speed, precipitation
            FROM historical_load_data
            WHERE datetime IS NOT NULL
            ORDER BY datetime
            """
            
            df = self.db_manager.execute_query(query)
            
            if df.empty:
                raise ValueError("No historical data found in database")
            
            # Convert datetime column
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Loaded {len(df)} records from database")
            logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Preprocess data for training.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Tuple of (features, targets, preprocessing_info)
        """
        logger.info("Starting data preprocessing...")
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create time-based features
        df = self._create_time_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Prepare feature columns
        feature_columns = [
            # Load data
            'DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES',
            # Weather data
            'temperature', 'humidity', 'wind_speed', 'precipitation',
            # Time features
            'hour', 'day_of_week', 'month', 'quarter',
            'is_weekend', 'is_peak_hour', 'season_sin', 'season_cos',
            # Lag features
            'DELHI_lag_1', 'DELHI_lag_24', 'DELHI_lag_168',
            'load_sum', 'load_mean', 'load_std'
        ]
        
        # Ensure all feature columns exist
        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with default values
            for col in missing_features:
                df[col] = 0
        
        # Prepare data for scaling
        features_df = df[feature_columns].copy()
        targets_df = df[self.targets].copy()
        
        # Scale features
        self.feature_scaler = StandardScaler()
        scaled_features = self.feature_scaler.fit_transform(features_df)
        
        # Scale each target separately
        scaled_targets = {}
        for target in self.targets:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_targets[target] = scaler.fit_transform(targets_df[[target]])
            self.target_scalers[target] = scaler
        
        # Create sequences for time series prediction
        X, y = self._create_sequences(scaled_features, scaled_targets, df['datetime'])
        
        # Save preprocessing information
        preprocessing_info = {
            'feature_columns': feature_columns,
            'target_columns': self.targets,
            'sequence_length': self.config.sequence_length,
            'prediction_horizon': self.config.prediction_horizon,
            'data_shape': df.shape,
            'feature_shape': X.shape,
            'target_shape': {target: y[target].shape for target in self.targets}
        }
        
        logger.info(f"Preprocessing completed. Feature shape: {X.shape}")
        
        return X, y, preprocessing_info
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime."""
        df = df.copy()
        
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['quarter'] = df['datetime'].dt.quarter
        df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        
        # Peak hours (6-11 AM and 6-11 PM)
        df['is_peak_hour'] = ((df['hour'].between(6, 11)) | (df['hour'].between(18, 23))).astype(int)
        
        # Cyclical encoding for seasonal patterns
        df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df = df.copy()
        
        # Forward fill for load data (carry last observation forward)
        for target in self.targets:
            if target in df.columns:
                df[target] = df[target].fillna(method='ffill')
                df[target] = df[target].fillna(method='bfill')
        
        # Interpolate weather data
        weather_columns = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        for col in weather_columns:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear')
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features."""
        df = df.copy()
        
        # Lag features for main target (DELHI)
        if 'DELHI' in df.columns:
            df['DELHI_lag_1'] = df['DELHI'].shift(1)
            df['DELHI_lag_24'] = df['DELHI'].shift(24)  # Same hour previous day
            df['DELHI_lag_168'] = df['DELHI'].shift(168)  # Same hour previous week
        
        # Aggregate load features
        load_columns = [col for col in self.targets if col in df.columns]
        if load_columns:
            df['load_sum'] = df[load_columns].sum(axis=1)
            df['load_mean'] = df[load_columns].mean(axis=1)
            df['load_std'] = df[load_columns].std(axis=1)
        
        # Rolling statistics
        if 'DELHI' in df.columns:
            df['DELHI_rolling_mean_24'] = df['DELHI'].rolling(window=24, center=True).mean()
            df['DELHI_rolling_std_24'] = df['DELHI'].rolling(window=24, center=True).std()
        
        # Fill NaN values created by lag and rolling operations
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _create_sequences(self, features: np.ndarray, targets: Dict, datetimes: pd.Series) -> Tuple[np.ndarray, Dict]:
        """Create sequences for time series prediction."""
        X = []
        y = {target: [] for target in self.targets}
        
        seq_len = self.config.sequence_length
        pred_horizon = self.config.prediction_horizon
        
        for i in range(seq_len, len(features) - pred_horizon + 1):
            # Input sequence
            X.append(features[i-seq_len:i])
            
            # Target sequences (next pred_horizon hours)
            for target in self.targets:
                if target in targets:
                    y[target].append(targets[target][i:i+pred_horizon].flatten())
        
        X = np.array(X)
        for target in self.targets:
            if target in targets:
                y[target] = np.array(y[target])
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: Dict) -> Dict:
        """Split data into train, validation, and test sets."""
        n_samples = X.shape[0]
        
        # Calculate split indices
        test_size = int(n_samples * self.config.test_split)
        val_size = int(n_samples * self.config.validation_split)
        train_size = n_samples - test_size - val_size
        
        # Split data chronologically
        train_end = train_size
        val_end = train_size + val_size
        
        data_splits = {
            'X_train': X[:train_end],
            'X_val': X[train_end:val_end],
            'X_test': X[val_end:],
        }
        
        for target in self.targets:
            if target in y:
                data_splits[f'y_train_{target}'] = y[target][:train_end]
                data_splits[f'y_val_{target}'] = y[target][train_end:val_end]
                data_splits[f'y_test_{target}'] = y[target][val_end:]
        
        logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        return data_splits
    
    def build_model(self, input_shape: Tuple, output_shape: int) -> tf.keras.Model:
        """Build GRU model architecture."""
        model = Sequential([
            GRU(self.config.gru_units[0], return_sequences=True, input_shape=input_shape),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            GRU(self.config.gru_units[1], return_sequences=True),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            GRU(self.config.gru_units[2], return_sequences=False),
            Dropout(self.config.dropout_rate),
            
            Dense(64, activation='relu'),
            Dropout(self.config.dropout_rate),
            
            Dense(32, activation='relu'),
            Dense(output_shape, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_models(self, data_splits: Dict) -> Dict:
        """Train models for each target."""
        results = {}
        
        # Create training session
        session_info = {
            'start_time': datetime.datetime.now(),
            'config': self.config.__dict__,
            'data_info': {
                'train_samples': data_splits['X_train'].shape[0],
                'val_samples': data_splits['X_val'].shape[0],
                'test_samples': data_splits['X_test'].shape[0],
                'feature_dim': data_splits['X_train'].shape[-1]
            }
        }
        
        self.current_session_id = self.db_manager.create_training_session(
            model_type='GRU',
            targets=self.targets,
            config=session_info['config'],
            data_info=session_info['data_info']
        )
        
        for target in self.targets:
            if f'y_train_{target}' not in data_splits:
                logger.warning(f"No training data for target {target}")
                continue
                
            logger.info(f"Training model for {target}...")
            
            # Get target-specific data
            y_train = data_splits[f'y_train_{target}']
            y_val = data_splits[f'y_val_{target}']
            y_test = data_splits[f'y_test_{target}']
            
            # Build model
            input_shape = (data_splits['X_train'].shape[1], data_splits['X_train'].shape[2])
            output_shape = y_train.shape[1]
            
            model = self.build_model(input_shape, output_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    filepath=str(self.models_dir / f'best_{target}_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Train model
            history = model.fit(
                data_splits['X_train'], y_train,
                validation_data=(data_splits['X_val'], y_val),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_predictions = model.predict(data_splits['X_test'])
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, test_predictions, target)
            
            # Save model and results
            model_path = self.models_dir / f'{target}_forecast_model.h5'
            model.save(str(model_path))
            
            # Save scaler
            scaler_path = self.scalers_dir / f'{target}_scaler.pkl'
            joblib.dump(self.target_scalers[target], str(scaler_path))
            
            # Save training history
            history_path = self.metrics_dir / f'{target}_training_history.json'
            with open(history_path, 'w') as f:
                json.dump({
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'mae': history.history['mae'],
                    'val_mae': history.history['val_mae']
                }, f)
            
            # Create plots
            self._create_training_plots(history, target)
            self._create_prediction_plots(y_test, test_predictions, target)
            
            results[target] = {
                'model': model,
                'history': history,
                'metrics': metrics,
                'test_predictions': test_predictions,
                'test_actual': y_test
            }
            
            # Update training session with results
            self.db_manager.update_training_session(
                self.current_session_id,
                target,
                metrics
            )
            
            logger.info(f"Training completed for {target}. MSE: {metrics['mse']:.4f}, MAE: {metrics['mae']:.4f}")
        
        # Save feature scaler
        feature_scaler_path = self.scalers_dir / 'feature_scaler.pkl'
        joblib.dump(self.feature_scaler, str(feature_scaler_path))
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, target: str) -> Dict:
        """Calculate evaluation metrics."""
        # Flatten arrays for metric calculation
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Inverse transform to original scale
        y_true_orig = self.target_scalers[target].inverse_transform(y_true_flat.reshape(-1, 1)).flatten()
        y_pred_orig = self.target_scalers[target].inverse_transform(y_pred_flat.reshape(-1, 1)).flatten()
        
        metrics = {
            'mse': float(mean_squared_error(y_true_orig, y_pred_orig)),
            'mae': float(mean_absolute_error(y_true_orig, y_pred_orig)),
            'rmse': float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))),
            'r2': float(r2_score(y_true_orig, y_pred_orig)),
            'mape': float(np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1))) * 100),
            'accuracy': float(100 - np.mean(np.abs((y_true_orig - y_pred_orig) / np.maximum(y_true_orig, 1))) * 100)
        }
        
        return metrics
    
    def _create_training_plots(self, history, target: str):
        """Create training history plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {target}', fontsize=16)
        
        # Loss plot
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        
        # MAPE (if available)
        if 'mape' in history.history:
            axes[1, 1].plot(history.history['mape'], label='Training MAPE')
            axes[1, 1].plot(history.history['val_mape'], label='Validation MAPE')
            axes[1, 1].set_title('Model MAPE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MAPE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{target}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_prediction_plots(self, y_true: np.ndarray, y_pred: np.ndarray, target: str):
        """Create prediction comparison plots."""
        # Take first prediction sequence for visualization
        y_true_sample = y_true[0]
        y_pred_sample = y_pred[0]
        
        # Inverse transform to original scale
        y_true_orig = self.target_scalers[target].inverse_transform(y_true_sample.reshape(-1, 1)).flatten()
        y_pred_orig = self.target_scalers[target].inverse_transform(y_pred_sample.reshape(-1, 1)).flatten()
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'Prediction Results - {target}', fontsize=16)
        
        # Time series comparison
        hours = range(len(y_true_orig))
        axes[0].plot(hours, y_true_orig, label='Actual', marker='o', alpha=0.7)
        axes[0].plot(hours, y_pred_orig, label='Predicted', marker='s', alpha=0.7)
        axes[0].set_title('Actual vs Predicted Load')
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Load (MW)')
        axes[0].legend()
        axes[0].grid(True)
        
        # Scatter plot
        axes[1].scatter(y_true_orig, y_pred_orig, alpha=0.6)
        min_val = min(y_true_orig.min(), y_pred_orig.min())
        max_val = max(y_true_orig.max(), y_pred_orig.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1].set_xlabel('Actual Load (MW)')
        axes[1].set_ylabel('Predicted Load (MW)')
        axes[1].set_title('Actual vs Predicted Scatter Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'{target}_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_training_pipeline(self) -> Dict:
        """Run the complete training pipeline."""
        try:
            logger.info("Starting full training pipeline...")
            
            # Step 1: Load data
            df = self.load_historical_data()
            
            # Step 2: Preprocess data
            X, y, preprocessing_info = self.preprocess_data(df)
            
            # Step 3: Split data
            data_splits = self.split_data(X, y)
            
            # Step 4: Train models
            results = self.train_models(data_splits)
            
            # Step 5: Generate summary report
            summary = self._generate_training_summary(results, preprocessing_info)
            
            # Save summary
            summary_path = self.metrics_dir / 'training_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info("Training pipeline completed successfully!")
            
            return {
                'results': results,
                'summary': summary,
                'session_id': self.current_session_id,
                'preprocessing_info': preprocessing_info
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            if self.current_session_id:
                self.db_manager.mark_training_session_failed(self.current_session_id, str(e))
            raise
    
    def _generate_training_summary(self, results: Dict, preprocessing_info: Dict) -> Dict:
        """Generate training summary report."""
        summary = {
            'training_session_id': self.current_session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'config': self.config.__dict__,
            'preprocessing_info': preprocessing_info,
            'model_results': {}
        }
        
        for target, result in results.items():
            summary['model_results'][target] = {
                'metrics': result['metrics'],
                'best_epoch': int(np.argmin(result['history'].history['val_loss'])),
                'final_train_loss': float(result['history'].history['loss'][-1]),
                'final_val_loss': float(result['history'].history['val_loss'][-1])
            }
        
        # Overall performance
        avg_accuracy = np.mean([result['metrics']['accuracy'] for result in results.values()])
        summary['overall_performance'] = {
            'average_accuracy': float(avg_accuracy),
            'models_trained': len(results),
            'total_targets': len(self.targets)
        }
        
        return summary

if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        sequence_length=24,
        prediction_horizon=24,
        epochs=50,
        batch_size=32
    )
    
    pipeline = ModelTrainingPipeline(config=config)
    
    # Run training pipeline
    try:
        results = pipeline.run_full_training_pipeline()
        
        print("Training completed successfully!")
        print(f"Session ID: {results['session_id']}")
        print(f"Overall accuracy: {results['summary']['overall_performance']['average_accuracy']:.2f}%")
        
        for target, metrics in results['summary']['model_results'].items():
            print(f"{target}: Accuracy = {metrics['metrics']['accuracy']:.2f}%")
            
    except Exception as e:
        print(f"Training failed: {e}")
