"""
Model Training Script for Delhi Load Forecasting
Phase 3 Week 1: Baseline Model Development
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

# Set matplotlib to use non-interactive backend before any imports
import matplotlib
matplotlib.use('Agg')

import sys
import time
import numpy as np
import pandas as pd
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our custom modules
from data_loader import DelhiDataLoader
from model_builder import DelhiModelBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DelhiModelTrainer:
    """
    Comprehensive model trainer for Delhi Load Forecasting.
    """
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader = DelhiDataLoader(config_path)
        self.model_builder = DelhiModelBuilder(config_path)
        self.training_results = {}
        self.trained_models = {}
        
        # Create directories
        self._create_directories()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _create_directories(self):
        """Create necessary directories for training."""
        directories = [
            'models/trained_models', 'models/checkpoints', 'models/scalers',
            'evaluation/figures', 'evaluation/results', 'logs'
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(self, data_path: str = None) -> Tuple:
        """
        Prepare data for training.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Tuple of prepared data splits
        """
        logger.info("Preparing training data...")
        return self.data_loader.prepare_data(data_path)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Flatten arrays for overall metrics
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Overall metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # MAPE calculation (avoid division by zero)
        mape = np.mean(np.abs((y_true_flat - y_pred_flat) / np.maximum(np.abs(y_true_flat), 1e-8))) * 100
        
        # Normalized RMSE
        nrmse = rmse / (np.max(y_true_flat) - np.min(y_true_flat)) * 100
        
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'NRMSE': float(nrmse)
        }
        
        # Per-target metrics
        target_names = self.config['data']['target_variables']
        for i, target in enumerate(target_names):
            if y_true.shape[-1] > i:
                target_true = y_true[:, :, i].flatten() if len(y_true.shape) == 3 else y_true[:, i].flatten()
                target_pred = y_pred[:, :, i].flatten() if len(y_pred.shape) == 3 else y_pred[:, i].flatten()
                
                target_mse = mean_squared_error(target_true, target_pred)
                target_rmse = np.sqrt(target_mse)
                target_mae = mean_absolute_error(target_true, target_pred)
                target_r2 = r2_score(target_true, target_pred)
                target_mape = np.mean(np.abs((target_true - target_pred) / np.maximum(np.abs(target_true), 1e-8))) * 100
                
                metrics[f'{target}_MSE'] = float(target_mse)
                metrics[f'{target}_RMSE'] = float(target_rmse)
                metrics[f'{target}_MAE'] = float(target_mae)
                metrics[f'{target}_R2'] = float(target_r2)
                metrics[f'{target}_MAPE'] = float(target_mape)
        
        return metrics
    
    def train_model(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Tuple[tf.keras.Model, Dict]:
        """
        Train a single model.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Tuple of (trained_model, training_history)
        """
        logger.info(f"Training {model_type.upper()} model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.model_builder.build_model(model_type, input_shape)
        
        # Create callbacks
        callbacks = self.model_builder.create_callbacks(model_type)
        
        # Training configuration
        training_config = self.config['training']
        
        # Train model
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Save trained model
        model_save_path = f"models/trained_models/{model_type}_model.h5"
        model.save(model_save_path)
        
        # Save model weights separately
        weights_save_path = f"models/trained_models/{model_type}_weights.h5"
        model.save_weights(weights_save_path)
        
        logger.info(f"{model_type.upper()} model training completed in {training_time:.2f} seconds")
        logger.info(f"Model saved to {model_save_path}")
        
        return model, history
    
    def evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, 
                      y_test: np.ndarray, model_type: str) -> Dict:
        """
        Evaluate trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_type: Type of model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_type.upper()} model...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Add model info
        model_info = self.model_builder.get_model_info(model)
        metrics.update({
            'model_type': model_type,
            'total_params': model_info['total_params'],
            'training_samples': len(X_test)
        })
        
        logger.info(f"{model_type.upper()} evaluation completed:")
        logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  MAE: {metrics['MAE']:.4f}")
        logger.info(f"  MAPE: {metrics['MAPE']:.4f}%")
        logger.info(f"  R²: {metrics['R2']:.4f}")
        
        return metrics, y_pred
    
    def plot_training_history(self, history: tf.keras.callbacks.History, 
                            model_type: str):
        """
        Plot and save training history.
        
        Args:
            history: Training history from model.fit()
            model_type: Type of model
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title(f'{model_type.upper()} Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE plot
        if 'mae' in history.history:
            axes[0, 1].plot(history.history['mae'], label='Training MAE')
            axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
            axes[0, 1].set_title(f'{model_type.upper()} Model MAE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MAE')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Learning rate plot
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'])
            axes[1, 0].set_title(f'{model_type.upper()} Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No LR data available', 
                          ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Convergence plot
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        
        axes[1, 1].plot(epochs, loss, 'b-', label='Training')
        axes[1, 1].plot(epochs, val_loss, 'r-', label='Validation')
        axes[1, 1].set_title(f'{model_type.upper()} Convergence')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"evaluation/figures/{model_type}_training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training history plot saved: {plot_path}")
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        model_type: str, n_samples: int = 100):
        """
        Plot prediction results.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Type of model
            n_samples: Number of samples to plot
        """
        target_names = self.config['data']['target_variables']
        n_targets = min(len(target_names), y_true.shape[-1])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(min(n_targets, 5)):  # Plot up to 5 targets
            target_name = target_names[i]
            
            # Get data for this target
            true_vals = y_true[:n_samples, 0, i] if len(y_true.shape) == 3 else y_true[:n_samples, i]
            pred_vals = y_pred[:n_samples, 0, i] if len(y_pred.shape) == 3 else y_pred[:n_samples, i]
            
            # Time series plot
            axes[i].plot(true_vals, label='Actual', alpha=0.7)
            axes[i].plot(pred_vals, label='Predicted', alpha=0.7)
            axes[i].set_title(f'{model_type.upper()}: {target_name} Predictions')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Load (MW)')
            axes[i].legend()
            axes[i].grid(True)
        
        # Overall scatter plot
        if len(axes) > 5:
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
            
            axes[5].scatter(y_true_flat, y_pred_flat, alpha=0.3, s=1)
            axes[5].plot([y_true_flat.min(), y_true_flat.max()], 
                        [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
            axes[5].set_title(f'{model_type.upper()}: Actual vs Predicted')
            axes[5].set_xlabel('Actual Load (MW)')
            axes[5].set_ylabel('Predicted Load (MW)')
            axes[5].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"evaluation/figures/{model_type}_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Predictions plot saved: {plot_path}")
    
    def train_all_models(self, data_path: str = None) -> Dict:
        """
        Train all configured models.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Dictionary of training results
        """
        logger.info("Starting comprehensive model training...")
        
        # Prepare data
        data_splits = self.prepare_training_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        
        logger.info(f"Data prepared - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Get model types from config
        model_types = list(self.config['models'].keys())
        if 'gru_baseline' in model_types:
            model_types[model_types.index('gru_baseline')] = 'gru'
        if 'lstm_baseline' in model_types:
            model_types[model_types.index('lstm_baseline')] = 'lstm'
        if 'cnn_gru_hybrid' in model_types:
            model_types[model_types.index('cnn_gru_hybrid')] = 'cnn_gru'
        
        # Add additional model types
        additional_models = ['attention', 'ensemble']
        for model in additional_models:
            if model not in model_types:
                model_types.append(model)
        
        results_summary = {}
        
        # Train each model
        for model_type in model_types:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Training {model_type.upper()} Model")
                logger.info(f"{'='*60}")
                
                # Train model
                model, history = self.train_model(model_type, X_train, y_train, X_val, y_val)
                
                # Evaluate model
                metrics, y_pred = self.evaluate_model(model, X_test, y_test, model_type)
                
                # Plot results
                self.plot_training_history(history, model_type)
                self.plot_predictions(y_test, y_pred, model_type)
                
                # Store results
                self.trained_models[model_type] = model
                self.training_results[model_type] = {
                    'metrics': metrics,
                    'history': {key: [float(val) for val in values] 
                               for key, values in history.history.items()}
                }
                
                # Add to summary
                results_summary[model_type] = {
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2'],
                    'params': metrics['total_params']
                }
                
                logger.info(f"{model_type.upper()} training completed successfully!")
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                continue
        
        # Save results
        self._save_results(results_summary)
        
        # Create comparison plots
        self._plot_model_comparison(results_summary)
        
        logger.info(f"\nTraining completed! Successfully trained {len(results_summary)} models.")
        return results_summary
    
    def _save_results(self, results_summary: Dict):
        """Save training results to files."""
        
        # Save detailed results
        results_path = f"evaluation/results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(self.training_results, f, indent=2)
        
        # Save summary
        summary_path = f"evaluation/results/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Results saved: {results_path}")
        logger.info(f"Summary saved: {summary_path}")
    
    def _plot_model_comparison(self, results_summary: Dict):
        """Create model comparison plots."""
        
        if not results_summary:
            return
        
        models = list(results_summary.keys())
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results_summary[model][metric] for model in models]
            
            bars = axes[i].bar(models, values)
            axes[i].set_title(f'Model Comparison: {metric}')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('evaluation/figures/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create parameter count comparison
        plt.figure(figsize=(12, 6))
        params = [results_summary[model]['params'] for model in models]
        bars = plt.bar(models, params)
        plt.title('Model Parameter Count Comparison')
        plt.ylabel('Number of Parameters')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, params):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:,}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('evaluation/figures/parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plots saved")

def main():
    """Main training function."""
    trainer = DelhiModelTrainer()
    
    # Train all models
    results = trainer.train_all_models()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if results:
        # Sort by RMSE
        sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])
        
        print(f"{'Model':<15} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'R²':<8} {'Params':<10}")
        print("-" * 65)
        
        for model, metrics in sorted_results:
            print(f"{model:<15} "
                  f"{metrics['RMSE']:<8.3f} "
                  f"{metrics['MAE']:<8.3f} "
                  f"{metrics['MAPE']:<8.1f} "
                  f"{metrics['R2']:<8.3f} "
                  f"{metrics['params']:<10,}")
        
        print(f"\nBest model: {sorted_results[0][0].upper()} (RMSE: {sorted_results[0][1]['RMSE']:.3f})")
        
    else:
        print("No models were successfully trained.")
    
    print("\nTraining completed! Check evaluation/figures/ for visualizations.")

if __name__ == "__main__":
    main()
