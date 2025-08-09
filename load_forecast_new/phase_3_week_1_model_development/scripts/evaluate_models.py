"""
Model Evaluation Script for Delhi Load Forecasting
Phase 3 Week 1: Baseline Model Development
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import sys
import numpy as np
import pandas as pd
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Import our custom modules
from data_loader import DelhiDataLoader
from model_builder import DelhiModelBuilder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DelhiModelEvaluator:
    """
    Comprehensive model evaluator for Delhi Load Forecasting.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """
        Initialize the model evaluator.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_loader = DelhiDataLoader(config_path)
        self.evaluation_results = {}
        
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
        """Create necessary directories for evaluation."""
        directories = ['evaluation/figures/forecasts', 
                      'evaluation/results', 
                      'evaluation/reports',
                      'evaluation/comparison']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_path: str, model_type: str = None) -> tf.keras.Model:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            model_type: Optional model type identifier
            
        Returns:
            Loaded Keras model
        """
        try:
            logger.info(f"Loading model from {model_path}")
            model = tf.keras.models.load_model(model_path)
            
            # Print model summary
            model.summary()
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_test_data(self, data_path: str = None) -> Tuple:
        """
        Prepare data for evaluation.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Test data tuple (X_test, y_test)
        """
        logger.info("Preparing test data for evaluation...")
        
        # Use data loader to prepare all data
        data_splits = self.data_loader.prepare_data(data_path)
        
        # We only need the test data for evaluation
        X_train, X_val, X_test, y_train, y_val, y_test = data_splits
        
        logger.info(f"Test data prepared - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        return X_test, y_test
    
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
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Overall metrics
        mse = mean_squared_error(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        
        # MAPE calculation (avoid division by zero)
        mask = y_true_flat != 0
        mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / y_true_flat[mask])) * 100
        
        # Normalized RMSE
        range_y = np.max(y_true_flat) - np.min(y_true_flat)
        nrmse = rmse / range_y if range_y > 0 else 0
        
        # Calculate metrics
        metrics = {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2),
            'MAPE': float(mape),
            'NRMSE': float(nrmse),
        }
        
        # Per-target metrics (for each utility)
        target_names = self.config['data']['target_variables']
        for i, target in enumerate(target_names):
            if y_true.shape[-1] > i:
                # Extract values for this target
                if len(y_true.shape) == 3:  # If predictions are 3D (samples, timesteps, features)
                    target_true = y_true[:, :, i].reshape(-1)
                    target_pred = y_pred[:, :, i].reshape(-1)
                else:  # If predictions are 2D (samples, features)
                    target_true = y_true[:, i].reshape(-1)
                    target_pred = y_pred[:, i].reshape(-1)
                
                # Calculate metrics
                target_mse = mean_squared_error(target_true, target_pred)
                target_rmse = np.sqrt(target_mse)
                target_mae = mean_absolute_error(target_true, target_pred)
                target_r2 = r2_score(target_true, target_pred)
                
                # MAPE with zero handling
                mask = target_true != 0
                target_mape = np.mean(np.abs((target_true[mask] - target_pred[mask]) / target_true[mask])) * 100
                
                # Store metrics
                metrics[f'{target}_MSE'] = float(target_mse)
                metrics[f'{target}_RMSE'] = float(target_rmse)
                metrics[f'{target}_MAE'] = float(target_mae)
                metrics[f'{target}_R2'] = float(target_r2)
                metrics[f'{target}_MAPE'] = float(target_mape)
        
        return metrics
    
    def evaluate_model(self, model: tf.keras.Model, X_test: np.ndarray, 
                      y_test: np.ndarray, model_type: str) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_type: Type of model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_type} model...")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)
        
        # Add model type to metrics
        metrics['model_type'] = model_type
        
        # Log metrics
        logger.info(f"Evaluation results for {model_type} model:")
        logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  MAE: {metrics['MAE']:.4f}")
        logger.info(f"  MAPE: {metrics['MAPE']:.4f}%")
        logger.info(f"  R²: {metrics['R2']:.4f}")
        
        # Save results
        self._save_evaluation_results(metrics, model_type)
        
        # Create plots
        self._plot_predictions(y_test, y_pred, model_type)
        self._plot_error_distribution(y_test, y_pred, model_type)
        
        return metrics, y_pred
    
    def _save_evaluation_results(self, metrics: Dict, model_type: str):
        """
        Save evaluation results to file.
        
        Args:
            metrics: Dictionary of metrics
            model_type: Type of model
        """
        # Create results directory if it doesn't exist
        Path('evaluation/results').mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = f"evaluation/results/{model_type}_evaluation_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Evaluation results saved to {results_path}")
    
    def _plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         model_type: str, n_samples: int = 200):
        """
        Plot and save prediction results.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Type of model
            n_samples: Number of samples to plot
        """
        target_names = self.config['data']['target_variables']
        n_targets = min(len(target_names), y_true.shape[-1])
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate grid dimensions
        n_rows = (n_targets // 2) + (n_targets % 2)
        n_cols = min(2, n_targets)
        
        # Plot each target
        for i in range(n_targets):
            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, i+1)
            
            # Extract values for this target
            if len(y_true.shape) == 3:
                target_true = y_true[:n_samples, 0, i]  # Use first timestep for each sample
                target_pred = y_pred[:n_samples, 0, i]
            else:
                target_true = y_true[:n_samples, i]
                target_pred = y_pred[:n_samples, i]
            
            # Plot actual vs predicted
            ax.plot(target_true, 'b-', label='Actual', alpha=0.7)
            ax.plot(target_pred, 'r-', label='Predicted', alpha=0.7)
            
            # Calculate metrics for this subset
            mse = mean_squared_error(target_true, target_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((target_true - target_pred) / np.maximum(np.abs(target_true), 1e-8))) * 100
            
            # Add title and labels
            ax.set_title(f'{target_names[i]} - RMSE: {rmse:.2f}, MAPE: {mape:.2f}%')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Load (MW)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle(f'{model_type.upper()} Model Predictions', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save figure
        plot_path = f"evaluation/figures/forecasts/{model_type}_forecasts.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forecast plot saved to {plot_path}")
        
        # Create scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 8))
        
        # Flatten arrays for scatter plot
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Create scatter plot
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.3, s=10)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
        max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Add labels and title
        plt.xlabel('Actual Load (MW)')
        plt.ylabel('Predicted Load (MW)')
        plt.title(f'{model_type.upper()} - Actual vs Predicted Load')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        scatter_path = f"evaluation/figures/forecasts/{model_type}_scatter.png"
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scatter plot saved to {scatter_path}")
    
    def _plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, model_type: str):
        """
        Plot and save error distribution.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_type: Type of model
        """
        # Flatten arrays
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)
        
        # Calculate errors
        errors = y_pred_flat - y_true_flat
        percentage_errors = 100 * errors / np.maximum(np.abs(y_true_flat), 1e-8)
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Error histogram
        sns.histplot(errors, kde=True, ax=axes[0, 0], bins=50)
        axes[0, 0].set_title(f'{model_type.upper()} - Error Distribution')
        axes[0, 0].set_xlabel('Prediction Error (MW)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Percentage error histogram
        sns.histplot(percentage_errors, kde=True, ax=axes[0, 1], bins=50)
        axes[0, 1].set_title(f'{model_type.upper()} - Percentage Error Distribution')
        axes[0, 1].set_xlabel('Percentage Error (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error vs actual value
        axes[1, 0].scatter(y_true_flat, errors, alpha=0.3, s=10)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_title(f'{model_type.upper()} - Error vs Actual Value')
        axes[1, 0].set_xlabel('Actual Load (MW)')
        axes[1, 0].set_ylabel('Prediction Error (MW)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        from scipy import stats
        stats.probplot(errors, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'{model_type.upper()} - Q-Q Plot of Errors')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = f"evaluation/figures/forecasts/{model_type}_error_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error analysis plots saved to {plot_path}")
    
    def evaluate_all_models(self, model_dir: str = "models/trained_models", data_path: str = None) -> Dict:
        """
        Evaluate all available models.
        
        Args:
            model_dir: Directory containing trained models
            data_path: Optional path to data file
            
        Returns:
            Dictionary of evaluation results for all models
        """
        logger.info(f"Evaluating all models in {model_dir}...")
        
        # Prepare test data
        X_test, y_test = self.prepare_test_data(data_path)
        
        # Find all model files
        model_files = list(Path(model_dir).glob("*_model.h5"))
        
        if not model_files:
            logger.warning(f"No model files found in {model_dir}")
            return {}
        
        results = {}
        
        for model_path in model_files:
            model_type = model_path.stem.split('_')[0]  # Extract model type from filename
            
            try:
                # Load model
                model = self.load_model(str(model_path), model_type)
                
                # Evaluate model
                metrics, y_pred = self.evaluate_model(model, X_test, y_test, model_type)
                
                # Store results
                results[model_type] = metrics
                
            except Exception as e:
                logger.error(f"Error evaluating {model_type} model: {e}")
                continue
        
        # Create comparison plots
        if results:
            self._plot_model_comparison(results)
        
        return results
    
    def _plot_model_comparison(self, results: Dict):
        """
        Create comparison plots for all evaluated models.
        
        Args:
            results: Dictionary of evaluation results
        """
        # Extract model types and metrics
        models = list(results.keys())
        metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
        
        if not models:
            return
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            # Create bar chart
            bars = axes[i].bar(models, values, color='skyblue')
            
            # Add labels
            axes[i].set_title(f'Model Comparison - {metric}')
            axes[i].set_xlabel('Model')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                axes[i].annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = "evaluation/comparison/model_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate Delhi-specific comparison
        target = 'DELHI'
        metrics = ['DELHI_RMSE', 'DELHI_MAE', 'DELHI_MAPE', 'DELHI_R2']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if all(metric in results[model] for model in models):
                values = [results[model][metric] for model in models]
                
                # Create bar chart
                bars = axes[i].bar(models, values, color='lightgreen')
                
                # Add labels
                axes[i].set_title(f'Model Comparison - {metric.replace("DELHI_", "")} for Delhi')
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel(metric.replace('DELHI_', ''))
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    axes[i].annotate(f'{height:.2f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save figure
        plot_path = "evaluation/comparison/delhi_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plots saved to evaluation/comparison/")
        
        # Save comparison results
        results_path = f"evaluation/comparison/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Comparison results saved to {results_path}")

def main():
    """Main evaluation function."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Delhi Load Forecasting models')
    parser.add_argument('--model_dir', type=str, default='models/trained_models',
                       help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to data file (optional)')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model type to evaluate (optional)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = DelhiModelEvaluator()
    
    # Evaluate models
    if args.model:
        # Evaluate specific model
        logger.info(f"Evaluating specific model: {args.model}")
        
        # Prepare test data
        X_test, y_test = evaluator.prepare_test_data(args.data_path)
        
        # Model path
        model_path = f"{args.model_dir}/{args.model}_model.h5"
        
        # Check if model file exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found: {model_path}")
            return
        
        # Load and evaluate model
        model = evaluator.load_model(model_path, args.model)
        metrics, _ = evaluator.evaluate_model(model, X_test, y_test, args.model)
        
        # Print results
        print("\n" + "="*60)
        print(f"EVALUATION RESULTS: {args.model.upper()}")
        print("="*60)
        
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
        
    else:
        # Evaluate all models
        results = evaluator.evaluate_all_models(args.model_dir, args.data_path)
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if results:
            # Sort models by RMSE
            sorted_results = sorted(results.items(), key=lambda x: x[1]['RMSE'])
            
            print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R²':<10}")
            print("-" * 65)
            
            for model, metrics in sorted_results:
                print(f"{model:<15} "
                      f"{metrics['RMSE']:<10.4f} "
                      f"{metrics['MAE']:<10.4f} "
                      f"{metrics['MAPE']:<10.2f}% "
                      f"{metrics['R2']:<10.4f}")
            
            print(f"\nBest model: {sorted_results[0][0].upper()} (RMSE: {sorted_results[0][1]['RMSE']:.4f})")
            
        else:
            print("No models were evaluated.")
    
    print("\nEvaluation completed! Check evaluation/figures/ for visualizations.")

if __name__ == "__main__":
    main()
