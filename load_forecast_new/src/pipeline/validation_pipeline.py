"""
Validation Pipeline for Delhi SLDC Load Forecasting.
Handles real-time data validation, accuracy monitoring, and model performance tracking.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import joblib
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import warnings
import schedule
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import our components
from database.db_manager import DatabaseManager, create_database_manager
from enhanced_fetcher import EnhancedDataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for validation pipeline."""
    validation_window: int = 24  # Hours to validate against
    min_accuracy_threshold: float = 85.0  # Minimum acceptable accuracy %
    max_error_threshold: float = 500.0  # Maximum acceptable error in MW
    data_quality_checks: bool = True
    real_time_monitoring: bool = True
    alert_thresholds: Dict = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'low_accuracy': 80.0,
                'high_error': 1000.0,
                'data_missing': 0.1,  # 10% missing data threshold
                'model_drift': 0.15   # 15% performance degradation
            }

@dataclass
class ValidationResult:
    """Result from validation check."""
    timestamp: datetime.datetime
    target: str
    accuracy: float
    mae: float
    mse: float
    rmse: float
    r2: float
    mape: float
    data_quality_score: float
    alerts: List[str]
    is_valid: bool
    predictions: Optional[np.ndarray] = None
    actuals: Optional[np.ndarray] = None

class ValidationPipeline:
    """Comprehensive validation pipeline for model monitoring."""
    
    def __init__(self, db_manager: DatabaseManager = None, 
                 config: ValidationConfig = None):
        """
        Initialize validation pipeline.
        
        Args:
            db_manager: Database manager instance
            config: Validation configuration
        """
        self.db_manager = db_manager or create_database_manager()
        self.config = config or ValidationConfig()
        self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        
        # Initialize data fetcher
        self.data_fetcher = EnhancedDataFetcher(self.db_manager)
        
        # Model and scaler paths
        self.models_dir = Path('data_pipeline/models/trained_models')
        self.scalers_dir = Path('data_pipeline/models/scalers')
        self.validation_dir = Path('data_pipeline/validation')
        
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models and scalers
        self.models = {}
        self.target_scalers = {}
        self.feature_scaler = None
        
        self._load_models_and_scalers()
        
        # Validation tracking
        self.validation_history = []
        self.baseline_performance = {}
        
        # Alert system
        self.alert_handlers = []
    
    def _load_models_and_scalers(self):
        """Load trained models and scalers."""
        try:
            # Load feature scaler
            feature_scaler_path = self.scalers_dir / 'feature_scaler.pkl'
            if feature_scaler_path.exists():
                self.feature_scaler = joblib.load(str(feature_scaler_path))
                logger.info("Feature scaler loaded successfully")
            
            # Load models and target scalers
            for target in self.targets:
                # Load model
                model_path = self.models_dir / f'{target}_forecast_model.h5'
                if model_path.exists():
                    self.models[target] = tf.keras.models.load_model(str(model_path))
                    logger.info(f"Model loaded for {target}")
                
                # Load target scaler
                scaler_path = self.scalers_dir / f'{target}_scaler.pkl'
                if scaler_path.exists():
                    self.target_scalers[target] = joblib.load(str(scaler_path))
                    logger.info(f"Scaler loaded for {target}")
            
            logger.info(f"Loaded {len(self.models)} models and {len(self.target_scalers)} scalers")
            
        except Exception as e:
            logger.error(f"Error loading models and scalers: {e}")
            raise
    
    def prepare_validation_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare data for validation (same preprocessing as training).
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Preprocessed feature array
        """
        try:
            # Create time-based features
            df = df.copy()
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['quarter'] = df['datetime'].dt.quarter
            df['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
            df['is_peak_hour'] = ((df['hour'].between(6, 11)) | (df['hour'].between(18, 23))).astype(int)
            df['season_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['season_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Create lag features
            if 'DELHI' in df.columns:
                df['DELHI_lag_1'] = df['DELHI'].shift(1)
                df['DELHI_lag_24'] = df['DELHI'].shift(24)
                df['DELHI_lag_168'] = df['DELHI'].shift(168)
            
            # Aggregate features
            load_columns = [col for col in self.targets if col in df.columns]
            if load_columns:
                df['load_sum'] = df[load_columns].sum(axis=1)
                df['load_mean'] = df[load_columns].mean(axis=1)
                df['load_std'] = df[load_columns].std(axis=1)
            
            # Handle missing values
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Feature columns (same as training)
            feature_columns = [
                'DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES',
                'temperature', 'humidity', 'wind_speed', 'precipitation',
                'hour', 'day_of_week', 'month', 'quarter',
                'is_weekend', 'is_peak_hour', 'season_sin', 'season_cos',
                'DELHI_lag_1', 'DELHI_lag_24', 'DELHI_lag_168',
                'load_sum', 'load_mean', 'load_std'
            ]
            
            # Ensure all columns exist
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Scale features
            features = df[feature_columns].values
            if self.feature_scaler:
                features = self.feature_scaler.transform(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing validation data: {e}")
            raise
    
    def validate_real_time_data(self) -> Dict[str, ValidationResult]:
        """Validate current real-time predictions against actual data."""
        try:
            logger.info("Starting real-time validation...")
            
            # Get recent data for validation
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=self.config.validation_window + 24)
            
            # Fetch recent data
            recent_data = self._fetch_recent_data(start_time, end_time)
            
            if recent_data.empty:
                logger.warning("No recent data available for validation")
                return {}
            
            # Prepare features
            features = self.prepare_validation_data(recent_data)
            
            validation_results = {}
            
            for target in self.targets:
                if target not in self.models or target not in self.target_scalers:
                    logger.warning(f"Model or scaler not available for {target}")
                    continue
                
                result = self._validate_target(target, features, recent_data)
                if result:
                    validation_results[target] = result
                    
                    # Log validation result to database
                    self._log_validation_result(result)
            
            # Check for system-wide alerts
            self._check_system_alerts(validation_results)
            
            logger.info(f"Real-time validation completed for {len(validation_results)} targets")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Real-time validation failed: {e}")
            return {}
    
    def _fetch_recent_data(self, start_time: datetime.datetime, 
                          end_time: datetime.datetime) -> pd.DataFrame:
        """Fetch recent data from database."""
        try:
            query = """
            SELECT datetime, DELHI, BRPL, BYPL, NDMC, MES,
                   temperature, humidity, wind_speed, precipitation
            FROM historical_load_data
            WHERE datetime BETWEEN ? AND ?
            ORDER BY datetime
            """
            
            df = self.db_manager.execute_query(query, (start_time, end_time))
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching recent data: {e}")
            return pd.DataFrame()
    
    def _validate_target(self, target: str, features: np.ndarray, 
                        data: pd.DataFrame) -> Optional[ValidationResult]:
        """Validate predictions for a specific target."""
        try:
            # Get model and scaler
            model = self.models[target]
            scaler = self.target_scalers[target]
            
            # Create sequences for prediction
            sequence_length = 24  # Same as training
            prediction_horizon = 24
            
            if len(features) < sequence_length + prediction_horizon:
                logger.warning(f"Insufficient data for {target} validation")
                return None
            
            # Prepare sequences
            X_sequences = []
            y_actual = []
            
            for i in range(sequence_length, len(features) - prediction_horizon + 1):
                X_sequences.append(features[i-sequence_length:i])
                
                # Get actual values for comparison
                actual_values = data[target].iloc[i:i+prediction_horizon].values
                y_actual.append(actual_values)
            
            if not X_sequences:
                return None
            
            X_sequences = np.array(X_sequences)
            y_actual = np.array(y_actual)
            
            # Make predictions
            predictions = model.predict(X_sequences, verbose=0)
            
            # Inverse transform predictions and actuals
            pred_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
            actual_orig = y_actual
            
            # Calculate metrics
            metrics = self._calculate_validation_metrics(actual_orig, pred_orig)
            
            # Data quality assessment
            data_quality_score = self._assess_data_quality(data[target].values)
            
            # Check for alerts
            alerts = self._check_target_alerts(target, metrics, data_quality_score)
            
            # Determine if validation passed
            is_valid = (
                metrics['accuracy'] >= self.config.min_accuracy_threshold and
                metrics['mae'] <= self.config.max_error_threshold and
                data_quality_score >= 0.8
            )
            
            result = ValidationResult(
                timestamp=datetime.datetime.now(),
                target=target,
                accuracy=metrics['accuracy'],
                mae=metrics['mae'],
                mse=metrics['mse'],
                rmse=metrics['rmse'],
                r2=metrics['r2'],
                mape=metrics['mape'],
                data_quality_score=data_quality_score,
                alerts=alerts,
                is_valid=is_valid,
                predictions=pred_orig,
                actuals=actual_orig
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating {target}: {e}")
            return None
    
    def _calculate_validation_metrics(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict:
        """Calculate validation metrics."""
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
        y_true_clean = y_true_flat[mask]
        y_pred_clean = y_pred_flat[mask]
        
        if len(y_true_clean) == 0:
            return {
                'mse': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'r2': -float('inf'),
                'mape': float('inf'),
                'accuracy': 0.0
            }
        
        metrics = {
            'mse': float(mean_squared_error(y_true_clean, y_pred_clean)),
            'mae': float(mean_absolute_error(y_true_clean, y_pred_clean)),
            'rmse': float(np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))),
            'r2': float(r2_score(y_true_clean, y_pred_clean)),
            'mape': float(np.mean(np.abs((y_true_clean - y_pred_clean) / np.maximum(y_true_clean, 1))) * 100),
        }
        
        metrics['accuracy'] = max(0.0, 100.0 - metrics['mape'])
        
        return metrics
    
    def _assess_data_quality(self, values: np.ndarray) -> float:
        """Assess data quality score (0-1)."""
        try:
            if len(values) == 0:
                return 0.0
            
            # Check for missing values
            missing_ratio = np.isnan(values).sum() / len(values)
            
            # Check for unrealistic values
            realistic_mask = (values >= 0) & (values <= 10000)
            unrealistic_ratio = (~realistic_mask).sum() / len(values)
            
            # Check for constant values (indicating sensor issues)
            unique_values = len(np.unique(values[~np.isnan(values)]))
            diversity_score = min(1.0, unique_values / 10.0)
            
            # Combined quality score
            quality_score = (
                (1.0 - missing_ratio) * 0.4 +
                (1.0 - unrealistic_ratio) * 0.4 +
                diversity_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _check_target_alerts(self, target: str, metrics: Dict, 
                           data_quality: float) -> List[str]:
        """Check for target-specific alerts."""
        alerts = []
        
        # Accuracy alerts
        if metrics['accuracy'] < self.config.alert_thresholds['low_accuracy']:
            alerts.append(f"Low accuracy: {metrics['accuracy']:.1f}%")
        
        # Error alerts
        if metrics['mae'] > self.config.alert_thresholds['high_error']:
            alerts.append(f"High MAE: {metrics['mae']:.1f} MW")
        
        # Data quality alerts
        if data_quality < (1.0 - self.config.alert_thresholds['data_missing']):
            alerts.append(f"Poor data quality: {data_quality:.1f}")
        
        # Model drift detection
        if target in self.baseline_performance:
            baseline_acc = self.baseline_performance[target]['accuracy']
            if metrics['accuracy'] < baseline_acc * (1 - self.config.alert_thresholds['model_drift']):
                alerts.append(f"Model drift detected: {metrics['accuracy']:.1f}% vs baseline {baseline_acc:.1f}%")
        
        return alerts
    
    def _check_system_alerts(self, validation_results: Dict[str, ValidationResult]):
        """Check for system-wide alerts."""
        if not validation_results:
            return
        
        # Check if multiple targets are failing
        failed_targets = [target for target, result in validation_results.items() 
                         if not result.is_valid]
        
        if len(failed_targets) >= len(validation_results) * 0.6:  # 60% failure rate
            logger.critical(f"System-wide validation failure: {len(failed_targets)}/{len(validation_results)} targets failed")
            self._trigger_alert("SYSTEM_FAILURE", f"Multiple targets failing: {failed_targets}")
        
        # Check average accuracy across all targets
        avg_accuracy = np.mean([result.accuracy for result in validation_results.values()])
        if avg_accuracy < self.config.alert_thresholds['low_accuracy']:
            logger.warning(f"Low system accuracy: {avg_accuracy:.1f}%")
            self._trigger_alert("LOW_SYSTEM_ACCURACY", f"Average accuracy: {avg_accuracy:.1f}%")
    
    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger system alert."""
        alert_data = {
            'timestamp': datetime.datetime.now(),
            'type': alert_type,
            'message': message
        }
        
        # Log to database
        try:
            self.db_manager.log_validation_alert(alert_type, message)
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def _log_validation_result(self, result: ValidationResult):
        """Log validation result to database."""
        try:
            self.db_manager.log_validation_result(
                target=result.target,
                accuracy=result.accuracy,
                mae=result.mae,
                mse=result.mse,
                data_quality_score=result.data_quality_score,
                alerts=result.alerts,
                is_valid=result.is_valid
            )
        except Exception as e:
            logger.error(f"Failed to log validation result: {e}")
    
    def run_validation_report(self) -> Dict:
        """Generate comprehensive validation report."""
        try:
            logger.info("Generating validation report...")
            
            # Run real-time validation
            validation_results = self.validate_real_time_data()
            
            # Get historical validation data
            historical_results = self._get_historical_validation_data()
            
            # Generate report
            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'current_validation': {
                    target: {
                        'accuracy': result.accuracy,
                        'mae': result.mae,
                        'data_quality': result.data_quality_score,
                        'alerts': result.alerts,
                        'is_valid': result.is_valid
                    }
                    for target, result in validation_results.items()
                },
                'system_summary': self._generate_system_summary(validation_results),
                'historical_trends': historical_results,
                'recommendations': self._generate_recommendations(validation_results)
            }
            
            # Save report
            report_path = self.validation_dir / f"validation_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Validation report saved to {report_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return {}
    
    def _get_historical_validation_data(self, days: int = 7) -> Dict:
        """Get historical validation data."""
        try:
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(days=days)
            
            query = """
            SELECT target, accuracy, mae, data_quality_score, timestamp
            FROM validation_logs
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            """
            
            df = self.db_manager.execute_query(query, (start_time, end_time))
            
            if df.empty:
                return {}
            
            # Group by target and calculate trends
            trends = {}
            for target in self.targets:
                target_data = df[df['target'] == target]
                if not target_data.empty:
                    trends[target] = {
                        'avg_accuracy': float(target_data['accuracy'].mean()),
                        'avg_mae': float(target_data['mae'].mean()),
                        'avg_data_quality': float(target_data['data_quality_score'].mean()),
                        'validation_count': len(target_data)
                    }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting historical validation data: {e}")
            return {}
    
    def _generate_system_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict:
        """Generate system performance summary."""
        if not validation_results:
            return {'status': 'NO_DATA'}
        
        accuracies = [result.accuracy for result in validation_results.values()]
        quality_scores = [result.data_quality_score for result in validation_results.values()]
        valid_targets = [target for target, result in validation_results.items() if result.is_valid]
        
        return {
            'overall_status': 'HEALTHY' if len(valid_targets) >= len(validation_results) * 0.8 else 'DEGRADED',
            'average_accuracy': float(np.mean(accuracies)),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'average_data_quality': float(np.mean(quality_scores)),
            'valid_targets': len(valid_targets),
            'total_targets': len(validation_results),
            'success_rate': float(len(valid_targets) / len(validation_results) * 100)
        }
    
    def _generate_recommendations(self, validation_results: Dict[str, ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if not validation_results:
            recommendations.append("No validation data available - check data pipeline")
            return recommendations
        
        # Check for low accuracy targets
        low_accuracy_targets = [
            target for target, result in validation_results.items() 
            if result.accuracy < self.config.min_accuracy_threshold
        ]
        
        if low_accuracy_targets:
            recommendations.append(f"Retrain models for low accuracy targets: {low_accuracy_targets}")
        
        # Check for data quality issues
        poor_quality_targets = [
            target for target, result in validation_results.items()
            if result.data_quality_score < 0.8
        ]
        
        if poor_quality_targets:
            recommendations.append(f"Investigate data quality issues for: {poor_quality_targets}")
        
        # Check for high error rates
        high_error_targets = [
            target for target, result in validation_results.items()
            if result.mae > self.config.max_error_threshold
        ]
        
        if high_error_targets:
            recommendations.append(f"Review model architecture for high error targets: {high_error_targets}")
        
        # System-wide recommendations
        avg_accuracy = np.mean([result.accuracy for result in validation_results.values()])
        if avg_accuracy < 90:
            recommendations.append("Consider collecting more training data or feature engineering")
        
        if not recommendations:
            recommendations.append("System performing well - continue monitoring")
        
        return recommendations
    
    def setup_scheduled_validation(self):
        """Setup scheduled validation checks."""
        # Schedule validation every hour
        schedule.every().hour.do(self.validate_real_time_data)
        
        # Schedule daily reports
        schedule.every().day.at("06:00").do(self.run_validation_report)
        
        logger.info("Scheduled validation configured")
    
    def run_validation_scheduler(self):
        """Run the validation scheduler."""
        logger.info("Starting validation scheduler...")
        
        while True:
            schedule.run_pending()
            time.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    # Example usage
    config = ValidationConfig(
        validation_window=24,
        min_accuracy_threshold=85.0,
        real_time_monitoring=True
    )
    
    pipeline = ValidationPipeline(config=config)
    
    # Run validation
    try:
        results = pipeline.validate_real_time_data()
        
        if results:
            print("Validation Results:")
            for target, result in results.items():
                status = "✓" if result.is_valid else "✗"
                print(f"{status} {target}: {result.accuracy:.1f}% accuracy, MAE: {result.mae:.1f} MW")
                
                if result.alerts:
                    print(f"  Alerts: {', '.join(result.alerts)}")
        else:
            print("No validation results available")
        
        # Generate report
        report = pipeline.run_validation_report()
        if report:
            print(f"\nSystem Status: {report['system_summary']['overall_status']}")
            print(f"Average Accuracy: {report['system_summary']['average_accuracy']:.1f}%")
            
    except Exception as e:
        print(f"Validation failed: {e}")
