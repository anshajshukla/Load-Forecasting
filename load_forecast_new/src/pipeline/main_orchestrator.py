"""
Main Pipeline Orchestrator for Delhi SLDC Load Forecasting.
Enhanced for SIH 2024 Problem Statement 1624 - AI-based Electricity Demand Projection.
Coordinates data fetching, training, validation, monitoring, and peak demand optimization.
"""

import os
import sys
import asyncio
import threading
import schedule
import time
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import argparse

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))  # Add parent for SIH enhancements

# Import our pipeline components
from database.db_manager import DatabaseManager, create_database_manager
from enhanced_fetcher import EnhancedDataFetcher
from training_pipeline import ModelTrainingPipeline, TrainingConfig
from validation_pipeline import ValidationPipeline, ValidationConfig

# Import SIH 2024 enhancements
try:
    from sih_2024_enhancements import (
        SIHForecastingConfig, 
        PeakDemandForecaster, 
        DelhiHolidayCalendar,
        SpikeDetectionSystem
    )
    SIH_ENHANCEMENTS_AVAILABLE = True
except ImportError:
    SIH_ENHANCEMENTS_AVAILABLE = False
    logging.warning("SIH 2024 enhancements not available. Using basic forecasting.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline/logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline with SIH 2024 enhancements."""
    # Data fetching
    fetch_historical_years: int = 5
    fetch_parallel_workers: int = 3
    
    # Training
    retrain_schedule: str = "weekly"  # daily, weekly, monthly
    auto_retrain_threshold: float = 80.0  # Retrain if accuracy drops below this
    
    # Validation
    validation_interval_minutes: int = 60
    validation_window_hours: int = 24
    
    # Monitoring
    enable_real_time_monitoring: bool = True
    enable_alerts: bool = True
    
    # SIH 2024 Enhancements
    enable_spike_detection: bool = True
    enable_peak_demand_forecasting: bool = True
    enable_holiday_awareness: bool = True
    spike_detection_interval_minutes: int = 15  # 15-minute intervals for spikes
    peak_forecast_horizon_hours: int = 48  # 48-hour peak demand forecast
    
    # Paths
    base_path: Path = None
    
    def __post_init__(self):
        if self.base_path is None:
            self.base_path = Path('data_pipeline')

class PipelineOrchestrator:
    """Main orchestrator for the complete forecasting pipeline with SIH 2024 enhancements."""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize the pipeline orchestrator with SIH 2024 capabilities."""
        self.config = config or PipelineConfig()
        
        # Create logs directory
        logs_dir = self.config.base_path / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.db_manager = create_database_manager()
        self.data_fetcher = EnhancedDataFetcher(self.db_manager, self.config.fetch_parallel_workers)
        
        # Training configuration
        training_config = TrainingConfig(
            sequence_length=24,
            prediction_horizon=24,
            epochs=100,
            batch_size=32,
            patience=15
        )
        self.training_pipeline = ModelTrainingPipeline(self.db_manager, training_config)
        
        # Validation configuration
        validation_config = ValidationConfig(
            validation_window=self.config.validation_window_hours,
            min_accuracy_threshold=85.0,
            real_time_monitoring=self.config.enable_real_time_monitoring
        )
        self.validation_pipeline = ValidationPipeline(self.db_manager, validation_config)
        
        # SIH 2024 Enhancements
        self.sih_components = None
        if SIH_ENHANCEMENTS_AVAILABLE and self.config.enable_spike_detection:
            try:
                sih_config = SIHForecastingConfig()
                self.sih_components = {
                    'peak_forecaster': PeakDemandForecaster(sih_config),
                    'holiday_calendar': DelhiHolidayCalendar(),
                    'spike_detector': SpikeDetectionSystem(sih_config),
                    'config': sih_config
                }
                logger.info("SIH 2024 enhancements initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SIH enhancements: {e}")
                self.sih_components = None
        
        # Pipeline state
        self.is_running = False
        self.current_tasks = {}
        self.scheduler_thread = None
        
        # Performance tracking
        self.performance_history = []
        
        # SIH-specific tracking
        self.spike_alerts = []
        self.peak_forecasts = []
        
        logger.info("Pipeline orchestrator initialized with SIH 2024 capabilities")
    
    def initialize_pipeline(self) -> bool:
        """Initialize the complete pipeline."""
        try:
            logger.info("Initializing data pipeline...")
            
            # Check database connection
            if not self._check_database_connection():
                logger.error("Database connection failed")
                return False
            
            # Check if models exist
            models_exist = self._check_models_exist()
            
            if not models_exist:
                logger.info("No trained models found. Starting with historical data collection...")
                
                # Fetch historical data first
                if not self._initial_data_setup():
                    logger.error("Initial data setup failed")
                    return False
                
                # Train initial models
                if not self._initial_model_training():
                    logger.error("Initial model training failed")
                    return False
            
            # Setup monitoring
            self._setup_monitoring()
            
            logger.info("Pipeline initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _check_database_connection(self) -> bool:
        """Check database connection and create tables if needed."""
        try:
            # Test connection and create tables
            self.db_manager.create_tables()
            
            # Test query
            result = self.db_manager.execute_query("SELECT COUNT(*) as count FROM historical_load_data")
            
            logger.info(f"Database connected. Historical records: {result['count'].iloc[0] if not result.empty else 0}")
            return True
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False
    
    def _check_models_exist(self) -> bool:
        """Check if trained models exist."""
        models_dir = self.config.base_path / 'models' / 'trained_models'
        
        if not models_dir.exists():
            return False
        
        # Check for model files
        targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        for target in targets:
            model_file = models_dir / f'{target}_forecast_model.h5'
            if not model_file.exists():
                return False
        
        logger.info("Trained models found")
        return True
    
    def _initial_data_setup(self) -> bool:
        """Perform initial historical data collection."""
        try:
            logger.info(f"Starting {self.config.fetch_historical_years}-year historical data collection...")
            
            # Test with recent data first
            test_start = datetime.date.today() - datetime.timedelta(days=3)
            test_end = datetime.date.today()
            
            logger.info("Testing data fetcher with recent dates...")
            test_df = self.data_fetcher.fetch_date_range_parallel(test_start, test_end)
            
            if test_df.empty:
                logger.error("Test data fetch failed - check network connection")
                return False
            
            logger.info(f"Test successful! Collected {len(test_df)} records")
            
            # Proceed with full historical fetch
            logger.info("Starting full historical data collection...")
            historical_df = self.data_fetcher.fetch_last_n_years(self.config.fetch_historical_years)
            
            if historical_df.empty:
                logger.error("Historical data collection failed")
                return False
            
            logger.info(f"Historical data collection completed! Total records: {len(historical_df)}")
            return True
            
        except Exception as e:
            logger.error(f"Initial data setup failed: {e}")
            return False
    
    def _initial_model_training(self) -> bool:
        """Perform initial model training."""
        try:
            logger.info("Starting initial model training...")
            
            # Check if we have enough data
            query = "SELECT COUNT(*) as count FROM historical_load_data"
            result = self.db_manager.execute_query(query)
            
            if result.empty or result['count'].iloc[0] < 1000:
                logger.error("Insufficient data for training (minimum 1000 records required)")
                return False
            
            # Run training pipeline
            training_results = self.training_pipeline.run_full_training_pipeline()
            
            if not training_results or 'results' not in training_results:
                logger.error("Model training failed")
                return False
            
            # Log training results
            summary = training_results['summary']
            avg_accuracy = summary['overall_performance']['average_accuracy']
            
            logger.info(f"Initial training completed! Average accuracy: {avg_accuracy:.2f}%")
            
            # Validate initial models
            validation_results = self.validation_pipeline.validate_real_time_data()
            if validation_results:
                logger.info("Initial model validation completed")
            
            return True
            
        except Exception as e:
            logger.error(f"Initial model training failed: {e}")
            return False
    
    def _setup_monitoring(self):
        """Setup monitoring and scheduling with SIH 2024 enhancements."""
        try:
            # Setup data fetcher schedules
            self.data_fetcher.setup_scheduled_updates()
            
            # Setup validation schedules
            self.validation_pipeline.setup_scheduled_validation()
            
            # Setup training schedules
            if self.config.retrain_schedule == "daily":
                schedule.every().day.at("02:00").do(self._scheduled_retrain)
            elif self.config.retrain_schedule == "weekly":
                schedule.every().monday.at("02:00").do(self._scheduled_retrain)
            elif self.config.retrain_schedule == "monthly":
                schedule.every(30).days.at("02:00").do(self._scheduled_retrain)
            
            # Setup performance monitoring
            schedule.every(30).minutes.do(self._monitor_performance)
            
            # Setup system health checks
            schedule.every().hour.do(self._system_health_check)
            
            # SIH 2024 Enhanced Monitoring
            if self.sih_components:
                # Spike detection monitoring
                if self.config.enable_spike_detection:
                    schedule.every(self.config.spike_detection_interval_minutes).minutes.do(self._monitor_spikes)
                
                # Peak demand forecasting
                if self.config.enable_peak_demand_forecasting:
                    schedule.every().hour.do(self._update_peak_forecasts)
                
                # Holiday-aware adjustments
                if self.config.enable_holiday_awareness:
                    schedule.every().day.at("00:30").do(self._check_holiday_adjustments)
                
                logger.info("SIH 2024 enhanced monitoring scheduled")
            
            logger.info("Monitoring and scheduling setup completed")
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
    
    def start_pipeline(self):
        """Start the complete pipeline."""
        try:
            if self.is_running:
                logger.warning("Pipeline is already running")
                return
            
            logger.info("Starting data pipeline...")
            
            # Initialize if not done
            if not self.initialize_pipeline():
                logger.error("Pipeline initialization failed")
                return
            
            self.is_running = True
            
            # Start scheduler in separate thread
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            
            # Start data fetcher scheduler
            fetcher_thread = threading.Thread(target=self.data_fetcher.run_scheduler, daemon=True)
            fetcher_thread.start()
            
            # Start validation scheduler
            validation_thread = threading.Thread(target=self.validation_pipeline.run_validation_scheduler, daemon=True)
            validation_thread.start()
            
            logger.info("Pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start pipeline: {e}")
            self.is_running = False
    
    def stop_pipeline(self):
        """Stop the pipeline."""
        logger.info("Stopping data pipeline...")
        
        self.is_running = False
        
        # Stop data fetcher
        self.data_fetcher.stop_fetch()
        
        # Clear schedules
        schedule.clear()
        
        logger.info("Pipeline stopped")
    
    def _run_scheduler(self):
        """Run the main scheduler."""
        logger.info("Scheduler started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)
        
        logger.info("Scheduler stopped")
    
    def _scheduled_retrain(self):
        """Perform scheduled retraining."""
        try:
            logger.info("Starting scheduled retraining...")
            
            # Check if retraining is needed
            if not self._should_retrain():
                logger.info("Retraining not needed based on current performance")
                return
            
            # Run training pipeline
            results = self.training_pipeline.run_full_training_pipeline()
            
            if results:
                avg_accuracy = results['summary']['overall_performance']['average_accuracy']
                logger.info(f"Scheduled retraining completed! New average accuracy: {avg_accuracy:.2f}%")
            else:
                logger.error("Scheduled retraining failed")
                
        except Exception as e:
            logger.error(f"Scheduled retraining error: {e}")
    
    def _should_retrain(self) -> bool:
        """Check if retraining is needed based on performance."""
        try:
            # Get recent validation results
            validation_results = self.validation_pipeline.validate_real_time_data()
            
            if not validation_results:
                return False
            
            # Check average accuracy
            avg_accuracy = sum(result.accuracy for result in validation_results.values()) / len(validation_results)
            
            return avg_accuracy < self.config.auto_retrain_threshold
            
        except Exception as e:
            logger.error(f"Error checking retrain need: {e}")
            return False
    
    def _monitor_performance(self):
        """Monitor system performance."""
        try:
            # Get validation results
            validation_results = self.validation_pipeline.validate_real_time_data()
            
            if validation_results:
                # Calculate performance metrics
                performance = {
                    'timestamp': datetime.datetime.now(),
                    'targets': {},
                    'system_average': 0.0
                }
                
                total_accuracy = 0
                for target, result in validation_results.items():
                    performance['targets'][target] = {
                        'accuracy': result.accuracy,
                        'mae': result.mae,
                        'data_quality': result.data_quality_score,
                        'alerts': len(result.alerts)
                    }
                    total_accuracy += result.accuracy
                
                performance['system_average'] = total_accuracy / len(validation_results)
                
                # Store performance history
                self.performance_history.append(performance)
                
                # Keep only last 24 hours of data
                cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)
                self.performance_history = [
                    p for p in self.performance_history 
                    if p['timestamp'] > cutoff_time
                ]
                
                # Log performance
                logger.info(f"System performance: {performance['system_average']:.1f}% average accuracy")
                
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
    
    def _system_health_check(self):
        """Perform system health check."""
        try:
            health_status = {
                'database': False,
                'models': False,
                'data_quality': False,
                'overall': False
            }
            
            # Database health
            try:
                result = self.db_manager.execute_query("SELECT 1")
                health_status['database'] = not result.empty
            except:
                health_status['database'] = False
            
            # Models health
            health_status['models'] = self._check_models_exist()
            
            # Data quality check
            try:
                recent_data = self.db_manager.execute_query("""
                    SELECT COUNT(*) as count 
                    FROM historical_load_data 
                    WHERE datetime > datetime('now', '-1 hour')
                """)
                health_status['data_quality'] = (
                    not recent_data.empty and 
                    recent_data['count'].iloc[0] > 0
                )
            except:
                health_status['data_quality'] = False
            
            # Overall health
            health_status['overall'] = all([
                health_status['database'],
                health_status['models'],
                health_status['data_quality']
            ])
            
            if not health_status['overall']:
                logger.warning(f"System health check failed: {health_status}")
            else:
                logger.debug("System health check passed")
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    # SIH 2024 Enhanced Monitoring Methods
    def _monitor_spikes(self):
        """Monitor for electrical load spikes using SIH 2024 algorithms."""
        if not self.sih_components:
            return
        
        try:
            # Get recent load data for spike detection
            query = """
                SELECT datetime, DELHI, BRPL, BYPL, NDMC, MES
                FROM historical_load_data 
                WHERE datetime > datetime('now', '-2 hours')
                ORDER BY datetime
            """
            
            recent_data = self.db_manager.execute_query(query)
            
            if not recent_data.empty:
                spike_detector = self.sih_components['spike_detector']
                
                # Check each target for spikes
                targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
                available_targets = [t for t in targets if t in recent_data.columns]
                
                for target in available_targets:
                    if recent_data[target].notna().any():
                        spikes = spike_detector.detect_spikes(recent_data[target])
                        
                        if spikes['statistical_spikes']:
                            alert = {
                                'timestamp': datetime.datetime.now(),
                                'target': target,
                                'spike_type': 'statistical',
                                'magnitude': max(spikes['statistical_spikes'].values()),
                                'description': f'Statistical spike detected in {target}'
                            }
                            self.spike_alerts.append(alert)
                            logger.warning(f"SPIKE ALERT: {alert['description']} - {alert['magnitude']:.2f} MW")
                
                # Keep only last 24 hours of alerts
                cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)
                self.spike_alerts = [
                    alert for alert in self.spike_alerts 
                    if alert['timestamp'] > cutoff_time
                ]
                
        except Exception as e:
            logger.error(f"Spike monitoring error: {e}")
    
    def _update_peak_forecasts(self):
        """Update peak demand forecasts for grid optimization."""
        if not self.sih_components:
            return
        
        try:
            # Get recent data for peak forecasting
            query = """
                SELECT datetime, DELHI, BRPL, BYPL, NDMC, MES,
                       temperature, humidity, wind_speed, precipitation
                FROM historical_load_data 
                WHERE datetime > datetime('now', '-7 days')
                ORDER BY datetime
            """
            
            data = self.db_manager.execute_query(query)
            
            if not data.empty:
                peak_forecaster = self.sih_components['peak_forecaster']
                
                # Analyze peak patterns
                analysis = peak_forecaster.analyze_peak_patterns(data)
                
                # Generate 48-hour peak forecast
                forecast = {
                    'timestamp': datetime.datetime.now(),
                    'forecast_horizon': self.config.peak_forecast_horizon_hours,
                    'analysis': analysis,
                    'recommendations': self._generate_grid_recommendations(analysis)
                }
                
                self.peak_forecasts.append(forecast)
                
                # Keep only last 7 days of forecasts
                cutoff_time = datetime.datetime.now() - datetime.timedelta(days=7)
                self.peak_forecasts = [
                    f for f in self.peak_forecasts 
                    if f['timestamp'] > cutoff_time
                ]
                
                logger.info(f"Peak demand forecast updated. Next peak expected: {self._get_next_peak_time(analysis)}")
                
        except Exception as e:
            logger.error(f"Peak forecasting error: {e}")
    
    def _check_holiday_adjustments(self):
        """Check for upcoming holidays and adjust forecasting parameters."""
        if not self.sih_components:
            return
        
        try:
            holiday_calendar = self.sih_components['holiday_calendar']
            
            # Check next 7 days for holidays
            today = datetime.date.today()
            upcoming_holidays = []
            
            for i in range(7):
                check_date = today + datetime.timedelta(days=i)
                holiday_features = holiday_calendar.get_holiday_features(check_date)
                
                if holiday_features['is_holiday']:
                    upcoming_holidays.append({
                        'date': check_date,
                        'features': holiday_features
                    })
            
            if upcoming_holidays:
                logger.info(f"Upcoming holidays detected: {len(upcoming_holidays)} holidays in next 7 days")
                
                # Adjust forecasting parameters for holidays
                for holiday in upcoming_holidays:
                    impact = holiday['features']['holiday_impact']
                    logger.info(f"Holiday on {holiday['date']}: {impact} impact expected")
                
        except Exception as e:
            logger.error(f"Holiday adjustment check error: {e}")
    
    def _generate_grid_recommendations(self, analysis: Dict) -> List[str]:
        """Generate grid optimization recommendations based on analysis."""
        recommendations = []
        
        try:
            # Peak load recommendations
            if 'daily_peaks' in analysis:
                for target, stats in analysis['daily_peaks'].items():
                    if stats['max_peak'] > stats['mean_peak'] * 1.2:
                        recommendations.append(
                            f"High peak variability in {target}. Consider load balancing strategies."
                        )
            
            # Weather-based recommendations
            if 'weather_correlations' in analysis:
                recommendations.append("Monitor weather forecasts for temperature extremes affecting AC/heating load.")
            
            # Holiday recommendations
            if 'holiday_impacts' in analysis:
                recommendations.append("Adjust generation scheduling for upcoming holidays based on historical patterns.")
            
            # Spike recommendations
            if 'spike_analysis' in analysis:
                for target, spike_info in analysis['spike_analysis'].items():
                    if spike_info['spike_count'] > 5:
                        recommendations.append(
                            f"Frequent spikes detected in {target}. Implement preventive measures."
                        )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _get_next_peak_time(self, analysis: Dict) -> str:
        """Estimate next peak time based on analysis."""
        try:
            # Simple heuristic based on typical Delhi load patterns
            current_hour = datetime.datetime.now().hour
            
            if current_hour < 10:
                return "Morning peak expected around 10-11 AM"
            elif current_hour < 19:
                return "Evening peak expected around 7-9 PM"
            else:
                return "Next morning peak expected around 10-11 AM"
                
        except Exception as e:
            logger.error(f"Error estimating next peak: {e}")
            return "Unable to estimate next peak time"
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status with SIH 2024 enhancements."""
        try:
            # Get data fetcher status
            fetch_status = self.data_fetcher.get_fetch_status()
            
            # Get recent validation results
            validation_results = self.validation_pipeline.validate_real_time_data()
            
            # Get database stats
            try:
                db_stats = self.db_manager.execute_query("""
                    SELECT 
                        COUNT(*) as total_records,
                        MIN(datetime) as earliest_date,
                        MAX(datetime) as latest_date
                    FROM historical_load_data
                """)
            except:
                db_stats = None
            
            status = {
                'is_running': self.is_running,
                'timestamp': datetime.datetime.now().isoformat(),
                'data_fetcher': fetch_status,
                'validation': {
                    target: {
                        'accuracy': result.accuracy,
                        'is_valid': result.is_valid,
                        'alerts': len(result.alerts)
                    }
                    for target, result in validation_results.items()
                } if validation_results else {},
                'database': {
                    'total_records': int(db_stats['total_records'].iloc[0]) if db_stats is not None and not db_stats.empty else 0,
                    'earliest_date': str(db_stats['earliest_date'].iloc[0]) if db_stats is not None and not db_stats.empty else None,
                    'latest_date': str(db_stats['latest_date'].iloc[0]) if db_stats is not None and not db_stats.empty else None
                },
                'performance_history': self.performance_history[-10:] if self.performance_history else []
            }
            
            # Add SIH 2024 status information
            if self.sih_components:
                sih_status = {
                    'spike_detection_enabled': self.config.enable_spike_detection,
                    'peak_forecasting_enabled': self.config.enable_peak_demand_forecasting,
                    'holiday_awareness_enabled': self.config.enable_holiday_awareness,
                    'recent_spike_alerts': len([
                        alert for alert in self.spike_alerts 
                        if alert['timestamp'] > datetime.datetime.now() - datetime.timedelta(hours=1)
                    ]),
                    'total_spike_alerts_24h': len(self.spike_alerts),
                    'latest_peak_forecast': self.peak_forecasts[-1]['timestamp'].isoformat() if self.peak_forecasts else None,
                    'grid_recommendations': self.peak_forecasts[-1]['recommendations'] if self.peak_forecasts else []
                }
                
                # Check today's holiday status
                today = datetime.date.today()
                holiday_features = self.sih_components['holiday_calendar'].get_holiday_features(today)
                sih_status['today_holiday_status'] = holiday_features
                
                status['sih_2024_enhancements'] = sih_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting pipeline status: {e}")
            return {'error': str(e)}
    
    def manual_retrain(self) -> Dict:
        """Manually trigger model retraining."""
        try:
            logger.info("Manual retraining triggered...")
            
            results = self.training_pipeline.run_full_training_pipeline()
            
            if results:
                logger.info("Manual retraining completed successfully")
                return {
                    'success': True,
                    'session_id': results['session_id'],
                    'summary': results['summary']
                }
            else:
                return {'success': False, 'error': 'Training failed'}
                
        except Exception as e:
            logger.error(f"Manual retraining failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def manual_data_fetch(self, days: int = 7) -> Dict:
        """Manually trigger data fetching."""
        try:
            logger.info(f"Manual data fetch triggered for {days} days...")
            
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=days)
            
            df = self.data_fetcher.fetch_date_range_parallel(start_date, end_date)
            
            return {
                'success': True,
                'records_collected': len(df),
                'date_range': f"{start_date} to {end_date}"
            }
            
        except Exception as e:
            logger.error(f"Manual data fetch failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # SIH 2024 Specific Methods
    def get_spike_alerts(self, hours: int = 24) -> List[Dict]:
        """Get spike alerts from the last N hours."""
        if not self.sih_components:
            return []
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        return [
            alert for alert in self.spike_alerts 
            if alert['timestamp'] > cutoff_time
        ]
    
    def get_peak_demand_forecast(self, hours: int = 48) -> Dict:
        """Get peak demand forecast for the next N hours."""
        if not self.sih_components or not self.peak_forecasts:
            return {'error': 'Peak forecasting not available'}
        
        latest_forecast = self.peak_forecasts[-1]
        
        return {
            'forecast_time': latest_forecast['timestamp'].isoformat(),
            'horizon_hours': hours,
            'analysis': latest_forecast['analysis'],
            'recommendations': latest_forecast['recommendations'],
            'next_peak_estimate': self._get_next_peak_time(latest_forecast['analysis'])
        }
    
    def get_holiday_impact_analysis(self, days: int = 30) -> Dict:
        """Get holiday impact analysis for grid planning."""
        if not self.sih_components:
            return {'error': 'Holiday analysis not available'}
        
        try:
            holiday_calendar = self.sih_components['holiday_calendar']
            
            # Analyze upcoming holidays
            today = datetime.date.today()
            upcoming_holidays = []
            
            for i in range(days):
                check_date = today + datetime.timedelta(days=i)
                holiday_features = holiday_calendar.get_holiday_features(check_date)
                
                if holiday_features['is_holiday'] or holiday_features['holiday_season']:
                    upcoming_holidays.append({
                        'date': check_date.isoformat(),
                        'features': holiday_features
                    })
            
            return {
                'analysis_period_days': days,
                'upcoming_holidays': upcoming_holidays,
                'total_holidays': len(upcoming_holidays),
                'planning_recommendations': self._generate_holiday_recommendations(upcoming_holidays)
            }
            
        except Exception as e:
            logger.error(f"Holiday analysis error: {e}")
            return {'error': str(e)}
    
    def _generate_holiday_recommendations(self, holidays: List[Dict]) -> List[str]:
        """Generate recommendations based on upcoming holidays."""
        recommendations = []
        
        high_impact_count = sum(1 for h in holidays if h['features']['holiday_impact'] == 'high_reduction')
        medium_impact_count = sum(1 for h in holidays if h['features']['holiday_impact'] == 'medium_reduction')
        increased_load_count = sum(1 for h in holidays if h['features']['holiday_impact'] == 'increased_load')
        
        if high_impact_count > 0:
            recommendations.append(f"Expect significant load reduction during {high_impact_count} major holidays. Plan maintenance schedules accordingly.")
        
        if medium_impact_count > 0:
            recommendations.append(f"Moderate load changes expected during {medium_impact_count} regional festivals. Adjust generation planning.")
        
        if increased_load_count > 0:
            recommendations.append(f"Increased load expected during {increased_load_count} celebration days. Ensure adequate generation capacity.")
        
        return recommendations
    
    def generate_grid_optimization_report(self) -> Dict:
        """Generate comprehensive grid optimization report for Delhi."""
        try:
            report = {
                'timestamp': datetime.datetime.now().isoformat(),
                'report_type': 'SIH_2024_Grid_Optimization',
                'executive_summary': {},
                'detailed_analysis': {},
                'recommendations': {},
                'alerts': {}
            }
            
            # Get current status
            status = self.get_pipeline_status()
            
            # Executive Summary
            report['executive_summary'] = {
                'system_health': 'Healthy' if status.get('is_running', False) else 'Issues Detected',
                'data_coverage': f"Historical data from {status.get('database', {}).get('earliest_date', 'N/A')} to {status.get('database', {}).get('latest_date', 'N/A')}",
                'total_records': status.get('database', {}).get('total_records', 0),
                'recent_accuracy': self._calculate_average_accuracy(status.get('validation', {}))
            }
            
            # SIH-specific analysis
            if self.sih_components:
                # Spike analysis
                recent_spikes = self.get_spike_alerts(24)
                report['detailed_analysis']['spike_analysis'] = {
                    'spikes_last_24h': len(recent_spikes),
                    'spike_targets': list(set(spike['target'] for spike in recent_spikes)),
                    'max_spike_magnitude': max([spike['magnitude'] for spike in recent_spikes], default=0)
                }
                
                # Peak demand analysis
                peak_forecast = self.get_peak_demand_forecast(48)
                if 'error' not in peak_forecast:
                    report['detailed_analysis']['peak_demand'] = peak_forecast
                
                # Holiday impact
                holiday_analysis = self.get_holiday_impact_analysis(14)
                if 'error' not in holiday_analysis:
                    report['detailed_analysis']['holiday_impact'] = holiday_analysis
                
                # Grid recommendations
                report['recommendations']['grid_optimization'] = self._generate_comprehensive_recommendations(report['detailed_analysis'])
            
            # Alerts
            report['alerts'] = {
                'spike_alerts': len(self.get_spike_alerts(24)),
                'system_alerts': len(status.get('validation', {})),
                'critical_issues': self._identify_critical_issues(status)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating grid optimization report: {e}")
            return {'error': str(e)}
    
    def _calculate_average_accuracy(self, validation_data: Dict) -> float:
        """Calculate average accuracy across all targets."""
        if not validation_data:
            return 0.0
        
        accuracies = [target_data.get('accuracy', 0) for target_data in validation_data.values()]
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    def _identify_critical_issues(self, status: Dict) -> List[str]:
        """Identify critical issues requiring immediate attention."""
        issues = []
        
        # Check system health
        if not status.get('is_running', False):
            issues.append("Pipeline not running")
        
        # Check data freshness
        latest_date = status.get('database', {}).get('latest_date')
        if latest_date:
            try:
                latest_datetime = datetime.datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
                if datetime.datetime.now() - latest_datetime > datetime.timedelta(hours=2):
                    issues.append("Data not updated in last 2 hours")
            except:
                pass
        
        # Check accuracy
        avg_accuracy = self._calculate_average_accuracy(status.get('validation', {}))
        if avg_accuracy < 75:
            issues.append(f"Low accuracy detected: {avg_accuracy:.1f}%")
        
        # Check spike alerts
        if self.sih_components and len(self.get_spike_alerts(1)) > 3:
            issues.append("Multiple spikes detected in last hour")
        
        return issues
    
    def _generate_comprehensive_recommendations(self, analysis: Dict) -> List[str]:
        """Generate comprehensive grid optimization recommendations."""
        recommendations = []
        
        # Spike-based recommendations
        spike_analysis = analysis.get('spike_analysis', {})
        if spike_analysis.get('spikes_last_24h', 0) > 5:
            recommendations.append("High spike activity detected. Implement real-time load balancing measures.")
        
        # Peak demand recommendations
        peak_demand = analysis.get('peak_demand', {})
        if 'recommendations' in peak_demand:
            recommendations.extend(peak_demand['recommendations'])
        
        # Holiday recommendations
        holiday_impact = analysis.get('holiday_impact', {})
        if 'planning_recommendations' in holiday_impact:
            recommendations.extend(holiday_impact['planning_recommendations'])
        
        # General grid optimization
        recommendations.extend([
            "Maintain 15-minute data collection intervals for optimal spike detection",
            "Monitor weather forecasts for extreme temperature events",
            "Coordinate with distribution companies for balanced load management",
            "Implement predictive maintenance during low-demand periods"
        ])
        
        return recommendations

def main():
    """Main entry point with SIH 2024 enhanced functionality."""
    parser = argparse.ArgumentParser(description='Delhi SLDC Load Forecasting Pipeline - SIH 2024 Enhanced')
    parser.add_argument('--action', choices=['start', 'status', 'retrain', 'fetch', 'spikes', 'peaks', 'holidays', 'report'], 
                       default='start', help='Action to perform')
    parser.add_argument('--days', type=int, default=7, 
                       help='Number of days for data fetching or analysis')
    parser.add_argument('--hours', type=int, default=24,
                       help='Number of hours for spike alerts or peak forecasting')
    parser.add_argument('--enable-sih', action='store_true', default=True,
                       help='Enable SIH 2024 enhancements (default: True)')
    
    args = parser.parse_args()
    
    # Create pipeline configuration with SIH 2024 enhancements
    config = PipelineConfig(
        fetch_historical_years=5,
        fetch_parallel_workers=3,
        retrain_schedule="weekly",
        validation_interval_minutes=60,
        enable_real_time_monitoring=True,
        enable_spike_detection=args.enable_sih,
        enable_peak_demand_forecasting=args.enable_sih,
        enable_holiday_awareness=args.enable_sih,
        spike_detection_interval_minutes=15,
        peak_forecast_horizon_hours=48
    )
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    if args.action == 'start':
        print("🚀 Starting Delhi SLDC Load Forecasting Pipeline (SIH 2024 Enhanced)...")
        
        if args.enable_sih and SIH_ENHANCEMENTS_AVAILABLE:
            print("✅ SIH 2024 Enhancements: Peak demand forecasting, spike detection, holiday awareness")
        
        orchestrator.start_pipeline()
        
        if orchestrator.is_running:
            print("✅ Pipeline started successfully!")
            print("📊 Features available:")
            print("   • Real-time load forecasting")
            print("   • 15-minute spike detection")
            print("   • Peak demand optimization")
            print("   • Holiday-aware predictions")
            print("   • Weather-integrated forecasting")
            print("\nPress Ctrl+C to stop...")
            
            try:
                while orchestrator.is_running:
                    time.sleep(10)
                    
                    # Print status every 10 minutes
                    if int(time.time()) % 600 == 0:
                        status = orchestrator.get_pipeline_status()
                        print(f"📈 Status: {status.get('validation', {})}")
                        
                        # Print SIH-specific status
                        if 'sih_2024_enhancements' in status:
                            sih_status = status['sih_2024_enhancements']
                            print(f"🎯 SIH Status: {sih_status.get('recent_spike_alerts', 0)} recent spikes, "
                                  f"Holiday: {sih_status.get('today_holiday_status', {}).get('is_holiday', False)}")
                        
            except KeyboardInterrupt:
                print("\n🛑 Stopping pipeline...")
                orchestrator.stop_pipeline()
                print("✅ Pipeline stopped.")
        else:
            print("❌ Failed to start pipeline!")
    
    elif args.action == 'status':
        print("📊 Getting pipeline status...")
        status = orchestrator.get_pipeline_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.action == 'retrain':
        print("🔄 Triggering manual retraining...")
        result = orchestrator.manual_retrain()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == 'fetch':
        print(f"📥 Triggering manual data fetch for {args.days} days...")
        result = orchestrator.manual_data_fetch(args.days)
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == 'spikes':
        print(f"⚡ Getting spike alerts for last {args.hours} hours...")
        if not SIH_ENHANCEMENTS_AVAILABLE:
            print("❌ SIH 2024 enhancements not available")
            return
        spikes = orchestrator.get_spike_alerts(args.hours)
        print(f"Found {len(spikes)} spike alerts:")
        for spike in spikes:
            print(f"  • {spike['timestamp']}: {spike['target']} - {spike['magnitude']:.2f} MW ({spike['spike_type']})")
    
    elif args.action == 'peaks':
        print(f"📈 Getting peak demand forecast for next {args.hours} hours...")
        if not SIH_ENHANCEMENTS_AVAILABLE:
            print("❌ SIH 2024 enhancements not available")
            return
        forecast = orchestrator.get_peak_demand_forecast(args.hours)
        print(json.dumps(forecast, indent=2, default=str))
    
    elif args.action == 'holidays':
        print(f"📅 Getting holiday impact analysis for next {args.days} days...")
        if not SIH_ENHANCEMENTS_AVAILABLE:
            print("❌ SIH 2024 enhancements not available")
            return
        holiday_analysis = orchestrator.get_holiday_impact_analysis(args.days)
        print(json.dumps(holiday_analysis, indent=2, default=str))
    
    elif args.action == 'report':
        print("📋 Generating comprehensive grid optimization report...")
        if not SIH_ENHANCEMENTS_AVAILABLE:
            print("❌ SIH 2024 enhancements not available")
            return
        report = orchestrator.generate_grid_optimization_report()
        print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
