"""
Feature Engineering Pipeline Orchestrator
Delhi Load Forecasting - SIH 2024

Main orchestrator that coordinates all feature engineering modules
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
import time
from datetime import datetime
import json
from pathlib import Path

# Import core components
from config.settings import settings
from core.data_loader import DataLoader, DataValidator
from core.base_module import BaseFeatureModule

# Import feature modules
from modules.dual_peaks import DualPeaksModule
from modules.geographic_distribution import GeographicDistributionModule
from modules.domestic_commercial import DomesticCommercialModule
from modules.growth_trends import GrowthTrendsModule
from modules.intraday_stability import IntradayStabilityModule
from modules.duck_curve import DuckCurveModule
from modules.seasonal_variation import SeasonalVariationModule
from modules.weather_compensation import WeatherCompensationModule
from modules.holiday_events import HolidayEventsModule
from modules.procurement_patterns import ProcurementPatternsModule
from modules.advanced_lag_features import run_advanced_lag_pipeline

# Import validation and reporting
from validation.validator import FeatureValidator
from reporting.reporter import FeatureEngineeringReporter


class FeatureEngineeringOrchestrator:
    """Main orchestrator for the feature engineering pipeline"""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the orchestrator
        
        Args:
            config_override: Optional configuration overrides
        """
        self.logger = logging.getLogger("feature_engineering.orchestrator")
        self.config = settings.to_dict()
        
        # Apply config overrides
        if config_override:
            self._apply_config_override(config_override)
        
        # Initialize components
        self.data_loader = DataLoader(self.config['modules']['data_loading'])
        self.data_validator = DataValidator(self.config.get('validation', {}))
        self.feature_validator = FeatureValidator(self.config.get('validation', {}))
        self.reporter = FeatureEngineeringReporter(self.config.get('reporting', {}))
        
        # Initialize feature modules
        self.modules = self._initialize_modules()
        
        # Execution tracking
        self.execution_stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'modules_executed': [],
            'total_features_created': 0,
            'errors': [],
            'warnings': []
        }
        
        self.logger.info(f"Feature Engineering Orchestrator initialized with {len(self.modules)} modules")
    
    def _apply_config_override(self, override: Dict[str, Any]) -> None:
        """Apply configuration overrides"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict:
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, override)
        self.logger.info("Applied configuration overrides")
    
    def _initialize_modules(self) -> List[BaseFeatureModule]:
        """Initialize all enabled feature modules"""
        modules = []
        module_configs = self.config.get('modules', {})
        
        # Module registry - add new modules here
        module_registry = {
            'dual_peaks': DualPeaksModule,
            'geographic_distribution': GeographicDistributionModule,
            'domestic_commercial': DomesticCommercialModule,
            'growth_trends': GrowthTrendsModule,
            'intraday_stability': IntradayStabilityModule,
            'duck_curve': DuckCurveModule,
            'seasonal_variation': SeasonalVariationModule,
            'weather_compensation': WeatherCompensationModule,
            'holiday_events': HolidayEventsModule,
            'procurement_patterns': ProcurementPatternsModule
        }
        
        # Initialize enabled modules
        for module_name, module_class in module_registry.items():
            if module_name in module_configs:
                module_config = module_configs[module_name]
                if module_config.get('enabled', False):
                    try:
                        module = module_class(module_config)
                        modules.append(module)
                        self.logger.info(f"Initialized module: {module_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize module {module_name}: {e}")
                        self.execution_stats['errors'].append(f"Module init error {module_name}: {e}")
                else:
                    self.logger.info(f"Module {module_name} is disabled")
            else:
                self.logger.warning(f"No configuration found for module: {module_name}")
        
        return modules
    
    def run_pipeline(self, input_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline
        
        Args:
            input_data: Optional input DataFrame (if None, loads from configured source)
            
        Returns:
            DataFrame with all engineered features
        """
        self.execution_stats['start_time'] = datetime.now()
        self.logger.info("Starting feature engineering pipeline")
        
        try:
            # STEP 1: Load and validate data
            df = self._load_and_validate_data(input_data)
            
            # STEP 2: Run feature modules
            df = self._run_feature_modules(df)
            
            # STEP 3: Validate final features
            df = self._validate_final_features(df)
            
            # STEP 4: Generate report
            self._generate_final_report(df)
            
            # Update execution stats
            self.execution_stats['end_time'] = datetime.now()
            self.execution_stats['total_duration'] = (
                self.execution_stats['end_time'] - self.execution_stats['start_time']
            ).total_seconds()
            
            self.logger.info(
                f"Feature engineering pipeline completed in "
                f"{self.execution_stats['total_duration']:.2f} seconds"
            )
            
            return df
            
        except Exception as e:
            self.execution_stats['errors'].append(f"Pipeline error: {str(e)}")
            self.execution_stats['end_time'] = datetime.now()
            self.logger.error(f"Pipeline failed: {e}")
            raise
    
    def _load_and_validate_data(self, input_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Load and validate input data"""
        self.logger.info("Loading and validating data")
        
        # Load data
        if input_data is not None:
            df = self.data_loader.process(input_data)
        else:
            df = self.data_loader.process()
        
        # Validate data quality
        validation_results = self.data_validator.validate_data_quality(df)
        
        if not validation_results['passed']:
            error_msg = f"Data validation failed: {validation_results['errors']}"
            self.execution_stats['errors'].append(error_msg)
            if settings.strict_mode:
                raise ValueError(error_msg)
            else:
                self.logger.warning(error_msg)
        
        # Add warnings
        if validation_results['warnings']:
            self.execution_stats['warnings'].extend(validation_results['warnings'])
        
        # Get data summary
        data_summary = self.data_loader.get_data_summary(df)
        self.logger.info(f"Data loaded: {data_summary['total_records']} records, {data_summary['columns']['total']} columns")
        
        return df
    
    def _run_feature_modules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run all feature engineering modules"""
        self.logger.info(f"Running {len(self.modules)} feature modules")
        
        # FIRST: Apply advanced lag features (critical for forecasting)
        self.logger.info("Applying advanced lag features...")
        try:
            df = run_advanced_lag_pipeline(df)
            self.execution_stats['total_features_created'] += 20  # Approximate lag features count
            self.logger.info("Advanced lag features applied successfully")
        except Exception as e:
            error_msg = f"Advanced lag features failed: {str(e)}"
            self.execution_stats['errors'].append(error_msg)
            self.logger.error(error_msg)
        
        # THEN: Run other feature modules
        for module in self.modules:
            try:
                self.logger.info(f"Executing module: {module.module_name}")
                
                # Execute module
                df = module.execute(df)
                
                # Track execution
                module_stats = module.get_stats()
                self.execution_stats['modules_executed'].append({
                    'module': module.module_name,
                    'features_created': module_stats['features_created'],
                    'duration_seconds': module_stats['duration_seconds'],
                    'errors': module_stats['errors'],
                    'warnings': module_stats['warnings']
                })
                
                self.execution_stats['total_features_created'] += module_stats['features_created']
                
                # Add any module errors/warnings to global tracking
                if module_stats['errors']:
                    self.execution_stats['errors'].extend(
                        [f"{module.module_name}: {error}" for error in module_stats['errors']]
                    )
                
                if module_stats['warnings']:
                    self.execution_stats['warnings'].extend(
                        [f"{module.module_name}: {warning}" for warning in module_stats['warnings']]
                    )
                
                self.logger.info(
                    f"Module {module.module_name} completed: "
                    f"{module_stats['features_created']} features, "
                    f"{module_stats['duration_seconds']:.2f}s"
                )
                
            except Exception as e:
                error_msg = f"Module {module.module_name} failed: {str(e)}"
                self.execution_stats['errors'].append(error_msg)
                self.logger.error(error_msg)
                
                if settings.strict_mode:
                    raise
                else:
                    self.logger.warning(f"Continuing pipeline without {module.module_name}")
        
        return df
    
    def _validate_final_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate all created features"""
        self.logger.info("Validating final features")
        
        # Get all feature names created by modules
        all_features = []
        for module in self.modules:
            all_features.extend(module.get_feature_names())
        
        # Validate features
        validation_results = self.feature_validator.validate_features(df, all_features)
        
        if not validation_results['passed']:
            error_msg = f"Feature validation failed: {validation_results['errors']}"
            self.execution_stats['errors'].append(error_msg)
            if settings.strict_mode:
                raise ValueError(error_msg)
            else:
                self.logger.warning(error_msg)
        
        # Add feature validation warnings
        if validation_results['warnings']:
            self.execution_stats['warnings'].extend(validation_results['warnings'])
        
        return df
    
    def _generate_final_report(self, df: pd.DataFrame) -> None:
        """Generate final pipeline report"""
        if not settings.reporting_enabled:
            return
        
        self.logger.info("Generating final report")
        
        try:
            # Generate comprehensive report
            report_data = {
                'execution_stats': self.execution_stats,
                'data_summary': self.data_loader.get_data_summary(df),
                'module_results': []
            }
            
            # Add module-specific results
            for module in self.modules:
                module_data = {
                    'module_name': module.module_name,
                    'features_created': module.get_feature_names(),
                    'feature_metadata': module.get_feature_metadata(),
                    'stats': module.get_stats()
                }
                
                # Add module-specific analysis if available
                if hasattr(module, 'get_peak_statistics'):
                    module_data['peak_statistics'] = module.get_peak_statistics(df)
                
                report_data['module_results'].append(module_data)
            
            # Generate report
            self.reporter.generate_report(df, report_data)
            
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            self.execution_stats['warnings'].append(f"Report generation failed: {e}")
    
    def run_single_module(self, module_name: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run a single feature module
        
        Args:
            module_name: Name of the module to run
            df: Input DataFrame
            
        Returns:
            DataFrame with features from that module
        """
        module = next((m for m in self.modules if m.module_name == module_name), None)
        
        if module is None:
            raise ValueError(f"Module '{module_name}' not found or not enabled")
        
        self.logger.info(f"Running single module: {module_name}")
        return module.execute(df)
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get information about all modules"""
        module_info = {}
        
        for module in self.modules:
            module_info[module.module_name] = {
                'enabled': module.enabled,
                'features': module.get_feature_names(),
                'config': module.config
            }
        
        return module_info
    
    def save_features(self, df: pd.DataFrame, output_path: str, 
                     include_original: bool = True) -> None:
        """
        Save engineered features to file
        
        Args:
            df: DataFrame with features
            output_path: Path to save the features
            include_original: Whether to include original columns
        """
        if not include_original:
            # Only save engineered features
            original_features = set()
            engineered_features = set()
            
            for module in self.modules:
                engineered_features.update(module.get_feature_names())
            
            # Include basic time features from data loader
            engineered_features.update(self.data_loader.get_feature_names())
            
            # Save only engineered features
            feature_df = df[list(engineered_features)]
        else:
            feature_df = df
        
        # Save based on file extension
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.csv':
            feature_df.to_csv(output_path, index=True)
        elif output_path.suffix.lower() in ['.pkl', '.pickle']:
            feature_df.to_pickle(output_path)
        elif output_path.suffix.lower() == '.parquet':
            feature_df.to_parquet(output_path, index=True)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")
        
        self.logger.info(f"Features saved to {output_path}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution"""
        return {
            'execution_stats': self.execution_stats,
            'modules_info': self.get_module_info(),
            'total_features': self.execution_stats['total_features_created'],
            'success': len(self.execution_stats['errors']) == 0
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics for the pipeline"""
        return self.execution_stats
