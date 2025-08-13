"""
Phase 2.6.2: Cross-Validation Strategy for Delhi Load Forecasting
================================================================

Walk-Forward Time Series Cross-Validation with Seasonal Stratification
Designed specifically for Delhi's load patterns and weather dependencies

Author: SIH 2024 Team
Date: January 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
import json
import os
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CrossValidationConfig:
    """Configuration for Delhi-specific cross-validation strategy."""
    
    # Walk-Forward Parameters
    training_window_months: int = 12  # 12 months rolling training window
    validation_window_months: int = 1  # 1 month ahead validation
    step_size_months: int = 1         # 1 month forward movement
    total_cv_folds: int = 24          # 24 folds for 2 years of validation
    
    # Seasonal Stratification
    summer_months: List[str] = None
    winter_months: List[str] = None
    monsoon_months: List[str] = None
    transition_months: List[str] = None
    
    # Peak Period Emphasis
    peak_hour_weight: float = 3.0     # 3x weight for peak hours
    peak_summer_hours: List[int] = None  # 14:00-17:00, 20:00-23:00
    peak_winter_hours: List[int] = None  # 19:00-22:00
    
    # Validation Requirements
    min_samples_per_fold: int = 720   # Minimum 30 days * 24 hours
    weather_extreme_inclusion: bool = True
    load_component_validation: bool = True
    
    def __post_init__(self):
        if self.summer_months is None:
            self.summer_months = ['April', 'May', 'June', 'July', 'August', 'September']
        if self.winter_months is None:
            self.winter_months = ['November', 'December', 'January', 'February']
        if self.monsoon_months is None:
            self.monsoon_months = ['June', 'July', 'August', 'September']
        if self.transition_months is None:
            self.transition_months = ['March', 'October']
        if self.peak_summer_hours is None:
            self.peak_summer_hours = list(range(14, 18)) + list(range(20, 24))
        if self.peak_winter_hours is None:
            self.peak_winter_hours = list(range(19, 23))

class DelhiTimeSeriesCV:
    """
    Time series cross-validation specifically designed for Delhi load forecasting.
    Implements walk-forward validation with seasonal awareness.
    """
    
    def __init__(self, config: CrossValidationConfig = None):
        self.config = config or CrossValidationConfig()
        self.cv_results = []
        self.seasonal_performance = {}
        self.validation_summary = {}
    
    def create_time_splits(self, data: pd.DataFrame, 
                          date_column: str = 'datetime') -> List[Dict]:
        """
        Create walk-forward time splits with seasonal stratification.
        
        Args:
            data: DataFrame with datetime index or column
            date_column: Name of datetime column
            
        Returns:
            List of dictionaries with train/validation split information
        """
        print("🔄 Creating walk-forward time splits...")
        
        # Ensure datetime index
        if date_column in data.columns:
            data = data.set_index(date_column)
        
        # Sort by datetime
        data = data.sort_index()
        
        # Get date range
        start_date = data.index.min()
        end_date = data.index.max()
        
        print(f"📅 Data range: {start_date.date()} to {end_date.date()}")
        
        splits = []
        current_date = start_date + pd.DateOffset(months=self.config.training_window_months)
        
        fold_number = 1
        while current_date <= end_date and fold_number <= self.config.total_cv_folds:
            # Define training period
            train_start = current_date - pd.DateOffset(months=self.config.training_window_months)
            train_end = current_date - pd.Timedelta(days=1)
            
            # Define validation period  
            val_start = current_date
            val_end = current_date + pd.DateOffset(months=self.config.validation_window_months) - pd.Timedelta(days=1)
            
            # Ensure validation period doesn't exceed data range
            val_end = min(val_end, end_date)
            
            # Extract data for this split
            train_data = data[(data.index >= train_start) & (data.index <= train_end)]
            val_data = data[(data.index >= val_start) & (data.index <= val_end)]
            
            # Check minimum sample requirements
            if len(train_data) < self.config.min_samples_per_fold or len(val_data) < 24:
                current_date += pd.DateOffset(months=self.config.step_size_months)
                continue
            
            # Determine validation season
            val_season = self._determine_season(val_start.month)
            
            # Calculate seasonal coverage in training
            train_seasonal_coverage = self._calculate_seasonal_coverage(train_data)
            
            # Check for weather extremes in validation period
            weather_extremes = self._detect_weather_extremes(val_data)
            
            split_info = {
                'fold_number': fold_number,
                'train_start': train_start,
                'train_end': train_end,
                'val_start': val_start,
                'val_end': val_end,
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'validation_season': val_season,
                'train_seasonal_coverage': train_seasonal_coverage,
                'weather_extremes': weather_extremes,
                'train_indices': train_data.index,
                'val_indices': val_data.index
            }
            
            splits.append(split_info)
            
            # Move to next fold
            current_date += pd.DateOffset(months=self.config.step_size_months)
            fold_number += 1
        
        print(f"✅ Created {len(splits)} cross-validation folds")
        return splits
    
    def _determine_season(self, month: int) -> str:
        """Determine season based on month number."""
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
        month_name = month_names[month]
        
        if month_name in self.config.summer_months:
            return 'Summer'
        elif month_name in self.config.winter_months:
            return 'Winter'
        elif month_name in self.config.monsoon_months:
            return 'Monsoon'
        else:
            return 'Transition'
    
    def _calculate_seasonal_coverage(self, data: pd.DataFrame) -> Dict:
        """Calculate seasonal coverage in training data."""
        month_counts = data.index.month.value_counts().to_dict()
        
        coverage = {
            'summer_coverage': sum(month_counts.get(m, 0) for m in [4, 5, 6, 7, 8, 9]) / len(data) * 100,
            'winter_coverage': sum(month_counts.get(m, 0) for m in [11, 12, 1, 2]) / len(data) * 100,
            'monsoon_coverage': sum(month_counts.get(m, 0) for m in [6, 7, 8, 9]) / len(data) * 100,
            'transition_coverage': sum(month_counts.get(m, 0) for m in [3, 10]) / len(data) * 100
        }
        
        return coverage
    
    def _detect_weather_extremes(self, data: pd.DataFrame) -> Dict:
        """Detect weather extremes in the validation period."""
        extremes = {
            'heat_wave_days': 0,
            'cold_wave_days': 0,
            'high_humidity_days': 0,
            'extreme_weather_present': False
        }
        
        # Check if weather columns exist
        temp_col = None
        humidity_col = None
        
        for col in data.columns:
            if 'temp' in col.lower() and temp_col is None:
                temp_col = col
            if 'humidity' in col.lower() and humidity_col is None:
                humidity_col = col
        
        if temp_col is not None:
            # Heat wave: Temperature > 45°C (converted if needed)
            if data[temp_col].max() > 45 or data[temp_col].max() > 318:  # Kelvin check
                extremes['heat_wave_days'] = (data[temp_col] > 45).sum()
            
            # Cold wave: Temperature < 5°C
            if data[temp_col].min() < 5 or data[temp_col].min() < 278:  # Kelvin check
                extremes['cold_wave_days'] = (data[temp_col] < 5).sum()
        
        if humidity_col is not None:
            # High humidity: > 90%
            extremes['high_humidity_days'] = (data[humidity_col] > 90).sum()
        
        # Mark if any extreme weather present
        extremes['extreme_weather_present'] = any([
            extremes['heat_wave_days'] > 0,
            extremes['cold_wave_days'] > 0,
            extremes['high_humidity_days'] > 0
        ])
        
        return extremes
    
    def create_peak_period_weights(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create sample weights emphasizing peak periods.
        
        Args:
            data: DataFrame with datetime index
            
        Returns:
            Array of weights for each sample
        """
        weights = np.ones(len(data))
        
        for i, timestamp in enumerate(data.index):
            hour = timestamp.hour
            month = timestamp.month
            
            # Determine if this is a peak hour based on season
            is_peak = False
            
            if month in [4, 5, 6, 7, 8, 9]:  # Summer months
                if hour in self.config.peak_summer_hours:
                    is_peak = True
            else:  # Winter months
                if hour in self.config.peak_winter_hours:
                    is_peak = True
            
            if is_peak:
                weights[i] = self.config.peak_hour_weight
        
        return weights
    
    def validate_seasonal_representation(self, splits: List[Dict]) -> Dict:
        """
        Validate that each season is adequately represented across folds.
        
        Args:
            splits: List of cross-validation splits
            
        Returns:
            Dictionary with seasonal representation analysis
        """
        print("🔍 Validating seasonal representation...")
        
        seasonal_counts = {'Summer': 0, 'Winter': 0, 'Monsoon': 0, 'Transition': 0}
        seasonal_coverage = {'Summer': [], 'Winter': [], 'Monsoon': [], 'Transition': []}
        
        for split in splits:
            season = split['validation_season']
            seasonal_counts[season] += 1
            
            # Track training seasonal coverage for each validation season
            coverage = split['train_seasonal_coverage']
            seasonal_coverage[season].append(coverage)
        
        # Calculate average coverage for each validation season
        avg_coverage = {}
        for season, coverages in seasonal_coverage.items():
            if coverages:  # Only if there are validation folds for this season
                avg_coverage[season] = {
                    'summer_in_train': np.mean([c['summer_coverage'] for c in coverages]),
                    'winter_in_train': np.mean([c['winter_coverage'] for c in coverages]),
                    'monsoon_in_train': np.mean([c['monsoon_coverage'] for c in coverages]),
                    'transition_in_train': np.mean([c['transition_coverage'] for c in coverages])
                }
        
        validation_summary = {
            'total_folds': len(splits),
            'seasonal_fold_distribution': seasonal_counts,
            'seasonal_representation_percentage': {k: (v/len(splits)*100) for k, v in seasonal_counts.items()},
            'average_training_coverage_by_validation_season': avg_coverage,
            'weather_extremes_included': sum(1 for s in splits if s['weather_extremes']['extreme_weather_present']),
            'min_validation_samples': min(s['val_samples'] for s in splits),
            'max_validation_samples': max(s['val_samples'] for s in splits),
            'avg_validation_samples': np.mean([s['val_samples'] for s in splits])
        }
        
        print(f"✅ Seasonal validation complete:")
        print(f"   Summer folds: {seasonal_counts['Summer']} ({seasonal_counts['Summer']/len(splits)*100:.1f}%)")
        print(f"   Winter folds: {seasonal_counts['Winter']} ({seasonal_counts['Winter']/len(splits)*100:.1f}%)")
        print(f"   Monsoon folds: {seasonal_counts['Monsoon']} ({seasonal_counts['Monsoon']/len(splits)*100:.1f}%)")
        print(f"   Transition folds: {seasonal_counts['Transition']} ({seasonal_counts['Transition']/len(splits)*100:.1f}%)")
        print(f"   Extreme weather folds: {validation_summary['weather_extremes_included']}")
        
        return validation_summary
    
    def create_multi_horizon_splits(self, data: pd.DataFrame, 
                                  horizons: List[int] = [1, 6, 24, 168]) -> Dict:
        """
        Create validation splits for multiple forecasting horizons.
        
        Args:
            data: Input data with datetime index
            horizons: List of forecast horizons in hours
            
        Returns:
            Dictionary with splits for each horizon
        """
        print(f"🔄 Creating multi-horizon validation splits for horizons: {horizons}")
        
        multi_horizon_splits = {}
        
        for horizon in horizons:
            print(f"   Creating splits for {horizon}-hour horizon...")
            
            # Adjust validation window based on horizon
            horizon_config = CrossValidationConfig(
                training_window_months=self.config.training_window_months,
                validation_window_months=max(1, horizon // (24 * 7)),  # At least 1 month
                step_size_months=self.config.step_size_months,
                total_cv_folds=self.config.total_cv_folds
            )
            
            # Create CV instance for this horizon
            horizon_cv = DelhiTimeSeriesCV(horizon_config)
            horizon_splits = horizon_cv.create_time_splits(data)
            
            # Add horizon-specific information
            for split in horizon_splits:
                split['forecast_horizon_hours'] = horizon
                split['horizon_name'] = f"{horizon}h"
            
            multi_horizon_splits[f"{horizon}h"] = horizon_splits
        
        return multi_horizon_splits
    
    def generate_cv_iterator(self, data: pd.DataFrame) -> Generator:
        """
        Generate cross-validation iterator for model training.
        
        Args:
            data: Input data with datetime index
            
        Yields:
            Tuple of (train_indices, validation_indices) for each fold
        """
        splits = self.create_time_splits(data)
        
        for split in splits:
            yield split['train_indices'], split['val_indices'], split
    
    def save_cv_configuration(self, splits: List[Dict], 
                            validation_summary: Dict,
                            output_dir: str = 'outputs'):
        """Save cross-validation configuration and splits."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save CV configuration
        config_dict = {
            'training_window_months': self.config.training_window_months,
            'validation_window_months': self.config.validation_window_months,
            'step_size_months': self.config.step_size_months,
            'total_cv_folds': self.config.total_cv_folds,
            'peak_hour_weight': self.config.peak_hour_weight,
            'min_samples_per_fold': self.config.min_samples_per_fold,
            'summer_months': self.config.summer_months,
            'winter_months': self.config.winter_months,
            'monsoon_months': self.config.monsoon_months,
            'transition_months': self.config.transition_months,
            'peak_summer_hours': self.config.peak_summer_hours,
            'peak_winter_hours': self.config.peak_winter_hours
        }
        
        with open(f'{output_dir}/cross_validation_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save splits information (without actual data indices)
        splits_info = []
        for split in splits:
            split_info = {
                'fold_number': split['fold_number'],
                'train_start': split['train_start'].isoformat(),
                'train_end': split['train_end'].isoformat(),
                'val_start': split['val_start'].isoformat(),
                'val_end': split['val_end'].isoformat(),
                'train_samples': split['train_samples'],
                'val_samples': split['val_samples'],
                'validation_season': split['validation_season'],
                'train_seasonal_coverage': split['train_seasonal_coverage'],
                'weather_extremes': split['weather_extremes']
            }
            splits_info.append(split_info)
        
        with open(f'{output_dir}/cv_splits_info.json', 'w') as f:
            json.dump(splits_info, f, indent=2, default=str)
        
        # Save validation summary
        with open(f'{output_dir}/cv_validation_summary.json', 'w') as f:
            json.dump(validation_summary, f, indent=2, default=str)
        
        print("✅ Cross-validation configuration saved successfully!")
        print(f"📄 Files saved in: {output_dir}/")
        return True

def main():
    """
    Execute cross-validation strategy design and testing.
    """
    print("🚀 Starting Phase 2.6.2: Cross-Validation Strategy Design")
    print("=" * 70)
    
    # Create sample dataset for testing (replace with actual data loading)
    print("📊 Creating sample dataset for CV strategy testing...")
    
    # Generate sample datetime range (2022-2023 for testing)
    date_range = pd.date_range(start='2022-01-01', end='2023-12-31', freq='H')
    
    # Create sample data structure
    sample_data = pd.DataFrame({
        'datetime': date_range,
        'delhi_load': np.random.normal(4000, 1000, len(date_range)),  # Sample load data
        'temp_c': np.random.normal(25, 10, len(date_range)),          # Sample temperature
        'humidity': np.random.normal(60, 20, len(date_range)),        # Sample humidity
        'brpl_load': np.random.normal(1200, 300, len(date_range)),
        'bypl_load': np.random.normal(1000, 250, len(date_range)),
        'ndpl_load': np.random.normal(1500, 400, len(date_range)),
        'ndmc_load': np.random.normal(200, 50, len(date_range)),
        'mes_load': np.random.normal(100, 25, len(date_range))
    }).set_index('datetime')
    
    print(f"   📅 Sample data range: {sample_data.index.min().date()} to {sample_data.index.max().date()}")
    print(f"   📊 Sample data shape: {sample_data.shape}")
    
    # Initialize cross-validation
    cv_config = CrossValidationConfig()
    delhi_cv = DelhiTimeSeriesCV(cv_config)
    
    # Create time splits
    splits = delhi_cv.create_time_splits(sample_data)
    
    # Validate seasonal representation
    validation_summary = delhi_cv.validate_seasonal_representation(splits)
    
    # Create multi-horizon splits
    multi_horizon_splits = delhi_cv.create_multi_horizon_splits(sample_data)
    
    print(f"\n🔄 Multi-horizon splits created:")
    for horizon, h_splits in multi_horizon_splits.items():
        print(f"   {horizon}: {len(h_splits)} folds")
    
    # Test CV iterator
    print(f"\n🔄 Testing CV iterator...")
    fold_count = 0
    for train_idx, val_idx, split_info in delhi_cv.generate_cv_iterator(sample_data):
        fold_count += 1
        if fold_count <= 3:  # Show first 3 folds
            print(f"   Fold {fold_count}: Train={len(train_idx)}, Val={len(val_idx)}, Season={split_info['validation_season']}")
    
    # Save configuration
    delhi_cv.save_cv_configuration(splits, validation_summary, '../outputs')
    
    print(f"\n✅ Phase 2.6.2 Complete!")
    print("🔄 Walk-forward validation strategy implemented")
    print("🌤️  Seasonal stratification validated")
    print("⚡ Peak period emphasis configured")
    print("🎯 Multi-horizon framework ready")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Cross-Validation Strategy ready for model training!")
    else:
        print("\n❌ Cross-validation setup failed. Please check configuration.")
