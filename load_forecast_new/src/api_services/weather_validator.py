"""
Data Validator for Weather and Solar Data
Comprehensive validation before database insertion.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

class DataValidator:
    """Validates weather and solar data quality before database insertion."""
    
    def __init__(self):
        """Initialize data validator."""
        self.logger = logging.getLogger(__name__)
        
        # Delhi-specific validation ranges
        self.delhi_weather_ranges = {
            'temperature': (-5, 50),  # Delhi temperature range (°C)
            'humidity': (10, 100),    # Humidity percentage
            'precipitation_mm': (0, 200),  # Max daily rainfall (mm)
            'wind_speed_kmh': (0, 80),     # Wind speed (km/h)
            'atmospheric_pressure_hpa': (980, 1040),  # Pressure range
            'cloud_cover_percent': (0, 100),
            'visibility_km': (0.1, 20),
            'uv_index': (0, 14),
            'dew_point': (-10, 35)
        }
        
        self.delhi_solar_ranges = {
            'solar_ghi': (0, 1200),        # Global Horizontal Irradiance (W/m²)
            'solar_dni': (0, 1000),        # Direct Normal Irradiance (W/m²)
            'solar_dhi': (0, 500),         # Diffuse Horizontal Irradiance (W/m²)
            'solar_zenith_angle': (0, 90), # Solar zenith angle (degrees)
            'uv_index': (0, 14),
            'solar_potential_kwh': (0, 10), # Daily solar potential (kWh)
            'delhi_solar_capacity_factor': (0, 1),
            'duck_curve_impact': (0, 1),
            'net_load_factor': (0, 2)
        }
        
        # Required columns for different data types
        self.required_weather_columns = [
            'timestamp', 'temperature', 'humidity', 'weather_condition'
        ]
        
        self.required_solar_columns = [
            'timestamp', 'solar_ghi'
        ]
        
        self.logger.info("🔍 Data Validator initialized with Delhi-specific validation rules")
    
    def validate_weather_data(self, data_path: str) -> Dict:
        """Validate weather data from CSV file."""
        try:
            # Load data
            df = pd.read_csv(data_path)
            validation_result = {
                'filepath': data_path,
                'data_type': 'weather',
                'is_valid': True,
                'record_count': len(df),
                'validation_timestamp': datetime.now(),
                'issues': [],
                'quality_score': 1.0,
                'quality_metrics': {}
            }
            
            # Check 1: Required columns
            missing_columns = [col for col in self.required_weather_columns if col not in df.columns]
            if missing_columns:
                validation_result['issues'].append(f"Missing required columns: {missing_columns}")
                validation_result['is_valid'] = False
                validation_result['quality_score'] -= 0.3
            
            # Check 2: Data type validation
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except:
                    validation_result['issues'].append("Invalid timestamp format")
                    validation_result['quality_score'] -= 0.2
            
            # Check 3: Data range validation
            range_issues = 0
            for column, (min_val, max_val) in self.delhi_weather_ranges.items():
                if column in df.columns:
                    out_of_range = df[(df[column] < min_val) | (df[column] > max_val)][column].count()
                    if out_of_range > 0:
                        range_issues += 1
                        validation_result['issues'].append(
                            f"{column}: {out_of_range} values out of range ({min_val}-{max_val})"
                        )
            
            if range_issues > 3:  # More than 3 columns with range issues
                validation_result['quality_score'] -= 0.2
            
            # Check 4: Missing values
            missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
            validation_result['quality_metrics']['missing_data_percentage'] = missing_percentage
            
            if missing_percentage > 20:
                validation_result['issues'].append(f"High missing data: {missing_percentage:.1f}%")
                validation_result['quality_score'] -= 0.2
            elif missing_percentage > 10:
                validation_result['quality_score'] -= 0.1
            
            # Final quality score
            validation_result['quality_score'] = max(0, validation_result['quality_score'])
            
            if validation_result['quality_score'] < 0.6:
                validation_result['is_valid'] = False
                validation_result['issues'].append("Overall quality score too low")
            
            self.logger.info(f"🔍 Weather validation: {validation_result['quality_score']:.2f} quality score")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Weather validation failed: {str(e)}")
            return {
                'filepath': data_path,
                'data_type': 'weather',
                'is_valid': False,
                'record_count': 0,
                'validation_timestamp': datetime.now(),
                'issues': [f"Validation error: {str(e)}"],
                'quality_score': 0.0,
                'quality_metrics': {}
            }
    
    def validate_file(self, filepath: str, data_type: str = None) -> Dict:
        """Validate file based on data type."""
        if data_type is None:
            # Try to infer data type from filename
            filename = filepath.lower()
            if 'weather' in filename:
                data_type = 'weather'
            elif 'solar' in filename:
                data_type = 'solar'
            else:
                data_type = 'weather'  # Default
        
        if data_type == 'weather':
            return self.validate_weather_data(filepath)
        else:
            raise ValueError(f"Data type {data_type} not yet implemented")
    
    def get_validation_summary(self, validation_results: List[Dict]) -> Dict:
        """Get summary of multiple validation results."""
        if not validation_results:
            return {}
        
        total_files = len(validation_results)
        valid_files = sum(1 for result in validation_results if result['is_valid'])
        avg_quality = sum(result['quality_score'] for result in validation_results) / total_files
        total_records = sum(result['record_count'] for result in validation_results)
        
        summary = {
            'total_files': total_files,
            'valid_files': valid_files,
            'validation_rate': valid_files / total_files * 100,
            'average_quality_score': avg_quality,
            'total_records': total_records
        }
        
        return summary
