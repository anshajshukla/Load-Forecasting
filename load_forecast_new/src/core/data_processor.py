"""
Core Data Processing Module
Handles data preprocessing, cleaning, and validation for the load forecasting system
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Advanced data processor for load forecasting with SIH 2024 enhancements
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataProcessor
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or self._get_default_config()
        self.processed_data = None
        self.scalers = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for data processing"""
        return {
            'missing_value_threshold': 0.1,
            'outlier_detection_method': 'iqr',
            'feature_engineering': True,
            'time_window': 24,  # hours
            'validation_split': 0.2,
            'preprocessing_steps': [
                'clean_missing_values',
                'detect_outliers',
                'normalize_data',
                'engineer_features'
            ]
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                data = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.json':
                data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"✅ Data loaded successfully: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def clean_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean missing values using advanced strategies
        
        Args:
            data: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("🧹 Cleaning missing values...")
        
        # Calculate missing value percentage
        missing_pct = data.isnull().sum() / len(data)
        
        # Drop columns with high missing values
        high_missing_cols = missing_pct[missing_pct > self.config['missing_value_threshold']].index
        if len(high_missing_cols) > 0:
            logger.warning(f"Dropping columns with >{self.config['missing_value_threshold']*100}% missing: {list(high_missing_cols)}")
            data = data.drop(columns=high_missing_cols)
        
        # Fill remaining missing values
        for col in data.columns:
            if data[col].isnull().any():
                if data[col].dtype in ['int64', 'float64']:
                    # Forward fill then backward fill for numerical data
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # Mode for categorical data
                    data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown')
        
        logger.info(f"✅ Missing values cleaned. Shape: {data.shape}")
        return data
    
    def detect_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using IQR method
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("🔍 Detecting outliers...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
            outlier_count = outliers.sum()
            outlier_counts[col] = outlier_count
            
            if outlier_count > 0:
                # Cap outliers instead of removing them
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound
        
        logger.info(f"✅ Outliers detected and capped: {outlier_counts}")
        return data
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer time-based and domain-specific features
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("⚙️ Engineering features...")
        
        # Ensure datetime column exists
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True)
        elif data.index.name != 'timestamp':
            logger.warning("No timestamp column found, using index as timestamp")
        
        # Time-based features
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for time features
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        
        # Lag features for load columns
        load_columns = [col for col in data.columns if 'load' in col.lower() or 'demand' in col.lower()]
        for col in load_columns:
            if col in data.columns:
                for lag in [1, 2, 24, 168]:  # 1h, 2h, 1day, 1week
                    data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        # Rolling statistics
        for col in load_columns:
            if col in data.columns:
                data[f'{col}_rolling_mean_24h'] = data[col].rolling(window=24).mean()
                data[f'{col}_rolling_std_24h'] = data[col].rolling(window=24).std()
        
        logger.info(f"✅ Features engineered. New shape: {data.shape}")
        return data
    
    def normalize_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (normalized_data, scalers_dict)
        """
        from sklearn.preprocessing import StandardScaler
        
        logger.info("📊 Normalizing data...")
        
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        scalers = {}
        
        for col in numerical_cols:
            scaler = StandardScaler()
            data[col] = scaler.fit_transform(data[[col]])
            scalers[col] = scaler
        
        logger.info(f"✅ Data normalized for {len(numerical_cols)} columns")
        return data, scalers
    
    def process_data(self, file_path: str) -> pd.DataFrame:
        """
        Complete data processing pipeline
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Processed DataFrame ready for modeling
        """
        logger.info("🚀 Starting data processing pipeline...")
        
        # Load data
        data = self.load_data(file_path)
        
        # Execute processing steps
        for step in self.config['preprocessing_steps']:
            if hasattr(self, step):
                if step == 'normalize_data':
                    data, self.scalers = getattr(self, step)(data)
                else:
                    data = getattr(self, step)(data)
            else:
                logger.warning(f"⚠️ Unknown processing step: {step}")
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        self.processed_data = data
        logger.info(f"✅ Data processing completed. Final shape: {data.shape}")
        
        return data
    
    def get_feature_info(self) -> Dict:
        """
        Get information about processed features
        
        Returns:
            Dictionary with feature information
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        return {
            "total_features": len(self.processed_data.columns),
            "numerical_features": len(self.processed_data.select_dtypes(include=[np.number]).columns),
            "categorical_features": len(self.processed_data.select_dtypes(include=['object']).columns),
            "data_shape": self.processed_data.shape,
            "date_range": {
                "start": str(self.processed_data.index.min()),
                "end": str(self.processed_data.index.max())
            },
            "feature_names": list(self.processed_data.columns)
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize processor
    processor = DataProcessor()
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'load_demand': np.random.normal(1000, 200, 100),
        'temperature': np.random.normal(25, 5, 100),
        'humidity': np.random.normal(60, 10, 100)
    })
    
    # Save sample data
    sample_data.to_csv('sample_data.csv', index=False)
    
    # Process data
    processed = processor.process_data('sample_data.csv')
    
    # Get feature info
    info = processor.get_feature_info()
    print("📊 Feature Information:")
    for key, value in info.items():
        print(f"   {key}: {value}")
