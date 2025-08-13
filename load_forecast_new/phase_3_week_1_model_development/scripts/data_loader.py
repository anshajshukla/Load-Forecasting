"""
Data Loading and Preprocessing Module for Delhi Load Forecasting
Phase 3 Week 1: Baseline Model Development
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DelhiDataLoader:
    """
    Comprehensive data loading and preprocessing for Delhi Load Forecasting.
    """
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        """
        Initialize the data loader with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.scalers = {}
        self.feature_names = []
        
        # Create necessary directories
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
        """Create necessary directories for data processing."""
        directories = ['data', 'models/scalers', 'evaluation/figures', 'logs']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            DataFrame with loaded data
        """
        # Try to load from multiple possible locations
        possible_paths = [
            data_path,
            "../data/delhi_interaction_enhanced_cleaned.csv",
            "../phase_2_5_3_outputs/delhi_selected_features.csv",
            "../data_preprocessing/phase 2/delhi_interaction_enhanced.csv",
            "../final_dataset.csv"
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                logger.info(f"Loading data from {path}")
                df = pd.read_csv(path)
                
                # Convert datetime column
                datetime_cols = ['datetime', 'Datetime', 'Date', 'timestamp']
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                        df = df.rename(columns={col: 'datetime'})
                        break
                
                logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
                
                return df
        
        raise FileNotFoundError("No valid data file found. Please check data paths.")
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive time-based features.
        
        Args:
            df: Input DataFrame with datetime column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Basic time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['quarter'] = df['datetime'].dt.quarter
        df['day_of_year'] = df['datetime'].dt.dayofyear
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Peak hour indicators for Delhi
        morning_peak = df['hour'].isin(self.config['delhi_config']['peak_hours']['morning'])
        evening_peak = df['hour'].isin(self.config['delhi_config']['peak_hours']['evening'])
        df['is_peak_hour'] = (morning_peak | evening_peak).astype(int)
        df['is_morning_peak'] = morning_peak.astype(int)
        df['is_evening_peak'] = evening_peak.astype(int)
        
        # Seasonal features (cyclical encoding)
        df['season_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['season_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Daily cycle features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Weekly cycle features
        df['week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Time features created successfully")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'delhi_load') -> pd.DataFrame:
        """
        Create lag features for load forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Target column for creating lags
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Create lag features
        lag_periods = [1, 2, 3, 6, 12, 24, 48, 168]  # 1h, 2h, 3h, 6h, 12h, 1d, 2d, 1w
        
        for lag in lag_periods:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        windows = [3, 6, 12, 24]
        for window in windows:
            df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_roll_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_roll_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_roll_max_{window}'] = df[target_col].rolling(window=window).max()
        
        # Load aggregation features
        load_cols = self.config['data']['target_variables']
        if all(col in df.columns for col in load_cols):
            df['load_sum'] = df[load_cols].sum(axis=1)
            df['load_mean'] = df[load_cols].mean(axis=1)
            df['load_std'] = df[load_cols].std(axis=1)
            df['load_max'] = df[load_cols].max(axis=1)
            df['load_min'] = df[load_cols].min(axis=1)
        
        logger.info("Lag features created successfully")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Log missing values before handling
        missing_before = df.isnull().sum()
        logger.info(f"Missing values before handling: {missing_before[missing_before > 0]}")
        
        # Forward fill for load data (recent values are good predictors)
        load_cols = self.config['data']['target_variables']
        for col in load_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()
        
        # Interpolate for weather data
        weather_cols = self.config['data']['feature_columns']['weather_features']
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear').ffill().bfill()
        
        # Fill remaining missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Log missing values after handling
        missing_after = df.isnull().sum()
        logger.info(f"Missing values after handling: {missing_after[missing_after > 0]}")
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Tuple of (X, y) sequences
        """
        # Get feature columns
        all_features = []
        for feature_group in self.config['data']['feature_columns'].values():
            all_features.extend(feature_group)
        
        # Add any additional engineered features that exist in the dataframe
        additional_features = ['dual_peak_intensity', 'thermal_comfort_index', 
                             'temporal_pattern_score', 'hour_sin', 'hour_cos',
                             'week_sin', 'week_cos']
        for feature in additional_features:
            if feature in df.columns:
                all_features.append(feature)
        
        # Remove duplicates and ensure features exist
        feature_cols = list(set([col for col in all_features if col in df.columns]))
        target_cols = self.config['data']['target_variables']
        
        # Sort by datetime
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Prepare features and targets
        X_data = df[feature_cols].values
        y_data = df[target_cols].values
        
        # Create sequences
        seq_length = self.config['data']['sequence_length']
        pred_horizon = self.config['data']['prediction_horizon']
        
        X_sequences = []
        y_sequences = []
        
        for i in range(len(df) - seq_length - pred_horizon + 1):
            # Input sequence
            X_seq = X_data[i:(i + seq_length)]
            # Target sequence (predict next pred_horizon steps)
            y_seq = y_data[i + seq_length:i + seq_length + pred_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def scale_data(self, X: np.ndarray, y: np.ndarray, fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features and targets.
        
        Args:
            X: Feature sequences
            y: Target sequences
            fit_scalers: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            Tuple of scaled (X, y)
        """
        # Reshape for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])
        y_reshaped = y.reshape(-1, y.shape[-1])
        
        if fit_scalers:
            # Fit scalers
            self.scalers['features'] = StandardScaler()
            self.scalers['targets'] = MinMaxScaler(feature_range=(0, 1))
            
            X_scaled = self.scalers['features'].fit_transform(X_reshaped)
            y_scaled = self.scalers['targets'].fit_transform(y_reshaped)
            
            # Save scalers
            joblib.dump(self.scalers, 'models/scalers/data_scalers.pkl')
            logger.info("Scalers fitted and saved")
        else:
            # Load and use existing scalers
            self.scalers = joblib.load('models/scalers/data_scalers.pkl')
            X_scaled = self.scalers['features'].transform(X_reshaped)
            y_scaled = self.scalers['targets'].transform(y_reshaped)
            logger.info("Existing scalers loaded and applied")
        
        # Reshape back to sequence format
        X_scaled = X_scaled.reshape(X.shape)
        y_scaled = y_scaled.reshape(y.shape)
        
        return X_scaled, y_scaled
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Feature sequences
            y: Target sequences
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['validation_split']
        
        # Calculate split indices
        n_samples = len(X)
        train_idx = int(n_samples * train_split)
        val_idx = int(n_samples * (train_split + val_split))
        
        # Split data (maintaining temporal order)
        X_train = X[:train_idx]
        X_val = X[train_idx:val_idx]
        X_test = X[val_idx:]
        
        y_train = y[:train_idx]
        y_val = y[train_idx:val_idx]
        y_test = y[val_idx:]
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data(self, data_path: str = None) -> Tuple:
        """
        Complete data preparation pipeline.
        
        Args:
            data_path: Optional path to data file
            
        Returns:
            Tuple of prepared data splits
        """
        logger.info("Starting data preparation pipeline...")
        
        # Load data
        df = self.load_data(data_path)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create sequences
        X, y = self.create_sequences(df)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale data
        X_train_scaled, y_train_scaled = self.scale_data(X_train, y_train, fit_scalers=True)
        X_val_scaled, y_val_scaled = self.scale_data(X_val, y_val, fit_scalers=False)
        X_test_scaled, y_test_scaled = self.scale_data(X_test, y_test, fit_scalers=False)
        
        logger.info("Data preparation pipeline completed successfully!")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train_scaled, y_val_scaled, y_test_scaled)
    
    def visualize_data(self, df: pd.DataFrame):
        """
        Create data visualization plots.
        
        Args:
            df: DataFrame to visualize
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Load patterns
        target_cols = self.config['data']['target_variables']
        for col in target_cols:
            if col in df.columns:
                axes[0, 0].plot(df['datetime'][:1000], df[col][:1000], label=col, alpha=0.7)
        axes[0, 0].set_title('Load Patterns (First 1000 Hours)')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Load (MW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Daily load pattern
        if 'delhi_load' in df.columns:
            hourly_avg = df.groupby('hour')['delhi_load'].mean()
            axes[0, 1].bar(hourly_avg.index, hourly_avg.values)
            axes[0, 1].set_title('Average Delhi Load by Hour')
            axes[0, 1].set_xlabel('Hour of Day')
            axes[0, 1].set_ylabel('Average Load (MW)')
            axes[0, 1].grid(True)
        
        # Weather correlation
        weather_cols = ['temperature_2m (°C)', 'relative_humidity_2m (%)']
        if all(col in df.columns for col in weather_cols) and 'delhi_load' in df.columns:
            correlation_data = df[weather_cols + ['delhi_load']].corr()
            sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
            axes[1, 0].set_title('Weather-Load Correlation')
        
        # Missing values
        missing_data = df.isnull().sum().head(10)
        if missing_data.sum() > 0:
            axes[1, 1].bar(range(len(missing_data)), missing_data.values)
            axes[1, 1].set_xticks(range(len(missing_data)))
            axes[1, 1].set_xticklabels(missing_data.index, rotation=45)
            axes[1, 1].set_title('Missing Values by Column')
            axes[1, 1].set_ylabel('Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Missing Values Status')
        
        plt.tight_layout()
        plt.savefig('evaluation/figures/data_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Data visualization saved to evaluation/figures/data_overview.png")

if __name__ == "__main__":
    # Example usage
    data_loader = DelhiDataLoader()
    
    # Prepare data
    data_splits = data_loader.prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    
    print(f"Data preparation completed!")
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data shape: X={X_val.shape}, y={y_val.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Feature names: {data_loader.feature_names}")
