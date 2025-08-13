"""
Advanced Lag Features Module
Delhi Load Forecasting - Enhanced Pipeline

Creates comprehensive lag features for time series forecasting:
- Load lag features (1h, 24h, 168h)
- Weather lag features
- Utility-specific lag features
- Rolling statistics and trends
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive lag features for load forecasting
    
    Args:
        df: Input DataFrame with datetime sorted
        
    Returns:
        DataFrame with lag features added
    """
    print("🔧 Creating lag features...")
    
    # Ensure data is sorted by datetime
    if 'datetime' in df.columns:
        df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. LOAD LAG FEATURES
    if 'delhi_load' in df.columns:
        df['delhi_load_lag_1h'] = df['delhi_load'].shift(1)     # Previous hour
        df['delhi_load_lag_24h'] = df['delhi_load'].shift(24)   # Same hour yesterday
        df['delhi_load_lag_168h'] = df['delhi_load'].shift(168) # Same hour last week
        df['delhi_load_lag_336h'] = df['delhi_load'].shift(336) # Same hour 2 weeks ago
    
    # 2. TEMPERATURE LAG FEATURES
    if 'temperature_2m (°C)' in df.columns:
        df['temp_lag_1h'] = df['temperature_2m (°C)'].shift(1)
        df['temp_lag_24h'] = df['temperature_2m (°C)'].shift(24)
        df['temp_lag_168h'] = df['temperature_2m (°C)'].shift(168)
    
    # 3. UTILITY LAG FEATURES
    utility_cols = ['brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
    for col in utility_cols:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_24h'] = df[col].shift(24)
            df[f'{col}_lag_168h'] = df[col].shift(168)
    
    # 4. WEATHER LAG FEATURES
    weather_cols = ['relative_humidity_2m (%)', 'wind_speed_10m (km/h)', 'apparent_temperature (°C)']
    for col in weather_cols:
        if col in df.columns:
            df[f'{col.split()[0]}_lag_1h'] = df[col].shift(1)
            df[f'{col.split()[0]}_lag_24h'] = df[col].shift(24)
    
    # 5. SOLAR LAG FEATURES
    if 'solar_generation_mw' in df.columns:
        df['solar_lag_1h'] = df['solar_generation_mw'].shift(1)
        df['solar_lag_24h'] = df['solar_generation_mw'].shift(24)
    
    # 6. NET LOAD LAG FEATURES
    if 'net_load_mw' in df.columns:
        df['net_load_lag_1h'] = df['net_load_mw'].shift(1)
        df['net_load_lag_24h'] = df['net_load_mw'].shift(24)
    
    print(f"✅ Created lag features - Dataset shape: {df.shape}")
    return df


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling statistics features
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with rolling features added
    """
    print("📈 Creating rolling statistics...")
    
    # 1. DELHI LOAD ROLLING FEATURES
    if 'delhi_load' in df.columns:
        # Moving averages
        df['delhi_load_ma_3h'] = df['delhi_load'].rolling(3, min_periods=1).mean()
        df['delhi_load_ma_6h'] = df['delhi_load'].rolling(6, min_periods=1).mean()
        df['delhi_load_ma_24h'] = df['delhi_load'].rolling(24, min_periods=1).mean()
        df['delhi_load_ma_168h'] = df['delhi_load'].rolling(168, min_periods=1).mean()
        
        # Standard deviations
        df['delhi_load_std_24h'] = df['delhi_load'].rolling(24, min_periods=1).std()
        df['delhi_load_std_168h'] = df['delhi_load'].rolling(168, min_periods=1).std()
        
        # Min/Max rolling
        df['delhi_load_min_24h'] = df['delhi_load'].rolling(24, min_periods=1).min()
        df['delhi_load_max_24h'] = df['delhi_load'].rolling(24, min_periods=1).max()
        
        # Rolling range
        df['delhi_load_range_24h'] = df['delhi_load_max_24h'] - df['delhi_load_min_24h']
    
    # 2. TEMPERATURE ROLLING FEATURES
    if 'temperature_2m (°C)' in df.columns:
        df['temp_ma_24h'] = df['temperature_2m (°C)'].rolling(24, min_periods=1).mean()
        df['temp_ma_168h'] = df['temperature_2m (°C)'].rolling(168, min_periods=1).mean()
        df['temp_std_24h'] = df['temperature_2m (°C)'].rolling(24, min_periods=1).std()
        df['temp_min_24h'] = df['temperature_2m (°C)'].rolling(24, min_periods=1).min()
        df['temp_max_24h'] = df['temperature_2m (°C)'].rolling(24, min_periods=1).max()
        df['temp_range_24h'] = df['temp_max_24h'] - df['temp_min_24h']
    
    # 3. UTILITY ROLLING FEATURES
    utility_cols = ['brpl_load', 'bypl_load', 'ndpl_load']
    for col in utility_cols:
        if col in df.columns:
            df[f'{col}_ma_24h'] = df[col].rolling(24, min_periods=1).mean()
            df[f'{col}_std_24h'] = df[col].rolling(24, min_periods=1).std()
    
    # 4. SOLAR ROLLING FEATURES
    if 'solar_generation_mw' in df.columns:
        df['solar_ma_24h'] = df['solar_generation_mw'].rolling(24, min_periods=1).mean()
        df['solar_max_24h'] = df['solar_generation_mw'].rolling(24, min_periods=1).max()
    
    print(f"✅ Created rolling features - Dataset shape: {df.shape}")
    return df


def create_difference_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create difference features for trend analysis
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with difference features
    """
    print("⚡ Creating difference features...")
    
    # Load differences
    if all(col in df.columns for col in ['delhi_load', 'delhi_load_lag_1h', 'delhi_load_lag_24h']):
        df['load_diff_1h'] = df['delhi_load'] - df['delhi_load_lag_1h']
        df['load_diff_24h'] = df['delhi_load'] - df['delhi_load_lag_24h']
        df['load_diff_168h'] = df['delhi_load'] - df['delhi_load_lag_168h']
        
        # Percentage changes
        df['load_pct_change_1h'] = df['load_diff_1h'] / df['delhi_load_lag_1h'] * 100
        df['load_pct_change_24h'] = df['load_diff_24h'] / df['delhi_load_lag_24h'] * 100
    
    # Temperature differences
    if all(col in df.columns for col in ['temperature_2m (°C)', 'temp_lag_1h', 'temp_lag_24h']):
        df['temp_diff_1h'] = df['temperature_2m (°C)'] - df['temp_lag_1h']
        df['temp_diff_24h'] = df['temperature_2m (°C)'] - df['temp_lag_24h']
    
    # Solar differences
    if all(col in df.columns for col in ['solar_generation_mw', 'solar_lag_1h']):
        df['solar_diff_1h'] = df['solar_generation_mw'] - df['solar_lag_1h']
        df['solar_ramp_rate'] = df['solar_diff_1h']  # Solar ramp rate
    
    print(f"✅ Created difference features - Dataset shape: {df.shape}")
    return df


def run_advanced_lag_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run complete advanced lag features pipeline
    
    Args:
        df: Input DataFrame
        
    Returns:
        Enhanced DataFrame with all lag features
    """
    print("🚀 Starting Advanced Lag Features Pipeline...")
    
    # Create features in sequence
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_difference_features(df)
    
    print(f"✅ Advanced lag pipeline complete - Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Test the module
    print("Testing Advanced Lag Features Module...")
    
    # Load data
    df = pd.read_csv('../optimized_delhi_load_dataset.csv', parse_dates=['datetime'])
    print(f"Original shape: {df.shape}")
    
    # Run pipeline
    enhanced_df = run_advanced_lag_pipeline(df)
    
    # Save results
    enhanced_df.to_csv('../lag_enhanced_dataset.csv', index=False)
    print(f"Enhanced dataset saved with shape: {enhanced_df.shape}")
