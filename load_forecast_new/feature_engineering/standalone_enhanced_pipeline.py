"""
Standalone Enhanced Feature Engineering Pipeline
Creates all features without external dependencies
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

def create_enhanced_features(df):
    """Create all enhanced features in one place"""
    print("🔧 Creating enhanced features...")
    
    # Ensure datetime is parsed and sorted
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # 1. LAG FEATURES
    print("   Creating lag features...")
    if 'delhi_load' in df.columns:
        df['delhi_load_lag_1h'] = df['delhi_load'].shift(1)
        df['delhi_load_lag_24h'] = df['delhi_load'].shift(24)
        df['delhi_load_lag_168h'] = df['delhi_load'].shift(168)
    
    if 'temperature_2m (°C)' in df.columns:
        df['temp_lag_1h'] = df['temperature_2m (°C)'].shift(1)
        df['temp_lag_24h'] = df['temperature_2m (°C)'].shift(24)
    
    utility_cols = ['brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
    for col in utility_cols:
        if col in df.columns:
            df[f'{col}_lag_1h'] = df[col].shift(1)
            df[f'{col}_lag_24h'] = df[col].shift(24)
    
    # 2. ROLLING STATISTICS
    print("   Creating rolling statistics...")
    if 'delhi_load' in df.columns:
        df['delhi_load_ma_24h'] = df['delhi_load'].rolling(24, min_periods=1).mean()
        df['delhi_load_ma_168h'] = df['delhi_load'].rolling(168, min_periods=1).mean()
        df['delhi_load_std_24h'] = df['delhi_load'].rolling(24, min_periods=1).std()
        df['delhi_load_min_24h'] = df['delhi_load'].rolling(24, min_periods=1).min()
        df['delhi_load_max_24h'] = df['delhi_load'].rolling(24, min_periods=1).max()
    
    if 'temperature_2m (°C)' in df.columns:
        df['temp_ma_24h'] = df['temperature_2m (°C)'].rolling(24, min_periods=1).mean()
        df['temp_std_24h'] = df['temperature_2m (°C)'].rolling(24, min_periods=1).std()
    
    # 3. DIFFERENCE FEATURES
    print("   Creating difference features...")
    if all(col in df.columns for col in ['delhi_load', 'delhi_load_lag_1h', 'delhi_load_lag_24h']):
        df['load_diff_1h'] = df['delhi_load'] - df['delhi_load_lag_1h']
        df['load_diff_24h'] = df['delhi_load'] - df['delhi_load_lag_24h']
        df['load_pct_change_1h'] = (df['load_diff_1h'] / df['delhi_load_lag_1h'] * 100).fillna(0)
    
    # 4. WEATHER INTERACTIONS
    print("   Creating weather interactions...")
    if all(col in df.columns for col in ['temperature_2m (°C)', 'relative_humidity_2m (%)']):
        df['temp_humidity_interaction'] = df['temperature_2m (°C)'] * df['relative_humidity_2m (%)']
    
    if 'apparent_temperature (°C)' in df.columns:
        df['apparent_temp_squared'] = df['apparent_temperature (°C)'] ** 2
    
    if 'temperature_2m (°C)' in df.columns:
        df['cooling_degree_hours'] = np.maximum(df['temperature_2m (°C)'] - 24, 0)
        df['heating_degree_hours'] = np.maximum(18 - df['temperature_2m (°C)'], 0)
    
    # 5. TIME FEATURES (Enhanced)
    print("   Creating enhanced time features...")
    if 'hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    if 'datetime' in df.columns:
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
    
    # 6. PEAK DETECTION
    print("   Creating peak detection features...")
    if 'hour' in df.columns:
        df['is_morning_peak'] = ((df['hour'] >= 9) & (df['hour'] <= 12)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        df['is_peak_period'] = (df['is_morning_peak'] | df['is_evening_peak']).astype(int)
    
    # 7. SOLAR FEATURES
    print("   Creating solar features...")
    if 'solar_generation_mw' in df.columns:
        df['solar_lag_1h'] = df['solar_generation_mw'].shift(1)
        df['solar_ramp_rate'] = df['solar_generation_mw'] - df['solar_lag_1h']
        solar_max = df['solar_generation_mw'].max()
        if solar_max > 0:
            df['solar_capacity_factor'] = df['solar_generation_mw'] / solar_max
    
    # 8. NET LOAD FEATURES
    print("   Creating net load features...")
    if 'net_load_mw' in df.columns:
        df['net_load_lag_1h'] = df['net_load_mw'].shift(1)
        df['net_load_ma_24h'] = df['net_load_mw'].rolling(24, min_periods=1).mean()
        df['duck_curve_severity'] = df['net_load_mw'] - df['net_load_ma_24h']
    
    # 9. UTILITY INTERACTIONS
    print("   Creating utility interaction features...")
    utility_main = ['brpl_load', 'bypl_load', 'ndpl_load']
    if all(col in df.columns for col in utility_main):
        df['major_utilities_total'] = df['brpl_load'] + df['bypl_load'] + df['ndpl_load']
        df['brpl_share'] = df['brpl_load'] / df['major_utilities_total']
        df['bypl_share'] = df['bypl_load'] / df['major_utilities_total']
        df['ndpl_share'] = df['ndpl_load'] / df['major_utilities_total']
    
    # 10. LOAD RELATIVE FEATURES
    print("   Creating relative load features...")
    if 'delhi_load' in df.columns and 'datetime' in df.columns:
        daily_avg = df.groupby(df['datetime'].dt.date)['delhi_load'].transform('mean')
        df['load_relative_to_daily_avg'] = df['delhi_load'] / daily_avg
    
    return df


def run_standalone_feature_engineering():
    """Run the complete feature engineering pipeline"""
    print("🚀 Starting Standalone Enhanced Feature Engineering...")
    
    # Load dataset
    input_path = '../optimized_delhi_load_dataset.csv'
    output_path = '../enhanced_delhi_load_dataset.csv'
    
    try:
        df = pd.read_csv(input_path, parse_dates=['datetime'])
        print(f"📊 Original dataset: {df.shape[0]:,} records × {df.shape[1]} features")
        
        # Create enhanced features
        df_enhanced = create_enhanced_features(df)
        
        print(f"✅ Enhanced dataset: {df_enhanced.shape[0]:,} records × {df_enhanced.shape[1]} features")
        
        # Clean dataset - remove rows with critical NaN values
        critical_features = ['delhi_load_lag_1h', 'delhi_load_lag_24h', 'delhi_load_lag_168h']
        available_critical = [f for f in critical_features if f in df_enhanced.columns]
        
        if available_critical:
            initial_rows = len(df_enhanced)
            df_clean = df_enhanced.dropna(subset=available_critical)
            rows_removed = initial_rows - len(df_clean)
            print(f"🧹 Cleaned dataset: removed {rows_removed:,} rows with missing lag values")
        else:
            df_clean = df_enhanced
        
        # Save enhanced dataset
        df_clean.to_csv(output_path, index=False)
        
        # Feature summary
        original_features = df.shape[1]
        new_features = df_clean.shape[1] - original_features
        
        print(f"\n🎯 FEATURE ENGINEERING COMPLETE!")
        print(f"📈 Final dataset: {df_clean.shape[0]:,} records × {df_clean.shape[1]} features")
        print(f"🆕 New features created: {new_features}")
        print(f"💾 Saved to: {output_path}")
        
        # Show some new features
        original_cols = set(df.columns)
        new_feature_names = [col for col in df_clean.columns if col not in original_cols]
        
        print(f"\n🔍 Sample new features:")
        for i, feature in enumerate(new_feature_names[:15], 1):
            print(f"  {i:2d}. {feature}")
        if len(new_feature_names) > 15:
            print(f"  ... and {len(new_feature_names)-15} more features")
        
        print(f"\n✅ Ready for model training!")
        return df_clean
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("DELHI LOAD FORECASTING - ENHANCED FEATURE ENGINEERING")
    print("=" * 60)
    
    # Run the pipeline
    result = run_standalone_feature_engineering()
    
    if result is not None:
        print("\n" + "=" * 60)
        print("✅ SUCCESS: Enhanced dataset ready for model training!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ FAILED: Check errors above")
        print("=" * 60)
