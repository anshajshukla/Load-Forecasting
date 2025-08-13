"""
SYNTHETIC DATA ANALYSIS - DELHI LOAD FORECASTING
===============================================
Analyze how much of the dataset is synthetic vs real data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_synthetic_data():
    """Analyze the proportion of synthetic vs real data"""
    print("🔍 SYNTHETIC DATA ANALYSIS")
    print("="*50)
    
    # Load the enhanced dataset
    try:
        df = pd.read_csv('../enhanced_delhi_load_dataset.csv')
        print(f"✅ Dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Enhanced dataset not found.")
        return
    
    # Convert datetime column
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Analyze data sources
    print(f"\n📊 DATA SOURCE ANALYSIS:")
    print(f"Total records: {len(df):,}")
    
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        print(f"\nData Source Distribution:")
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count:,} records ({percentage:.2f}%)")
        
        # Categorize as synthetic vs real
        synthetic_keywords = ['simulation', 'synthetic', 'generated', 'artificial', 'mock']
        real_keywords = ['real', 'actual', 'measured', 'observed', 'historical']
        
        synthetic_count = 0
        real_count = 0
        unknown_count = 0
        
        for source in source_counts.index:
            source_lower = str(source).lower()
            count = source_counts[source]
            
            if any(keyword in source_lower for keyword in synthetic_keywords):
                synthetic_count += count
            elif any(keyword in source_lower for keyword in real_keywords):
                real_count += count
            else:
                unknown_count += count
        
        print(f"\n🎯 SYNTHETIC vs REAL DATA BREAKDOWN:")
        print(f"  Synthetic Data: {synthetic_count:,} records ({(synthetic_count/len(df))*100:.2f}%)")
        print(f"  Real Data: {real_count:,} records ({(real_count/len(df))*100:.2f}%)")
        print(f"  Unknown/Other: {unknown_count:,} records ({(unknown_count/len(df))*100:.2f}%)")
        
    else:
        print("❌ No 'data_source' column found")
    
    # Analyze date ranges
    print(f"\n📅 DATE RANGE ANALYSIS:")
    print(f"Start Date: {df['datetime'].min()}")
    print(f"End Date: {df['datetime'].max()}")
    print(f"Total Duration: {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    # Check if we have future dates (likely synthetic)
    current_date = datetime(2025, 8, 1)  # Current date from context
    future_mask = df['datetime'] > current_date
    past_mask = df['datetime'] <= current_date
    
    future_count = future_mask.sum()
    past_count = past_mask.sum()
    
    print(f"\n🔮 TEMPORAL ANALYSIS (relative to {current_date.strftime('%Y-%m-%d')}):")
    print(f"  Past/Current Data: {past_count:,} records ({(past_count/len(df))*100:.2f}%)")
    print(f"  Future Data: {future_count:,} records ({(future_count/len(df))*100:.2f}%)")
    
    if future_count > 0:
        print(f"  ⚠️  Future data detected - likely synthetic/simulated")
    
    # Analyze missing values pattern (synthetic data often has fewer missing values)
    print(f"\n❓ MISSING VALUES PATTERN:")
    missing_total = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (missing_total / total_cells) * 100
    
    print(f"  Total Missing Values: {missing_total:,}")
    print(f"  Missing Percentage: {missing_percentage:.3f}%")
    
    # Check specific columns that might indicate synthetic data
    synthetic_indicators = [
        'solar_generation_mw', 'solar_irradiance_ghi_wm2', 'solar_capacity_factor_percent',
        'duck_curve_depth_mw', 'net_load_mw', 'solar_penetration_percent'
    ]
    
    print(f"\n🔬 SYNTHETIC FEATURE INDICATORS:")
    for col in synthetic_indicators:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            percentage = (non_null_count / len(df)) * 100
            print(f"  {col}: {non_null_count:,} records ({percentage:.1f}% populated)")
            
            # Check for perfectly regular patterns (indicator of synthetic data)
            if df[col].dtype in ['float64', 'int64']:
                unique_count = df[col].nunique()
                if unique_count < len(df) * 0.1:  # Less than 10% unique values might indicate synthetic
                    print(f"    ⚠️  Low variance detected ({unique_count} unique values)")
    
    # Check load data patterns
    print(f"\n⚡ LOAD DATA ANALYSIS:")
    load_columns = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
    
    for col in load_columns:
        if col in df.columns:
            non_zero_count = (df[col] > 0).sum()
            zero_count = (df[col] == 0).sum()
            percentage_active = (non_zero_count / len(df)) * 100
            
            print(f"  {col}:")
            print(f"    Active periods: {non_zero_count:,} ({percentage_active:.1f}%)")
            print(f"    Zero/inactive: {zero_count:,}")
            
            # Check for unrealistic patterns
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                mean_load = df[col].mean()
                std_load = df[col].std()
                cv = std_load / mean_load if mean_load > 0 else 0
                print(f"    Mean: {mean_load:.2f} MW, Std: {std_load:.2f} MW, CV: {cv:.3f}")
    
    # Generate summary
    print(f"\n" + "="*50)
    print("📋 SUMMARY:")
    print("="*50)
    
    if 'data_source' in df.columns:
        synthetic_pct = (synthetic_count / len(df)) * 100
        real_pct = (real_count / len(df)) * 100
        
        print(f"🎭 Synthetic Data: ~{synthetic_pct:.1f}% of dataset")
        print(f"📊 Real Data: ~{real_pct:.1f}% of dataset")
        
        if synthetic_pct > 50:
            print("⚠️  Dataset is primarily synthetic/simulated")
        elif real_pct > 50:
            print("✅ Dataset is primarily real/measured data")
        else:
            print("🔄 Mixed dataset with significant synthetic and real components")
    
    future_pct = (future_count / len(df)) * 100
    if future_pct > 0:
        print(f"🔮 Future projections: {future_pct:.1f}% (likely synthetic)")
    
    print(f"📈 Data quality: {(100-missing_percentage):.2f}% complete")
    
    return df

if __name__ == "__main__":
    analyze_synthetic_data()
