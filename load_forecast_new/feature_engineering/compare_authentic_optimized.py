"""
AUTHENTIC vs OPTIMIZED DATASET COMPARISON
========================================
Compare authentic data load with optimized_delhi_load.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_datasets():
    """Load both authentic and optimized datasets"""
    print("📊 LOADING DATASETS FOR COMPARISON")
    print("="*50)
    
    # Load authentic dataset
    try:
        authentic_df = pd.read_csv('../../Load-Forecasting/final_dataset_with_authentic_load.csv')
        print(f"✅ Authentic dataset: {authentic_df.shape[0]:,} records × {authentic_df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Authentic dataset not found")
        return None, None
    
    # Load optimized dataset
    try:
        optimized_df = pd.read_csv('../../Load-Forecasting/optimized_delhi_load_dataset.csv')
        print(f"✅ Optimized dataset: {optimized_df.shape[0]:,} records × {optimized_df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Optimized dataset not found")
        return authentic_df, None
    
    return authentic_df, optimized_df

def compare_basic_info(authentic_df, optimized_df):
    """Compare basic information between datasets"""
    print("\n" + "="*60)
    print("📈 BASIC DATASET COMPARISON")
    print("="*60)
    
    # Convert datetime columns
    authentic_df['datetime'] = pd.to_datetime(authentic_df['datetime'])
    optimized_df['datetime'] = pd.to_datetime(optimized_df['datetime'])
    
    print(f"\n📊 SIZE COMPARISON:")
    print(f"  Authentic Dataset: {authentic_df.shape[0]:,} records × {authentic_df.shape[1]} features")
    print(f"  Optimized Dataset: {optimized_df.shape[0]:,} records × {optimized_df.shape[1]} features")
    print(f"  Record difference: {authentic_df.shape[0] - optimized_df.shape[0]:,} records")
    print(f"  Feature difference: {authentic_df.shape[1] - optimized_df.shape[1]} features")
    
    print(f"\n📅 DATE RANGE COMPARISON:")
    print(f"  Authentic Dataset:")
    print(f"    Start: {authentic_df['datetime'].min()}")
    print(f"    End: {authentic_df['datetime'].max()}")
    print(f"    Duration: {(authentic_df['datetime'].max() - authentic_df['datetime'].min()).days} days")
    
    print(f"  Optimized Dataset:")
    print(f"    Start: {optimized_df['datetime'].min()}")
    print(f"    End: {optimized_df['datetime'].max()}")
    print(f"    Duration: {(optimized_df['datetime'].max() - optimized_df['datetime'].min()).days} days")

def compare_load_data(authentic_df, optimized_df):
    """Compare load data between datasets"""
    print(f"\n⚡ LOAD DATA COMPARISON:")
    print("="*50)
    
    load_cols = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
    
    print(f"  {'Utility':<15} {'Auth Mean':<12} {'Auth Std':<12} {'Opt Mean':<12} {'Opt Std':<12} {'Diff %'}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    
    for col in load_cols:
        if col in authentic_df.columns and col in optimized_df.columns:
            auth_mean = authentic_df[col].mean()
            auth_std = authentic_df[col].std()
            opt_mean = optimized_df[col].mean()
            opt_std = optimized_df[col].std()
            diff_pct = ((opt_mean - auth_mean) / auth_mean) * 100 if auth_mean > 0 else 0
            
            print(f"  {col:<15} {auth_mean:<12.2f} {auth_std:<12.2f} {opt_mean:<12.2f} {opt_std:<12.2f} {diff_pct:>6.2f}%")

def compare_data_sources(authentic_df, optimized_df):
    """Compare data sources"""
    print(f"\n🏷️ DATA SOURCE COMPARISON:")
    print("="*50)
    
    if 'data_source' in authentic_df.columns:
        auth_sources = authentic_df['data_source'].value_counts()
        print(f"  Authentic Data Sources:")
        for source, count in auth_sources.items():
            percentage = (count / len(authentic_df)) * 100
            print(f"    {source}: {count:,} ({percentage:.1f}%)")
    
    if 'data_source' in optimized_df.columns:
        opt_sources = optimized_df['data_source'].value_counts()
        print(f"  Optimized Data Sources:")
        for source, count in opt_sources.items():
            percentage = (count / len(optimized_df)) * 100
            print(f"    {source}: {count:,} ({percentage:.1f}%)")

def compare_features(authentic_df, optimized_df):
    """Compare features between datasets"""
    print(f"\n🔧 FEATURE COMPARISON:")
    print("="*50)
    
    auth_features = set(authentic_df.columns)
    opt_features = set(optimized_df.columns)
    
    common_features = auth_features.intersection(opt_features)
    auth_only_features = auth_features - opt_features
    opt_only_features = opt_features - auth_features
    
    print(f"  Common features: {len(common_features)}")
    print(f"  Authentic-only features: {len(auth_only_features)}")
    if len(auth_only_features) > 0:
        print(f"    Sample: {list(auth_only_features)[:5]}")
    
    print(f"  Optimized-only features: {len(opt_only_features)}")
    if len(opt_only_features) > 0:
        print(f"    Sample: {list(opt_only_features)[:5]}")

def compare_missing_values(authentic_df, optimized_df):
    """Compare missing values"""
    print(f"\n❓ MISSING VALUES COMPARISON:")
    print("="*50)
    
    auth_missing = authentic_df.isnull().sum().sum()
    auth_missing_pct = (auth_missing / (authentic_df.shape[0] * authentic_df.shape[1])) * 100
    
    opt_missing = optimized_df.isnull().sum().sum()
    opt_missing_pct = (opt_missing / (optimized_df.shape[0] * optimized_df.shape[1])) * 100
    
    print(f"  Authentic Dataset Missing: {auth_missing:,} ({auth_missing_pct:.3f}%)")
    print(f"  Optimized Dataset Missing: {opt_missing:,} ({opt_missing_pct:.3f}%)")
    print(f"  Difference: {auth_missing - opt_missing:,} values")

def analyze_data_overlap(authentic_df, optimized_df):
    """Analyze time overlap between datasets"""
    print(f"\n⏰ TIME OVERLAP ANALYSIS:")
    print("="*50)
    
    auth_dates = set(authentic_df['datetime'].dt.date)
    opt_dates = set(optimized_df['datetime'].dt.date)
    
    overlap_dates = auth_dates.intersection(opt_dates)
    auth_only_dates = auth_dates - opt_dates
    opt_only_dates = opt_dates - auth_dates
    
    print(f"  Overlapping dates: {len(overlap_dates):,}")
    print(f"  Authentic-only dates: {len(auth_only_dates):,}")
    print(f"  Optimized-only dates: {len(opt_only_dates):,}")
    
    if len(overlap_dates) > 0:
        overlap_pct = (len(overlap_dates) / len(auth_dates)) * 100
        print(f"  Overlap percentage: {overlap_pct:.1f}% of authentic data")
    
    # Show sample dates if available
    if len(auth_only_dates) > 0:
        sample_auth_only = sorted(list(auth_only_dates))[:5]
        print(f"  Sample authentic-only dates: {sample_auth_only}")
    
    if len(opt_only_dates) > 0:
        sample_opt_only = sorted(list(opt_only_dates))[:5]
        print(f"  Sample optimized-only dates: {sample_opt_only}")

def compare_load_patterns(authentic_df, optimized_df):
    """Compare load patterns"""
    print(f"\n📊 LOAD PATTERN COMPARISON:")
    print("="*50)
    
    if 'delhi_load' in authentic_df.columns and 'delhi_load' in optimized_df.columns:
        # Daily patterns
        auth_hourly_avg = authentic_df.groupby(authentic_df['datetime'].dt.hour)['delhi_load'].mean()
        opt_hourly_avg = optimized_df.groupby(optimized_df['datetime'].dt.hour)['delhi_load'].mean()
        
        correlation = np.corrcoef(auth_hourly_avg, opt_hourly_avg)[0,1]
        print(f"  Hourly pattern correlation: {correlation:.4f}")
        
        # Statistical comparison
        auth_stats = authentic_df['delhi_load'].describe()
        opt_stats = optimized_df['delhi_load'].describe()
        
        print(f"\n  Delhi Load Statistics Comparison:")
        print(f"    {'Statistic':<10} {'Authentic':<12} {'Optimized':<12} {'Difference'}")
        print(f"    {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        
        for stat in ['min', '25%', '50%', '75%', 'max', 'mean', 'std']:
            auth_val = auth_stats[stat]
            opt_val = opt_stats[stat]
            diff = opt_val - auth_val
            print(f"    {stat:<10} {auth_val:<12.2f} {opt_val:<12.2f} {diff:>8.2f}")

def generate_recommendation(authentic_df, optimized_df):
    """Generate recommendation based on comparison"""
    print(f"\n" + "="*60)
    print("💡 RECOMMENDATION")
    print("="*60)
    
    # Calculate key metrics
    record_diff = authentic_df.shape[0] - optimized_df.shape[0]
    feature_diff = authentic_df.shape[1] - optimized_df.shape[1]
    
    auth_missing_pct = (authentic_df.isnull().sum().sum() / (authentic_df.shape[0] * authentic_df.shape[1])) * 100
    opt_missing_pct = (optimized_df.isnull().sum().sum() / (optimized_df.shape[0] * optimized_df.shape[1])) * 100
    
    # Time overlap
    auth_dates = set(authentic_df['datetime'].dt.date)
    opt_dates = set(optimized_df['datetime'].dt.date)
    overlap_pct = (len(auth_dates.intersection(opt_dates)) / len(auth_dates)) * 100
    
    print(f"📋 ANALYSIS SUMMARY:")
    print(f"  Authentic Dataset: {authentic_df.shape[0]:,} records, {authentic_df.shape[1]} features")
    print(f"  Optimized Dataset: {optimized_df.shape[0]:,} records, {optimized_df.shape[1]} features")
    print(f"  Data overlap: {overlap_pct:.1f}%")
    print(f"  Missing values: Auth {auth_missing_pct:.2f}%, Opt {opt_missing_pct:.2f}%")
    
    print(f"\n✅ RECOMMENDATIONS:")
    
    if record_diff > 0:
        print(f"   • Authentic dataset has {record_diff:,} more records")
        print(f"   • Use authentic dataset for comprehensive training")
    
    if feature_diff > 0:
        print(f"   • Authentic dataset has {feature_diff} more features")
        print(f"   • Additional features may provide better model performance")
    elif feature_diff < 0:
        print(f"   • Optimized dataset has {abs(feature_diff)} more features")
        print(f"   • May include engineered features for better modeling")
    
    if auth_missing_pct > opt_missing_pct:
        print(f"   • Optimized dataset has better data completeness")
    
    print(f"\n🎯 FINAL RECOMMENDATION:")
    if overlap_pct > 95 and opt_missing_pct < auth_missing_pct:
        print(f"   → Use OPTIMIZED dataset for modeling")
        print(f"   → Better data quality and completeness")
    else:
        print(f"   → Use AUTHENTIC dataset for modeling")
        print(f"   → More comprehensive and authentic data")
        print(f"   → Apply data cleaning if needed")

def main():
    """Main comparison function"""
    print("🔍 AUTHENTIC vs OPTIMIZED DATASET COMPARISON")
    print("="*65)
    
    # Load datasets
    authentic_df, optimized_df = load_datasets()
    
    if authentic_df is None or optimized_df is None:
        print("❌ Cannot proceed with comparison - missing datasets")
        return
    
    # Run comparisons
    compare_basic_info(authentic_df, optimized_df)
    compare_load_data(authentic_df, optimized_df)
    compare_data_sources(authentic_df, optimized_df)
    compare_features(authentic_df, optimized_df)
    compare_missing_values(authentic_df, optimized_df)
    analyze_data_overlap(authentic_df, optimized_df)
    compare_load_patterns(authentic_df, optimized_df)
    generate_recommendation(authentic_df, optimized_df)
    
    print(f"\n" + "="*65)
    print("✅ COMPARISON COMPLETE!")
    print("="*65)

if __name__ == "__main__":
    main()
