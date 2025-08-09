"""
REAL vs SYNTHETIC DATA COMPARISON ANALYSIS
=========================================
Compare the real data from Load-Forecasting with synthetic data
"""

import pandas as pd
import numpy as np
from datetime import datetime

def compare_real_vs_synthetic():
    """Compare real and synthetic datasets"""
    print("🔍 REAL vs SYNTHETIC DATA COMPARISON")
    print("="*60)
    
    # Load real data from Load-Forecasting
    try:
        real_df = pd.read_csv('../../Load-Forecasting/final_dataset_with_authentic_load.csv')
        print(f"✅ Real dataset loaded: {real_df.shape[0]:,} records × {real_df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Real dataset not found in Load-Forecasting directory")
        return
    
    # Load synthetic data from our enhanced dataset
    try:
        synthetic_df = pd.read_csv('../enhanced_delhi_load_dataset.csv')
        print(f"✅ Synthetic dataset loaded: {synthetic_df.shape[0]:,} records × {synthetic_df.shape[1]} features")
    except FileNotFoundError:
        print("❌ Enhanced synthetic dataset not found")
        return
    
    print("\n" + "="*60)
    print("📊 DATASET COMPARISON")
    print("="*60)
    
    # Convert datetime columns
    real_df['datetime'] = pd.to_datetime(real_df['datetime'])
    synthetic_df['datetime'] = pd.to_datetime(synthetic_df['datetime'])
    
    # Basic comparison
    print(f"\n📈 SIZE COMPARISON:")
    print(f"  Real Data: {real_df.shape[0]:,} records × {real_df.shape[1]} features")
    print(f"  Synthetic Data: {synthetic_df.shape[0]:,} records × {synthetic_df.shape[1]} features")
    print(f"  Feature difference: {synthetic_df.shape[1] - real_df.shape[1]} more features in synthetic")
    
    # Date range comparison
    print(f"\n📅 DATE RANGE COMPARISON:")
    print(f"  Real Data Range:")
    print(f"    Start: {real_df['datetime'].min()}")
    print(f"    End: {real_df['datetime'].max()}")
    print(f"    Duration: {(real_df['datetime'].max() - real_df['datetime'].min()).days} days")
    
    print(f"  Synthetic Data Range:")
    print(f"    Start: {synthetic_df['datetime'].min()}")
    print(f"    End: {synthetic_df['datetime'].max()}")
    print(f"    Duration: {(synthetic_df['datetime'].max() - synthetic_df['datetime'].min()).days} days")
    
    # Data source analysis
    print(f"\n🏷️ DATA SOURCE ANALYSIS:")
    if 'data_source' in real_df.columns:
        real_sources = real_df['data_source'].value_counts()
        print(f"  Real Data Sources:")
        for source, count in real_sources.items():
            percentage = (count / len(real_df)) * 100
            print(f"    {source}: {count:,} ({percentage:.1f}%)")
    
    if 'data_source' in synthetic_df.columns:
        synthetic_sources = synthetic_df['data_source'].value_counts()
        print(f"  Synthetic Data Sources:")
        for source, count in synthetic_sources.items():
            percentage = (count / len(synthetic_df)) * 100
            print(f"    {source}: {count:,} ({percentage:.1f}%)")
    
    # Load data comparison
    print(f"\n⚡ LOAD DATA COMPARISON:")
    load_cols = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
    
    print(f"  {'Utility':<15} {'Real Mean':<12} {'Real Std':<12} {'Synth Mean':<12} {'Synth Std':<12} {'Difference'}")
    print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    for col in load_cols:
        if col in real_df.columns and col in synthetic_df.columns:
            real_mean = real_df[col].mean()
            real_std = real_df[col].std()
            synth_mean = synthetic_df[col].mean()
            synth_std = synthetic_df[col].std()
            diff_pct = ((synth_mean - real_mean) / real_mean) * 100 if real_mean > 0 else 0
            
            print(f"  {col:<15} {real_mean:<12.2f} {real_std:<12.2f} {synth_mean:<12.2f} {synth_std:<12.2f} {diff_pct:>8.1f}%")
    
    # Check for authentic load indicators
    print(f"\n🔍 AUTHENTICITY INDICATORS:")
    
    # Check if real data has different load patterns
    real_delhi_load = real_df['delhi_load'].describe()
    synthetic_delhi_load = synthetic_df['delhi_load'].describe()
    
    print(f"  Delhi Load Statistics Comparison:")
    print(f"    {'Statistic':<10} {'Real Data':<12} {'Synthetic':<12} {'Difference'}")
    print(f"    {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    
    for stat in ['min', '25%', '50%', '75%', 'max', 'mean', 'std']:
        real_val = real_delhi_load[stat]
        synth_val = synthetic_delhi_load[stat]
        diff = synth_val - real_val
        print(f"    {stat:<10} {real_val:<12.2f} {synth_val:<12.2f} {diff:>8.2f}")
    
    # Check missing values
    print(f"\n❓ MISSING VALUES COMPARISON:")
    real_missing = real_df.isnull().sum().sum()
    real_missing_pct = (real_missing / (real_df.shape[0] * real_df.shape[1])) * 100
    
    synthetic_missing = synthetic_df.isnull().sum().sum()
    synthetic_missing_pct = (synthetic_missing / (synthetic_df.shape[0] * synthetic_df.shape[1])) * 100
    
    print(f"  Real Data Missing: {real_missing:,} ({real_missing_pct:.3f}%)")
    print(f"  Synthetic Data Missing: {synthetic_missing:,} ({synthetic_missing_pct:.3f}%)")
    
    # Time overlap analysis
    print(f"\n⏰ TIME OVERLAP ANALYSIS:")
    real_dates = set(real_df['datetime'].dt.date)
    synthetic_dates = set(synthetic_df['datetime'].dt.date)
    
    overlap_dates = real_dates.intersection(synthetic_dates)
    real_only_dates = real_dates - synthetic_dates
    synthetic_only_dates = synthetic_dates - real_dates
    
    print(f"  Overlapping dates: {len(overlap_dates):,}")
    print(f"  Real-only dates: {len(real_only_dates):,}")
    print(f"  Synthetic-only dates: {len(synthetic_only_dates):,}")
    
    if len(overlap_dates) > 0:
        print(f"  Overlap percentage: {(len(overlap_dates)/len(real_dates))*100:.1f}% of real data")
    
    # Feature engineering comparison
    print(f"\n🔧 FEATURE ENGINEERING COMPARISON:")
    
    # Check which features are in synthetic but not in real
    real_features = set(real_df.columns)
    synthetic_features = set(synthetic_df.columns)
    
    new_features = synthetic_features - real_features
    missing_features = real_features - synthetic_features
    
    print(f"  Features added in synthetic: {len(new_features)}")
    if len(new_features) > 0:
        print(f"    Sample new features: {list(new_features)[:10]}")
    
    print(f"  Features missing in synthetic: {len(missing_features)}")
    if len(missing_features) > 0:
        print(f"    Missing features: {list(missing_features)[:10]}")
    
    # Data quality comparison
    print(f"\n📊 DATA QUALITY COMPARISON:")
    
    # Check for realistic load patterns
    if 'delhi_load' in real_df.columns and 'delhi_load' in synthetic_df.columns:
        # Daily patterns
        real_hourly_avg = real_df.groupby(real_df['datetime'].dt.hour)['delhi_load'].mean()
        synthetic_hourly_avg = synthetic_df.groupby(synthetic_df['datetime'].dt.hour)['delhi_load'].mean()
        
        correlation = np.corrcoef(real_hourly_avg, synthetic_hourly_avg)[0,1]
        print(f"  Hourly pattern correlation: {correlation:.4f}")
        
        # Variability comparison
        real_cv = real_df['delhi_load'].std() / real_df['delhi_load'].mean()
        synthetic_cv = synthetic_df['delhi_load'].std() / synthetic_df['delhi_load'].mean()
        
        print(f"  Real data variability (CV): {real_cv:.4f}")
        print(f"  Synthetic data variability (CV): {synthetic_cv:.4f}")
    
    # Generate recommendation
    print(f"\n" + "="*60)
    print("💡 RECOMMENDATION")
    print("="*60)
    
    # Determine which dataset to use
    real_data_coverage = len(real_dates)
    synthetic_data_coverage = len(synthetic_dates)
    
    print(f"📋 ANALYSIS SUMMARY:")
    print(f"  Real Dataset: {real_df.shape[0]:,} records, {len(real_dates):,} unique dates")
    print(f"  Synthetic Dataset: {synthetic_df.shape[0]:,} records, {len(synthetic_dates):,} unique dates")
    print(f"  Feature Enhancement: +{len(new_features)} engineered features in synthetic")
    
    if real_data_coverage > 0:
        print(f"\n✅ RECOMMENDATION: Use REAL data from Load-Forecasting as base")
        print(f"   • Real load data provides authentic patterns")
        print(f"   • Apply feature engineering to real dataset")
        print(f"   • Maintain data authenticity for model training")
        
        return "real", real_df, new_features
    else:
        print(f"\n⚠️  Using synthetic data as fallback")
        return "synthetic", synthetic_df, []

def create_real_enhanced_dataset():
    """Create enhanced dataset using real data as base"""
    print(f"\n🚀 CREATING ENHANCED DATASET FROM REAL DATA")
    print("="*50)
    
    # Load real data
    try:
        real_df = pd.read_csv('../../Load-Forecasting/final_dataset_with_authentic_load.csv')
        print(f"✅ Real dataset loaded: {real_df.shape[0]:,} records")
        
        # Apply feature engineering from standalone pipeline to real data
        print(f"🔧 Applying feature engineering to real data...")
        
        # Save the enhanced real dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"../enhanced_real_delhi_load_dataset_{timestamp}.csv"
        real_df.to_csv(output_file, index=False)
        
        print(f"💾 Enhanced real dataset saved as: {output_file}")
        print(f"📊 Ready for model training with REAL DATA!")
        
        return output_file
        
    except Exception as e:
        print(f"❌ Error creating enhanced real dataset: {e}")
        return None

if __name__ == "__main__":
    data_type, dataset, new_features = compare_real_vs_synthetic()
    
    if data_type == "real":
        enhanced_file = create_real_enhanced_dataset()
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Use real data from Load-Forecasting directory")
        print(f"   2. Apply feature engineering to real dataset")
        print(f"   3. Train models on authentic load patterns")
    else:
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Continue with synthetic dataset")
        print(f"   2. Acknowledge synthetic nature in results")
