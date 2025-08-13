"""
DELHI LOAD FORECASTING - ENHANCED DATASET ANALYSIS
==================================================
Comprehensive analysis of the enhanced dataset with 115 features
Created after feature engineering pipeline execution
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib to use a non-GUI backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_enhanced_dataset():
    """Load the enhanced dataset"""
    print("🔄 Loading enhanced dataset...")
    try:
        # Load from parent directory where it was saved
        df = pd.read_csv('../enhanced_delhi_load_dataset.csv')
        print(f"✅ Dataset loaded successfully: {df.shape[0]:,} records × {df.shape[1]} features")
        return df
    except FileNotFoundError:
        print("❌ Enhanced dataset not found. Please run standalone_enhanced_pipeline.py first.")
        return None

def basic_dataset_info(df):
    """Display basic information about the dataset"""
    print("\n" + "="*60)
    print("📊 BASIC DATASET INFORMATION")
    print("="*60)
    
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Data types
    print(f"\nData Types:")
    print(f"  Numeric columns: {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"  Object columns: {df.select_dtypes(include=['object']).shape[1]}")
    print(f"  DateTime columns: {df.select_dtypes(include=['datetime64']).shape[1]}")

def feature_categories_analysis(df):
    """Analyze features by categories"""
    print("\n" + "="*60)
    print("🏷️ FEATURE CATEGORIES ANALYSIS")
    print("="*60)
    
    # Categorize features
    lag_features = [col for col in df.columns if 'lag' in col]
    rolling_features = [col for col in df.columns if any(x in col for x in ['rolling', 'ma_', 'std_'])]
    diff_features = [col for col in df.columns if 'diff' in col]
    weather_features = [col for col in df.columns if any(x in col for x in ['temp', 'humidity', 'pressure', 'wind', 'weather'])]
    time_features = [col for col in df.columns if any(x in col for x in ['hour', 'day', 'month', 'year', 'weekend', 'season'])]
    load_features = [col for col in df.columns if 'load' in col and col not in lag_features]
    interaction_features = [col for col in df.columns if '_x_' in col or 'interaction' in col]
    
    categories = {
        'Lag Features': lag_features,
        'Rolling Statistics': rolling_features,
        'Difference Features': diff_features,
        'Weather Features': weather_features,
        'Time Features': time_features,
        'Load Features': load_features,
        'Interaction Features': interaction_features
    }
    
    for category, features in categories.items():
        print(f"\n{category}: {len(features)} features")
        if len(features) > 0:
            print(f"  Sample features: {features[:5]}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more")

def missing_values_analysis(df):
    """Analyze missing values in the dataset"""
    print("\n" + "="*60)
    print("🔍 MISSING VALUES ANALYSIS")
    print("="*60)
    
    missing_stats = df.isnull().sum()
    missing_percent = (missing_stats / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_stats,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    
    # Show only columns with missing values
    missing_with_values = missing_df[missing_df['Missing Count'] > 0]
    
    if len(missing_with_values) > 0:
        print("Columns with missing values:")
        print(missing_with_values.head(10))
    else:
        print("✅ No missing values found in the dataset!")
    
    return missing_df

def correlation_analysis(df):
    """Analyze correlations between features"""
    print("\n" + "="*60)
    print("🔗 CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation with target variable (delhi_load)
    if 'delhi_load' in df.columns:
        target_corr = df[numeric_cols].corr()['delhi_load'].abs().sort_values(ascending=False)
        
        print("Top 15 features most correlated with Delhi Load:")
        print(target_corr.head(15).round(4))
        
        # High correlation pairs (excluding self-correlation)
        corr_matrix = df[numeric_cols].corr().abs()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            print(f"\n⚠️ High correlation pairs (>0.8): {len(high_corr_pairs)} found")
            high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
            print(high_corr_df.head(10))
        else:
            print("\n✅ No high correlation pairs (>0.8) found")

def statistical_summary(df):
    """Generate statistical summary of key features"""
    print("\n" + "="*60)
    print("📈 STATISTICAL SUMMARY")
    print("="*60)
    
    # Key load features
    load_cols = [col for col in df.columns if 'load' in col and 'lag' not in col][:10]
    
    if load_cols:
        summary = df[load_cols].describe()
        print("Load Features Summary:")
        print(summary.round(2))
    
    # Weather features summary
    weather_cols = [col for col in df.columns if any(x in col for x in ['temp', 'humidity', 'pressure']) and 'lag' not in col][:5]
    
    if weather_cols:
        print(f"\nWeather Features Summary:")
        weather_summary = df[weather_cols].describe()
        print(weather_summary.round(2))

def data_quality_checks(df):
    """Perform data quality checks"""
    print("\n" + "="*60)
    print("🔎 DATA QUALITY CHECKS")
    print("="*60)
    
    # Check for infinite values
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            inf_cols.append(col)
    
    if inf_cols:
        print(f"⚠️ Columns with infinite values: {inf_cols}")
    else:
        print("✅ No infinite values found")
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check datetime continuity
    if 'datetime' in df.columns:
        df_sorted = df.sort_values('datetime')
        time_diff = pd.to_datetime(df_sorted['datetime']).diff().dropna()
        expected_freq = pd.Timedelta(hours=1)  # Assuming hourly data
        
        irregular_intervals = time_diff[time_diff != expected_freq]
        print(f"Irregular time intervals: {len(irregular_intervals)}")
        
        if len(irregular_intervals) > 0:
            print(f"Sample irregular intervals: {irregular_intervals.head()}")

def feature_importance_analysis(df):
    """Analyze feature importance using simple correlation"""
    print("\n" + "="*60)
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    if 'delhi_load' not in df.columns:
        print("❌ Target variable 'delhi_load' not found")
        return
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'delhi_load']
    
    # Calculate absolute correlation with target
    correlations = df[numeric_cols + ['delhi_load']].corr()['delhi_load'].abs().sort_values(ascending=False)
    
    print("Top 20 Most Important Features (by correlation):")
    print(correlations.drop('delhi_load').head(20).round(4))
    
    # Categorize by feature type
    lag_importance = correlations[[col for col in correlations.index if 'lag' in col]]
    rolling_importance = correlations[[col for col in correlations.index if any(x in col for x in ['rolling', 'ma_', 'std_'])]]
    weather_importance = correlations[[col for col in correlations.index if any(x in col for x in ['temp', 'humidity', 'pressure'])]]
    
    print(f"\nBest performing feature types:")
    if len(lag_importance) > 0:
        print(f"  Best Lag Feature: {lag_importance.idxmax()} ({lag_importance.max():.4f})")
    if len(rolling_importance) > 0:
        print(f"  Best Rolling Feature: {rolling_importance.idxmax()} ({rolling_importance.max():.4f})")
    if len(weather_importance) > 0:
        print(f"  Best Weather Feature: {weather_importance.idxmax()} ({weather_importance.max():.4f})")

def create_visualizations(df):
    """Create key visualizations"""
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Dataset Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Load distribution
    if 'delhi_load' in df.columns:
        axes[0, 0].hist(df['delhi_load'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Delhi Load Distribution')
        axes[0, 0].set_xlabel('Load (MW)')
        axes[0, 0].set_ylabel('Frequency')
    
    # 2. Feature count by category
    categories = ['lag', 'rolling', 'ma_', 'std_', 'diff', 'temp', 'load']
    category_counts = []
    category_names = []
    
    for cat in categories:
        count = len([col for col in df.columns if cat in col])
        if count > 0:
            category_counts.append(count)
            category_names.append(cat.replace('_', '').title())
    
    axes[0, 1].bar(category_names, category_counts, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Feature Count by Category')
    axes[0, 1].set_ylabel('Number of Features')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Missing values heatmap (if any)
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_top = missing_data[missing_data > 0].head(10)
        axes[1, 0].bar(range(len(missing_top)), missing_top.values, color='orange', alpha=0.7)
        axes[1, 0].set_title('Missing Values by Feature')
        axes[1, 0].set_ylabel('Missing Count')
        axes[1, 0].set_xticks(range(len(missing_top)))
        axes[1, 0].set_xticklabels(missing_top.index, rotation=45)
    else:
        axes[1, 0].text(0.5, 0.5, 'No Missing Values!', 
                       transform=axes[1, 0].transAxes, ha='center', va='center',
                       fontsize=14, color='green', fontweight='bold')
        axes[1, 0].set_title('Missing Values Status')
    
    # 4. Top correlations with target
    if 'delhi_load' in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_corr = df[numeric_cols].corr()['delhi_load'].abs().sort_values(ascending=False)
        top_corr = target_corr.drop('delhi_load').head(10)
        
        axes[1, 1].barh(range(len(top_corr)), top_corr.values, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Top 10 Features Correlated with Delhi Load')
        axes[1, 1].set_xlabel('Absolute Correlation')
        axes[1, 1].set_yticks(range(len(top_corr)))
        axes[1, 1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in top_corr.index])
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"enhanced_dataset_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved as: {plot_filename}")
    
    # Don't show plot in headless environment
    # plt.show()

def save_analysis_report(df, missing_df):
    """Save detailed analysis report"""
    print("\n" + "="*60)
    print("💾 SAVING ANALYSIS REPORT")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"enhanced_dataset_analysis_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("DELHI LOAD FORECASTING - ENHANCED DATASET ANALYSIS REPORT\n")
        f.write("="*65 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"DATASET OVERVIEW:\n")
        f.write(f"- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
        f.write(f"- Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n")
        f.write(f"- Date Range: {df['datetime'].min()} to {df['datetime'].max()}\n\n")
        
        # Feature categories
        lag_features = [col for col in df.columns if 'lag' in col]
        rolling_features = [col for col in df.columns if any(x in col for x in ['rolling', 'ma_', 'std_'])]
        diff_features = [col for col in df.columns if 'diff' in col]
        
        f.write("FEATURE CATEGORIES:\n")
        f.write(f"- Lag Features: {len(lag_features)}\n")
        f.write(f"- Rolling Features: {len(rolling_features)}\n")
        f.write(f"- Difference Features: {len(diff_features)}\n\n")
        
        # Data quality
        f.write("DATA QUALITY:\n")
        f.write(f"- Missing Values: {df.isnull().sum().sum()}\n")
        f.write(f"- Duplicate Rows: {df.duplicated().sum()}\n")
        
        # Top correlations
        if 'delhi_load' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            target_corr = df[numeric_cols].corr()['delhi_load'].abs().sort_values(ascending=False)
            f.write(f"\nTOP 10 CORRELATED FEATURES:\n")
            for feature, corr in target_corr.drop('delhi_load').head(10).items():
                f.write(f"- {feature}: {corr:.4f}\n")
    
    print(f"📄 Analysis report saved as: {report_filename}")

def main():
    """Main analysis function"""
    print("🚀 DELHI LOAD FORECASTING - ENHANCED DATASET ANALYSIS")
    print("="*65)
    
    # Load dataset
    df = load_enhanced_dataset()
    if df is None:
        return
    
    # Run all analyses
    basic_dataset_info(df)
    feature_categories_analysis(df)
    missing_values_analysis(df)
    correlation_analysis(df)
    statistical_summary(df)
    data_quality_checks(df)
    feature_importance_analysis(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save report
    missing_df = df.isnull().sum()
    save_analysis_report(df, missing_df)
    
    print("\n" + "="*65)
    print("✅ ENHANCED DATASET ANALYSIS COMPLETE!")
    print("="*65)
    print("📊 Key insights:")
    print(f"   • Dataset size: {df.shape[0]:,} records × {df.shape[1]} features")
    print(f"   • Feature engineering success: {df.shape[1] - 69} new features added")
    print(f"   • Data quality: {df.isnull().sum().sum()} missing values")
    print(f"   • Ready for model training: ✅")
    print("="*65)

if __name__ == "__main__":
    main()
