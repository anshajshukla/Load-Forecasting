#!/usr/bin/env python3
"""
Integrate Delhi-Specific Features to Final Dataset
Enhance our production dataset with critical temporal features for Delhi load forecasting
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delhi_specific_features import DelhiSpecificFeatures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrate_delhi_features():
    """
    Integrate Delhi-specific features into our final dataset
    """
    
    print("🎯 DELHI-SPECIFIC FEATURES INTEGRATION")
    print("=" * 60)
    
    # Load existing final dataset
    dataset_path = "c:/Users/ansha/Desktop/SIH_new/load_forecast/dataset/delhi_weather_load_v2_complete.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset not found at {dataset_path}")
        return False
    
    print(f"📂 Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Dataset info
    print(f"📊 Current Dataset Info:")
    print(f"   - Records: {len(df):,}")
    print(f"   - Features: {len(df.columns)}")
    print(f"   - Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize Delhi features
    print("\n🏛️ Initializing Delhi-specific features...")
    delhi_features = DelhiSpecificFeatures()
    
    # Add Delhi-specific features
    print("🔄 Adding Delhi-specific temporal features...")
    df_enhanced = delhi_features.add_delhi_features(df.copy())
    
    # Validate enhancement
    validation = delhi_features.validate_delhi_features(df_enhanced)
    
    print("\n✅ DELHI FEATURES VALIDATION")
    print("-" * 40)
    for key, value in validation.items():
        print(f"   {key}: {value:,}")
    
    # Feature comparison
    print(f"\n📈 FEATURE ENHANCEMENT SUMMARY")
    print("-" * 40)
    print(f"   Original Features: {len(df.columns)}")
    print(f"   Enhanced Features: {len(df_enhanced.columns)}")
    print(f"   Delhi Features Added: {len(df_enhanced.columns) - len(df.columns)}")
    
    # Show new features
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"\n🆕 NEW DELHI-SPECIFIC FEATURES ({len(new_features)}):")
    print("-" * 40)
    for i, feature in enumerate(new_features, 1):
        count = df_enhanced[feature].sum() if df_enhanced[feature].dtype == bool else "N/A"
        print(f"   {i:2d}. {feature} (Active: {count})")
    
    # Save enhanced dataset
    output_path = "c:/Users/ansha/Desktop/SIH_new/load_forecast/dataset/delhi_weather_load_v3_with_delhi_features.csv"
    print(f"\n💾 Saving enhanced dataset to: {output_path}")
    df_enhanced.to_csv(output_path, index=False)
    
    # Feature impact analysis
    print("\n🔍 FESTIVAL IMPACT ANALYSIS")
    print("-" * 40)
    
    # Diwali impact
    diwali_data = df_enhanced[df_enhanced['is_diwali_period'] == True]
    if len(diwali_data) > 0:
        avg_load_normal = df_enhanced[df_enhanced['is_diwali_period'] == False]['delhi_load'].mean()
        avg_load_diwali = diwali_data['delhi_load'].mean()
        diwali_impact = ((avg_load_diwali - avg_load_normal) / avg_load_normal) * 100
        print(f"   🪔 Diwali Load Impact: {diwali_impact:+.1f}%")
    
    # Major festivals impact
    festival_data = df_enhanced[df_enhanced['is_major_festival'] == True]
    if len(festival_data) > 0:
        avg_load_normal = df_enhanced[df_enhanced['is_major_festival'] == False]['delhi_load'].mean()
        avg_load_festival = festival_data['delhi_load'].mean()
        festival_impact = ((avg_load_festival - avg_load_normal) / avg_load_normal) * 100
        print(f"   🎉 Festival Load Impact: {festival_impact:+.1f}%")
    
    # Political events impact
    political_data = df_enhanced[df_enhanced['is_political_event'] == True]
    if len(political_data) > 0:
        avg_load_normal = df_enhanced[df_enhanced['is_political_event'] == False]['delhi_load'].mean()
        avg_load_political = political_data['delhi_load'].mean()
        political_impact = ((avg_load_political - avg_load_normal) / avg_load_normal) * 100
        print(f"   🏛️ Political Event Impact: {political_impact:+.1f}%")
    
    # Dataset quality check
    print("\n🎯 ENHANCED DATASET QUALITY")
    print("-" * 40)
    missing_percentage = (df_enhanced.isnull().sum().sum() / (len(df_enhanced) * len(df_enhanced.columns))) * 100
    print(f"   Missing Values: {missing_percentage:.3f}%")
    print(f"   Data Completeness: {100 - missing_percentage:.3f}%")
    print(f"   Ready for ML: {'✅ YES' if missing_percentage < 1 else '⚠️ NEEDS CLEANUP'}")
    
    # Success summary
    print("\n🎉 INTEGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📁 Enhanced Dataset: {output_path}")
    print(f"📊 Total Features: {len(df_enhanced.columns)} (Original: {len(df.columns)} + Delhi: {len(new_features)})")
    print(f"🎯 Delhi Load Forecasting: Ready for Feature Engineering with comprehensive temporal features")
    
    return True

def create_feature_summary():
    """
    Create comprehensive feature summary for documentation
    """
    
    print("\n📋 CREATING COMPREHENSIVE FEATURE DOCUMENTATION...")
    
    # Load enhanced dataset
    dataset_path = "c:/Users/ansha/Desktop/SIH_new/load_forecast/dataset/delhi_weather_load_v3_with_delhi_features.csv"
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        
        # Feature categories
        load_features = [col for col in df.columns if 'load' in col]
        weather_features = [col for col in df.columns if any(w in col for w in ['temperature', 'humidity', 'wind', 'pressure', 'rain', 'cloud', 'radiation'])]
        temporal_features = [col for col in df.columns if any(t in col for t in ['timestamp', 'hour', 'day', 'month', 'is_', 'festival'])]
        delhi_features = [col for col in df.columns if any(d in col for d in ['diwali', 'festival', 'political', 'stubble', 'pollution'])]
        
        print(f"\n📊 COMPREHENSIVE FEATURE BREAKDOWN:")
        print(f"   🔌 Load Features: {len(load_features)}")
        print(f"   🌤️ Weather Features: {len(weather_features)}")
        print(f"   ⏰ Temporal Features: {len(temporal_features)}")
        print(f"   🏛️ Delhi-Specific: {len(delhi_features)}")
        print(f"   📈 Total Features: {len(df.columns)}")
        
        return {
            'total_features': len(df.columns),
            'load_features': len(load_features),
            'weather_features': len(weather_features),
            'temporal_features': len(temporal_features),
            'delhi_features': len(delhi_features),
            'dataset_size': len(df)
        }
    
    return None

def main():
    """
    Main execution function
    """
    
    # Integrate Delhi features
    success = integrate_delhi_features()
    
    if success:
        # Create feature summary
        summary = create_feature_summary()
        
        if summary:
            print(f"\n🎯 FINAL DATASET STATUS")
            print("=" * 50)
            print(f"✅ Enhanced dataset created with {summary['total_features']} features")
            print(f"✅ Delhi-specific temporal features integrated")
            print(f"✅ Ready for advanced feature engineering phase")
            print(f"✅ Production-ready for Delhi load forecasting ML models")
    else:
        print("\n❌ Integration failed. Please check the dataset path and try again.")

if __name__ == "__main__":
    main()
