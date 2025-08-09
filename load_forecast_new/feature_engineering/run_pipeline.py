"""
Enhanced Feature Engineering Pipeline
Generates advanced features from production dataset including lag features
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_feature_engineering(input_csv=None, output_csv=None):
    """
    Run enhanced feature engineering pipeline
    
    Args:
        input_csv: Input dataset path 
        output_csv: Output dataset path
    """
    
    if input_csv is None:
        input_csv = '../optimized_delhi_load_dataset.csv'
    
    if output_csv is None:
        output_csv = '../enhanced_delhi_load_dataset.csv'
    
    try:
        print("🚀 Starting Enhanced Feature Engineering Pipeline...")
        
        # Load the optimized dataset
        df = pd.read_csv(input_csv, parse_dates=['datetime'])
        print(f"📊 Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} features")
        
        # Import and run advanced lag features first
        from modules.advanced_lag_features import run_advanced_lag_pipeline
        df_enhanced = run_advanced_lag_pipeline(df)
        
        # Then run the orchestrated pipeline for other features
        from orchestrator.pipeline import FeatureEngineeringOrchestrator
        
        # Initialize orchestrator
        orchestrator = FeatureEngineeringOrchestrator()
        
        # Run pipeline on already lag-enhanced data
        try:
            final_df = orchestrator.run_pipeline(df_enhanced)
        except Exception as orchestrator_error:
            print(f"Orchestrator pipeline had issues: {orchestrator_error}")
            print("Continuing with lag-enhanced dataset only...")
            final_df = df_enhanced
        
        # Clean dataset - remove rows with critical NaN values
        critical_features = ['delhi_load_lag_1h', 'delhi_load_lag_24h', 'delhi_load_lag_168h']
        available_critical = [f for f in critical_features if f in final_df.columns]
        
        if available_critical:
            initial_rows = len(final_df)
            final_df = final_df.dropna(subset=available_critical)
            rows_removed = initial_rows - len(final_df)
            print(f"🧹 Cleaned dataset: removed {rows_removed:,} rows with missing lag values")
        
        # Save enhanced dataset
        final_df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Feature engineering completed!")
        print(f"📈 Enhanced dataset: {final_df.shape}")
        print(f"💾 Saved to: {output_csv}")
        
        # Show feature summary
        original_cols = pd.read_csv(input_csv).columns.tolist()
        new_features = [col for col in final_df.columns if col not in original_cols]
        
        print(f"\n🆕 New features created: {len(new_features)}")
        if len(new_features) <= 20:
            for i, feature in enumerate(new_features, 1):
                print(f"  {i:2d}. {feature}")
        else:
            print("First 20 new features:")
            for i, feature in enumerate(new_features[:20], 1):
                print(f"  {i:2d}. {feature}")
            print(f"  ... and {len(new_features)-20} more features")
        
        print(f"\n🎯 Dataset ready for model training!")
        return final_df
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_feature_engineering()
