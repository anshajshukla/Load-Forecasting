"""
STEP 2.5.1.5: DATA LEAKAGE FEATURE REMOVAL
==========================================
Removes identified leakage features from Delhi load forecasting dataset.
Creates a clean dataset ready for modeling.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Set, Tuple
import warnings
warnings.filterwarnings('ignore')


class LeakageFeatureRemover:
    """
    Removes data leakage features identified by the leakage detection analysis.
    
    Critical Actions:
    - Remove net_load_mw (severe leakage across 4 targets)
    - Remove identical/duplicate features
    - Clean suspicious temporal features
    - Generate clean dataset for modeling
    """
    
    def __init__(self, input_path: str, output_dir: str = "phase_2_5_1_outputs"):
        self.input_path = input_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Features to remove based on leakage analysis
        self.severe_leakage_features = [
            'net_load_mw'  # 99.92% correlation with delhi_load
        ]
        
        # Identical feature pairs - keep one, remove the other
        self.identical_features_to_remove = [
            'rain (mm)',  # identical to precipitation (mm)
            'heat_wave_season',  # identical to is_summer
            'is_summer_session',  # identical to is_summer
            'is_monsoon_month',  # identical to is_monsoon
            'is_monsoon_session',  # identical to is_monsoon
            'delhi_morning_peak',  # near-identical to is_morning_peak
            'is_stubble_burning_period',  # identical to is_pollution_emergency
            'is_diwali_season',  # identical to is_pollution_emergency
        ]
        
        # Target columns to preserve
        self.target_columns = [
            'delhi_load', 'brpl_load', 'bypl_load',
            'ndpl_load', 'ndmc_load', 'mes_load'
        ]
        
        # Results storage
        self.removal_stats = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load the original dataset"""
        try:
            print("Loading original dataset...")
            
            if self.input_path.endswith('.parquet'):
                df = pd.read_parquet(self.input_path)
            else:
                df = pd.read_csv(self.input_path)
            
            print(f"Original dataset shape: {df.shape}")
            print(f"Original features: {df.shape[1] - len(self.target_columns)} features + {len(self.target_columns)} targets")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def remove_severe_leakage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features with severe correlation leakage"""
        print("\n" + "="*60)
        print("REMOVING SEVERE LEAKAGE FEATURES")
        print("="*60)
        
        removed_features = []
        
        for feature in self.severe_leakage_features:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                removed_features.append(feature)
                print(f"✓ Removed severe leakage feature: {feature}")
            else:
                print(f"⚠ Feature not found: {feature}")
        
        self.removal_stats['severe_leakage_removed'] = removed_features
        print(f"\nRemoved {len(removed_features)} severe leakage features")
        
        return df
    
    def remove_identical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove identical/duplicate features"""
        print("\n" + "="*60)
        print("REMOVING IDENTICAL/DUPLICATE FEATURES")
        print("="*60)
        
        removed_features = []
        
        for feature in self.identical_features_to_remove:
            if feature in df.columns:
                df = df.drop(columns=[feature])
                removed_features.append(feature)
                print(f"✓ Removed identical feature: {feature}")
            else:
                print(f"⚠ Feature not found: {feature}")
        
        self.removal_stats['identical_features_removed'] = removed_features
        print(f"\nRemoved {len(removed_features)} identical/duplicate features")
        
        return df
    
    def detect_and_remove_remaining_high_correlations(self, df: pd.DataFrame, 
                                                    threshold: float = 0.99) -> pd.DataFrame:
        """Detect and remove any remaining high correlation features"""
        print("\n" + "="*60)
        print("DETECTING REMAINING HIGH CORRELATIONS")
        print("="*60)
        
        # Get numeric features only (exclude targets and datetime)
        exclude_cols = self.target_columns + ['datetime', 'date', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        if len(feature_cols) < 2:
            print("Not enough features for correlation analysis")
            return df
        
        feature_df = df[feature_cols]
        
        # Calculate correlation matrix
        corr_matrix = feature_df.corr().abs()
        
        # Find high correlation pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if corr_val > threshold:
                    feature1 = corr_matrix.columns[i]
                    feature2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_val))
        
        # Remove one feature from each high correlation pair
        additional_removed = []
        for feature1, feature2, corr_val in high_corr_pairs:
            if feature2 in df.columns:  # Remove the second feature by default
                df = df.drop(columns=[feature2])
                additional_removed.append(feature2)
                print(f"✓ Removed high correlation feature: {feature2} (corr with {feature1}: {corr_val:.4f})")
        
        self.removal_stats['additional_high_corr_removed'] = additional_removed
        print(f"\nRemoved {len(additional_removed)} additional high correlation features")
        
        return df
    
    def validate_target_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all target columns are present"""
        print("\n" + "="*60)
        print("VALIDATING TARGET COLUMNS")
        print("="*60)
        
        missing_targets = []
        present_targets = []
        
        for target in self.target_columns:
            if target in df.columns:
                present_targets.append(target)
                print(f"✓ Target found: {target}")
            else:
                missing_targets.append(target)
                print(f"⚠ Target missing: {target}")
        
        self.removal_stats['targets_present'] = present_targets
        self.removal_stats['targets_missing'] = missing_targets
        
        print(f"\nTargets available: {len(present_targets)}/6")
        
        return df
    
    def generate_feature_summary(self, original_df: pd.DataFrame, 
                               cleaned_df: pd.DataFrame) -> Dict:
        """Generate summary of cleaning process"""
        print("\n" + "="*60)
        print("GENERATING CLEANING SUMMARY")
        print("="*60)
        
        # Calculate statistics
        original_shape = original_df.shape
        cleaned_shape = cleaned_df.shape
        
        features_removed = original_shape[1] - cleaned_shape[1]
        removal_percentage = (features_removed / original_shape[1]) * 100
        
        summary = {
            'original_shape': original_shape,
            'cleaned_shape': cleaned_shape,
            'features_removed_count': features_removed,
            'features_removed_percentage': round(removal_percentage, 2),
            'rows_affected': 0,  # We don't remove rows, only columns
            'cleaning_date': datetime.now().isoformat()
        }
        
        # Add detailed removal stats
        summary.update(self.removal_stats)
        
        print(f"Original shape: {original_shape}")
        print(f"Cleaned shape: {cleaned_shape}")
        print(f"Features removed: {features_removed} ({removal_percentage:.2f}%)")
        
        return summary
    
    def save_cleaned_dataset(self, df: pd.DataFrame, 
                           summary: Dict) -> Tuple[str, str]:
        """Save the cleaned dataset and summary report"""
        print("\n" + "="*60)
        print("SAVING CLEANED DATASET")
        print("="*60)
        
        # Generate output filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cleaned_csv_path = os.path.join(
            self.output_dir, 
            'delhi_interaction_enhanced_cleaned.csv'
        )
        
        summary_path = os.path.join(
            self.output_dir,
            'feature_removal_summary.json'
        )
        
        # Save cleaned dataset
        df.to_csv(cleaned_csv_path, index=False)
        print(f"✓ Cleaned dataset saved: {cleaned_csv_path}")
        
        # Save summary report
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"✓ Summary report saved: {summary_path}")
        
        return cleaned_csv_path, summary_path
    
    def run_complete_cleaning(self) -> Tuple[str, str]:
        """Run the complete data cleaning process"""
        print("Starting Data Leakage Feature Removal...")
        print("="*80)
        print("DATA LEAKAGE FEATURE REMOVAL - DELHI LOAD FORECASTING")
        print("="*80)
        print(f"Analysis started at: {datetime.now()}")
        
        # Load original data
        original_df = self.load_data()
        df = original_df.copy()
        
        # Step 1: Remove severe leakage features
        df = self.remove_severe_leakage_features(df)
        
        # Step 2: Remove identical features
        df = self.remove_identical_features(df)
        
        # Step 3: Detect and remove any remaining high correlations
        df = self.detect_and_remove_remaining_high_correlations(df)
        
        # Step 4: Validate target columns
        df = self.validate_target_columns(df)
        
        # Step 5: Generate summary
        summary = self.generate_feature_summary(original_df, df)
        
        # Step 6: Save cleaned dataset
        cleaned_path, summary_path = self.save_cleaned_dataset(df, summary)
        
        # Final summary
        print("\n" + "="*80)
        print("DATA CLEANING COMPLETED")
        print("="*80)
        print(f"Original features: {original_df.shape[1]}")
        print(f"Cleaned features: {df.shape[1]}")
        print(f"Features removed: {original_df.shape[1] - df.shape[1]}")
        print(f"Targets preserved: {len(self.removal_stats.get('targets_present', []))}/6")
        print(f"\nCleaned dataset: {cleaned_path}")
        print(f"Summary report: {summary_path}")
        print(f"Cleaning completed at: {datetime.now()}")
        
        return cleaned_path, summary_path


def main():
    """Main execution function"""
    # Set up paths
    input_path = "data_preprocessing/phase 2/delhi_interaction_enhanced.csv"
    output_dir = "phase_2_5_1_outputs"
    
    # Initialize cleaner
    cleaner = LeakageFeatureRemover(input_path, output_dir)
    
    # Run complete cleaning
    cleaned_path, summary_path = cleaner.run_complete_cleaning()
    
    print(f"\n🎉 SUCCESS! Clean dataset ready for modeling:")
    print(f"📁 Clean data: {cleaned_path}")
    print(f"📊 Summary: {summary_path}")
    
    return cleaned_path, summary_path


if __name__ == "__main__":
    main()
