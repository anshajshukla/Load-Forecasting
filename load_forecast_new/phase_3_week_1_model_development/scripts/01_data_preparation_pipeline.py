"""
Delhi Load Forecasting - Phase 3 Week 1
Day 1-2: Data Preparation Pipeline Implementation

This script implements the comprehensive data preparation pipeline for baseline model establishment.
Includes time-based splits, feature scaling, cross-validation setup, and evaluation metrics.

Target: Prepare 111 validated features for baseline modeling
Timeline: Days 1-2 of Week 1 baseline establishment
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import joblib
import json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DelhiLoadDataPreparation:
    """
    Comprehensive data preparation pipeline for Delhi Load Forecasting
    
    Features:
    - Time-based train/validation/test splits (70/15/15)
    - Feature scaling and normalization for 111 selected features
    - Walk-forward cross-validation framework
    - Delhi-specific evaluation metrics implementation
    """
    
    def __init__(self, data_path, output_dir):
        """Initialize data preparation pipeline"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.create_output_directories()
        
        # Data containers
        self.data = None
        self.features = None
        self.target_columns = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
        
        # Split containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Scalers
        self.feature_scaler = None
        self.target_scaler = None
        
        # Cross-validation
        self.cv_splits = None
        
        # Metadata
        self.preparation_metadata = {}
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            os.path.join(self.output_dir, 'data'),
            os.path.join(self.output_dir, 'scalers'),
            os.path.join(self.output_dir, 'metadata'),
            os.path.join(self.output_dir, 'visualizations')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print("[OK] Output directories created successfully")
    
    def load_and_validate_data(self):
        """Load and validate the feature-selected dataset"""
        print("\n🔄 Loading Delhi feature-selected dataset...")
        
        try:
            # Load the 111 validated features dataset
            self.data = pd.read_csv(self.data_path)
            
            # Convert datetime column
            self.data['datetime'] = pd.to_datetime(self.data['datetime'])
            self.data.set_index('datetime', inplace=True)
            
            # Sort by datetime
            self.data.sort_index(inplace=True)
            
            print(f"[OK] Data loaded successfully")
            print(f"   📊 Shape: {self.data.shape}")
            print(f"   📅 Date range: {self.data.index.min()} to {self.data.index.max()}")
            print(f"   🎯 Target variables: {len(self.target_columns)}")
            
            # Extract features (exclude target columns)
            self.features = [col for col in self.data.columns if col not in self.target_columns]
            print(f"   🔧 Features available: {len(self.features)}")
            
            # Validate data quality
            self.validate_data_quality()
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise
    
    def validate_data_quality(self):
        """Validate data quality and completeness"""
        print("\n🔄 Validating data quality...")
        
        # Check for missing values
        missing_info = self.data.isnull().sum()
        missing_features = missing_info[missing_info > 0]
        
        if len(missing_features) > 0:
            print(f"⚠️  Missing values found in {len(missing_features)} features:")
            for feature, count in missing_features.head(10).items():
                pct = (count / len(self.data)) * 100
                print(f"     {feature}: {count} ({pct:.2f}%)")
        else:
            print("[OK] No missing values found")
        
        # Check for infinite values
        inf_check = np.isinf(self.data.select_dtypes(include=[np.number])).sum().sum()
        print(f"   Infinite values: {inf_check}")
        
        # Data completeness
        completeness = (1 - self.data.isnull().sum().sum() / (self.data.shape[0] * self.data.shape[1])) * 100
        print(f"   Data completeness: {completeness:.2f}%")
        
        # Store validation results
        self.preparation_metadata['data_quality'] = {
            'total_records': len(self.data),
            'total_features': len(self.features),
            'missing_values': int(missing_info.sum()),
            'infinite_values': int(inf_check),
            'completeness_percent': round(completeness, 2),
            'date_range': {
                'start': str(self.data.index.min()),
                'end': str(self.data.index.max())
            }
        }
    
    def create_time_based_splits(self, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
        """Create time-based train/validation/test splits (70/15/15)"""
        print(f"\n🔄 Creating time-based splits ({train_ratio*100:.0f}%/{val_ratio*100:.0f}%/{test_ratio*100:.0f}%)...")
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
        
        # Calculate split indices
        n_samples = len(self.data)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)
        
        # Create splits based on time order
        train_data = self.data.iloc[:train_size]
        val_data = self.data.iloc[train_size:train_size + val_size]
        test_data = self.data.iloc[train_size + val_size:]
        
        # Separate features and targets
        self.X_train = train_data[self.features]
        self.X_val = val_data[self.features]
        self.X_test = test_data[self.features]
        
        self.y_train = train_data[self.target_columns]
        self.y_val = val_data[self.target_columns]
        self.y_test = test_data[self.target_columns]
        
        print("[OK] Time-based splits created successfully")
        print(f"   📊 Training set: {len(self.X_train)} samples ({self.X_train.index.min()} to {self.X_train.index.max()})")
        print(f"   📊 Validation set: {len(self.X_val)} samples ({self.X_val.index.min()} to {self.X_val.index.max()})")
        print(f"   📊 Test set: {len(self.X_test)} samples ({self.X_test.index.min()} to {self.X_test.index.max()})")
        
        # Store split metadata
        self.preparation_metadata['data_splits'] = {
            'train_size': len(self.X_train),
            'val_size': len(self.X_val),
            'test_size': len(self.X_test),
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'train_period': {
                'start': str(self.X_train.index.min()),
                'end': str(self.X_train.index.max())
            },
            'val_period': {
                'start': str(self.X_val.index.min()),
                'end': str(self.X_val.index.max())
            },
            'test_period': {
                'start': str(self.X_test.index.min()),
                'end': str(self.X_test.index.max())
            }
        }
    
    def setup_feature_scaling(self, scaler_type='robust'):
        """Setup feature scaling and normalization"""
        print(f"\n🔄 Setting up feature scaling ({scaler_type})...")
        
        # Initialize scaler based on type
        if scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit scaler on training data only
        self.feature_scaler.fit(self.X_train)
        
        # Transform all sets
        X_train_scaled = self.feature_scaler.transform(self.X_train)
        X_val_scaled = self.feature_scaler.transform(self.X_val)
        X_test_scaled = self.feature_scaler.transform(self.X_test)
        
        # Convert back to DataFrames with original indices and column names
        self.X_train_scaled = pd.DataFrame(
            X_train_scaled, 
            index=self.X_train.index, 
            columns=self.X_train.columns
        )
        self.X_val_scaled = pd.DataFrame(
            X_val_scaled, 
            index=self.X_val.index, 
            columns=self.X_val.columns
        )
        self.X_test_scaled = pd.DataFrame(
            X_test_scaled, 
            index=self.X_test.index, 
            columns=self.X_test.columns
        )
        
        print("[OK] Feature scaling completed successfully")
        print(f"   🔧 Scaler type: {scaler_type}")
        print(f"   📊 Features scaled: {len(self.features)}")
        
        # Store scaling metadata
        self.preparation_metadata['feature_scaling'] = {
            'scaler_type': scaler_type,
            'features_count': len(self.features),
            'scaling_stats': {
                'train_mean': float(self.X_train_scaled.mean().mean()),
                'train_std': float(self.X_train_scaled.std().mean()),
                'val_mean': float(self.X_val_scaled.mean().mean()),
                'val_std': float(self.X_val_scaled.std().mean())
            }
        }
    
    def setup_walk_forward_cv(self, n_splits=5, max_train_size=None):
        """Setup walk-forward time series cross-validation"""
        print(f"\n🔄 Setting up walk-forward cross-validation (n_splits={n_splits})...")
        
        # Initialize TimeSeriesSplit
        self.cv_splits = TimeSeriesSplit(
            n_splits=n_splits,
            max_train_size=max_train_size
        )
        
        # Test CV splits on training data
        cv_info = []
        for i, (train_idx, val_idx) in enumerate(self.cv_splits.split(self.X_train)):
            train_period = (
                self.X_train.index[train_idx[0]],
                self.X_train.index[train_idx[-1]]
            )
            val_period = (
                self.X_train.index[val_idx[0]],
                self.X_train.index[val_idx[-1]]
            )
            
            cv_info.append({
                'fold': i + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_period': train_period,
                'val_period': val_period
            })
            
            print(f"   Fold {i+1}: Train {len(train_idx)} samples, Val {len(val_idx)} samples")
        
        print("[OK] Walk-forward cross-validation setup completed")
        
        # Store CV metadata
        self.preparation_metadata['cross_validation'] = {
            'type': 'TimeSeriesSplit',
            'n_splits': n_splits,
            'max_train_size': max_train_size,
            'cv_folds': cv_info
        }
    
    def implement_evaluation_metrics(self):
        """Implement comprehensive evaluation metrics including Delhi-specific ones"""
        print("\n🔄 Setting up evaluation metrics framework...")
        
        def calculate_metrics(y_true, y_pred, target_name=""):
            """Calculate comprehensive metrics for a single target"""
            # Handle any NaN or infinite values
            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {f'{target_name}_mape': np.nan, f'{target_name}_mae': np.nan, f'{target_name}_rmse': np.nan}
            
            # Core metrics
            mape = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            
            return {
                f'{target_name}_mape': mape,
                f'{target_name}_mae': mae,
                f'{target_name}_rmse': rmse
            }
        
        def calculate_delhi_specific_metrics(y_true_df, y_pred_df):
            """Calculate Delhi-specific evaluation metrics"""
            metrics = {}
            
            # Overall delhi_load metrics
            delhi_metrics = calculate_metrics(
                y_true_df['delhi_load'].values, 
                y_pred_df['delhi_load'].values, 
                'delhi'
            )
            metrics.update(delhi_metrics)
            
            # Component DISCOM metrics
            for discom in ['brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']:
                if discom in y_true_df.columns and discom in y_pred_df.columns:
                    discom_name = discom.replace('_load', '')
                    discom_metrics = calculate_metrics(
                        y_true_df[discom].values,
                        y_pred_df[discom].values,
                        discom_name
                    )
                    metrics.update(discom_metrics)
            
            # Peak hour analysis (hours 10-12 and 18-22)
            peak_hours = [10, 11, 18, 19, 20, 21]
            if hasattr(y_true_df.index, 'hour'):
                peak_mask = y_true_df.index.hour.isin(peak_hours)
                if peak_mask.sum() > 0:
                    peak_metrics = calculate_metrics(
                        y_true_df.loc[peak_mask, 'delhi_load'].values,
                        y_pred_df.loc[peak_mask, 'delhi_load'].values,
                        'delhi_peak'
                    )
                    metrics.update(peak_metrics)
            
            # Average MAPE across all DISCOMs
            discom_mapes = [metrics.get(f'{discom.replace("_load", "")}_mape', np.nan) 
                           for discom in ['brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']]
            valid_mapes = [mape for mape in discom_mapes if not np.isnan(mape)]
            if valid_mapes:
                metrics['average_discom_mape'] = np.mean(valid_mapes)
            
            return metrics
        
        # Store evaluation function
        self.evaluate_predictions = calculate_delhi_specific_metrics
        
        print("✅ Evaluation metrics framework implemented")
        print("   📊 Core metrics: MAPE, MAE, RMSE")
        print("   🎯 Delhi-specific: Peak hour analysis, DISCOM components")
        print("   ⚡ Target performance: MAPE <10% for baselines")
    
    def save_prepared_data(self):
        """Save all prepared data and metadata"""
        print("\n🔄 Saving prepared data and artifacts...")
        
        data_dir = os.path.join(self.output_dir, 'data')
        scalers_dir = os.path.join(self.output_dir, 'scalers')
        metadata_dir = os.path.join(self.output_dir, 'metadata')
        
        # Save data splits (both original and scaled)
        data_files = {
            'X_train.csv': self.X_train,
            'X_val.csv': self.X_val,
            'X_test.csv': self.X_test,
            'y_train.csv': self.y_train,
            'y_val.csv': self.y_val,
            'y_test.csv': self.y_test,
            'X_train_scaled.csv': self.X_train_scaled,
            'X_val_scaled.csv': self.X_val_scaled,
            'X_test_scaled.csv': self.X_test_scaled
        }
        
        for filename, data in data_files.items():
            filepath = os.path.join(data_dir, filename)
            data.to_csv(filepath)
            print(f"   💾 Saved: {filename}")
        
        # Save scalers
        scaler_path = os.path.join(scalers_dir, 'feature_scaler.pkl')
        joblib.dump(self.feature_scaler, scaler_path)
        print(f"   💾 Saved: feature_scaler.pkl")
        
        # Save metadata
        self.preparation_metadata['preparation_timestamp'] = datetime.now().isoformat()
        self.preparation_metadata['feature_list'] = self.features
        self.preparation_metadata['target_columns'] = self.target_columns
        
        metadata_path = os.path.join(metadata_dir, 'preparation_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.preparation_metadata, f, indent=2, default=str)
        print(f"   💾 Saved: preparation_metadata.json")
        
        print("✅ All prepared data and artifacts saved successfully")
    
    def create_data_visualization(self):
        """Create visualizations of the prepared data"""
        print("\n🔄 Creating data visualizations...")
        
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        
        # 1. Data splits timeline
        plt.figure(figsize=(15, 8))
        
        # Plot delhi_load for all splits
        plt.subplot(2, 2, 1)
        plt.plot(self.y_train.index, self.y_train['delhi_load'], label='Training', alpha=0.7)
        plt.plot(self.y_val.index, self.y_val['delhi_load'], label='Validation', alpha=0.7)
        plt.plot(self.y_test.index, self.y_test['delhi_load'], label='Test', alpha=0.7)
        plt.title('Delhi Load - Data Splits Timeline')
        plt.xlabel('Date')
        plt.ylabel('Delhi Load (MW)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 2. Feature scaling comparison
        plt.subplot(2, 2, 2)
        feature_sample = self.features[:5]  # Plot first 5 features
        before_scaling = self.X_train[feature_sample].values.flatten()
        after_scaling = self.X_train_scaled[feature_sample].values.flatten()
        
        plt.hist(before_scaling, bins=50, alpha=0.7, label='Before Scaling', density=True)
        plt.hist(after_scaling, bins=50, alpha=0.7, label='After Scaling', density=True)
        plt.title('Feature Scaling Effect (Sample Features)')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
        
        # 3. Target variable distributions
        plt.subplot(2, 2, 3)
        for target in self.target_columns:
            if target in self.y_train.columns:
                plt.hist(self.y_train[target], bins=30, alpha=0.6, label=target.replace('_load', ''))
        plt.title('Target Variable Distributions (Training Set)')
        plt.xlabel('Load (MW)')
        plt.ylabel('Frequency')
        plt.legend()
        
        # 4. Cross-validation splits illustration
        plt.subplot(2, 2, 4)
        for i, (train_idx, val_idx) in enumerate(self.cv_splits.split(self.X_train)):
            plt.scatter([i] * len(train_idx), train_idx, alpha=0.3, s=1, label='Train' if i == 0 else "")
            plt.scatter([i] * len(val_idx), val_idx, alpha=0.7, s=1, label='Validation' if i == 0 else "")
        plt.title('Cross-Validation Splits')
        plt.xlabel('CV Fold')
        plt.ylabel('Sample Index')
        plt.legend()
        
        plt.tight_layout()
        viz_path = os.path.join(viz_dir, 'data_preparation_overview.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved: data_preparation_overview.png")
        print("✅ Data visualizations created successfully")
    
    def run_complete_pipeline(self):
        """Run the complete data preparation pipeline"""
        print("[STARTING] Delhi Load Forecasting Data Preparation Pipeline")
        print("=" * 80)
        
        try:
            # Step 1: Load and validate data
            self.load_and_validate_data()
            
            # Step 2: Create time-based splits
            self.create_time_based_splits()
            
            # Step 3: Setup feature scaling
            self.setup_feature_scaling()
            
            # Step 4: Setup cross-validation
            self.setup_walk_forward_cv()
            
            # Step 5: Implement evaluation metrics
            self.implement_evaluation_metrics()
            
            # Step 6: Save prepared data
            self.save_prepared_data()
            
            # Step 7: Create visualizations
            self.create_data_visualization()
            
            print("\n🎉 Data Preparation Pipeline Completed Successfully!")
            print("=" * 80)
            print(f"📊 Training samples: {len(self.X_train)}")
            print(f"📊 Validation samples: {len(self.X_val)}")
            print(f"📊 Test samples: {len(self.X_test)}")
            print(f"🔧 Features prepared: {len(self.features)}")
            print(f"🎯 Target variables: {len(self.target_columns)}")
            print(f"💾 Output directory: {self.output_dir}")
            print("\n✅ Ready for baseline model development!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    # Configuration
    data_path = r"C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_2_5_3_outputs\delhi_selected_features.csv"
    output_dir = r"C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_1_model_development"
    
    # Initialize and run pipeline
    pipeline = DelhiLoadDataPreparation(data_path, output_dir)
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Proceed to Day 3-4: Linear & Tree-Based Baselines")
        print("   2. Use prepared data in 'data/' directory")
        print("   3. Use feature_scaler.pkl for consistent preprocessing")
        print("   4. Target baseline MAPE <10% for model progression")

if __name__ == "__main__":
    main()
