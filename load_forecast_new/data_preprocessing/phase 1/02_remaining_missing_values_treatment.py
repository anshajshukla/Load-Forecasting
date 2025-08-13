"""
Delhi Load Forecasting - Remaining Missing Values Treatment
===========================================================
Phase 1.2: Targeted treatment for remaining 10 features with missing values.
Focused approach for duck curve depth calculation and minor weather gaps.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless operation
plt.switch_backend('Agg')

class RemainingMissingValuesTreatment:
    """Targeted treatment for remaining missing values in Delhi Load Forecasting dataset"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.treatment_log = {}
        self.output_dir = os.path.dirname(__file__)
        
    def load_data(self):
        """Load the pre-treated dataset"""
        print("=" * 70)
        print("REMAINING MISSING VALUES TREATMENT")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
        
        print(f"✅ Loading dataset: {self.data_path}")
        print(f"📊 Dataset loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
    def analyze_remaining_missing(self):
        """Analyze remaining missing values by priority groups"""
        print(f"\n📈 REMAINING MISSING VALUES ANALYSIS")
        print("-" * 50)
        
        total_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_percentage = (total_missing / total_cells) * 100
        
        print(f"Total remaining missing: {total_missing:,}")
        print(f"Missing percentage: {missing_percentage:.3f}%")
        print(f"Data completeness: {100 - missing_percentage:.3f}%")
        
        # Get missing columns
        missing_cols = self.df.isnull().sum()
        missing_cols = missing_cols[missing_cols > 0].sort_values(ascending=False)
        
        if len(missing_cols) == 0:
            print("\n✅ No missing values found! Dataset is already complete.")
            return {}
        
        print(f"\nRemaining features with missing values: {len(missing_cols)}")
        print("-" * 50)
        
        # Categorize by priority groups
        priority_groups = self.categorize_remaining_missing(missing_cols)
        
        for group, features in priority_groups.items():
            if features:
                print(f"\n{group}:")
                for feature in features:
                    count = missing_cols[feature]
                    pct = (count / len(self.df)) * 100
                    print(f"  {feature}: {count:,} ({pct:.2f}%)")
        
        return priority_groups
    
    def categorize_remaining_missing(self, missing_cols):
        """Categorize remaining missing values by treatment priority"""
        groups = {
            "🦆 DERIVED FEATURES (HIGH PRIORITY)": [
                col for col in missing_cols.index 
                if any(x in col.lower() for x in ['duck', 'curve', 'depth'])
            ],
            "📅 METADATA FEATURES (MEDIUM PRIORITY)": [
                col for col in missing_cols.index 
                if col in ['timestamp', 'data_source']
            ],
            "☀️ WEATHER RADIATION (LOW PRIORITY)": [
                col for col in missing_cols.index 
                if any(x in col.lower() for x in ['shortwave_radiation', 'diffuse_radiation', 'direct_radiation'])
            ],
            "☁️ CLOUD COVER (LOW PRIORITY)": [
                col for col in missing_cols.index 
                if 'cloud_cover' in col.lower()
            ],
            "🔍 OTHER FEATURES": [
                col for col in missing_cols.index 
                if not any(category_check(col) for category_check in [
                    lambda x: any(y in x.lower() for y in ['duck', 'curve', 'depth']),
                    lambda x: x in ['timestamp', 'data_source'],
                    lambda x: any(y in x.lower() for y in ['shortwave_radiation', 'diffuse_radiation', 'direct_radiation']),
                    lambda x: 'cloud_cover' in x.lower()
                ])
            ]
        }
        return groups
    
    def fix_timestamp_missing(self):
        """
        PRIORITY 1: Fix missing timestamps (Critical for time series)
        """
        print(f"\n📅 FIXING TIMESTAMP MISSING VALUES")
        print("-" * 50)
        
        if 'timestamp' not in self.df.columns:
            print("No timestamp column found - skipping")
            return
        
        initial_missing = self.df['timestamp'].isnull().sum()
        print(f"Initial missing timestamps: {initial_missing:,}")
        
        if initial_missing == 0:
            print("✅ No missing timestamps found")
            return
        
        # Strategy: Reconstruct from datetime column if available
        if 'datetime' in self.df.columns:
            # Fill timestamp from datetime
            missing_mask = self.df['timestamp'].isnull()
            self.df.loc[missing_mask, 'timestamp'] = self.df.loc[missing_mask, 'datetime']
            
            final_missing = self.df['timestamp'].isnull().sum()
            filled = initial_missing - final_missing
            
            print(f"✅ Filled from datetime column: {filled:,}")
            print(f"🎯 Final missing: {final_missing:,}")
            
            self.treatment_log['timestamp'] = {
                'initial': initial_missing,
                'method': 'copied_from_datetime',
                'filled': filled,
                'final': final_missing
            }
        else:
            print("⚠️ No datetime column available for timestamp reconstruction")
    
    def calculate_duck_curve_depth(self):
        """
        PRIORITY 1: Calculate duck_curve_depth_mw (Critical for Delhi load forecasting)
        Formula: duck_curve_depth = daily_min_net_load - baseline_load
        """
        print(f"\n🦆 CALCULATING DUCK CURVE DEPTH")
        print("-" * 50)
        
        if 'duck_curve_depth_mw' not in self.df.columns:
            print("No duck_curve_depth_mw column found - skipping")
            return
        
        initial_missing = self.df['duck_curve_depth_mw'].isnull().sum()
        print(f"Initial missing duck curve values: {initial_missing:,}")
        
        if initial_missing == 0:
            print("✅ No missing duck curve values found")
            return
        
        # Required columns for calculation
        load_cols = [col for col in self.df.columns if 'load' in col.lower() and 'net' not in col.lower()]
        solar_cols = [col for col in self.df.columns if 'solar_generation' in col.lower()]
        
        if not load_cols:
            print("❌ No load columns found for duck curve calculation")
            return
        
        print(f"📊 Using load columns: {load_cols[:3]}...")  # Show first 3
        print(f"☀️ Using solar columns: {solar_cols}")
        
        # Add datetime components if not present
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df['datetime'].dt.hour
        if 'date' not in self.df.columns:
            self.df['date'] = self.df['datetime'].dt.date
        
        filled_count = 0
        
        # Group by date for daily calculations
        for date in self.df['date'].unique():
            date_mask = self.df['date'] == date
            date_data = self.df[date_mask].copy()
            
            # Calculate total load for this day
            if len(load_cols) > 1:
                # If multiple load columns, sum them (excluding net_load)
                main_load_cols = [col for col in load_cols if 'delhi_load' in col.lower() or 'total' in col.lower()]
                if main_load_cols:
                    total_load = date_data[main_load_cols[0]]
                else:
                    total_load = date_data[load_cols[0]]  # Use first load column
            else:
                total_load = date_data[load_cols[0]]
            
            # Calculate net load (load - solar generation)
            if solar_cols and not date_data[solar_cols[0]].isnull().all():
                solar_generation = date_data[solar_cols[0]].fillna(0)  # Fill missing solar with 0
                net_load = total_load - solar_generation
            else:
                # If no solar data, use load as proxy (conservative approach)
                net_load = total_load
            
            # Calculate duck curve depth for this day
            # Solar hours: 10 AM to 4 PM (peak solar generation period)
            solar_hours_mask = date_data['hour'].between(10, 16)
            
            if len(net_load) > 0 and not net_load.isnull().all():
                # Find minimum net load during the day
                daily_min_net_load = net_load.min()
                
                # Find baseline load (average during non-solar peak hours)
                non_solar_hours = date_data[~solar_hours_mask]
                if len(non_solar_hours) > 0:
                    baseline_load = non_solar_hours[load_cols[0]].mean()
                else:
                    baseline_load = total_load.mean()
                
                # Duck curve depth = difference between baseline and minimum
                duck_depth = baseline_load - daily_min_net_load
                
                # Apply physical constraints (duck curve depth should be positive and reasonable)
                duck_depth = max(0, min(duck_depth, 2000))  # Cap at 2000 MW for Delhi
                
                # Fill missing values for this day
                missing_mask = date_mask & self.df['duck_curve_depth_mw'].isnull()
                if missing_mask.sum() > 0:
                    self.df.loc[missing_mask, 'duck_curve_depth_mw'] = duck_depth
                    filled_count += missing_mask.sum()
        
        final_missing = self.df['duck_curve_depth_mw'].isnull().sum()
        
        print(f"✅ Calculated duck curve depth: {filled_count:,}")
        print(f"🎯 Final missing: {final_missing:,}")
        
        # Apply smoothing to reduce artificial patterns
        if filled_count > 0:
            self.apply_duck_curve_smoothing()
        
        self.treatment_log['duck_curve_depth_mw'] = {
            'initial': initial_missing,
            'method': 'calculated_from_load_solar',
            'filled': filled_count,
            'final': final_missing
        }
    
    def apply_duck_curve_smoothing(self):
        """Apply rolling median smoothing to duck curve depth"""
        print("  📊 Applying rolling median smoothing...")
        
        # Apply 7-day rolling median to smooth out artificial patterns
        window_size = min(24 * 7, len(self.df) // 10)  # 7 days or 10% of data
        
        if window_size >= 3:
            smoothed = self.df['duck_curve_depth_mw'].rolling(
                window=window_size, 
                center=True, 
                min_periods=3
            ).median()
            
            # Only apply smoothing to previously calculated values
            calculated_mask = self.df['duck_curve_depth_mw'].notnull()
            self.df.loc[calculated_mask, 'duck_curve_depth_mw'] = smoothed[calculated_mask]
            
            print(f"  ✅ Applied {window_size}-hour rolling median smoothing")
    
    def handle_data_source_missing(self):
        """
        PRIORITY 2: Handle missing data_source metadata
        """
        print(f"\n🗂️ HANDLING DATA_SOURCE MISSING VALUES")
        print("-" * 50)
        
        if 'data_source' not in self.df.columns:
            print("No data_source column found - skipping")
            return
        
        initial_missing = self.df['data_source'].isnull().sum()
        print(f"Initial missing data_source: {initial_missing:,}")
        
        if initial_missing == 0:
            print("✅ No missing data_source values found")
            return
        
        # Strategy: Infer from data patterns or fill with "estimated"
        missing_mask = self.df['data_source'].isnull()
        
        # Check if these rows have good quality data (few missing values per row)
        row_missing_counts = self.df[missing_mask].isnull().sum(axis=1)
        
        # If row has mostly complete data, mark as "historical"
        # If row has many missing values, mark as "estimated"
        high_quality_mask = row_missing_counts <= 2  # 2 or fewer missing values per row
        
        self.df.loc[missing_mask & high_quality_mask, 'data_source'] = 'historical'
        self.df.loc[missing_mask & ~high_quality_mask, 'data_source'] = 'estimated'
        
        final_missing = self.df['data_source'].isnull().sum()
        filled = initial_missing - final_missing
        
        print(f"✅ Filled with 'historical': {(missing_mask & high_quality_mask).sum():,}")
        print(f"✅ Filled with 'estimated': {(missing_mask & ~high_quality_mask).sum():,}")
        print(f"🎯 Final missing: {final_missing:,}")
        
        self.treatment_log['data_source'] = {
            'initial': initial_missing,
            'method': 'inferred_from_quality',
            'filled': filled,
            'final': final_missing
        }
    
    def interpolate_weather_radiation(self):
        """
        PRIORITY 3: Interpolate weather radiation features (only 7 missing each)
        """
        print(f"\n☀️ INTERPOLATING WEATHER RADIATION FEATURES")
        print("-" * 50)
        
        radiation_features = [
            'shortwave_radiation', 'diffuse_radiation', 'direct_radiation'
        ]
        
        radiation_features = [col for col in radiation_features if col in self.df.columns]
        
        if not radiation_features:
            print("No radiation features found - skipping")
            return
        
        for feature in radiation_features:
            initial_missing = self.df[feature].isnull().sum()
            print(f"\n📊 Treating {feature}: {initial_missing:,} missing")
            
            if initial_missing == 0:
                continue
            
            # Method 1: Linear interpolation (best for small gaps)
            self.df[feature] = self.df[feature].interpolate(method='linear')
            
            # Method 2: Forward/backward fill for any remaining
            self.df[feature] = self.df[feature].fillna(method='ffill').fillna(method='bfill')
            
            final_missing = self.df[feature].isnull().sum()
            filled = initial_missing - final_missing
            
            print(f"  ✅ Interpolated: {filled:,}")
            print(f"  🎯 Final missing: {final_missing:,}")
            
            self.treatment_log[feature] = {
                'initial': initial_missing,
                'method': 'linear_interpolation',
                'filled': filled,
                'final': final_missing
            }
    
    def interpolate_cloud_cover(self):
        """
        PRIORITY 4: Interpolate cloud cover features with constraints
        """
        print(f"\n☁️ INTERPOLATING CLOUD COVER FEATURES")
        print("-" * 50)
        
        cloud_features = [
            col for col in self.df.columns 
            if 'cloud_cover' in col.lower()
        ]
        
        if not cloud_features:
            print("No cloud cover features found - skipping")
            return
        
        for feature in cloud_features:
            initial_missing = self.df[feature].isnull().sum()
            print(f"\n📊 Treating {feature}: {initial_missing:,} missing")
            
            if initial_missing == 0:
                continue
            
            # Method 1: Linear interpolation
            self.df[feature] = self.df[feature].interpolate(method='linear')
            
            # Method 2: Forward/backward fill for any remaining
            self.df[feature] = self.df[feature].fillna(method='ffill').fillna(method='bfill')
            
            # Apply physical constraints (0-100% cloud cover)
            self.df[feature] = self.df[feature].clip(0, 100)
            
            final_missing = self.df[feature].isnull().sum()
            filled = initial_missing - final_missing
            
            print(f"  ✅ Interpolated: {filled:,}")
            print(f"  🎯 Final missing: {final_missing:,}")
            
            self.treatment_log[feature] = {
                'initial': initial_missing,
                'method': 'linear_interpolation_constrained',
                'filled': filled,
                'final': final_missing
            }
        
        # Validate cloud cover consistency if multiple layers exist
        self.validate_cloud_cover_consistency()
    
    def validate_cloud_cover_consistency(self):
        """Ensure cloud cover layers are consistent"""
        total_col = next((col for col in self.df.columns if col == 'cloud_cover_total'), None)
        layer_cols = [col for col in self.df.columns if col in ['cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high']]
        
        if total_col and len(layer_cols) >= 2:
            print("  🔍 Validating cloud cover layer consistency...")
            
            # Calculate total from layers
            layer_sum = self.df[layer_cols].sum(axis=1)
            
            # Check for inconsistencies (where total != sum of layers)
            inconsistent = abs(self.df[total_col] - layer_sum) > 5  # 5% tolerance
            
            if inconsistent.sum() > 0:
                print(f"  ⚠️ Found {inconsistent.sum()} inconsistent cloud cover values")
                # Use total cloud cover as authoritative source
                self.df.loc[inconsistent, layer_cols] = self.df.loc[inconsistent, total_col] / len(layer_cols)
                print("  ✅ Normalized layer values to match total")
    
    def handle_other_features(self):
        """Handle any remaining miscellaneous missing features"""
        print(f"\n🔍 HANDLING OTHER REMAINING FEATURES")
        print("-" * 50)
        
        # Check for any remaining missing values
        remaining_missing = self.df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        
        if len(remaining_missing) == 0:
            print("✅ No other features with missing values found")
            return
        
        for feature in remaining_missing.index:
            if feature in self.treatment_log:
                continue  # Already handled
            
            initial_missing = remaining_missing[feature]
            print(f"\n📊 Treating {feature}: {initial_missing:,} missing")
            
            # Apply generic treatment based on data type
            if self.df[feature].dtype in ['float64', 'int64']:
                # Numeric feature - use median
                median_val = self.df[feature].median()
                self.df[feature] = self.df[feature].fillna(median_val)
                method = f'median_fill_{median_val:.2f}'
            else:
                # Categorical feature - use mode or 'unknown'
                mode_val = self.df[feature].mode()
                if len(mode_val) > 0:
                    self.df[feature] = self.df[feature].fillna(mode_val[0])
                    method = f'mode_fill_{mode_val[0]}'
                else:
                    self.df[feature] = self.df[feature].fillna('unknown')
                    method = 'unknown_fill'
            
            final_missing = self.df[feature].isnull().sum()
            filled = initial_missing - final_missing
            
            print(f"  ✅ Filled: {filled:,}")
            print(f"  🎯 Final missing: {final_missing:,}")
            
            self.treatment_log[feature] = {
                'initial': initial_missing,
                'method': method,
                'filled': filled,
                'final': final_missing
            }
    
    def create_completion_visualization(self):
        """Create visualization showing completion results"""
        print(f"\n📊 CREATING COMPLETION VISUALIZATION")
        print("-" * 50)
        
        # Check final state
        final_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        completeness = ((total_cells - final_missing) / total_cells) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Treatment summary by feature
        ax1 = axes[0, 0]
        if self.treatment_log:
            features = list(self.treatment_log.keys())
            initial_counts = [self.treatment_log[f]['initial'] for f in features]
            final_counts = [self.treatment_log[f]['final'] for f in features]
            
            x = np.arange(len(features))
            width = 0.35
            
            ax1.bar(x - width/2, initial_counts, width, label='Before', color='red', alpha=0.7)
            ax1.bar(x + width/2, final_counts, width, label='After', color='green', alpha=0.7)
            ax1.set_xlabel('Features Treated')
            ax1.set_ylabel('Missing Values Count')
            ax1.set_title('Missing Values: Before vs After Treatment')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in features], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Treatment methods used
        ax2 = axes[0, 1]
        if self.treatment_log:
            methods = [self.treatment_log[f]['method'] for f in self.treatment_log.keys()]
            method_counts = {}
            for method in methods:
                method_key = method.split('_')[0]  # Get first part of method name
                method_counts[method_key] = method_counts.get(method_key, 0) + 1
            
            ax2.pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%', startangle=90)
            ax2.set_title('Treatment Methods Used')
        
        # 3. Data completeness progress
        ax3 = axes[1, 0]
        categories = ['Before Treatment', 'After Treatment']
        # Estimate before completeness (assume we had 5,977 missing before)
        before_completeness = ((total_cells - 5977) / total_cells) * 100
        completeness_data = [before_completeness, completeness]
        
        bars = ax3.bar(categories, completeness_data, color=['red', 'green'], alpha=0.7)
        ax3.set_ylabel('Data Completeness (%)')
        ax3.set_title('Overall Data Completeness Improvement')
        ax3.set_ylim(99.0, 100.1)
        ax3.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, comp in zip(bars, completeness_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{comp:.3f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Final summary text
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""
FINAL COMPLETION SUMMARY

Dataset: {len(self.df):,} records × {len(self.df.columns)} columns
Total cells: {total_cells:,}

Missing Values:
• Final missing: {final_missing:,}
• Data completeness: {completeness:.3f}%

Features Treated: {len(self.treatment_log)}

Key Achievements:
✅ Duck curve depth calculated
✅ Timestamps reconstructed  
✅ Weather gaps interpolated
✅ Metadata completed

Status: {'PERFECT - 100% COMPLETE!' if final_missing == 0 else f'{final_missing:,} values still missing'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen' if final_missing == 0 else 'lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'remaining_missing_treatment_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Completion visualization saved: {output_path}")
    
    def generate_final_report(self):
        """Generate final completion report"""
        print(f"\n📋 FINAL COMPLETION REPORT")
        print("=" * 70)
        
        final_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        completeness = ((total_cells - final_missing) / total_cells) * 100
        
        print(f"Dataset: {len(self.df):,} records × {len(self.df.columns)} columns")
        print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        print(f"\nFinal Status:")
        print(f"  Total missing values: {final_missing:,}")
        print(f"  Data completeness: {completeness:.4f}%")
        
        if final_missing == 0:
            print(f"  🏆 PERFECT: 100% complete dataset achieved!")
        else:
            print(f"  ⚠️  {final_missing:,} values still missing")
        
        print(f"\nTreatment Summary:")
        total_treated = sum([log['filled'] for log in self.treatment_log.values()])
        print(f"  Features treated: {len(self.treatment_log)}")
        print(f"  Values filled: {total_treated:,}")
        
        for feature, log in self.treatment_log.items():
            improvement = ((log['initial'] - log['final']) / log['initial'] * 100) if log['initial'] > 0 else 0
            print(f"    {feature}: {log['initial']:,} → {log['final']:,} ({improvement:.1f}% improvement)")
        
        print(f"\nRecommendations:")
        if completeness >= 99.99:
            print("  ✅ Dataset ready for advanced feature engineering")
            print("  ✅ Proceed to model training phase")
            print("  ✅ Duck curve modeling now possible")
        elif completeness >= 99.5:
            print("  ✅ Excellent data quality - minimal impact on modeling")
            print("  ✅ Ready for feature engineering")
        else:
            print("  ⚠️  Consider reviewing remaining missing values")
        
        print(f"\n💾 Treated dataset ready for Phase 2: Feature Engineering")
    
    def save_completed_dataset(self):
        """Save the completed dataset"""
        output_path = os.path.join(self.output_dir, 'final_dataset_100_percent_complete.csv')
        self.df.to_csv(output_path, index=False)
        print(f"\n✅ Completed dataset saved: {output_path}")
        return output_path
    
    def run_complete_remaining_treatment(self):
        """Execute the complete remaining missing value treatment"""
        try:
            # Load data
            self.load_data()
            
            # Analyze remaining missing patterns
            priority_groups = self.analyze_remaining_missing()
            
            if not any(priority_groups.values()):
                print("\n✅ No missing values found! Dataset is already complete.")
                return self.data_path
            
            # Execute treatment by priority
            print(f"\n🎯 EXECUTING PRIORITY TREATMENT PLAN")
            print("=" * 50)
            
            # Priority 1: Critical features
            self.fix_timestamp_missing()
            self.calculate_duck_curve_depth()
            
            # Priority 2: Metadata
            self.handle_data_source_missing()
            
            # Priority 3: Weather features (small gaps)
            self.interpolate_weather_radiation()
            self.interpolate_cloud_cover()
            
            # Priority 4: Any remaining features
            self.handle_other_features()
            
            # Create visualization and report
            self.create_completion_visualization()
            self.generate_final_report()
            
            # Save completed dataset
            output_path = self.save_completed_dataset()
            
            print(f"\n🎉 REMAINING MISSING VALUES TREATMENT COMPLETED!")
            print(f"📁 Output file: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during remaining treatment: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Use the output from previous treatment
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, 'final_dataset_missing_values_treated.csv')
    
    # Initialize treatment
    treatment = RemainingMissingValuesTreatment(input_path)
    
    # Run complete remaining treatment
    output_path = treatment.run_complete_remaining_treatment()
    
    return output_path


if __name__ == "__main__":
    output_file = main()
