"""
Delhi Load Forecasting - Phase 2: Delhi Dual Peak Feature Engineering
=====================================================================
Advanced feature engineering leveraging the 100% complete dataset.
Focus on Delhi's unique dual peak pattern and duck curve characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless operation
plt.switch_backend('Agg')

class DelhiDualPeakFeatureEngineering:
    """
    Advanced feature engineering for Delhi's dual peak load pattern
    Implements duck curve analysis and peak detection features
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.feature_log = {}
        self.output_dir = os.path.dirname(__file__)
        
        # Delhi-specific constants
        self.MORNING_PEAK_WINDOW = (6, 11)    # 6 AM - 11 AM
        self.EVENING_PEAK_WINDOW = (18, 23)   # 6 PM - 11 PM
        self.SOLAR_PEAK_WINDOW = (11, 16)     # 11 AM - 4 PM (solar generation peak)
        self.DUCK_CURVE_WINDOW = (10, 20)     # 10 AM - 8 PM (duck curve period)
        
    def load_complete_dataset(self):
        """Load the 100% complete dataset from Phase 1"""
        print("=" * 70)
        print("DELHI DUAL PEAK FEATURE ENGINEERING - PHASE 2")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Complete dataset not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
        
        print(f"✅ Loading 100% complete dataset: {os.path.basename(self.data_path)}")
        print(f"📊 Dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        # Verify completeness
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️  Warning: {missing_count} missing values found!")
        else:
            print(f"✅ Verified: 100% complete dataset")
        
        # Add time components if not present
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df['datetime'].dt.hour
        if 'date' not in self.df.columns:
            self.df['date'] = self.df['datetime'].dt.date
        if 'day_of_week' not in self.df.columns:
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
    
    def create_peak_detection_features(self):
        """
        A) Peak Detection Features for Delhi's Dual Peak Pattern
        """
        print(f"\n🏔️ CREATING PEAK DETECTION FEATURES")
        print("-" * 50)
        
        # Find load column
        load_cols = [col for col in self.df.columns if 'delhi_load' in col.lower()]
        if not load_cols:
            load_cols = [col for col in self.df.columns if 'load' in col.lower() and 'net' not in col.lower()]
        
        if not load_cols:
            print("❌ No load column found for peak detection")
            return
        
        load_col = load_cols[0]
        print(f"📊 Using load column: {load_col}")
        
        # Initialize new feature columns
        new_features = {}
        
        # Group by date for daily peak analysis
        for date in self.df['date'].unique():
            date_mask = self.df['date'] == date
            daily_data = self.df[date_mask].copy().sort_values('hour')
            
            if len(daily_data) < 20:  # Need sufficient hourly data
                continue
            
            daily_load = daily_data[load_col].values
            hours = daily_data['hour'].values
            
            # A1) Morning Peak Intensity (6 AM - 11 AM)
            morning_mask = (hours >= self.MORNING_PEAK_WINDOW[0]) & (hours <= self.MORNING_PEAK_WINDOW[1])
            if morning_mask.sum() > 0:
                morning_peak_intensity = daily_load[morning_mask].max()
                morning_peak_time = hours[morning_mask][np.argmax(daily_load[morning_mask])]
            else:
                morning_peak_intensity = np.nan
                morning_peak_time = np.nan
            
            # A2) Evening Peak Intensity (6 PM - 11 PM)
            evening_mask = (hours >= self.EVENING_PEAK_WINDOW[0]) & (hours <= self.EVENING_PEAK_WINDOW[1])
            if evening_mask.sum() > 0:
                evening_peak_intensity = daily_load[evening_mask].max()
                evening_peak_time = hours[evening_mask][np.argmax(daily_load[evening_mask])]
            else:
                evening_peak_intensity = np.nan
                evening_peak_time = np.nan
            
            # A3) Peak Amplitude Ratio (evening/morning)
            if not np.isnan(morning_peak_intensity) and morning_peak_intensity > 0:
                peak_amplitude_ratio = evening_peak_intensity / morning_peak_intensity
            else:
                peak_amplitude_ratio = np.nan
            
            # A4) Peak Timing Variations (shift from expected)
            expected_morning_peak = 9.0  # 9 AM typical for Delhi
            expected_evening_peak = 20.0  # 8 PM typical for Delhi
            
            morning_peak_shift = morning_peak_time - expected_morning_peak if not np.isnan(morning_peak_time) else np.nan
            evening_peak_shift = evening_peak_time - expected_evening_peak if not np.isnan(evening_peak_time) else np.nan
            
            # A5) Peak Duration Features
            # Morning peak duration (hours above 90% of peak)
            if not np.isnan(morning_peak_intensity):
                morning_threshold = 0.90 * morning_peak_intensity
                morning_duration = (daily_load[morning_mask] >= morning_threshold).sum()
            else:
                morning_duration = np.nan
            
            # Evening peak duration (hours above 90% of peak)
            if not np.isnan(evening_peak_intensity):
                evening_threshold = 0.90 * evening_peak_intensity
                evening_duration = (daily_load[evening_mask] >= evening_threshold).sum()
            else:
                evening_duration = np.nan
            
            # Assign features to all hours of this day
            for idx in daily_data.index:
                if 'morning_peak_intensity' not in new_features:
                    new_features['morning_peak_intensity'] = {}
                if 'evening_peak_intensity' not in new_features:
                    new_features['evening_peak_intensity'] = {}
                if 'peak_amplitude_ratio' not in new_features:
                    new_features['peak_amplitude_ratio'] = {}
                if 'morning_peak_shift' not in new_features:
                    new_features['morning_peak_shift'] = {}
                if 'evening_peak_shift' not in new_features:
                    new_features['evening_peak_shift'] = {}
                if 'morning_peak_duration' not in new_features:
                    new_features['morning_peak_duration'] = {}
                if 'evening_peak_duration' not in new_features:
                    new_features['evening_peak_duration'] = {}
                
                new_features['morning_peak_intensity'][idx] = morning_peak_intensity
                new_features['evening_peak_intensity'][idx] = evening_peak_intensity
                new_features['peak_amplitude_ratio'][idx] = peak_amplitude_ratio
                new_features['morning_peak_shift'][idx] = morning_peak_shift
                new_features['evening_peak_shift'][idx] = evening_peak_shift
                new_features['morning_peak_duration'][idx] = morning_duration
                new_features['evening_peak_duration'][idx] = evening_duration
        
        # Add features to dataframe
        feature_count = 0
        for feature_name, feature_data in new_features.items():
            if feature_data:
                self.df[feature_name] = pd.Series(feature_data)
                feature_count += 1
                
                # Log feature statistics
                valid_values = self.df[feature_name].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature_name] = {
                        'type': 'peak_detection',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
        
        print(f"✅ Created {feature_count} peak detection features")
        print(f"   📈 Morning/Evening peak intensity and timing")
        print(f"   📊 Peak amplitude ratios and duration analysis")
        print(f"   ⏰ Peak shift detection from expected times")
    
    def create_duck_curve_analysis_features(self):
        """
        B) Duck Curve Analysis Features for Solar Integration Impact
        """
        print(f"\n🦆 CREATING DUCK CURVE ANALYSIS FEATURES")
        print("-" * 50)
        
        # Find required columns
        load_col = next((col for col in self.df.columns if 'delhi_load' in col.lower()), None)
        if not load_col:
            load_col = next((col for col in self.df.columns if 'load' in col.lower() and 'net' not in col.lower()), None)
        
        solar_col = next((col for col in self.df.columns if 'solar_generation' in col.lower()), None)
        duck_depth_col = 'duck_curve_depth_mw'  # We calculated this in Phase 1
        
        if not load_col:
            print("❌ No load column found for duck curve analysis")
            return
        
        print(f"📊 Using load column: {load_col}")
        print(f"☀️ Using solar column: {solar_col}")
        print(f"🦆 Using duck curve depth: {duck_depth_col}")
        
        # Initialize new feature columns
        new_features = {}
        
        # Group by date for daily duck curve analysis
        for date in self.df['date'].unique():
            date_mask = self.df['date'] == date
            daily_data = self.df[date_mask].copy().sort_values('hour')
            
            if len(daily_data) < 20:
                continue
            
            daily_load = daily_data[load_col].values
            hours = daily_data['hour'].values
            
            # Calculate net load if solar data available
            if solar_col and solar_col in daily_data.columns:
                daily_solar = daily_data[solar_col].fillna(0).values
                net_load = daily_load - daily_solar
            else:
                net_load = daily_load
            
            # Duck curve analysis window (10 AM - 8 PM)
            duck_mask = (hours >= self.DUCK_CURVE_WINDOW[0]) & (hours <= self.DUCK_CURVE_WINDOW[1])
            
            if duck_mask.sum() < 5:  # Need at least 5 hours
                continue
            
            duck_period_net_load = net_load[duck_mask]
            duck_period_hours = hours[duck_mask]
            
            # B1) Evening Ramp Severity (steepest duck curve rise)
            # Calculate hourly ramp rates during duck curve period
            ramp_rates = np.diff(duck_period_net_load)  # MW/hour change
            
            if len(ramp_rates) > 0:
                evening_ramp_severity = np.max(ramp_rates)  # Steepest upward ramp
                max_ramp_hour = duck_period_hours[np.argmax(ramp_rates) + 1]
            else:
                evening_ramp_severity = np.nan
                max_ramp_hour = np.nan
            
            # B2) Net Load Minimum Timing (bottom of duck curve)
            min_net_load_idx = np.argmin(duck_period_net_load)
            net_load_minimum_timing = duck_period_hours[min_net_load_idx]
            net_load_minimum_value = duck_period_net_load[min_net_load_idx]
            
            # B3) Solar Decline Rate Impact
            if solar_col and solar_col in daily_data.columns:
                solar_decline_window = (hours >= 14) & (hours <= 18)  # 2 PM - 6 PM
                if solar_decline_window.sum() > 2:
                    solar_decline_period = daily_data[solar_col].values[solar_decline_window]
                    solar_decline_rate = np.mean(np.diff(solar_decline_period))  # Average decline rate
                    max_solar_decline = np.min(np.diff(solar_decline_period))    # Steepest decline
                else:
                    solar_decline_rate = np.nan
                    max_solar_decline = np.nan
            else:
                solar_decline_rate = np.nan
                max_solar_decline = np.nan
            
            # B4) Grid Flexibility Requirement Index
            # Based on ramp rate requirements and duck curve depth
            duck_depth = daily_data[duck_depth_col].iloc[0] if duck_depth_col in daily_data.columns else np.nan
            
            if not np.isnan(evening_ramp_severity) and not np.isnan(duck_depth):
                # Normalized index: higher values = more flexibility needed
                grid_flexibility_index = (evening_ramp_severity / 100) + (duck_depth / 1000)
            else:
                grid_flexibility_index = np.nan
            
            # B5) Renewable Integration Stress Level
            # Based on solar variability and net load fluctuations
            if solar_col and solar_col in daily_data.columns:
                solar_variability = np.std(daily_data[solar_col].values)
                net_load_variability = np.std(net_load)
                
                # Stress level increases with higher variability
                renewable_stress_level = (solar_variability / 50) + (net_load_variability / 200)
            else:
                renewable_stress_level = np.nan
            
            # Assign features to all hours of this day
            for idx in daily_data.index:
                for feature_name in ['evening_ramp_severity', 'net_load_minimum_timing', 
                                   'net_load_minimum_value', 'solar_decline_rate', 
                                   'max_solar_decline', 'grid_flexibility_index', 
                                   'renewable_stress_level', 'max_ramp_hour']:
                    if feature_name not in new_features:
                        new_features[feature_name] = {}
                
                new_features['evening_ramp_severity'][idx] = evening_ramp_severity
                new_features['net_load_minimum_timing'][idx] = net_load_minimum_timing
                new_features['net_load_minimum_value'][idx] = net_load_minimum_value
                new_features['solar_decline_rate'][idx] = solar_decline_rate
                new_features['max_solar_decline'][idx] = max_solar_decline
                new_features['grid_flexibility_index'][idx] = grid_flexibility_index
                new_features['renewable_stress_level'][idx] = renewable_stress_level
                new_features['max_ramp_hour'][idx] = max_ramp_hour
        
        # Add features to dataframe
        feature_count = 0
        for feature_name, feature_data in new_features.items():
            if feature_data:
                self.df[feature_name] = pd.Series(feature_data)
                feature_count += 1
                
                # Log feature statistics
                valid_values = self.df[feature_name].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature_name] = {
                        'type': 'duck_curve_analysis',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
        
        print(f"✅ Created {feature_count} duck curve analysis features")
        print(f"   📈 Evening ramp severity and timing analysis")
        print(f"   🦆 Net load minimum detection and solar decline rates")
        print(f"   ⚡ Grid flexibility and renewable stress indicators")
    
    def create_advanced_temporal_features(self):
        """
        Create advanced temporal features specific to Delhi patterns
        """
        print(f"\n⏰ CREATING ADVANCED TEMPORAL FEATURES")
        print("-" * 50)
        
        # Delhi-specific time features
        self.df['is_peak_month'] = self.df['datetime'].dt.month.isin([5, 6, 10, 11])  # May, June, Oct, Nov
        self.df['is_monsoon_month'] = self.df['datetime'].dt.month.isin([7, 8, 9])    # July, Aug, Sep
        self.df['is_winter_month'] = self.df['datetime'].dt.month.isin([12, 1, 2])    # Dec, Jan, Feb
        
        # Hour-based features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # Day of year cyclical features
        day_of_year = self.df['datetime'].dt.dayofyear
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Delhi-specific load patterns
        self.df['is_morning_ramp'] = (self.df['hour'] >= 5) & (self.df['hour'] <= 8)
        self.df['is_midday_low'] = (self.df['hour'] >= 12) & (self.df['hour'] <= 15)
        self.df['is_evening_ramp'] = (self.df['hour'] >= 17) & (self.df['hour'] <= 21)
        self.df['is_night_valley'] = (self.df['hour'] >= 23) | (self.df['hour'] <= 4)
        
        # Check if is_weekend exists and handle data types properly
        if 'is_weekend' not in self.df.columns:
            # Create is_weekend if it doesn't exist
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])  # Saturday=5, Sunday=6
        
        # Ensure boolean type and handle NaN values
        is_weekend_bool = self.df['is_weekend'].fillna(False).astype(bool)
        
        # Weekend vs weekday interactions
        self.df['weekend_evening'] = is_weekend_bool & (self.df['hour'] >= 18)
        self.df['weekday_morning'] = (~is_weekend_bool) & (self.df['hour'] <= 10)
        
        temporal_features = ['is_peak_month', 'is_monsoon_month', 'is_winter_month',
                           'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
                           'is_morning_ramp', 'is_midday_low', 'is_evening_ramp', 'is_night_valley',
                           'weekend_evening', 'weekday_morning']
        
        print(f"✅ Created {len(temporal_features)} advanced temporal features")
        print(f"   📅 Seasonal and cyclical patterns")
        print(f"   ⏰ Delhi-specific time-of-day indicators")
        print(f"   📊 Weekend/weekday interaction features")
        
        # Log temporal features
        for feature in temporal_features:
            if feature in self.df.columns:
                self.feature_log[feature] = {
                    'type': 'temporal',
                    'count': len(self.df),
                    'unique_values': self.df[feature].nunique()
                }
    
    def calculate_rolling_features(self):
        """
        Calculate rolling/lag features for temporal dependencies
        """
        print(f"\n📊 CREATING ROLLING AND LAG FEATURES")
        print("-" * 50)
        
        # Find load column
        load_col = next((col for col in self.df.columns if 'delhi_load' in col.lower()), None)
        if not load_col:
            load_col = next((col for col in self.df.columns if 'load' in col.lower() and 'net' not in col.lower()), None)
        
        if not load_col:
            print("❌ No load column found for rolling features")
            return
        
        # Sort by datetime for proper rolling calculations
        self.df = self.df.sort_values('datetime').reset_index(drop=True)
        
        # Rolling averages (24-hour, 7-day, 30-day)
        self.df['load_24h_mean'] = self.df[load_col].rolling(window=24, min_periods=12).mean()
        self.df['load_7d_mean'] = self.df[load_col].rolling(window=24*7, min_periods=24*3).mean()
        self.df['load_30d_mean'] = self.df[load_col].rolling(window=24*30, min_periods=24*7).mean()
        
        # Rolling standard deviations (volatility)
        self.df['load_24h_std'] = self.df[load_col].rolling(window=24, min_periods=12).std()
        self.df['load_7d_std'] = self.df[load_col].rolling(window=24*7, min_periods=24*3).std()
        
        # Lag features (previous hours, days, weeks)
        self.df['load_lag_1h'] = self.df[load_col].shift(1)
        self.df['load_lag_24h'] = self.df[load_col].shift(24)
        self.df['load_lag_168h'] = self.df[load_col].shift(168)  # 1 week
        
        # Difference features (change from previous periods)
        self.df['load_diff_1h'] = self.df[load_col] - self.df['load_lag_1h']
        self.df['load_diff_24h'] = self.df[load_col] - self.df['load_lag_24h']
        self.df['load_diff_168h'] = self.df[load_col] - self.df['load_lag_168h']
        
        # Rolling min/max (daily extremes)
        self.df['load_24h_min'] = self.df[load_col].rolling(window=24, min_periods=12).min()
        self.df['load_24h_max'] = self.df[load_col].rolling(window=24, min_periods=12).max()
        self.df['load_24h_range'] = self.df['load_24h_max'] - self.df['load_24h_min']
        
        rolling_features = ['load_24h_mean', 'load_7d_mean', 'load_30d_mean',
                          'load_24h_std', 'load_7d_std',
                          'load_lag_1h', 'load_lag_24h', 'load_lag_168h',
                          'load_diff_1h', 'load_diff_24h', 'load_diff_168h',
                          'load_24h_min', 'load_24h_max', 'load_24h_range']
        
        print(f"✅ Created {len(rolling_features)} rolling and lag features")
        print(f"   📈 Rolling averages and volatility measures")
        print(f"   ⏮️ Lag features for temporal dependencies")
        print(f"   📊 Difference and range features")
        
        # Log rolling features
        for feature in rolling_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'rolling',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std()
                    }
    
    def create_feature_visualization(self):
        """Create comprehensive visualization of new features"""
        print(f"\n📊 CREATING FEATURE VISUALIZATION")
        print("-" * 50)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        # 1. Peak intensity comparison
        ax1 = axes[0]
        if 'morning_peak_intensity' in self.df.columns and 'evening_peak_intensity' in self.df.columns:
            morning_peaks = self.df['morning_peak_intensity'].dropna()
            evening_peaks = self.df['evening_peak_intensity'].dropna()
            
            ax1.scatter(morning_peaks, evening_peaks, alpha=0.6, s=20)
            ax1.plot([morning_peaks.min(), morning_peaks.max()], 
                    [morning_peaks.min(), morning_peaks.max()], 'r--', alpha=0.8)
            ax1.set_xlabel('Morning Peak Intensity (MW)')
            ax1.set_ylabel('Evening Peak Intensity (MW)')
            ax1.set_title('Delhi Dual Peak Comparison')
            ax1.grid(True, alpha=0.3)
        
        # 2. Duck curve depth over time
        ax2 = axes[1]
        if 'duck_curve_depth_mw' in self.df.columns:
            sample_data = self.df.sample(n=min(5000, len(self.df))).sort_values('datetime')
            ax2.plot(sample_data['datetime'], sample_data['duck_curve_depth_mw'], alpha=0.7, linewidth=0.5)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Duck Curve Depth (MW)')
            ax2.set_title('Duck Curve Depth Over Time')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
        
        # 3. Evening ramp severity distribution
        ax3 = axes[2]
        if 'evening_ramp_severity' in self.df.columns:
            ramp_data = self.df['evening_ramp_severity'].dropna()
            ax3.hist(ramp_data, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_xlabel('Evening Ramp Severity (MW/h)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Evening Ramp Severity Distribution')
            ax3.grid(True, alpha=0.3)
        
        # 4. Peak timing variations
        ax4 = axes[3]
        if 'morning_peak_shift' in self.df.columns and 'evening_peak_shift' in self.df.columns:
            morning_shifts = self.df['morning_peak_shift'].dropna()
            evening_shifts = self.df['evening_peak_shift'].dropna()
            
            ax4.scatter(morning_shifts, evening_shifts, alpha=0.6, s=20, color='green')
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Morning Peak Shift (hours)')
            ax4.set_ylabel('Evening Peak Shift (hours)')
            ax4.set_title('Peak Timing Variations')
            ax4.grid(True, alpha=0.3)
        
        # 5. Grid flexibility index
        ax5 = axes[4]
        if 'grid_flexibility_index' in self.df.columns:
            flex_data = self.df['grid_flexibility_index'].dropna()
            ax5.hist(flex_data, bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax5.set_xlabel('Grid Flexibility Index')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Grid Flexibility Requirement Distribution')
            ax5.grid(True, alpha=0.3)
        
        # 6. Hourly peak patterns
        ax6 = axes[5]
        if 'hour' in self.df.columns:
            load_col = next((col for col in self.df.columns if 'delhi_load' in col.lower()), None)
            if load_col:
                hourly_avg = self.df.groupby('hour')[load_col].mean()
                ax6.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=4)
                ax6.set_xlabel('Hour of Day')
                ax6.set_ylabel('Average Load (MW)')
                ax6.set_title('Delhi Daily Load Pattern')
                ax6.set_xticks(range(0, 24, 4))
                ax6.grid(True, alpha=0.3)
        
        # 7. Renewable stress level vs solar generation
        ax7 = axes[6]
        if 'renewable_stress_level' in self.df.columns:
            solar_col = next((col for col in self.df.columns if 'solar_generation' in col.lower()), None)
            if solar_col:
                sample_data = self.df.sample(n=min(3000, len(self.df)))
                ax7.scatter(sample_data[solar_col], sample_data['renewable_stress_level'], 
                          alpha=0.6, s=15, color='red')
                ax7.set_xlabel('Solar Generation (MW)')
                ax7.set_ylabel('Renewable Stress Level')
                ax7.set_title('Solar Generation vs Renewable Stress')
                ax7.grid(True, alpha=0.3)
        
        # 8. Feature correlation heatmap
        ax8 = axes[7]
        peak_features = [col for col in self.df.columns if any(x in col.lower() 
                        for x in ['peak', 'ramp', 'duck', 'flexibility', 'stress'])]
        if len(peak_features) > 2:
            corr_data = self.df[peak_features].corr()
            im = ax8.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
            ax8.set_xticks(range(len(peak_features)))
            ax8.set_yticks(range(len(peak_features)))
            ax8.set_xticklabels([f[:15] + '...' if len(f) > 15 else f for f in peak_features], rotation=45, ha='right')
            ax8.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in peak_features])
            ax8.set_title('Peak Features Correlation')
            plt.colorbar(im, ax=ax8, shrink=0.6)
        
        # 9. Summary statistics
        ax9 = axes[8]
        ax9.axis('off')
        
        # Count features by type
        feature_types = {}
        for feature, info in self.feature_log.items():
            ftype = info.get('type', 'other')
            feature_types[ftype] = feature_types.get(ftype, 0) + 1
        
        summary_text = f"""
DELHI DUAL PEAK FEATURES SUMMARY

Total Features Created: {len(self.feature_log)}

By Category:
• Peak Detection: {feature_types.get('peak_detection', 0)}
• Duck Curve Analysis: {feature_types.get('duck_curve_analysis', 0)}
• Temporal Features: {feature_types.get('temporal', 0)}
• Rolling Features: {feature_types.get('rolling', 0)}

Key Achievements:
✅ Dual peak intensity tracking
✅ Duck curve severity analysis  
✅ Grid flexibility assessment
✅ Renewable stress monitoring
✅ Advanced temporal patterns

Dataset Ready For:
🤖 Machine Learning Training
📊 Load Forecasting Models
⚡ Grid Operations Planning
        """
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, 'delhi_dual_peak_features_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Feature visualization saved: {output_path}")
    
    def generate_feature_report(self):
        """Generate comprehensive feature engineering report"""
        print(f"\n📋 DELHI DUAL PEAK FEATURES REPORT")
        print("=" * 70)
        
        print(f"Dataset: {len(self.df):,} records × {len(self.df.columns)} columns")
        print(f"Feature engineering period: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        print(f"\nFeature Engineering Summary:")
        print(f"  Total new features created: {len(self.feature_log)}")
        
        # Group features by type
        feature_types = {}
        for feature, info in self.feature_log.items():
            ftype = info.get('type', 'other')
            if ftype not in feature_types:
                feature_types[ftype] = []
            feature_types[ftype].append(feature)
        
        for ftype, features in feature_types.items():
            print(f"\n  {ftype.upper()} FEATURES ({len(features)}):")
            for feature in features[:5]:  # Show first 5
                if feature in self.feature_log:
                    info = self.feature_log[feature]
                    if 'mean' in info:
                        print(f"    {feature}: μ={info['mean']:.2f}, σ={info['std']:.2f}")
                    else:
                        print(f"    {feature}: {info.get('count', 'N/A')} values")
            if len(features) > 5:
                print(f"    ... and {len(features) - 5} more")
        
        print(f"\nDelhi-Specific Capabilities Enabled:")
        print(f"  ✅ Dual peak pattern analysis (morning 6-11 AM, evening 6-11 PM)")
        print(f"  ✅ Duck curve severity measurement (10 AM - 8 PM period)")
        print(f"  ✅ Grid flexibility requirement assessment")
        print(f"  ✅ Renewable integration stress monitoring")
        print(f"  ✅ Peak timing shift detection")
        print(f"  ✅ Evening ramp rate analysis (critical for Delhi)")
        
        print(f"\nNext Steps:")
        print(f"  🔄 Ready for Phase 3: Advanced Feature Engineering")
        print(f"  🤖 Ready for Model Training and Validation")
        print(f"  📊 Ready for Load Forecasting Implementation")
        
        print(f"\n💾 Enhanced dataset ready for Delhi load forecasting models!")
    
    def save_enhanced_dataset(self):
        """Save the enhanced dataset with new features"""
        output_path = os.path.join(self.output_dir, 'delhi_enhanced_dual_peak_features.csv')
        self.df.to_csv(output_path, index=False)
        print(f"\n✅ Enhanced dataset saved: {output_path}")
        return output_path
    
    def run_complete_feature_engineering(self):
        """Execute the complete Delhi dual peak feature engineering"""
        try:
            # Load 100% complete dataset
            self.load_complete_dataset()
            
            # Create Delhi-specific features
            print(f"\n🚀 EXECUTING DELHI DUAL PEAK FEATURE ENGINEERING")
            print("=" * 50)
            
            # Step 1: Peak Detection Features
            self.create_peak_detection_features()
            
            # Step 2: Duck Curve Analysis Features
            self.create_duck_curve_analysis_features()
            
            # Step 3: Advanced Temporal Features
            self.create_advanced_temporal_features()
            
            # Step 4: Rolling and Lag Features
            self.calculate_rolling_features()
            
            # Create visualizations and reports
            self.create_feature_visualization()
            self.generate_feature_report()
            
            # Save enhanced dataset
            output_path = self.save_enhanced_dataset()
            
            print(f"\n🎉 DELHI DUAL PEAK FEATURE ENGINEERING COMPLETED!")
            print(f"📁 Output file: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during feature engineering: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Use the 100% complete dataset from Phase 1
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, '..', 'phase 1', 'final_dataset_100_percent_complete.csv')
    
    # Initialize feature engineering
    feature_eng = DelhiDualPeakFeatureEngineering(input_path)
    
    # Run complete feature engineering
    output_path = feature_eng.run_complete_feature_engineering()
    
    return output_path


if __name__ == "__main__":
    output_file = main()
