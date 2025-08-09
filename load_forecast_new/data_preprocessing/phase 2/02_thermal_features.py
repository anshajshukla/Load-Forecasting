"""
Delhi Load Forecasting - Phase 2: Thermal Comfort Features
=========================================================
Advanced thermal comfort feature engineering for Delhi's unique climate.
Focus on heat index, cooling degree days, heat waves, and monsoon impacts.
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

class DelhiThermalComfortFeatures:
    """
    Advanced thermal comfort feature engineering for Delhi climate
    Implements heat index, cooling degree days, heat wave detection, and monsoon analysis
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.feature_log = {}
        self.output_dir = os.path.dirname(__file__)
        
        # Delhi-specific thermal constants
        self.DELHI_AC_BASE_TEMP = 24.0      # Base temperature for cooling degree days (°C)
        self.DELHI_HEAT_WAVE_TEMP = 42.0    # Heat wave threshold for Delhi (°C)
        self.DELHI_COMFORT_TEMP_MIN = 18.0  # Comfort zone minimum (°C)
        self.DELHI_COMFORT_TEMP_MAX = 28.0  # Comfort zone maximum (°C)
        self.DELHI_HIGH_HUMIDITY = 70.0     # High humidity threshold (%)
        
        # Monsoon months for Delhi
        self.MONSOON_MONTHS = [6, 7, 8, 9]  # June to September
        self.PRE_MONSOON_MONTHS = [4, 5]    # April, May
        self.POST_MONSOON_MONTHS = [10, 11] # October, November
        self.WINTER_MONTHS = [12, 1, 2]     # December, January, February
        
    def load_enhanced_dataset(self):
        """Load the enhanced dataset from Phase 2 Step 1"""
        print("=" * 70)
        print("DELHI THERMAL COMFORT FEATURES - PHASE 2 STEP 2")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Enhanced dataset not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
        
        print(f"✅ Loading enhanced dataset: {os.path.basename(self.data_path)}")
        print(f"📊 Dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        # Verify weather columns exist (check actual column names)
        temp_cols = [col for col in self.df.columns if 'temperature_2m' in col.lower()]
        humidity_cols = [col for col in self.df.columns if 'relative_humidity_2m' in col.lower()]
        
        if temp_cols and humidity_cols:
            self.temp_col = temp_cols[0]
            self.humidity_col = humidity_cols[0]
            print(f"✅ Weather data verified: {self.temp_col} and {self.humidity_col} available")
        else:
            print(f"⚠️  Warning: Missing weather columns. Available: {[col for col in self.df.columns if any(x in col.lower() for x in ['temp', 'humid'])]}")
            self.temp_col = None
            self.humidity_col = None
        
        # Add time components if not present
        if 'month' not in self.df.columns:
            self.df['month'] = self.df['datetime'].dt.month
        if 'day_of_year' not in self.df.columns:
            self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['datetime'].dt.year
    
    def calculate_heat_index(self):
        """
        A) Calculate Heat Index (Apparent Temperature) for Delhi Climate
        Uses temperature and humidity to determine perceived temperature
        """
        print(f"\n🌡️ CALCULATING HEAT INDEX FEATURES")
        print("-" * 50)
        
        if not hasattr(self, 'temp_col') or not hasattr(self, 'humidity_col') or not self.temp_col or not self.humidity_col:
            print("❌ Temperature or humidity data not available for heat index calculation")
            return
        
        temp_c = self.df[self.temp_col].fillna(method='ffill')  # Temperature in Celsius
        humidity = self.df[self.humidity_col].fillna(method='ffill')  # Relative humidity in %
        
        # Convert temperature to Fahrenheit for heat index calculation
        temp_f = (temp_c * 9/5) + 32
        
        # Heat Index calculation (Rothfusz equation)
        # Only calculate when temp >= 80°F (26.7°C) and humidity >= 40%
        heat_index_mask = (temp_f >= 80) & (humidity >= 40)
        
        # Initialize heat index with air temperature
        heat_index_f = temp_f.copy()
        
        # Calculate heat index for applicable conditions
        if heat_index_mask.sum() > 0:
            T = temp_f[heat_index_mask]
            RH = humidity[heat_index_mask]
            
            # Rothfusz regression equation
            HI = (-42.379 + 
                  2.04901523 * T + 
                  10.14333127 * RH + 
                  -0.22475541 * T * RH + 
                  -0.00683783 * T**2 + 
                  -0.05481717 * RH**2 + 
                  0.00122874 * T**2 * RH + 
                  0.00085282 * T * RH**2 + 
                  -0.00000199 * T**2 * RH**2)
            
            heat_index_f.loc[heat_index_mask] = HI
        
        # Convert back to Celsius
        self.df['heat_index_celsius'] = (heat_index_f - 32) * 5/9
        
        # A1) Heat Index Categories for Delhi
        self.df['heat_index_category'] = pd.cut(
            self.df['heat_index_celsius'],
            bins=[-np.inf, 27, 32, 39, 46, np.inf],
            labels=['Comfortable', 'Caution', 'Extreme_Caution', 'Danger', 'Extreme_Danger']
        )
        
        # A2) Heat Index Stress Level (0-100 scale)
        # Normalized based on Delhi climate extremes
        delhi_min_hi = 15.0  # Minimum expected heat index
        delhi_max_hi = 55.0  # Maximum expected heat index
        
        self.df['heat_index_stress'] = np.clip(
            ((self.df['heat_index_celsius'] - delhi_min_hi) / (delhi_max_hi - delhi_min_hi)) * 100,
            0, 100
        )
        
        # A3) Heat Index Delta (difference from comfort zone)
        delhi_comfort_hi = 26.0  # Ideal heat index for Delhi
        self.df['heat_index_delta'] = self.df['heat_index_celsius'] - delhi_comfort_hi
        
        # A4) Daily Heat Index Statistics
        self.df['date'] = self.df['datetime'].dt.date
        daily_hi_stats = self.df.groupby('date')['heat_index_celsius'].agg([
            'mean', 'max', 'min', 'std'
        ]).add_prefix('daily_heat_index_')
        
        # Merge back to main dataframe
        self.df = self.df.merge(daily_hi_stats, left_on='date', right_index=True, how='left')
        
        # A5) Heat Index Persistence (hours above threshold)
        threshold_35 = 35.0  # High heat index threshold
        self.df['heat_index_above_35'] = (self.df['heat_index_celsius'] > threshold_35).astype(int)
        
        # Calculate rolling persistence (24-hour window)
        self.df = self.df.sort_values('datetime')
        self.df['heat_index_persistence_24h'] = self.df['heat_index_above_35'].rolling(
            window=24, min_periods=12
        ).sum()
        
        print(f"✅ Heat index features calculated")
        print(f"   🌡️ Heat index range: {self.df['heat_index_celsius'].min():.1f}°C to {self.df['heat_index_celsius'].max():.1f}°C")
        print(f"   📊 Heat index stress levels: 0-100 scale")
        print(f"   ⏰ Daily statistics and persistence tracking")
        
        # Log heat index features
        heat_index_features = ['heat_index_celsius', 'heat_index_stress', 'heat_index_delta',
                             'daily_heat_index_mean', 'daily_heat_index_max', 'heat_index_persistence_24h']
        
        for feature in heat_index_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'heat_index',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
    
    def calculate_cooling_degree_days(self):
        """
        B) Calculate Cooling Degree Days for Delhi (Base 24°C for AC usage)
        """
        print(f"\n❄️ CALCULATING COOLING DEGREE DAYS")
        print("-" * 50)
        
        if not hasattr(self, 'temp_col') or not self.temp_col:
            print("❌ Temperature data not available for cooling degree days")
            return
        
        temp_c = self.df[self.temp_col].fillna(method='ffill')
        
        # B1) Hourly Cooling Degree Hours (CDH)
        # Delhi AC base temperature: 24°C
        self.df['cooling_degree_hours'] = np.maximum(temp_c - self.DELHI_AC_BASE_TEMP, 0)
        
        # B2) Daily Cooling Degree Days (CDD)
        # Calculate using daily average temperature
        daily_avg_temp = self.df.groupby('date')[self.temp_col].mean()
        daily_cdd = np.maximum(daily_avg_temp - self.DELHI_AC_BASE_TEMP, 0)
        
        # Map back to hourly data
        temp_to_cdd = dict(zip(daily_avg_temp.index, daily_cdd.values))
        self.df['daily_cooling_degree_days'] = self.df['date'].map(temp_to_cdd)
        
        # B3) Cumulative Cooling Degree Days (seasonal accumulation)
        self.df['month_year'] = self.df['datetime'].dt.to_period('M')
        monthly_cdd = self.df.groupby('month_year')['daily_cooling_degree_days'].transform('cumsum')
        self.df['cumulative_cdd_monthly'] = monthly_cdd
        
        # B4) Rolling Cooling Load Index (7-day average)
        self.df['cooling_load_index_7d'] = self.df['cooling_degree_hours'].rolling(
            window=24*7, min_periods=24*3
        ).mean()
        
        # B5) Peak Cooling Demand Indicator
        # Based on both temperature and humidity (discomfort index)
        if hasattr(self, 'humidity_col') and self.humidity_col:
            humidity = self.df[self.humidity_col].fillna(method='ffill')
            
            # Thom's Discomfort Index for cooling demand
            discomfort_index = temp_c - (0.55 - 0.0055 * humidity) * (temp_c - 14.5)
            self.df['cooling_discomfort_index'] = discomfort_index
            
            # Peak cooling demand periods (DI > 26.5)
            self.df['peak_cooling_demand'] = (discomfort_index > 26.5).astype(int)
        
        # B6) AC Load Probability (based on temperature and time)
        # Higher probability during hot hours (11 AM - 10 PM)
        peak_cooling_hours = (self.df['datetime'].dt.hour >= 11) & (self.df['datetime'].dt.hour <= 22)
        temp_factor = np.clip((temp_c - 20) / 25, 0, 1)  # 0 at 20°C, 1 at 45°C
        hour_factor = peak_cooling_hours.astype(float) * 0.8 + 0.2  # 0.2-1.0 range
        
        self.df['ac_load_probability'] = temp_factor * hour_factor
        
        print(f"✅ Cooling degree days calculated")
        print(f"   ❄️ CDH range: {self.df['cooling_degree_hours'].min():.1f} to {self.df['cooling_degree_hours'].max():.1f}")
        print(f"   📊 Daily CDD range: {self.df['daily_cooling_degree_days'].min():.1f} to {self.df['daily_cooling_degree_days'].max():.1f}")
        print(f"   🏠 AC load probability and discomfort index calculated")
        
        # Log cooling features
        cooling_features = ['cooling_degree_hours', 'daily_cooling_degree_days', 'cumulative_cdd_monthly',
                          'cooling_load_index_7d', 'cooling_discomfort_index', 'ac_load_probability']
        
        for feature in cooling_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'cooling_degree_days',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
    
    def detect_heat_waves(self):
        """
        C) Delhi Heat Wave Detection (>42°C consecutive days)
        """
        print(f"\n🔥 DETECTING DELHI HEAT WAVE PATTERNS")
        print("-" * 50)
        
        if not hasattr(self, 'temp_col') or not self.temp_col:
            print("❌ Temperature data not available for heat wave detection")
            return
        
        # C1) Daily Maximum Temperature
        daily_max_temp = self.df.groupby('date')[self.temp_col].max()
        
        # C2) Heat Wave Day Detection (>42°C)
        heat_wave_days = daily_max_temp > self.DELHI_HEAT_WAVE_TEMP
        
        # C3) Consecutive Heat Wave Days
        # Calculate consecutive days using groupby
        heat_wave_groups = (heat_wave_days != heat_wave_days.shift()).cumsum()
        consecutive_days = heat_wave_days.groupby(heat_wave_groups).transform('sum') * heat_wave_days
        
        # Map back to hourly data
        date_to_consecutive = dict(zip(consecutive_days.index, consecutive_days.values))
        self.df['consecutive_heat_wave_days'] = self.df['date'].map(date_to_consecutive)
        
        # C4) Heat Wave Intensity
        # How much above the threshold
        heat_wave_intensity = np.maximum(daily_max_temp - self.DELHI_HEAT_WAVE_TEMP, 0)
        date_to_intensity = dict(zip(heat_wave_intensity.index, heat_wave_intensity.values))
        self.df['heat_wave_intensity'] = self.df['date'].map(date_to_intensity)
        
        # C5) Heat Wave Status Categories
        self.df['heat_wave_status'] = 'Normal'
        self.df.loc[self.df['consecutive_heat_wave_days'] == 1, 'heat_wave_status'] = 'Heat_Day'
        self.df.loc[self.df['consecutive_heat_wave_days'] == 2, 'heat_wave_status'] = 'Heat_Wave'
        self.df.loc[self.df['consecutive_heat_wave_days'] >= 3, 'heat_wave_status'] = 'Severe_Heat_Wave'
        
        # C6) Pre-Heat Wave Conditions (day before heat wave)
        heat_wave_tomorrow = self.df['consecutive_heat_wave_days'].shift(-24)  # Next day
        self.df['pre_heat_wave'] = ((heat_wave_tomorrow > 0) & 
                                   (self.df['consecutive_heat_wave_days'] == 0)).astype(int)
        
        # C7) Post-Heat Wave Recovery (day after heat wave ends)
        heat_wave_yesterday = self.df['consecutive_heat_wave_days'].shift(24)  # Previous day
        self.df['post_heat_wave'] = ((heat_wave_yesterday > 0) & 
                                    (self.df['consecutive_heat_wave_days'] == 0)).astype(int)
        
        # C8) Heat Wave Season Indicator
        # Peak heat wave months in Delhi: April, May, June
        self.df['heat_wave_season'] = self.df['month'].isin([4, 5, 6]).astype(int)
        
        # C9) Cumulative Heat Stress (seasonal accumulation)
        self.df['cumulative_heat_stress'] = self.df.groupby(['month', 'year'])['heat_wave_intensity'].cumsum()
        
        # Calculate heat wave statistics
        total_heat_wave_days = (self.df['consecutive_heat_wave_days'] > 0).sum() / 24  # Convert to days
        max_consecutive = self.df['consecutive_heat_wave_days'].max()
        
        print(f"✅ Heat wave detection completed")
        print(f"   🔥 Total heat wave hours detected: {(self.df['consecutive_heat_wave_days'] > 0).sum():,}")
        print(f"   📊 Maximum consecutive heat wave days: {max_consecutive}")
        print(f"   🌡️ Heat wave intensity range: {self.df['heat_wave_intensity'].min():.1f} to {self.df['heat_wave_intensity'].max():.1f}°C")
        
        # Log heat wave features
        heat_wave_features = ['consecutive_heat_wave_days', 'heat_wave_intensity', 'pre_heat_wave',
                            'post_heat_wave', 'heat_wave_season', 'cumulative_heat_stress']
        
        for feature in heat_wave_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'heat_wave',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
    
    def create_monsoon_impact_features(self):
        """
        D) Monsoon Impact Features for Delhi Climate
        """
        print(f"\n🌧️ CREATING MONSOON IMPACT FEATURES")
        print("-" * 50)
        
        # D1) Monsoon Season Classification
        self.df['season_type'] = 'Other'
        self.df.loc[self.df['month'].isin(self.PRE_MONSOON_MONTHS), 'season_type'] = 'Pre_Monsoon'
        self.df.loc[self.df['month'].isin(self.MONSOON_MONTHS), 'season_type'] = 'Monsoon'
        self.df.loc[self.df['month'].isin(self.POST_MONSOON_MONTHS), 'season_type'] = 'Post_Monsoon'
        self.df.loc[self.df['month'].isin(self.WINTER_MONTHS), 'season_type'] = 'Winter'
        
        # D2) Monsoon Cooling Effect
        if hasattr(self, 'temp_col') and hasattr(self, 'humidity_col') and self.temp_col and self.humidity_col:
            temp = self.df[self.temp_col].fillna(method='ffill')
            humidity = self.df[self.humidity_col].fillna(method='ffill')
            
            # Calculate expected temperature without monsoon (based on trend)
            # Use pre-monsoon trend to predict monsoon temperatures
            pre_monsoon_data = self.df[self.df['month'].isin(self.PRE_MONSOON_MONTHS)]
            
            if len(pre_monsoon_data) > 0:
                # Calculate cooling effect during monsoon
                is_monsoon = self.df['month'].isin(self.MONSOON_MONTHS)
                
                # Average pre-monsoon temperature
                avg_pre_monsoon_temp = pre_monsoon_data.groupby('hour')[self.temp_col].mean()
                
                # Map expected temperature based on hour
                expected_temp = self.df['datetime'].dt.hour.map(avg_pre_monsoon_temp)
                
                # Monsoon cooling effect (negative values indicate cooling)
                self.df['monsoon_cooling_effect'] = np.where(
                    is_monsoon,
                    temp - expected_temp,
                    0
                )
        
        # D3) Humidity Impact on Comfort
        if hasattr(self, 'humidity_col') and self.humidity_col:
            humidity = self.df[self.humidity_col].fillna(method='ffill')
            
            # High humidity discomfort (especially during monsoon)
            self.df['humidity_discomfort'] = np.maximum(humidity - 60, 0)  # Discomfort above 60%
            
            # Monsoon humidity stress
            self.df['monsoon_humidity_stress'] = (
                (humidity > self.DELHI_HIGH_HUMIDITY) & 
                (self.df['month'].isin(self.MONSOON_MONTHS))
            ).astype(int)
        
        # D4) Seasonal Temperature Variations
        if hasattr(self, 'temp_col') and self.temp_col:
            temp = self.df[self.temp_col].fillna(method='ffill')
            
            # Calculate seasonal temperature anomalies
            seasonal_avg = self.df.groupby(['month', 'hour'])[self.temp_col].transform('mean')
            self.df['seasonal_temp_anomaly'] = temp - seasonal_avg
            
            # Temperature range variation by season
            daily_temp_range = self.df.groupby('date')[self.temp_col].agg(lambda x: x.max() - x.min())
            date_to_range = dict(zip(daily_temp_range.index, daily_temp_range.values))
            self.df['daily_temp_range'] = self.df['date'].map(date_to_range)
        
        # D5) Monsoon Transition Features
        # Detect monsoon onset and withdrawal periods
        self.df['is_monsoon_transition'] = (
            (self.df['month'].isin([5, 6])) |  # Pre-monsoon to monsoon
            (self.df['month'].isin([9, 10]))   # Monsoon to post-monsoon
        ).astype(int)
        
        # D6) Air Quality-Temperature Interaction
        # Assuming air quality is worse in winter and better during monsoon
        self.df['air_quality_temp_interaction'] = 0
        
        # Winter: high pollution, moderate cooling load
        winter_mask = self.df['month'].isin(self.WINTER_MONTHS)
        self.df.loc[winter_mask, 'air_quality_temp_interaction'] = 1
        
        # Pre-monsoon: high pollution, high cooling load
        pre_monsoon_mask = self.df['month'].isin(self.PRE_MONSOON_MONTHS)
        self.df.loc[pre_monsoon_mask, 'air_quality_temp_interaction'] = 3
        
        # Monsoon: low pollution, moderate cooling load
        monsoon_mask = self.df['month'].isin(self.MONSOON_MONTHS)
        self.df.loc[monsoon_mask, 'air_quality_temp_interaction'] = 2
        
        print(f"✅ Monsoon impact features created")
        print(f"   🌧️ Seasonal classifications and cooling effects")
        print(f"   💧 Humidity discomfort and stress indicators")
        print(f"   🌡️ Seasonal temperature anomalies")
        
        # Log monsoon features
        monsoon_features = ['monsoon_cooling_effect', 'humidity_discomfort', 'monsoon_humidity_stress',
                          'seasonal_temp_anomaly', 'daily_temp_range', 'air_quality_temp_interaction']
        
        for feature in monsoon_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'monsoon_impact',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
    
    def calculate_thermal_stress_accumulation(self):
        """
        E) Thermal Stress Accumulation Features
        """
        print(f"\n🌡️ CALCULATING THERMAL STRESS ACCUMULATION")
        print("-" * 50)
        
        if not hasattr(self, 'temp_col') or not self.temp_col:
            print("❌ Temperature data not available for thermal stress calculation")
            return
        
        temp = self.df[self.temp_col].fillna(method='ffill')
        
        # E1) Hourly Thermal Stress
        # Stress increases exponentially above comfort zone
        comfort_max = self.DELHI_COMFORT_TEMP_MAX
        self.df['hourly_thermal_stress'] = np.maximum(
            np.power(np.maximum(temp - comfort_max, 0), 1.5), 0
        )
        
        # E2) Daily Thermal Stress Accumulation
        daily_stress = self.df.groupby('date')['hourly_thermal_stress'].sum()
        date_to_stress = dict(zip(daily_stress.index, daily_stress.values))
        self.df['daily_thermal_stress'] = self.df['date'].map(date_to_stress)
        
        # E3) Weekly Thermal Stress (7-day rolling sum)
        self.df = self.df.sort_values('datetime')
        self.df['weekly_thermal_stress'] = self.df['daily_thermal_stress'].rolling(
            window=7, min_periods=3
        ).sum()
        
        # E4) Thermal Recovery Index
        # Lower temperatures help recovery from thermal stress
        recovery_temp = self.DELHI_COMFORT_TEMP_MIN
        self.df['thermal_recovery'] = np.maximum(recovery_temp - temp, 0)
        
        # E5) Net Thermal Balance (stress vs recovery)
        self.df['net_thermal_balance'] = self.df['hourly_thermal_stress'] - self.df['thermal_recovery']
        
        # E6) Cumulative Thermal Load (seasonal)
        # year column is already created in load_enhanced_dataset
        self.df['cumulative_thermal_load'] = self.df.groupby(['year', 'month'])['daily_thermal_stress'].cumsum()
        
        # E7) Thermal Stress Categories
        stress_percentiles = self.df['daily_thermal_stress'].quantile([0.33, 0.66, 0.9])
        
        self.df['thermal_stress_category'] = 'Low'
        self.df.loc[self.df['daily_thermal_stress'] > stress_percentiles.iloc[0], 'thermal_stress_category'] = 'Moderate'
        self.df.loc[self.df['daily_thermal_stress'] > stress_percentiles.iloc[1], 'thermal_stress_category'] = 'High'
        self.df.loc[self.df['daily_thermal_stress'] > stress_percentiles.iloc[2], 'thermal_stress_category'] = 'Extreme'
        
        print(f"✅ Thermal stress accumulation calculated")
        print(f"   🌡️ Daily thermal stress range: {self.df['daily_thermal_stress'].min():.1f} to {self.df['daily_thermal_stress'].max():.1f}")
        print(f"   📊 Thermal recovery and net balance computed")
        print(f"   📈 Cumulative seasonal thermal load tracking")
        
        # Log thermal stress features
        thermal_features = ['hourly_thermal_stress', 'daily_thermal_stress', 'weekly_thermal_stress',
                          'thermal_recovery', 'net_thermal_balance', 'cumulative_thermal_load']
        
        for feature in thermal_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'thermal_stress',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
    
    def create_thermal_visualization(self):
        """Create comprehensive visualization of thermal comfort features"""
        print(f"\n📊 CREATING THERMAL FEATURES VISUALIZATION")
        print("-" * 50)
        
        fig, axes = plt.subplots(4, 3, figsize=(24, 20))
        axes = axes.flatten()
        
        # 1. Heat Index vs Temperature
        ax1 = axes[0]
        if 'heat_index_celsius' in self.df.columns and hasattr(self, 'temp_col') and self.temp_col:
            sample_data = self.df.sample(n=min(5000, len(self.df)))
            ax1.scatter(sample_data[self.temp_col], sample_data['heat_index_celsius'], 
                       alpha=0.6, s=15, c='red')
            ax1.plot([0, 50], [0, 50], 'k--', alpha=0.5)  # y=x line
            ax1.set_xlabel('Temperature (°C)')
            ax1.set_ylabel('Heat Index (°C)')
            ax1.set_title('Heat Index vs Temperature')
            ax1.grid(True, alpha=0.3)
        
        # 2. Cooling Degree Days by Month
        ax2 = axes[1]
        if 'daily_cooling_degree_days' in self.df.columns:
            monthly_cdd = self.df.groupby('month')['daily_cooling_degree_days'].mean()
            ax2.bar(monthly_cdd.index, monthly_cdd.values, color='lightblue', edgecolor='blue')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Average Daily CDD')
            ax2.set_title('Cooling Degree Days by Month')
            ax2.set_xticks(range(1, 13))
            ax2.grid(True, alpha=0.3)
        
        # 3. Heat Wave Detection Timeline
        ax3 = axes[2]
        if 'consecutive_heat_wave_days' in self.df.columns:
            heat_wave_data = self.df[self.df['consecutive_heat_wave_days'] > 0].copy()
            if len(heat_wave_data) > 0:
                sample_hw = heat_wave_data.sample(n=min(1000, len(heat_wave_data)))
                scatter = ax3.scatter(sample_hw['datetime'], sample_hw['consecutive_heat_wave_days'],
                                    c=sample_hw['consecutive_heat_wave_days'], cmap='Reds', s=20)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Consecutive Heat Wave Days')
                ax3.set_title('Heat Wave Events Timeline')
                plt.colorbar(scatter, ax=ax3, shrink=0.6)
                ax3.tick_params(axis='x', rotation=45)
        
        # 4. Seasonal Temperature Patterns
        ax4 = axes[3]
        if hasattr(self, 'temp_col') and self.temp_col:
            hourly_seasonal = self.df.groupby(['month', 'hour'])[self.temp_col].mean().unstack()
            im = ax4.imshow(hourly_seasonal.values, cmap='RdYlBu_r', aspect='auto')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Month')
            ax4.set_title('Seasonal Temperature Patterns')
            ax4.set_xticks(range(0, 24, 4))
            ax4.set_xticklabels(range(0, 24, 4))
            ax4.set_yticks(range(12))
            ax4.set_yticklabels(range(1, 13))
            plt.colorbar(im, ax=ax4, shrink=0.6)
        
        # 5. Thermal Stress Accumulation
        ax5 = axes[4]
        if 'daily_thermal_stress' in self.df.columns:
            sample_data = self.df.sample(n=min(3000, len(self.df))).sort_values('datetime')
            ax5.plot(sample_data['datetime'], sample_data['daily_thermal_stress'], 
                    alpha=0.7, linewidth=0.5, color='orange')
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Daily Thermal Stress')
            ax5.set_title('Thermal Stress Over Time')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
        
        # 6. Monsoon Cooling Effect
        ax6 = axes[5]
        if 'monsoon_cooling_effect' in self.df.columns:
            monsoon_data = self.df[self.df['month'].isin(self.MONSOON_MONTHS)]
            if len(monsoon_data) > 0:
                cooling_effect = monsoon_data['monsoon_cooling_effect'].dropna()
                ax6.hist(cooling_effect, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
                ax6.set_xlabel('Monsoon Cooling Effect (°C)')
                ax6.set_ylabel('Frequency')
                ax6.set_title('Monsoon Cooling Effect Distribution')
                ax6.grid(True, alpha=0.3)
        
        # 7. AC Load Probability vs Temperature
        ax7 = axes[6]
        if 'ac_load_probability' in self.df.columns and hasattr(self, 'temp_col') and self.temp_col:
            sample_data = self.df.sample(n=min(3000, len(self.df)))
            ax7.scatter(sample_data[self.temp_col], sample_data['ac_load_probability'],
                       alpha=0.6, s=15, c='purple')
            ax7.set_xlabel('Temperature (°C)')
            ax7.set_ylabel('AC Load Probability')
            ax7.set_title('AC Load Probability vs Temperature')
            ax7.grid(True, alpha=0.3)
        
        # 8. Heat Index Categories Distribution
        ax8 = axes[7]
        if 'heat_index_category' in self.df.columns:
            category_counts = self.df['heat_index_category'].value_counts()
            colors = ['green', 'yellow', 'orange', 'red', 'darkred'][:len(category_counts)]
            ax8.bar(category_counts.index, category_counts.values, color=colors)
            ax8.set_xlabel('Heat Index Category')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Heat Index Categories Distribution')
            ax8.tick_params(axis='x', rotation=45)
        
        # 9. Humidity Discomfort by Season
        ax9 = axes[8]
        if 'humidity_discomfort' in self.df.columns and 'season_type' in self.df.columns:
            seasonal_humidity = self.df.groupby('season_type')['humidity_discomfort'].mean()
            ax9.bar(seasonal_humidity.index, seasonal_humidity.values, 
                   color='lightcoral', edgecolor='darkred')
            ax9.set_xlabel('Season')
            ax9.set_ylabel('Average Humidity Discomfort')
            ax9.set_title('Humidity Discomfort by Season')
            ax9.tick_params(axis='x', rotation=45)
            ax9.grid(True, alpha=0.3)
        
        # 10. Thermal feature correlation heatmap
        ax10 = axes[9]
        thermal_features = [col for col in self.df.columns if any(x in col.lower() 
                           for x in ['thermal', 'heat', 'cooling', 'monsoon', 'humidity'])]
        # Filter to only numeric columns
        numeric_thermal_features = [col for col in thermal_features 
                                  if self.df[col].dtype in ['int64', 'float64']][:10]
        if len(numeric_thermal_features) > 2:
            corr_data = self.df[numeric_thermal_features].corr()
            im = ax10.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
            ax10.set_xticks(range(len(numeric_thermal_features)))
            ax10.set_yticks(range(len(numeric_thermal_features)))
            ax10.set_xticklabels([f[:12] + '...' if len(f) > 12 else f for f in numeric_thermal_features], 
                               rotation=45, ha='right')
            ax10.set_yticklabels([f[:12] + '...' if len(f) > 12 else f for f in numeric_thermal_features])
            ax10.set_title('Thermal Features Correlation')
            plt.colorbar(im, ax=ax10, shrink=0.6)
        
        # 11. Daily temperature range by month
        ax11 = axes[10]
        if 'daily_temp_range' in self.df.columns:
            monthly_temp_range = self.df.groupby('month')['daily_temp_range'].mean()
            ax11.plot(monthly_temp_range.index, monthly_temp_range.values, 
                     marker='o', linewidth=2, markersize=6, color='darkblue')
            ax11.set_xlabel('Month')
            ax11.set_ylabel('Average Daily Temperature Range (°C)')
            ax11.set_title('Daily Temperature Range by Month')
            ax11.set_xticks(range(1, 13))
            ax11.grid(True, alpha=0.3)
        
        # 12. Summary statistics
        ax12 = axes[11]
        ax12.axis('off')
        
        # Count features by type
        feature_types = {}
        for feature, info in self.feature_log.items():
            ftype = info.get('type', 'other')
            feature_types[ftype] = feature_types.get(ftype, 0) + 1
        
        summary_text = f"""
DELHI THERMAL COMFORT FEATURES SUMMARY

Total Features Created: {len(self.feature_log)}

By Category:
• Heat Index: {feature_types.get('heat_index', 0)}
• Cooling Degree Days: {feature_types.get('cooling_degree_days', 0)}
• Heat Wave: {feature_types.get('heat_wave', 0)}
• Monsoon Impact: {feature_types.get('monsoon_impact', 0)}
• Thermal Stress: {feature_types.get('thermal_stress', 0)}

Key Achievements:
✅ Heat index calculation (Delhi-specific)
✅ Cooling degree days (24°C base)
✅ Heat wave detection (>42°C)
✅ Monsoon cooling effects
✅ Thermal stress accumulation

Climate Features Ready For:
🌡️ Temperature-Load Correlation
❄️ AC Demand Prediction
🔥 Heat Wave Impact Analysis
🌧️ Monsoon Load Patterns
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, 'delhi_thermal_comfort_features_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Thermal features visualization saved: {output_path}")
    
    def generate_thermal_report(self):
        """Generate comprehensive thermal features report"""
        print(f"\n📋 DELHI THERMAL COMFORT FEATURES REPORT")
        print("=" * 70)
        
        print(f"Dataset: {len(self.df):,} records × {len(self.df.columns)} columns")
        print(f"Thermal analysis period: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        print(f"\nThermal Feature Engineering Summary:")
        print(f"  Total new thermal features: {len(self.feature_log)}")
        
        # Group features by type
        feature_types = {}
        for feature, info in self.feature_log.items():
            ftype = info.get('type', 'other')
            if ftype not in feature_types:
                feature_types[ftype] = []
            feature_types[ftype].append(feature)
        
        for ftype, features in feature_types.items():
            print(f"\n  {ftype.upper().replace('_', ' ')} FEATURES ({len(features)}):")
            for feature in features[:5]:  # Show first 5
                if feature in self.feature_log:
                    info = self.feature_log[feature]
                    if 'mean' in info:
                        print(f"    {feature}: μ={info['mean']:.2f}, σ={info['std']:.2f}")
                    else:
                        print(f"    {feature}: {info.get('count', 'N/A')} values")
            if len(features) > 5:
                print(f"    ... and {len(features) - 5} more")
        
        # Climate insights
        if 'heat_index_celsius' in self.df.columns:
            max_heat_index = self.df['heat_index_celsius'].max()
            print(f"\n🌡️ Climate Insights:")
            print(f"  Maximum heat index recorded: {max_heat_index:.1f}°C")
        
        if 'consecutive_heat_wave_days' in self.df.columns:
            max_heat_wave = self.df['consecutive_heat_wave_days'].max()
            heat_wave_hours = (self.df['consecutive_heat_wave_days'] > 0).sum()
            print(f"  Longest heat wave: {max_heat_wave} consecutive days")
            print(f"  Total heat wave hours: {heat_wave_hours:,}")
        
        if 'daily_cooling_degree_days' in self.df.columns:
            total_cdd = self.df['daily_cooling_degree_days'].sum() / 24  # Convert to days
            print(f"  Total cooling degree days: {total_cdd:.1f}")
        
        print(f"\nDelhi-Specific Thermal Capabilities:")
        print(f"  ✅ Heat index with Delhi humidity patterns")
        print(f"  ✅ Cooling demand analysis (24°C base)")
        print(f"  ✅ Heat wave detection (>42°C threshold)")
        print(f"  ✅ Monsoon cooling effect quantification")
        print(f"  ✅ Thermal stress accumulation tracking")
        print(f"  ✅ Seasonal comfort zone analysis")
        
        print(f"\nThermal-Load Correlation Ready:")
        print(f"  🔄 AC load probability predictions")
        print(f"  📊 Peak cooling demand forecasting")
        print(f"  🌡️ Temperature-sensitive load modeling")
        print(f"  🌧️ Monsoon impact on electrical demand")
        
        print(f"\n💾 Thermal-enhanced dataset ready for advanced modeling!")
    
    def save_thermal_enhanced_dataset(self):
        """Save the dataset with thermal comfort features"""
        output_path = os.path.join(self.output_dir, 'delhi_thermal_comfort_enhanced.csv')
        self.df.to_csv(output_path, index=False)
        print(f"\n✅ Thermal-enhanced dataset saved: {output_path}")
        return output_path
    
    def run_complete_thermal_analysis(self):
        """Execute the complete thermal comfort feature engineering"""
        try:
            # Load enhanced dataset from Phase 2 Step 1
            self.load_enhanced_dataset()
            
            # Create Delhi-specific thermal features
            print(f"\n🚀 EXECUTING DELHI THERMAL COMFORT ANALYSIS")
            print("=" * 50)
            
            # Step 1: Heat Index Calculation
            self.calculate_heat_index()
            
            # Step 2: Cooling Degree Days
            self.calculate_cooling_degree_days()
            
            # Step 3: Heat Wave Detection
            self.detect_heat_waves()
            
            # Step 4: Monsoon Impact Features
            self.create_monsoon_impact_features()
            
            # Step 5: Thermal Stress Accumulation
            self.calculate_thermal_stress_accumulation()
            
            # Create visualizations and reports
            self.create_thermal_visualization()
            self.generate_thermal_report()
            
            # Save thermal-enhanced dataset
            output_path = self.save_thermal_enhanced_dataset()
            
            print(f"\n🎉 DELHI THERMAL COMFORT ANALYSIS COMPLETED!")
            print(f"📁 Output file: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during thermal analysis: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Use the enhanced dataset from Phase 2 Step 1
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, 'delhi_enhanced_dual_peak_features.csv')
    
    # Initialize thermal comfort analysis
    thermal_analysis = DelhiThermalComfortFeatures(input_path)
    
    # Run complete thermal comfort analysis
    output_path = thermal_analysis.run_complete_thermal_analysis()
    
    return output_path


if __name__ == "__main__":
    output_file = main()
