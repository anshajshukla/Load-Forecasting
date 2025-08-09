"""
Delhi Load Forecasting - Phase 2: Complex Interaction Features
=============================================================
Advanced multi-variable interaction feature engineering for Delhi.
Focus on complex relationships between weather, time, festivals, and load patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless operation
plt.switch_backend('Agg')

class DelhiInteractionFeatures:
    """
    Complex interaction feature engineering for Delhi load forecasting
    Implements multi-variable relationships and sophisticated feature combinations
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.feature_log = {}
        self.output_dir = os.path.dirname(__file__)
        
        # Delhi-specific interaction constants
        self.COMFORT_TEMP_RANGE = (20, 28)  # Comfortable temperature range (°C)
        self.HIGH_HUMIDITY_THRESHOLD = 70    # High humidity threshold (%)
        self.LOW_HUMIDITY_THRESHOLD = 30     # Low humidity threshold (%)
        self.AC_THRESHOLD_TEMP = 28          # Temperature when AC usage starts
        self.HEATING_THRESHOLD_TEMP = 18     # Temperature when heating might be needed
        
        # Solar generation patterns (typical Delhi)
        self.SOLAR_PEAK_HOURS = (11, 15)    # Peak solar generation hours
        self.SOLAR_ACTIVE_HOURS = (6, 18)   # Solar active hours
        
        # Load pattern characteristics
        self.PEAK_LOAD_HOURS = [(7, 11), (18, 22)]  # Morning and evening peaks
        self.OFF_PEAK_HOURS = (23, 5)       # Late night off-peak
        
        # Duck curve characteristics (solar impact on load)
        self.DUCK_CURVE_HOURS = (10, 16)    # Hours when solar reduces net load
        self.RAMP_UP_HOURS = (16, 19)       # Evening ramp-up when solar decreases
    
    def load_temporal_enhanced_dataset(self):
        """Load the temporal-enhanced dataset from Phase 2 Step 3"""
        print("=" * 70)
        print("DELHI COMPLEX INTERACTION FEATURES - PHASE 2 STEP 4")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Temporal-enhanced dataset not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
        
        print(f"✅ Loading temporal-enhanced dataset: {os.path.basename(self.data_path)}")
        print(f"📊 Dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        # Verify required columns exist
        required_weather_cols = ['temperature_2m (°C)', 'relative_humidity_2m (%)']
        required_time_cols = ['hour', 'day_of_week', 'month']
        
        missing_cols = []
        for col in required_weather_cols + required_time_cols:
            if col not in self.df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols}")
            
        print(f"✅ Weather and temporal columns verified")
    
    def create_temperature_humidity_hour_interactions(self):
        """
        A) Temperature × Humidity × Hour Triple Interactions
        """
        print(f"\n🌡️ CREATING TEMPERATURE-HUMIDITY-HOUR INTERACTIONS")
        print("-" * 50)
        
        # Ensure required columns exist with proper names
        temp_col = 'temperature_2m (°C)'
        humidity_col = 'relative_humidity_2m (%)'
        
        if temp_col not in self.df.columns or humidity_col not in self.df.columns:
            print(f"⚠️ Required weather columns not found. Available columns with 'temp' or 'humidity':")
            weather_cols = [col for col in self.df.columns if 'temp' in col.lower() or 'humid' in col.lower()]
            for col in weather_cols:
                print(f"   {col}")
            return
        
        # A1) Basic triple interaction
        self.df['temp_humidity_hour'] = (
            self.df[temp_col] * self.df[humidity_col] * self.df['hour']
        )
        
        # A2) Comfort index based on temperature and humidity
        # Heat Index approximation for Delhi conditions
        temp_f = self.df[temp_col] * 9/5 + 32  # Convert to Fahrenheit for heat index
        rh = self.df[humidity_col]
        
        # Simplified heat index calculation
        heat_index_f = (
            -42.379 + 2.04901523 * temp_f + 10.14333127 * rh
            - 0.22475541 * temp_f * rh - 6.83783e-3 * temp_f**2
            - 5.481717e-2 * rh**2 + 1.22874e-3 * temp_f**2 * rh
            + 8.5282e-4 * temp_f * rh**2 - 1.99e-6 * temp_f**2 * rh**2
        )
        
        self.df['heat_index_celsius'] = (heat_index_f - 32) * 5/9
        
        # A3) Hourly comfort patterns
        # Morning comfort (6-10 AM)
        morning_mask = (self.df['hour'] >= 6) & (self.df['hour'] <= 10)
        self.df['morning_comfort_index'] = np.where(
            morning_mask,
            self.df['heat_index_celsius'] * (1 + self.df[humidity_col] / 100),
            0
        )
        
        # Afternoon discomfort (12-16 PM)
        afternoon_mask = (self.df['hour'] >= 12) & (self.df['hour'] <= 16)
        self.df['afternoon_discomfort_index'] = np.where(
            afternoon_mask,
            (self.df[temp_col] - 25) * (self.df[humidity_col] / 50) * (self.df['hour'] - 12),
            0
        )
        
        # Evening cooling effect (18-22 PM)
        evening_mask = (self.df['hour'] >= 18) & (self.df['hour'] <= 22)
        self.df['evening_cooling_index'] = np.where(
            evening_mask,
            (35 - self.df[temp_col]) * (1 - self.df[humidity_col] / 100) * (22 - self.df['hour']),
            0
        )
        
        # A4) AC demand probability based on triple interaction
        # Higher temperature, humidity, and specific hours increase AC demand
        temp_factor = np.clip((self.df[temp_col] - 25) / 15, 0, 1)  # 0 at 25°C, 1 at 40°C+
        humidity_factor = np.clip(self.df[humidity_col] / 100, 0, 1)  # 0-1 scale
        
        # Hour factor (higher in afternoon/evening)
        hour_factor = np.where(
            (self.df['hour'] >= 12) & (self.df['hour'] <= 22),
            1.0,
            np.where(
                (self.df['hour'] >= 6) & (self.df['hour'] < 12),
                0.7,
                0.3
            )
        )
        
        self.df['ac_demand_probability'] = temp_factor * humidity_factor * hour_factor
        
        # A5) Thermal stress accumulation with hourly patterns
        # Cumulative thermal stress throughout the day
        self.df = self.df.sort_values('datetime')
        
        # Daily thermal stress (resets each day)
        self.df['date'] = self.df['datetime'].dt.date
        thermal_stress = np.maximum(0, self.df[temp_col] - 28) * np.maximum(0, self.df[humidity_col] - 60) / 100
        
        self.df['daily_thermal_stress_accumulation'] = self.df.groupby('date')['heat_index_celsius'].transform(
            lambda x: x.expanding().mean() - 30  # Subtract comfortable temperature
        ).fillna(0)
        
        # A6) Humidity-Temperature interaction by time of day
        # Different humidity effects at different times
        morning_humidity_effect = np.where(
            morning_mask,
            self.df[humidity_col] * 0.5,  # Lower impact in morning
            0
        )
        
        afternoon_humidity_effect = np.where(
            afternoon_mask,
            self.df[humidity_col] * 1.5,  # Higher impact in afternoon
            0
        )
        
        evening_humidity_effect = np.where(
            evening_mask,
            self.df[humidity_col] * 1.0,  # Moderate impact in evening
            0
        )
        
        self.df['humidity_time_interaction'] = (
            morning_humidity_effect + afternoon_humidity_effect + evening_humidity_effect
        )
        
        temp_humidity_features = [
            'temp_humidity_hour', 'heat_index_celsius', 'morning_comfort_index',
            'afternoon_discomfort_index', 'evening_cooling_index', 'ac_demand_probability',
            'daily_thermal_stress_accumulation', 'humidity_time_interaction'
        ]
        
        print(f"✅ Created {len(temp_humidity_features)} temperature-humidity-hour interaction features")
        print(f"   🌡️ Heat index and comfort calculations")
        print(f"   ⏰ Hourly thermal pattern interactions")
        print(f"   ❄️ AC demand probability modeling")
        
        # Log features
        for feature in temp_humidity_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'temp_humidity_hour',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'range': f"{valid_values.min():.2f} to {valid_values.max():.2f}"
                    }
    
    def create_solar_temperature_interactions(self):
        """
        B) Solar Generation × Temperature Cooling Offset Interactions
        """
        print(f"\n☀️ CREATING SOLAR-TEMPERATURE INTERACTIONS")
        print("-" * 50)
        
        temp_col = 'temperature_2m (°C)'
        
        # B1) Solar availability estimation
        # Based on hour and weather conditions
        solar_potential = np.where(
            (self.df['hour'] >= 6) & (self.df['hour'] <= 18),
            np.sin(np.pi * (self.df['hour'] - 6) / 12),  # Solar curve throughout day
            0
        )
        
        # Cloud cover impact (if available, otherwise use clear sky assumption)
        if 'cloud_cover' in self.df.columns:
            cloud_factor = 1 - self.df['cloud_cover'] / 100
        else:
            # Estimate cloud impact from humidity (rough approximation)
            cloud_factor = 1 - np.clip((self.df['relative_humidity_2m (%)'] - 60) / 40, 0, 0.4)
        
        self.df['estimated_solar_generation'] = solar_potential * cloud_factor
        
        # B2) Solar cooling offset
        # When solar is high and temperature is high, solar can offset cooling load
        high_temp_mask = self.df[temp_col] > 30
        high_solar_mask = self.df['estimated_solar_generation'] > 0.5
        
        self.df['solar_cooling_offset'] = np.where(
            high_temp_mask & high_solar_mask,
            self.df['estimated_solar_generation'] * (self.df[temp_col] - 30) * 0.1,  # Offset factor
            0
        )
        
        # B3) Net cooling demand (Temperature load minus solar offset)
        cooling_demand = np.maximum(0, self.df[temp_col] - 25)  # Basic cooling demand
        self.df['net_cooling_demand'] = np.maximum(0, cooling_demand - self.df['solar_cooling_offset'])
        
        # B4) Solar-Temperature efficiency
        # Solar panels are less efficient at higher temperatures
        panel_efficiency = 1 - np.clip((self.df[temp_col] - 25) * 0.004, 0, 0.2)  # 0.4%/°C efficiency loss
        self.df['temperature_adjusted_solar'] = self.df['estimated_solar_generation'] * panel_efficiency
        
        # B5) Duck curve effect
        # Solar generation creates duck curve pattern in net load
        duck_curve_hours = (self.df['hour'] >= 10) & (self.df['hour'] <= 16)
        
        self.df['duck_curve_depth'] = np.where(
            duck_curve_hours,
            self.df['estimated_solar_generation'] * (1 + self.df[temp_col] / 40),  # Deeper with higher temp
            0
        )
        
        # B6) Evening ramp requirement
        # When solar drops in evening, load ramps up (especially in hot weather)
        evening_ramp_hours = (self.df['hour'] >= 16) & (self.df['hour'] <= 19)
        
        self.df['evening_ramp_intensity'] = np.where(
            evening_ramp_hours,
            (self.df[temp_col] - 25) * (20 - self.df['hour']) * (1 - self.df['estimated_solar_generation']),
            0
        )
        
        # B7) Solar-Temperature load correlation
        # Complex relationship between solar availability and temperature-driven load
        solar_temp_correlation = (
            self.df['estimated_solar_generation'] * 
            (1 - np.clip((self.df[temp_col] - 20) / 20, 0, 1)) +  # Inverse correlation
            (1 - self.df['estimated_solar_generation']) * 
            np.clip((self.df[temp_col] - 25) / 15, 0, 1)  # Direct correlation when no solar
        )
        
        self.df['solar_temperature_load_factor'] = solar_temp_correlation
        
        solar_temp_features = [
            'estimated_solar_generation', 'solar_cooling_offset', 'net_cooling_demand',
            'temperature_adjusted_solar', 'duck_curve_depth', 'evening_ramp_intensity',
            'solar_temperature_load_factor'
        ]
        
        print(f"✅ Created {len(solar_temp_features)} solar-temperature interaction features")
        print(f"   ☀️ Solar generation estimation and efficiency modeling")
        print(f"   🦆 Duck curve depth and evening ramp calculations")
        print(f"   ❄️ Solar cooling offset and net demand modeling")
        
        # Log features
        for feature in solar_temp_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'solar_temperature',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'range': f"{valid_values.min():.2f} to {valid_values.max():.2f}"
                    }
    
    def create_festival_weather_day_interactions(self):
        """
        C) Festival × Weather × Day_type Complex Interactions
        """
        print(f"\n🎉 CREATING FESTIVAL-WEATHER-DAY INTERACTIONS")
        print("-" * 50)
        
        temp_col = 'temperature_2m (°C)'
        humidity_col = 'relative_humidity_2m (%)'
        
        # Check for required festival and day type columns
        festival_cols = [col for col in self.df.columns if 'festival' in col.lower()]
        day_type_cols = [col for col in self.df.columns if any(x in col.lower() for x in ['weekend', 'weekday', 'day_of_week'])]
        
        if not festival_cols:
            print("⚠️ No festival columns found. Creating basic festival indicators...")
            # Create basic festival periods
            self.df['is_festival_period'] = (
                ((self.df['month'] == 10) & (self.df['day'] >= 15)) |  # Diwali period
                ((self.df['month'] == 11) & (self.df['day'] <= 15)) |
                ((self.df['month'] == 3) & (self.df['day'] >= 10) & (self.df['day'] <= 20)) |  # Holi period
                ((self.df['month'] == 8) & (self.df['day'] == 15)) |  # Independence Day
                ((self.df['month'] == 1) & (self.df['day'] == 26))  # Republic Day
            ).astype(int)
            festival_intensity_col = 'is_festival_period'
        else:
            # Use existing festival intensity if available
            festival_intensity_col = 'festival_intensity' if 'festival_intensity' in self.df.columns else festival_cols[0]
        
        # Ensure weekend column exists
        if 'is_weekend' not in self.df.columns:
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # C1) Festival weather sensitivity
        # Festivals may be more/less affected by weather conditions
        festival_mask = self.df[festival_intensity_col] > 0
        
        # Hot weather during festivals (increased AC usage for celebrations)
        self.df['festival_hot_weather_load'] = np.where(
            festival_mask & (self.df[temp_col] > 30),
            self.df[festival_intensity_col] * (self.df[temp_col] - 30) * 1.5,
            0
        )
        
        # Humid weather during festivals (indoor celebrations)
        self.df['festival_humid_weather_load'] = np.where(
            festival_mask & (self.df[humidity_col] > 70),
            self.df[festival_intensity_col] * (self.df[humidity_col] - 70) * 0.1,
            0
        )
        
        # Pleasant weather during festivals (outdoor activities)
        pleasant_weather = (
            (self.df[temp_col] >= 20) & (self.df[temp_col] <= 30) &
            (self.df[humidity_col] >= 40) & (self.df[humidity_col] <= 70)
        )
        
        self.df['festival_pleasant_weather_activity'] = np.where(
            festival_mask & pleasant_weather,
            self.df[festival_intensity_col] * 2.0,  # Increased outdoor activity
            0
        )
        
        # C2) Weekend-Festival-Weather interactions
        weekend_festival = self.df['is_weekend'] & festival_mask
        
        # Weekend festivals in hot weather (peak load scenario)
        self.df['weekend_festival_hot_load'] = np.where(
            weekend_festival & (self.df[temp_col] > 32),
            self.df[festival_intensity_col] * self.df['is_weekend'] * (self.df[temp_col] - 32) * 2.0,
            0
        )
        
        # Weekend festivals in pleasant weather (mixed indoor/outdoor)
        self.df['weekend_festival_pleasant_activity'] = np.where(
            weekend_festival & pleasant_weather,
            self.df[festival_intensity_col] * self.df['is_weekend'] * 1.5,
            0
        )
        
        # C3) Weekday vs Weekend festival patterns
        weekday_festival = (1 - self.df['is_weekend']) & festival_mask
        
        # Weekday festivals (different pattern - evening focused)
        evening_hours = (self.df['hour'] >= 17) & (self.df['hour'] <= 22)
        
        self.df['weekday_festival_evening_load'] = np.where(
            weekday_festival & evening_hours,
            self.df[festival_intensity_col] * (1 - self.df['is_weekend']) * 1.8,
            0
        )
        
        # C4) Day type weather sensitivity
        # Weekend weather sensitivity (more leisure activities)
        self.df['weekend_weather_sensitivity'] = (
            self.df['is_weekend'] * 
            (np.abs(self.df[temp_col] - 25) + np.abs(self.df[humidity_col] - 50)) / 50
        )
        
        # Weekday weather sensitivity (work/commute patterns)
        self.df['weekday_weather_sensitivity'] = (
            (1 - self.df['is_weekend']) * 
            (np.maximum(0, self.df[temp_col] - 28) + np.maximum(0, self.df[humidity_col] - 60)) / 40
        )
        
        # C5) Complex three-way interactions
        # Festival intensity × Temperature comfort × Weekend factor
        comfort_deviation = np.abs(self.df[temp_col] - 25)  # Deviation from 25°C comfort
        
        self.df['festival_comfort_weekend_interaction'] = (
            self.df[festival_intensity_col] * 
            (1 + comfort_deviation / 10) *  # Higher load with temperature deviation
            (1 + self.df['is_weekend'] * 0.5)  # Weekend boost
        )
        
        # Weather extremes during festivals
        extreme_weather = (
            (self.df[temp_col] > 38) | (self.df[temp_col] < 10) |
            (self.df[humidity_col] > 85) | (self.df[humidity_col] < 20)
        )
        
        self.df['festival_extreme_weather_load'] = np.where(
            festival_mask & extreme_weather,
            self.df[festival_intensity_col] * 3.0,  # High load during extreme weather festivals
            0
        )
        
        festival_weather_features = [
            'festival_hot_weather_load', 'festival_humid_weather_load', 'festival_pleasant_weather_activity',
            'weekend_festival_hot_load', 'weekend_festival_pleasant_activity', 'weekday_festival_evening_load',
            'weekend_weather_sensitivity', 'weekday_weather_sensitivity', 'festival_comfort_weekend_interaction',
            'festival_extreme_weather_load'
        ]
        
        print(f"✅ Created {len(festival_weather_features)} festival-weather-day interaction features")
        print(f"   🎉 Festival weather sensitivity modeling")
        print(f"   📅 Weekend vs weekday festival patterns")
        print(f"   🌡️ Complex three-way interaction calculations")
        
        # Log features
        for feature in festival_weather_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'festival_weather_day',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'range': f"{valid_values.min():.2f} to {valid_values.max():.2f}"
                    }
    
    def create_peak_hour_weather_sensitivity(self):
        """
        D) Peak Hour × Weather Sensitivity Interactions
        """
        print(f"\n⚡ CREATING PEAK HOUR-WEATHER SENSITIVITY")
        print("-" * 50)
        
        temp_col = 'temperature_2m (°C)'
        humidity_col = 'relative_humidity_2m (%)'
        
        # D1) Peak hour identification
        morning_peak = (self.df['hour'] >= 7) & (self.df['hour'] <= 11)
        evening_peak = (self.df['hour'] >= 18) & (self.df['hour'] <= 22)
        off_peak = (self.df['hour'] >= 23) | (self.df['hour'] <= 6)
        mid_day = (self.df['hour'] >= 12) & (self.df['hour'] <= 17)
        
        # D2) Temperature sensitivity during different hours
        # Morning peak temperature sensitivity (AC startup)
        self.df['morning_peak_temp_sensitivity'] = np.where(
            morning_peak,
            np.maximum(0, self.df[temp_col] - 26) * 2.0,  # High sensitivity above 26°C
            0
        )
        
        # Evening peak temperature sensitivity (peak AC usage)
        self.df['evening_peak_temp_sensitivity'] = np.where(
            evening_peak,
            np.maximum(0, self.df[temp_col] - 25) * 3.0,  # Very high sensitivity
            0
        )
        
        # Mid-day temperature sensitivity (solar interaction)
        self.df['midday_temp_sensitivity'] = np.where(
            mid_day,
            np.maximum(0, self.df[temp_col] - 30) * 2.5,  # High sensitivity at peak heat
            0
        )
        
        # Off-peak temperature sensitivity (minimal)
        self.df['offpeak_temp_sensitivity'] = np.where(
            off_peak,
            np.maximum(0, self.df[temp_col] - 28) * 0.5,  # Low sensitivity
            0
        )
        
        # D3) Humidity sensitivity during peak hours
        # Morning humidity impact (comfort for commuting)
        self.df['morning_peak_humidity_impact'] = np.where(
            morning_peak,
            np.maximum(0, self.df[humidity_col] - 60) * 0.1,
            0
        )
        
        # Evening humidity impact (indoor comfort)
        self.df['evening_peak_humidity_impact'] = np.where(
            evening_peak,
            np.maximum(0, self.df[humidity_col] - 65) * 0.15,
            0
        )
        
        # D4) Combined weather stress during peaks
        # Weather stress index (temperature + humidity combined effect)
        weather_stress = (
            (self.df[temp_col] - 25) / 15 +  # Temperature stress (normalized)
            (self.df[humidity_col] - 50) / 50  # Humidity stress (normalized)
        )
        
        # Peak hour weather stress multipliers
        self.df['morning_peak_weather_stress'] = np.where(
            morning_peak,
            weather_stress * 1.5,
            0
        )
        
        self.df['evening_peak_weather_stress'] = np.where(
            evening_peak,
            weather_stress * 2.0,  # Highest multiplier for evening peak
            0
        )
        
        # D5) Weather-driven load volatility during peaks
        # Temperature volatility (how much temperature affects load during peaks)
        temp_volatility = np.abs(self.df[temp_col] - self.df[temp_col].rolling(24, center=True).mean().fillna(self.df[temp_col].mean()))
        
        self.df['peak_temp_volatility_impact'] = np.where(
            morning_peak | evening_peak,
            temp_volatility * (1 + np.maximum(0, self.df[temp_col] - 30) / 10),
            0
        )
        
        # D6) Extreme weather peak hour effects
        extreme_hot = self.df[temp_col] > 40
        extreme_humid = self.df[humidity_col] > 80
        
        # Extreme heat during peaks (system stress)
        self.df['extreme_heat_peak_stress'] = np.where(
            (morning_peak | evening_peak) & extreme_hot,
            (self.df[temp_col] - 40) * 5.0,  # Very high stress multiplier
            0
        )
        
        # Extreme humidity during peaks (comfort degradation)
        self.df['extreme_humidity_peak_stress'] = np.where(
            (morning_peak | evening_peak) & extreme_humid,
            (self.df[humidity_col] - 80) * 0.2,
            0
        )
        
        # D7) Peak hour weather adaptation patterns
        # How load adapts to weather during different peak periods
        
        # Morning adaptation (slower response)
        self.df['morning_weather_adaptation'] = np.where(
            morning_peak,
            0.7 * np.maximum(0, self.df[temp_col] - 25) + 0.3 * np.maximum(0, self.df[humidity_col] - 60),
            0
        )
        
        # Evening adaptation (faster response)
        self.df['evening_weather_adaptation'] = np.where(
            evening_peak,
            0.8 * np.maximum(0, self.df[temp_col] - 24) + 0.2 * np.maximum(0, self.df[humidity_col] - 55),
            0
        )
        
        peak_weather_features = [
            'morning_peak_temp_sensitivity', 'evening_peak_temp_sensitivity', 'midday_temp_sensitivity',
            'offpeak_temp_sensitivity', 'morning_peak_humidity_impact', 'evening_peak_humidity_impact',
            'morning_peak_weather_stress', 'evening_peak_weather_stress', 'peak_temp_volatility_impact',
            'extreme_heat_peak_stress', 'extreme_humidity_peak_stress', 'morning_weather_adaptation',
            'evening_weather_adaptation'
        ]
        
        print(f"✅ Created {len(peak_weather_features)} peak hour-weather sensitivity features")
        print(f"   ⚡ Peak period weather sensitivity modeling")
        print(f"   🌡️ Temperature and humidity impact quantification")
        print(f"   🔥 Extreme weather peak stress calculations")
        
        # Log features
        for feature in peak_weather_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'peak_weather_sensitivity',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'range': f"{valid_values.min():.2f} to {valid_values.max():.2f}"
                    }
    
    def create_duck_curve_temperature_interactions(self):
        """
        E) Duck Curve Depth × Temperature Advanced Interactions
        """
        print(f"\n🦆 CREATING DUCK CURVE-TEMPERATURE INTERACTIONS")
        print("-" * 50)
        
        temp_col = 'temperature_2m (°C)'
        
        # E1) Enhanced duck curve modeling
        # Solar generation curve (bell-shaped during day)
        solar_curve = np.where(
            (self.df['hour'] >= 6) & (self.df['hour'] <= 18),
            np.sin(np.pi * (self.df['hour'] - 6) / 12) ** 2,  # Squared sine for sharper peak
            0
        )
        
        # Temperature-adjusted solar efficiency
        temp_efficiency = 1 - np.clip((self.df[temp_col] - 25) * 0.004, 0, 0.25)
        adjusted_solar = solar_curve * temp_efficiency
        
        # E2) Duck curve depth calculation
        # Deeper duck curve when high solar meets high temperature load
        base_load = 0.6  # Assume 60% base load
        temperature_load = np.clip((self.df[temp_col] - 20) / 20, 0, 1.5)  # Temperature-driven load
        
        gross_load = base_load + temperature_load
        net_load = gross_load - adjusted_solar
        
        self.df['duck_curve_depth'] = np.where(
            (self.df['hour'] >= 10) & (self.df['hour'] <= 16),
            gross_load - net_load,  # How much solar reduces the load
            0
        )
        
        # E3) Temperature-dependent duck curve characteristics
        # Shallow duck curve (cloudy/cool days)
        cool_day = self.df[temp_col] < 25
        self.df['shallow_duck_curve'] = np.where(
            cool_day & (self.df['hour'] >= 10) & (self.df['hour'] <= 16),
            self.df['duck_curve_depth'] * 0.5,  # Reduced depth on cool days
            0
        )
        
        # Deep duck curve (hot sunny days)
        hot_day = self.df[temp_col] > 35
        self.df['deep_duck_curve'] = np.where(
            hot_day & (self.df['hour'] >= 10) & (self.df['hour'] <= 16),
            self.df['duck_curve_depth'] * 1.8,  # Increased depth on hot days
            0
        )
        
        # E4) Duck curve timing shifts with temperature
        # Earlier duck curve start in very hot weather (earlier solar impact)
        very_hot = self.df[temp_col] > 38
        early_duck_hours = (self.df['hour'] >= 9) & (self.df['hour'] <= 15)
        
        self.df['early_duck_curve'] = np.where(
            very_hot & early_duck_hours,
            self.df['duck_curve_depth'] * 1.2,
            0
        )
        
        # Later duck curve end in mild weather (extended solar hours)
        mild_weather = (self.df[temp_col] >= 20) & (self.df[temp_col] <= 30)
        extended_duck_hours = (self.df['hour'] >= 11) & (self.df['hour'] <= 17)
        
        self.df['extended_duck_curve'] = np.where(
            mild_weather & extended_duck_hours,
            self.df['duck_curve_depth'] * 1.1,
            0
        )
        
        # E5) Evening ramp-up intensity
        # How quickly load ramps up when solar decreases (temperature dependent)
        ramp_hours = (self.df['hour'] >= 16) & (self.df['hour'] <= 19)
        
        # Ramp intensity based on temperature and solar drop
        solar_drop = np.maximum(0, self.df.groupby(self.df['datetime'].dt.date)['duck_curve_depth'].transform('max') - adjusted_solar)
        
        self.df['temperature_driven_ramp'] = np.where(
            ramp_hours,
            solar_drop * (self.df[temp_col] - 20) / 20 * (20 - self.df['hour']),  # Intensity × temp factor × hour factor
            0
        )
        
        # E6) Duck curve stability metrics
        # How stable the duck curve pattern is with temperature variations
        
        # Daily temperature range impact on duck curve
        daily_temp_range = self.df.groupby(self.df['datetime'].dt.date)[temp_col].transform('max') - \
                          self.df.groupby(self.df['datetime'].dt.date)[temp_col].transform('min')
        
        self.df['duck_curve_temperature_stability'] = np.where(
            (self.df['hour'] >= 10) & (self.df['hour'] <= 16),
            1 / (1 + daily_temp_range / 10),  # More stable with smaller temp range
            0
        )
        
        # E7) Net load variability during duck curve hours
        # How much the net load varies due to temperature-solar interactions
        duck_hours = (self.df['hour'] >= 10) & (self.df['hour'] <= 16)
        
        # Calculate hourly net load variability
        hourly_net_load = gross_load - adjusted_solar
        self.df['duck_curve_net_load_variability'] = np.where(
            duck_hours,
            np.abs(hourly_net_load - hourly_net_load.rolling(3, center=True).mean().fillna(hourly_net_load)),
            0
        )
        
        # E8) Post-duck curve recovery patterns
        # How load recovers after duck curve hours (temperature dependent)
        recovery_hours = (self.df['hour'] >= 17) & (self.df['hour'] <= 20)
        
        max_duck_depth = self.df.groupby(self.df['datetime'].dt.date)['duck_curve_depth'].transform('max')
        
        self.df['post_duck_recovery_intensity'] = np.where(
            recovery_hours,
            max_duck_depth * (self.df[temp_col] - 25) / 15 * (21 - self.df['hour']) / 4,
            0
        )
        
        duck_curve_features = [
            'duck_curve_depth', 'shallow_duck_curve', 'deep_duck_curve', 'early_duck_curve',
            'extended_duck_curve', 'temperature_driven_ramp', 'duck_curve_temperature_stability',
            'duck_curve_net_load_variability', 'post_duck_recovery_intensity'
        ]
        
        print(f"✅ Created {len(duck_curve_features)} duck curve-temperature interaction features")
        print(f"   🦆 Advanced duck curve depth and timing modeling")
        print(f"   🌡️ Temperature-dependent curve characteristics")
        print(f"   📈 Evening ramp-up and recovery patterns")
        
        # Log features
        for feature in duck_curve_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'duck_curve_temperature',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'range': f"{valid_values.min():.2f} to {valid_values.max():.2f}"
                    }
    
    def create_interaction_visualization(self):
        """Create comprehensive visualization of interaction features"""
        print(f"\n📊 CREATING INTERACTION FEATURES VISUALIZATION")
        print("-" * 50)
        
        fig, axes = plt.subplots(4, 3, figsize=(24, 20))
        axes = axes.flatten()
        
        # 1. Temperature-Humidity-Hour interaction heatmap
        ax1 = axes[0]
        if 'ac_demand_probability' in self.df.columns:
            # Create 2D heatmap for temp vs hour, averaged over humidity ranges
            pivot_data = self.df.groupby(['hour', pd.cut(self.df['temperature_2m (°C)'], bins=10)])['ac_demand_probability'].mean().unstack()
            
            if not pivot_data.empty:
                im = ax1.imshow(pivot_data.values, cmap='viridis', aspect='auto')
                ax1.set_xlabel('Temperature Bins')
                ax1.set_ylabel('Hour of Day')
                ax1.set_title('AC Demand Probability: Hour × Temperature')
                ax1.set_yticks(range(0, 24, 4))
                ax1.set_yticklabels(range(0, 24, 4))
                plt.colorbar(im, ax=ax1, shrink=0.6)
        
        # 2. Solar-Temperature cooling offset
        ax2 = axes[1]
        if 'solar_cooling_offset' in self.df.columns and 'estimated_solar_generation' in self.df.columns:
            scatter_data = self.df.sample(min(1000, len(self.df)))  # Sample for performance
            scatter = ax2.scatter(scatter_data['estimated_solar_generation'], 
                                scatter_data['solar_cooling_offset'],
                                c=scatter_data['temperature_2m (°C)'], cmap='coolwarm', alpha=0.6)
            ax2.set_xlabel('Solar Generation')
            ax2.set_ylabel('Solar Cooling Offset')
            ax2.set_title('Solar Generation vs Cooling Offset\n(Color = Temperature)')
            plt.colorbar(scatter, ax=ax2, shrink=0.6)
        
        # 3. Festival-Weather interaction patterns
        ax3 = axes[2]
        if 'festival_hot_weather_load' in self.df.columns:
            festival_data = self.df[self.df['festival_hot_weather_load'] > 0]
            if len(festival_data) > 0:
                hourly_festival = festival_data.groupby('hour')['festival_hot_weather_load'].mean()
                ax3.plot(hourly_festival.index, hourly_festival.values, 
                        marker='o', linewidth=2, color='red')
                ax3.set_xlabel('Hour of Day')
                ax3.set_ylabel('Festival Hot Weather Load')
                ax3.set_title('Festival Load Pattern in Hot Weather')
                ax3.grid(True, alpha=0.3)
        
        # 4. Peak hour weather sensitivity
        ax4 = axes[3]
        if 'morning_peak_temp_sensitivity' in self.df.columns and 'evening_peak_temp_sensitivity' in self.df.columns:
            hourly_morning = self.df.groupby('hour')['morning_peak_temp_sensitivity'].mean()
            hourly_evening = self.df.groupby('hour')['evening_peak_temp_sensitivity'].mean()
            
            ax4.plot(hourly_morning.index, hourly_morning.values, 
                    label='Morning Peak', marker='o', linewidth=2, color='blue')
            ax4.plot(hourly_evening.index, hourly_evening.values, 
                    label='Evening Peak', marker='s', linewidth=2, color='red')
            ax4.set_xlabel('Hour of Day')
            ax4.set_ylabel('Temperature Sensitivity')
            ax4.set_title('Peak Hour Temperature Sensitivity')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Duck curve depth by temperature
        ax5 = axes[4]
        if 'duck_curve_depth' in self.df.columns:
            temp_bins = pd.cut(self.df['temperature_2m (°C)'], bins=8)
            duck_by_temp = self.df.groupby([temp_bins, 'hour'])['duck_curve_depth'].mean().unstack(level=0)
            
            if not duck_by_temp.empty:
                for i, col in enumerate(duck_by_temp.columns[:4]):  # Show first 4 temperature bins
                    ax5.plot(duck_by_temp.index, duck_by_temp[col], 
                            label=f'Temp: {col}', linewidth=2)
                
                ax5.set_xlabel('Hour of Day')
                ax5.set_ylabel('Duck Curve Depth')
                ax5.set_title('Duck Curve by Temperature Range')
                ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax5.grid(True, alpha=0.3)
        
        # 6. Heat index distribution
        ax6 = axes[5]
        if 'heat_index_celsius' in self.df.columns:
            ax6.hist(self.df['heat_index_celsius'].dropna(), bins=50, alpha=0.7, color='orange', edgecolor='red')
            ax6.set_xlabel('Heat Index (°C)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Heat Index Distribution')
            ax6.axvline(self.df['heat_index_celsius'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.df["heat_index_celsius"].mean():.1f}°C')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Weekend vs Weekday festival patterns
        ax7 = axes[6]
        if 'weekend_festival_hot_load' in self.df.columns and 'weekday_festival_evening_load' in self.df.columns:
            weekend_pattern = self.df[self.df['is_weekend'] == 1].groupby('hour')['weekend_festival_hot_load'].mean()
            weekday_pattern = self.df[self.df['is_weekend'] == 0].groupby('hour')['weekday_festival_evening_load'].mean()
            
            ax7.plot(weekend_pattern.index, weekend_pattern.values, 
                    label='Weekend Festival', marker='o', linewidth=2, color='purple')
            ax7.plot(weekday_pattern.index, weekday_pattern.values, 
                    label='Weekday Festival', marker='s', linewidth=2, color='green')
            ax7.set_xlabel('Hour of Day')
            ax7.set_ylabel('Festival Load')
            ax7.set_title('Weekend vs Weekday Festival Patterns')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # 8. Solar generation vs net cooling demand
        ax8 = axes[7]
        if 'estimated_solar_generation' in self.df.columns and 'net_cooling_demand' in self.df.columns:
            hourly_solar = self.df.groupby('hour')['estimated_solar_generation'].mean()
            hourly_cooling = self.df.groupby('hour')['net_cooling_demand'].mean()
            
            ax8_twin = ax8.twinx()
            line1 = ax8.plot(hourly_solar.index, hourly_solar.values, 
                           label='Solar Generation', marker='o', linewidth=2, color='yellow')
            line2 = ax8_twin.plot(hourly_cooling.index, hourly_cooling.values, 
                                label='Net Cooling Demand', marker='s', linewidth=2, color='blue')
            
            ax8.set_xlabel('Hour of Day')
            ax8.set_ylabel('Solar Generation', color='yellow')
            ax8_twin.set_ylabel('Net Cooling Demand', color='blue')
            ax8.set_title('Solar Generation vs Net Cooling Demand')
            
            # Combine legends
            lines1, labels1 = ax8.get_legend_handles_labels()
            lines2, labels2 = ax8_twin.get_legend_handles_labels()
            ax8.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax8.grid(True, alpha=0.3)
        
        # 9. Extreme weather interaction effects
        ax9 = axes[8]
        if 'extreme_heat_peak_stress' in self.df.columns and 'extreme_humidity_peak_stress' in self.df.columns:
            extreme_heat_data = self.df[self.df['extreme_heat_peak_stress'] > 0]
            extreme_humidity_data = self.df[self.df['extreme_humidity_peak_stress'] > 0]
            
            heat_pattern = None
            if len(extreme_heat_data) > 0:
                heat_pattern = extreme_heat_data.groupby('hour')['extreme_heat_peak_stress'].mean()
                ax9.bar(heat_pattern.index, heat_pattern.values, alpha=0.7, 
                       label='Extreme Heat Stress', color='red')
            
            if len(extreme_humidity_data) > 0:
                humidity_pattern = extreme_humidity_data.groupby('hour')['extreme_humidity_peak_stress'].mean()
                # Align indices for proper stacking
                if heat_pattern is not None:
                    # Reindex humidity pattern to match heat pattern
                    humidity_aligned = humidity_pattern.reindex(heat_pattern.index, fill_value=0)
                    heat_aligned = heat_pattern.reindex(humidity_pattern.index, fill_value=0)
                    ax9.bar(humidity_pattern.index, humidity_pattern.values, alpha=0.7,
                           label='Extreme Humidity Stress', color='blue', 
                           bottom=heat_aligned.values)
                else:
                    ax9.bar(humidity_pattern.index, humidity_pattern.values, alpha=0.7,
                           label='Extreme Humidity Stress', color='blue')
            
            ax9.set_xlabel('Hour of Day')
            ax9.set_ylabel('Extreme Weather Stress')
            ax9.set_title('Extreme Weather Peak Stress')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        # 10. Interaction features correlation matrix
        ax10 = axes[9]
        interaction_features = [col for col in self.df.columns if any(x in col.lower() 
                               for x in ['interaction', 'offset', 'sensitivity', 'duck', 'ramp'])]
        numeric_interaction_features = [col for col in interaction_features[:8]  # Limit to 8 for visibility
                                      if self.df[col].dtype in ['int64', 'float64']]
        
        if len(numeric_interaction_features) > 2:
            corr_data = self.df[numeric_interaction_features].corr()
            im = ax10.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
            ax10.set_xticks(range(len(numeric_interaction_features)))
            ax10.set_yticks(range(len(numeric_interaction_features)))
            ax10.set_xticklabels([f[:15] + '...' if len(f) > 15 else f 
                                for f in numeric_interaction_features], rotation=45, ha='right')
            ax10.set_yticklabels([f[:15] + '...' if len(f) > 15 else f 
                                for f in numeric_interaction_features])
            ax10.set_title('Interaction Features Correlation')
            plt.colorbar(im, ax=ax10, shrink=0.6)
        
        # 11. Temperature-driven ramp intensity
        ax11 = axes[10]
        if 'temperature_driven_ramp' in self.df.columns:
            ramp_data = self.df[self.df['temperature_driven_ramp'] > 0]
            if len(ramp_data) > 0:
                temp_bins = pd.cut(ramp_data['temperature_2m (°C)'], bins=6)
                ramp_by_temp = ramp_data.groupby([temp_bins, 'hour'])['temperature_driven_ramp'].mean().unstack(level=0)
                
                if not ramp_by_temp.empty:
                    for i, col in enumerate(ramp_by_temp.columns[:4]):
                        ax11.plot(ramp_by_temp.index, ramp_by_temp[col], 
                                label=f'Temp: {col}', linewidth=2, marker='o')
                    
                    ax11.set_xlabel('Hour of Day')
                    ax11.set_ylabel('Ramp Intensity')
                    ax11.set_title('Temperature-Driven Evening Ramp')
                    ax11.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
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
DELHI COMPLEX INTERACTION FEATURES SUMMARY

Total Features Created: {len(self.feature_log)}

By Category:
• Temp-Humidity-Hour: {feature_types.get('temp_humidity_hour', 0)}
• Solar-Temperature: {feature_types.get('solar_temperature', 0)}
• Festival-Weather-Day: {feature_types.get('festival_weather_day', 0)}
• Peak-Weather Sensitivity: {feature_types.get('peak_weather_sensitivity', 0)}
• Duck Curve-Temperature: {feature_types.get('duck_curve_temperature', 0)}

Advanced Interaction Capabilities:
✅ Multi-variable relationships
✅ Triple interaction modeling
✅ Solar-load offset calculations
✅ Festival-weather combinations
✅ Peak hour sensitivity analysis
✅ Duck curve depth variations

Ready For:
🔮 Complex Pattern Recognition
⚡ Peak Load Prediction
🦆 Solar Integration Modeling
🎉 Festival Load Forecasting
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, 'delhi_interaction_features_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Interaction features visualization saved: {output_path}")
    
    def generate_interaction_report(self):
        """Generate comprehensive interaction features report"""
        print(f"\n📋 DELHI COMPLEX INTERACTION FEATURES REPORT")
        print("=" * 70)
        
        print(f"Dataset: {len(self.df):,} records × {len(self.df.columns)} columns")
        print(f"Interaction analysis period: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        print(f"\nComplex Interaction Engineering Summary:")
        print(f"  Total new interaction features: {len(self.feature_log)}")
        
        # Group features by type
        feature_types = {}
        for feature, info in self.feature_log.items():
            ftype = info.get('type', 'other')
            if ftype not in feature_types:
                feature_types[ftype] = []
            feature_types[ftype].append(feature)
        
        for ftype, features in feature_types.items():
            print(f"\n  {ftype.upper().replace('_', ' ')} FEATURES ({len(features)}):")
            for feature in features[:3]:  # Show first 3
                if feature in self.feature_log:
                    info = self.feature_log[feature]
                    range_info = info.get('range', 'N/A')
                    print(f"    {feature}: {range_info}")
            if len(features) > 3:
                print(f"    ... and {len(features) - 3} more")
        
        # Interaction insights
        print(f"\n🔮 Complex Interaction Insights:")
        
        if 'ac_demand_probability' in self.df.columns:
            max_ac_prob = self.df['ac_demand_probability'].max()
            high_ac_hours = (self.df['ac_demand_probability'] > 0.7).sum()
            print(f"  Maximum AC demand probability: {max_ac_prob:.3f}")
            print(f"  High AC demand hours (>0.7): {high_ac_hours:,}")
        
        if 'duck_curve_depth' in self.df.columns:
            max_duck_depth = self.df['duck_curve_depth'].max()
            duck_hours = (self.df['duck_curve_depth'] > 0.1).sum()
            print(f"  Maximum duck curve depth: {max_duck_depth:.3f}")
            print(f"  Significant duck curve hours: {duck_hours:,}")
        
        if 'festival_hot_weather_load' in self.df.columns:
            festival_hot_hours = (self.df['festival_hot_weather_load'] > 0).sum()
            print(f"  Festival hot weather hours: {festival_hot_hours:,}")
        
        print(f"\nAdvanced Interaction Capabilities:")
        print(f"  ✅ Temperature × Humidity × Hour triple interactions")
        print(f"  ✅ Solar generation cooling offset modeling")
        print(f"  ✅ Festival × Weather × Day_type combinations")
        print(f"  ✅ Peak hour weather sensitivity analysis")
        print(f"  ✅ Duck curve depth temperature dependencies")
        print(f"  ✅ Multi-factor load prediction features")
        
        print(f"\nComplex Modeling Ready:")
        print(f"  🔮 Non-linear relationship capture")
        print(f"  ⚡ Peak load surge predictions")
        print(f"  🦆 Solar-load interaction modeling")
        print(f"  🎉 Festival weather impact analysis")
        print(f"  🌡️ Extreme weather stress calculations")
        
        print(f"\n💫 Interaction-enhanced dataset ready for advanced ML models!")
    
    def save_interaction_enhanced_dataset(self):
        """Save the dataset with complex interaction features"""
        output_path = os.path.join(self.output_dir, 'delhi_interaction_enhanced.csv')
        self.df.to_csv(output_path, index=False)
        print(f"\n✅ Interaction-enhanced dataset saved: {output_path}")
        return output_path
    
    def run_complete_interaction_analysis(self):
        """Execute the complete complex interaction feature engineering"""
        try:
            # Load temporal-enhanced dataset from Phase 2 Step 3
            self.load_temporal_enhanced_dataset()
            
            # Create Delhi-specific interaction features
            print(f"\n🚀 EXECUTING DELHI COMPLEX INTERACTION ANALYSIS")
            print("=" * 50)
            
            # Step 1: Temperature × Humidity × Hour Interactions
            self.create_temperature_humidity_hour_interactions()
            
            # Step 2: Solar × Temperature Interactions
            self.create_solar_temperature_interactions()
            
            # Step 3: Festival × Weather × Day Interactions
            self.create_festival_weather_day_interactions()
            
            # Step 4: Peak Hour × Weather Sensitivity
            self.create_peak_hour_weather_sensitivity()
            
            # Step 5: Duck Curve × Temperature Interactions
            self.create_duck_curve_temperature_interactions()
            
            # Create visualizations and reports
            self.create_interaction_visualization()
            self.generate_interaction_report()
            
            # Save interaction-enhanced dataset
            output_path = self.save_interaction_enhanced_dataset()
            
            print(f"\n🎉 DELHI COMPLEX INTERACTION ANALYSIS COMPLETED!")
            print(f"📁 Output file: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during interaction analysis: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Use the temporal-enhanced dataset from Phase 2 Step 3
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, 'delhi_temporal_pattern_enhanced.csv')
    
    # Initialize complex interaction analysis
    interaction_analysis = DelhiInteractionFeatures(input_path)
    
    # Run complete interaction analysis
    output_path = interaction_analysis.run_complete_interaction_analysis()
    
    return output_path


if __name__ == "__main__":
    output_file = main()
