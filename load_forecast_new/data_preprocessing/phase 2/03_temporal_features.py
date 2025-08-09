"""
Delhi Load Forecasting - Phase 2: Temporal Pattern Features
==========================================================
Advanced temporal feature engineering for Delhi's unique time patterns.
Focus on cyclical encoding, holidays, festivals, and cultural events.
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

class DelhiTemporalPatternFeatures:
    """
    Advanced temporal pattern feature engineering for Delhi
    Implements cyclical encoding, holiday effects, and cultural patterns
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.feature_log = {}
        self.output_dir = os.path.dirname(__file__)
        
        # Delhi-specific temporal constants
        self.DELHI_FESTIVALS = {
            # Major Hindu festivals (approximate dates, vary yearly)
            'diwali': [(10, 15, 11, 15)],  # October 15 - November 15
            'holi': [(2, 25, 3, 25)],      # February 25 - March 25
            'dussehra': [(9, 15, 10, 15)], # September 15 - October 15
            'karva_chauth': [(10, 1, 11, 1)], # October 1 - November 1
            'janmashtami': [(8, 1, 9, 1)], # August 1 - September 1
            'navratri': [(9, 1, 10, 10)],  # September 1 - October 10
        }
        
        self.DELHI_HOLIDAYS = {
            # Fixed holidays in Delhi
            'republic_day': (1, 26),
            'independence_day': (8, 15),
            'gandhi_jayanti': (10, 2),
            'christmas': (12, 25),
            'new_year': (1, 1),
        }
        
        # Wedding seasons in Delhi
        self.WEDDING_SEASONS = [
            (10, 1, 12, 31),  # October to December
            (1, 15, 3, 15),   # Mid January to Mid March
        ]
        
        # Academic calendar
        self.SCHOOL_SESSIONS = [
            (4, 1, 6, 30),    # Summer session
            (7, 1, 12, 20),   # Monsoon to winter session
            (1, 1, 3, 31),    # Winter to spring session
        ]
        
        # Delhi-specific commercial patterns
        self.COMMERCIAL_PEAK_HOURS = {
            'weekday_morning': (8, 12),
            'weekday_evening': (17, 21),
            'weekend_afternoon': (12, 18),
        }
    
    def load_thermal_enhanced_dataset(self):
        """Load the thermal-enhanced dataset from Phase 2 Step 2"""
        print("=" * 70)
        print("DELHI TEMPORAL PATTERN FEATURES - PHASE 2 STEP 3")
        print("=" * 70)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Thermal-enhanced dataset not found: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
        
        print(f"✅ Loading thermal-enhanced dataset: {os.path.basename(self.data_path)}")
        print(f"📊 Dataset: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        # Add basic time components if not present
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df['datetime'].dt.hour
        if 'day' not in self.df.columns:
            self.df['day'] = self.df['datetime'].dt.day
        if 'month' not in self.df.columns:
            self.df['month'] = self.df['datetime'].dt.month
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['datetime'].dt.year
        if 'day_of_week' not in self.df.columns:
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
        if 'day_of_year' not in self.df.columns:
            self.df['day_of_year'] = self.df['datetime'].dt.dayofyear
        if 'week_of_year' not in self.df.columns:
            self.df['week_of_year'] = self.df['datetime'].dt.isocalendar().week
        
        print(f"✅ Temporal components verified and added")
    
    def create_advanced_cyclical_encoding(self):
        """
        A) Advanced Cyclical Encoding for Time Components
        """
        print(f"\n🔄 CREATING ADVANCED CYCLICAL ENCODING")
        print("-" * 50)
        
        # A1) Hour encoding (24-hour cycle)
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        
        # A2) Day of week encoding (7-day cycle)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        
        # A3) Day of month encoding (30-day cycle)
        self.df['day_of_month_sin'] = np.sin(2 * np.pi * self.df['day'] / 30)
        self.df['day_of_month_cos'] = np.cos(2 * np.pi * self.df['day'] / 30)
        
        # A4) Month encoding (12-month cycle)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # A5) Day of year encoding (365-day cycle)
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        
        # A6) Week of year encoding (52-week cycle)
        self.df['week_of_year_sin'] = np.sin(2 * np.pi * self.df['week_of_year'] / 52)
        self.df['week_of_year_cos'] = np.cos(2 * np.pi * self.df['week_of_year'] / 52)
        
        # A7) Quarter encoding (4-quarter cycle)
        self.df['quarter'] = self.df['datetime'].dt.quarter
        self.df['quarter_sin'] = np.sin(2 * np.pi * self.df['quarter'] / 4)
        self.df['quarter_cos'] = np.cos(2 * np.pi * self.df['quarter'] / 4)
        
        # A8) Business hour encoding (for Delhi commercial patterns)
        business_hours = ((self.df['hour'] >= 9) & (self.df['hour'] <= 18) & 
                         (self.df['day_of_week'] < 6)).astype(int)
        self.df['is_business_hour'] = business_hours
        
        # A9) Delhi-specific time patterns
        # Morning peak hours (6-10 AM)
        self.df['is_morning_peak'] = ((self.df['hour'] >= 6) & (self.df['hour'] <= 10)).astype(int)
        
        # Evening peak hours (6-10 PM)
        self.df['is_evening_peak'] = ((self.df['hour'] >= 18) & (self.df['hour'] <= 22)).astype(int)
        
        # Late night hours (11 PM - 5 AM)
        self.df['is_late_night'] = ((self.df['hour'] >= 23) | (self.df['hour'] <= 5)).astype(int)
        
        # Weekend indicator
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        
        # Month-end effect (last 3 days of month)
        self.df['is_month_end'] = (self.df['day'] >= 28).astype(int)
        
        cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                           'day_of_month_sin', 'day_of_month_cos', 'month_sin', 'month_cos',
                           'day_of_year_sin', 'day_of_year_cos', 'week_of_year_sin', 'week_of_year_cos',
                           'quarter_sin', 'quarter_cos']
        
        pattern_features = ['is_business_hour', 'is_morning_peak', 'is_evening_peak', 
                          'is_late_night', 'is_weekend', 'is_month_end']
        
        print(f"✅ Created {len(cyclical_features)} cyclical encoding features")
        print(f"✅ Created {len(pattern_features)} time pattern features")
        print(f"   🔄 Hour, day, month, and yearly cycles encoded")
        print(f"   ⏰ Delhi-specific time patterns identified")
        
        # Log features
        all_features = cyclical_features + pattern_features
        for feature in all_features:
            if feature in self.df.columns:
                self.feature_log[feature] = {
                    'type': 'cyclical_encoding',
                    'count': len(self.df),
                    'unique_values': self.df[feature].nunique() if feature.startswith('is_') else 'continuous'
                }
    
    def create_delhi_holiday_features(self):
        """
        B) Delhi-Specific Holiday and Festival Features
        """
        print(f"\n🎉 CREATING DELHI HOLIDAY AND FESTIVAL FEATURES")
        print("-" * 50)
        
        # B1) Major Indian holidays
        self.df['is_republic_day'] = ((self.df['month'] == 1) & (self.df['day'] == 26)).astype(int)
        self.df['is_independence_day'] = ((self.df['month'] == 8) & (self.df['day'] == 15)).astype(int)
        self.df['is_gandhi_jayanti'] = ((self.df['month'] == 10) & (self.df['day'] == 2)).astype(int)
        self.df['is_christmas'] = ((self.df['month'] == 12) & (self.df['day'] == 25)).astype(int)
        self.df['is_new_year'] = ((self.df['month'] == 1) & (self.df['day'] == 1)).astype(int)
        
        # B2) Holiday proximity effects (±3 days)
        def create_proximity_feature(holiday_mask, proximity_days=3):
            proximity = np.zeros(len(self.df))
            holiday_indices = np.where(holiday_mask)[0]
            
            for idx in holiday_indices:
                start_idx = max(0, idx - proximity_days * 24)  # 3 days before (hourly data)
                end_idx = min(len(self.df), idx + proximity_days * 24 + 1)  # 3 days after
                
                for i in range(start_idx, end_idx):
                    distance = abs(i - idx) / 24  # Distance in days
                    if distance <= proximity_days:
                        proximity[i] = max(proximity[i], 1 - distance / proximity_days)
            
            return proximity
        
        # Apply proximity effects for major holidays
        self.df['republic_day_proximity'] = create_proximity_feature(self.df['is_republic_day'] == 1)
        self.df['independence_day_proximity'] = create_proximity_feature(self.df['is_independence_day'] == 1)
        self.df['christmas_proximity'] = create_proximity_feature(self.df['is_christmas'] == 1)
        
        # B3) Festival season indicators
        # Diwali season (October 15 - November 15)
        diwali_mask = ((self.df['month'] == 10) & (self.df['day'] >= 15)) | \
                      ((self.df['month'] == 11) & (self.df['day'] <= 15))
        self.df['is_diwali_season'] = diwali_mask.astype(int)
        
        # Holi season (February 25 - March 25)
        holi_mask = ((self.df['month'] == 2) & (self.df['day'] >= 25)) | \
                    ((self.df['month'] == 3) & (self.df['day'] <= 25))
        self.df['is_holi_season'] = holi_mask.astype(int)
        
        # Wedding season indicator
        wedding_season1 = ((self.df['month'] >= 10) & (self.df['month'] <= 12))  # Oct-Dec
        wedding_season2 = ((self.df['month'] >= 1) & (self.df['month'] <= 3) & 
                          ((self.df['month'] == 1) & (self.df['day'] >= 15) | 
                           (self.df['month'] == 2) | 
                           (self.df['month'] == 3) & (self.df['day'] <= 15)))  # Mid Jan - Mid Mar
        self.df['is_wedding_season'] = (wedding_season1 | wedding_season2).astype(int)
        
        # B4) Festival intensity scoring
        # Combine multiple festival indicators
        festival_score = (self.df['is_diwali_season'] * 3 +     # Highest intensity
                         self.df['is_holi_season'] * 2 +        # Medium intensity
                         self.df['is_wedding_season'] * 1.5 +   # Medium intensity
                         self.df['republic_day_proximity'] * 2 +
                         self.df['independence_day_proximity'] * 2 +
                         self.df['christmas_proximity'] * 2)
        
        self.df['festival_intensity'] = np.clip(festival_score, 0, 10)  # Scale 0-10
        
        # B5) Pre and post festival patterns
        # Pre-festival preparation (increased activity)
        self.df['pre_festival_prep'] = (
            ((self.df['is_diwali_season'] == 1) & (self.df['day'] <= 10)) |
            ((self.df['is_holi_season'] == 1) & (self.df['day'] <= 10)) |
            (self.df['republic_day_proximity'] > 0.5) |
            (self.df['independence_day_proximity'] > 0.5)
        ).astype(int)
        
        # Post-festival recovery (potentially decreased activity)
        self.df['post_festival_recovery'] = (
            ((self.df['is_diwali_season'] == 1) & (self.df['day'] >= 10)) |
            ((self.df['is_holi_season'] == 1) & (self.df['day'] >= 10))
        ).astype(int)
        
        holiday_features = ['is_republic_day', 'is_independence_day', 'is_gandhi_jayanti', 
                          'is_christmas', 'is_new_year', 'republic_day_proximity',
                          'independence_day_proximity', 'christmas_proximity',
                          'is_diwali_season', 'is_holi_season', 'is_wedding_season',
                          'festival_intensity', 'pre_festival_prep', 'post_festival_recovery']
        
        print(f"✅ Created {len(holiday_features)} holiday and festival features")
        print(f"   🎉 Major holidays with proximity effects")
        print(f"   🏮 Festival seasons (Diwali, Holi, Wedding)")
        print(f"   📊 Festival intensity scoring (0-10 scale)")
        
        # Log holiday features
        for feature in holiday_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'holiday_festival',
                        'count': len(valid_values),
                        'mean': valid_values.mean() if feature not in ['is_republic_day', 'is_independence_day', 'is_gandhi_jayanti', 'is_christmas', 'is_new_year'] else valid_values.sum(),
                        'max': valid_values.max()
                    }
    
    def create_commercial_activity_patterns(self):
        """
        C) Weekend and Commercial Activity Patterns
        """
        print(f"\n🏢 CREATING COMMERCIAL ACTIVITY PATTERNS")
        print("-" * 50)
        
        # C1) Enhanced weekend patterns
        self.df['is_friday'] = (self.df['day_of_week'] == 4).astype(int)  # Friday
        self.df['is_saturday'] = (self.df['day_of_week'] == 5).astype(int)  # Saturday
        self.df['is_sunday'] = (self.df['day_of_week'] == 6).astype(int)  # Sunday
        self.df['is_monday'] = (self.df['day_of_week'] == 0).astype(int)  # Monday
        
        # Weekend transition effects
        self.df['friday_evening'] = (self.df['is_friday'] & (self.df['hour'] >= 17)).astype(int)
        self.df['saturday_morning'] = (self.df['is_saturday'] & (self.df['hour'] <= 12)).astype(int)
        self.df['sunday_evening'] = (self.df['is_sunday'] & (self.df['hour'] >= 16)).astype(int)
        self.df['monday_morning'] = (self.df['is_monday'] & (self.df['hour'] <= 10)).astype(int)
        
        # C2) Commercial activity intensity
        # Delhi market timing patterns
        market_hours = ((self.df['hour'] >= 10) & (self.df['hour'] <= 20)).astype(float)
        weekday_factor = (1 - self.df['is_weekend']) * 1.0 + self.df['is_weekend'] * 0.7
        
        self.df['commercial_activity_index'] = market_hours * weekday_factor
        
        # Shopping patterns (higher on weekends and evenings)
        shopping_hours = ((self.df['hour'] >= 16) & (self.df['hour'] <= 22)).astype(float)
        weekend_shopping = self.df['is_weekend'] * 1.3 + (1 - self.df['is_weekend']) * 1.0
        
        self.df['shopping_activity_index'] = shopping_hours * weekend_shopping
        
        # C3) Delhi-specific commercial patterns
        # Connaught Place / Central Delhi activity (business hours)
        central_business_hours = ((self.df['hour'] >= 9) & (self.df['hour'] <= 19) & 
                                 (self.df['day_of_week'] < 6)).astype(int)
        self.df['central_business_activity'] = central_business_hours
        
        # Karol Bagh / Markets activity (extended hours, including weekends)
        market_activity = ((self.df['hour'] >= 10) & (self.df['hour'] <= 21)).astype(int)
        self.df['market_activity'] = market_activity
        
        # Restaurant/entertainment activity (evenings and weekends)
        restaurant_hours = ((self.df['hour'] >= 18) & (self.df['hour'] <= 23)).astype(float)
        restaurant_weekend_factor = self.df['is_weekend'] * 1.4 + (1 - self.df['is_weekend']) * 1.0
        self.df['restaurant_activity_index'] = restaurant_hours * restaurant_weekend_factor
        
        # C4) Transport and commute patterns
        # Morning commute (7-10 AM on weekdays)
        morning_commute = ((self.df['hour'] >= 7) & (self.df['hour'] <= 10) & 
                          (self.df['day_of_week'] < 6)).astype(int)
        self.df['morning_commute_activity'] = morning_commute
        
        # Evening commute (5-8 PM on weekdays)
        evening_commute = ((self.df['hour'] >= 17) & (self.df['hour'] <= 20) & 
                          (self.df['day_of_week'] < 6)).astype(int)
        self.df['evening_commute_activity'] = evening_commute
        
        # Metro operational hours impact
        metro_hours = ((self.df['hour'] >= 6) & (self.df['hour'] <= 23)).astype(int)
        self.df['metro_operational_hours'] = metro_hours
        
        commercial_features = ['is_friday', 'is_saturday', 'is_sunday', 'is_monday',
                             'friday_evening', 'saturday_morning', 'sunday_evening', 'monday_morning',
                             'commercial_activity_index', 'shopping_activity_index',
                             'central_business_activity', 'market_activity', 'restaurant_activity_index',
                             'morning_commute_activity', 'evening_commute_activity', 'metro_operational_hours']
        
        print(f"✅ Created {len(commercial_features)} commercial activity features")
        print(f"   🏢 Business district and market activity patterns")
        print(f"   🛍️ Shopping and restaurant activity indices")
        print(f"   🚇 Transport and commute pattern indicators")
        
        # Log commercial features
        for feature in commercial_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'commercial_activity',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std() if len(valid_values) > 1 else 0
                    }
    
    def create_academic_calendar_features(self):
        """
        D) Academic Calendar Impact Features
        """
        print(f"\n🎓 CREATING ACADEMIC CALENDAR FEATURES")
        print("-" * 50)
        
        # D1) School session periods
        # Summer session (April-June)
        summer_session = ((self.df['month'] >= 4) & (self.df['month'] <= 6)).astype(int)
        self.df['is_summer_session'] = summer_session
        
        # Monsoon session (July-September)
        monsoon_session = ((self.df['month'] >= 7) & (self.df['month'] <= 9)).astype(int)
        self.df['is_monsoon_session'] = monsoon_session
        
        # Winter session (October-March)
        winter_session = ((self.df['month'] >= 10) | (self.df['month'] <= 3)).astype(int)
        self.df['is_winter_session'] = winter_session
        
        # D2) School holidays and breaks
        # Summer vacation (May-June)
        summer_vacation = ((self.df['month'] == 5) | (self.df['month'] == 6)).astype(int)
        self.df['is_summer_vacation'] = summer_vacation
        
        # Winter vacation (December-January)
        winter_vacation = ((self.df['month'] == 12) | (self.df['month'] == 1)).astype(int)
        self.df['is_winter_vacation'] = winter_vacation
        
        # D3) Examination periods (typically March, May, October, December)
        exam_months = self.df['month'].isin([3, 5, 10, 12]).astype(int)
        self.df['is_exam_period'] = exam_months
        
        # D4) Academic activity intensity
        # Higher during active sessions, lower during vacations
        academic_intensity = (
            self.df['is_summer_session'] * 0.8 +
            self.df['is_monsoon_session'] * 1.0 +  # Peak session
            self.df['is_winter_session'] * 0.9
        )
        
        # Reduce intensity during vacations and exams
        vacation_factor = (1 - self.df['is_summer_vacation'] * 0.5 - 
                          self.df['is_winter_vacation'] * 0.3)
        exam_factor = (1 - self.df['is_exam_period'] * 0.2)  # Slight reduction during exams
        
        self.df['academic_activity_intensity'] = academic_intensity * vacation_factor * exam_factor
        
        # D5) School day patterns
        # School hours impact (8 AM - 2 PM on weekdays during sessions)
        school_hours = ((self.df['hour'] >= 8) & (self.df['hour'] <= 14) & 
                       (self.df['day_of_week'] < 6)).astype(float)
        
        session_factor = (self.df['is_summer_session'] + self.df['is_monsoon_session'] + 
                         self.df['is_winter_session']).clip(0, 1)
        
        self.df['school_hours_activity'] = school_hours * session_factor * (1 - self.df['is_summer_vacation'] * 0.8)
        
        # D6) College/University patterns (slightly different timing)
        # College hours (9 AM - 5 PM)
        college_hours = ((self.df['hour'] >= 9) & (self.df['hour'] <= 17) & 
                        (self.df['day_of_week'] < 6)).astype(float)
        
        self.df['college_hours_activity'] = college_hours * session_factor * (1 - self.df['is_summer_vacation'] * 0.6)
        
        # D7) Weekend academic activities (coaching classes, tuitions)
        weekend_academic = (self.df['is_weekend'] & 
                           ((self.df['hour'] >= 9) & (self.df['hour'] <= 18))).astype(float)
        
        self.df['weekend_academic_activity'] = weekend_academic * session_factor
        
        academic_features = ['is_summer_session', 'is_monsoon_session', 'is_winter_session',
                           'is_summer_vacation', 'is_winter_vacation', 'is_exam_period',
                           'academic_activity_intensity', 'school_hours_activity',
                           'college_hours_activity', 'weekend_academic_activity']
        
        print(f"✅ Created {len(academic_features)} academic calendar features")
        print(f"   🎓 School/college session and vacation periods")
        print(f"   📚 Academic activity intensity scoring")
        print(f"   ⏰ School and college hour activity patterns")
        
        # Log academic features
        for feature in academic_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'academic_calendar',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std() if len(valid_values) > 1 else 0
                    }
    
    def create_advanced_temporal_interactions(self):
        """
        E) Advanced Temporal Interaction Features
        """
        print(f"\n🔗 CREATING ADVANCED TEMPORAL INTERACTIONS")
        print("-" * 50)
        
        # E1) Hour-Weekend interactions
        self.df['weekend_morning'] = (self.df['is_weekend'] & (self.df['hour'] <= 10)).astype(int)
        self.df['weekend_afternoon'] = (self.df['is_weekend'] & 
                                       (self.df['hour'] >= 12) & (self.df['hour'] <= 16)).astype(int)
        self.df['weekend_evening'] = (self.df['is_weekend'] & (self.df['hour'] >= 18)).astype(int)
        
        self.df['weekday_morning_rush'] = ((1 - self.df['is_weekend']) & 
                                          (self.df['hour'] >= 7) & (self.df['hour'] <= 10)).astype(int)
        self.df['weekday_evening_rush'] = ((1 - self.df['is_weekend']) & 
                                          (self.df['hour'] >= 17) & (self.df['hour'] <= 20)).astype(int)
        
        # E2) Festival-Hour interactions
        self.df['festival_evening'] = (self.df['festival_intensity'] > 1) & (self.df['hour'] >= 18)
        self.df['festival_night'] = (self.df['festival_intensity'] > 1) & (self.df['hour'] >= 20)
        
        # E3) Season-Hour interactions
        # Summer evening patterns (AC usage peak)
        summer_mask = self.df['month'].isin([4, 5, 6])
        self.df['summer_evening_peak'] = (summer_mask & (self.df['hour'] >= 18) & (self.df['hour'] <= 22)).astype(int)
        
        # Winter morning patterns (heating load)
        winter_mask = self.df['month'].isin([12, 1, 2])
        self.df['winter_morning_peak'] = (winter_mask & (self.df['hour'] >= 6) & (self.df['hour'] <= 10)).astype(int)
        
        # Monsoon patterns (reduced activity)
        monsoon_mask = self.df['month'].isin([7, 8, 9])
        self.df['monsoon_reduced_activity'] = (monsoon_mask & 
                                              (self.df['hour'] >= 12) & (self.df['hour'] <= 18)).astype(int)
        
        # E4) Multi-factor interaction scores
        # Comprehensive activity score combining multiple factors
        base_activity = 0.5  # Base activity level
        
        # Hour effect
        hour_effect = np.where(
            (self.df['hour'] >= 7) & (self.df['hour'] <= 22), 1.0, 0.3
        )
        
        # Day effect
        day_effect = np.where(self.df['is_weekend'] == 1, 0.8, 1.0)
        
        # Festival effect
        festival_effect = 1.0 + self.df['festival_intensity'] * 0.1
        
        # Academic effect
        academic_effect = self.df['academic_activity_intensity']
        
        # Commercial effect
        commercial_effect = self.df['commercial_activity_index']
        
        # Combined activity score
        self.df['comprehensive_activity_score'] = (
            base_activity * hour_effect * day_effect * festival_effect * 
            (0.3 * academic_effect + 0.7 * commercial_effect)
        )
        
        # E5) Time-lag features for pattern recognition
        # Previous hour activity
        self.df = self.df.sort_values('datetime')
        self.df['prev_hour_is_peak'] = self.df['is_evening_peak'].shift(1).fillna(0)
        self.df['next_hour_is_peak'] = self.df['is_evening_peak'].shift(-1).fillna(0)
        
        # Previous day same hour pattern
        self.df['prev_day_same_hour_weekend'] = self.df['is_weekend'].shift(24).fillna(0)
        
        interaction_features = ['weekend_morning', 'weekend_afternoon', 'weekend_evening',
                              'weekday_morning_rush', 'weekday_evening_rush',
                              'festival_evening', 'festival_night',
                              'summer_evening_peak', 'winter_morning_peak', 'monsoon_reduced_activity',
                              'comprehensive_activity_score', 'prev_hour_is_peak', 'next_hour_is_peak',
                              'prev_day_same_hour_weekend']
        
        print(f"✅ Created {len(interaction_features)} temporal interaction features")
        print(f"   🔗 Hour-weekend and festival interactions")
        print(f"   📊 Comprehensive activity scoring")
        print(f"   ⏮️ Time-lag features for pattern recognition")
        
        # Log interaction features
        for feature in interaction_features:
            if feature in self.df.columns:
                valid_values = self.df[feature].dropna()
                if len(valid_values) > 0:
                    self.feature_log[feature] = {
                        'type': 'temporal_interactions',
                        'count': len(valid_values),
                        'mean': valid_values.mean(),
                        'std': valid_values.std() if len(valid_values) > 1 else 0
                    }
    
    def create_temporal_visualization(self):
        """Create comprehensive visualization of temporal features"""
        print(f"\n📊 CREATING TEMPORAL FEATURES VISUALIZATION")
        print("-" * 50)
        
        fig, axes = plt.subplots(4, 3, figsize=(24, 20))
        axes = axes.flatten()
        
        # 1. Cyclical encoding visualization
        ax1 = axes[0]
        hours = np.arange(24)
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)
        ax1.plot(hours, hour_sin, label='Hour Sin', marker='o', linewidth=2)
        ax1.plot(hours, hour_cos, label='Hour Cos', marker='s', linewidth=2)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Cyclical Value')
        ax1.set_title('Hour Cyclical Encoding')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Festival intensity by month
        ax2 = axes[1]
        if 'festival_intensity' in self.df.columns:
            monthly_festival = self.df.groupby('month')['festival_intensity'].mean()
            ax2.bar(monthly_festival.index, monthly_festival.values, 
                   color='gold', edgecolor='orange')
            ax2.set_xlabel('Month')
            ax2.set_ylabel('Average Festival Intensity')
            ax2.set_title('Festival Intensity by Month')
            ax2.set_xticks(range(1, 13))
            ax2.grid(True, alpha=0.3)
        
        # 3. Commercial activity patterns
        ax3 = axes[2]
        if 'commercial_activity_index' in self.df.columns:
            hourly_commercial = self.df.groupby('hour')['commercial_activity_index'].mean()
            ax3.plot(hourly_commercial.index, hourly_commercial.values, 
                    marker='o', linewidth=2, color='blue')
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Commercial Activity Index')
            ax3.set_title('Daily Commercial Activity Pattern')
            ax3.set_xticks(range(0, 24, 4))
            ax3.grid(True, alpha=0.3)
        
        # 4. Academic activity intensity
        ax4 = axes[3]
        if 'academic_activity_intensity' in self.df.columns:
            monthly_academic = self.df.groupby('month')['academic_activity_intensity'].mean()
            ax4.bar(monthly_academic.index, monthly_academic.values,
                   color='lightgreen', edgecolor='green')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Academic Activity Intensity')
            ax4.set_title('Academic Activity by Month')
            ax4.set_xticks(range(1, 13))
            ax4.grid(True, alpha=0.3)
        
        # 5. Weekend vs Weekday patterns
        ax5 = axes[4]
        if 'comprehensive_activity_score' in self.df.columns:
            weekend_pattern = self.df[self.df['is_weekend'] == 1].groupby('hour')['comprehensive_activity_score'].mean()
            weekday_pattern = self.df[self.df['is_weekend'] == 0].groupby('hour')['comprehensive_activity_score'].mean()
            
            ax5.plot(weekend_pattern.index, weekend_pattern.values, 
                    label='Weekend', marker='o', linewidth=2, color='red')
            ax5.plot(weekday_pattern.index, weekday_pattern.values, 
                    label='Weekday', marker='s', linewidth=2, color='blue')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('Activity Score')
            ax5.set_title('Weekend vs Weekday Activity Patterns')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Holiday proximity effects
        ax6 = axes[5]
        if 'republic_day_proximity' in self.df.columns:
            republic_day_data = self.df[self.df['republic_day_proximity'] > 0]
            if len(republic_day_data) > 0:
                proximity_pattern = republic_day_data.groupby('hour')['republic_day_proximity'].mean()
                ax6.plot(proximity_pattern.index, proximity_pattern.values,
                        marker='o', linewidth=2, color='orange')
                ax6.set_xlabel('Hour of Day')
                ax6.set_ylabel('Republic Day Proximity')
                ax6.set_title('Holiday Proximity Effect Pattern')
                ax6.grid(True, alpha=0.3)
        
        # 7. Seasonal temporal patterns heatmap
        ax7 = axes[6]
        if 'comprehensive_activity_score' in self.df.columns:
            seasonal_pattern = self.df.groupby(['month', 'hour'])['comprehensive_activity_score'].mean().unstack()
            im = ax7.imshow(seasonal_pattern.values, cmap='viridis', aspect='auto')
            ax7.set_xlabel('Hour of Day')
            ax7.set_ylabel('Month')
            ax7.set_title('Seasonal Activity Patterns')
            ax7.set_xticks(range(0, 24, 4))
            ax7.set_xticklabels(range(0, 24, 4))
            ax7.set_yticks(range(12))
            ax7.set_yticklabels(range(1, 13))
            plt.colorbar(im, ax=ax7, shrink=0.6)
        
        # 8. Festival vs Non-festival comparison
        ax8 = axes[7]
        if 'festival_intensity' in self.df.columns and 'comprehensive_activity_score' in self.df.columns:
            festival_days = self.df[self.df['festival_intensity'] > 1]
            normal_days = self.df[self.df['festival_intensity'] <= 1]
            
            if len(festival_days) > 0 and len(normal_days) > 0:
                festival_pattern = festival_days.groupby('hour')['comprehensive_activity_score'].mean()
                normal_pattern = normal_days.groupby('hour')['comprehensive_activity_score'].mean()
                
                ax8.plot(festival_pattern.index, festival_pattern.values,
                        label='Festival Days', marker='o', linewidth=2, color='red')
                ax8.plot(normal_pattern.index, normal_pattern.values,
                        label='Normal Days', marker='s', linewidth=2, color='blue')
                ax8.set_xlabel('Hour of Day')
                ax8.set_ylabel('Activity Score')
                ax8.set_title('Festival vs Normal Day Patterns')
                ax8.legend()
                ax8.grid(True, alpha=0.3)
        
        # 9. Day of week patterns
        ax9 = axes[8]
        if 'comprehensive_activity_score' in self.df.columns:
            daily_activity = self.df.groupby('day_of_week')['comprehensive_activity_score'].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            ax9.bar(daily_activity.index, daily_activity.values, 
                   color='lightblue', edgecolor='blue')
            ax9.set_xlabel('Day of Week')
            ax9.set_ylabel('Average Activity Score')
            ax9.set_title('Activity by Day of Week')
            ax9.set_xticks(range(7))
            ax9.set_xticklabels(day_names)
            ax9.grid(True, alpha=0.3)
        
        # 10. Temporal feature correlation
        ax10 = axes[9]
        temporal_features = [col for col in self.df.columns if any(x in col.lower() 
                           for x in ['hour', 'day', 'month', 'festival', 'commercial', 'academic'])]
        numeric_temporal_features = [col for col in temporal_features 
                                   if self.df[col].dtype in ['int64', 'float64']][:10]
        
        if len(numeric_temporal_features) > 2:
            corr_data = self.df[numeric_temporal_features].corr()
            im = ax10.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
            ax10.set_xticks(range(len(numeric_temporal_features)))
            ax10.set_yticks(range(len(numeric_temporal_features)))
            ax10.set_xticklabels([f[:10] + '...' if len(f) > 10 else f 
                                for f in numeric_temporal_features], rotation=45, ha='right')
            ax10.set_yticklabels([f[:10] + '...' if len(f) > 10 else f 
                                for f in numeric_temporal_features])
            ax10.set_title('Temporal Features Correlation')
            plt.colorbar(im, ax=ax10, shrink=0.6)
        
        # 11. Commute patterns
        ax11 = axes[10]
        if 'morning_commute_activity' in self.df.columns and 'evening_commute_activity' in self.df.columns:
            hourly_morning = self.df.groupby('hour')['morning_commute_activity'].mean()
            hourly_evening = self.df.groupby('hour')['evening_commute_activity'].mean()
            
            ax11.plot(hourly_morning.index, hourly_morning.values,
                     label='Morning Commute', marker='o', linewidth=2, color='green')
            ax11.plot(hourly_evening.index, hourly_evening.values,
                     label='Evening Commute', marker='s', linewidth=2, color='purple')
            ax11.set_xlabel('Hour of Day')
            ax11.set_ylabel('Commute Activity')
            ax11.set_title('Daily Commute Patterns')
            ax11.legend()
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
DELHI TEMPORAL PATTERN FEATURES SUMMARY

Total Features Created: {len(self.feature_log)}

By Category:
• Cyclical Encoding: {feature_types.get('cyclical_encoding', 0)}
• Holiday/Festival: {feature_types.get('holiday_festival', 0)}
• Commercial Activity: {feature_types.get('commercial_activity', 0)}
• Academic Calendar: {feature_types.get('academic_calendar', 0)}
• Temporal Interactions: {feature_types.get('temporal_interactions', 0)}

Key Achievements:
✅ Advanced cyclical time encoding
✅ Delhi festival and holiday effects
✅ Commercial activity patterns
✅ Academic calendar impacts
✅ Multi-factor temporal interactions

Temporal Features Ready For:
📊 Load Pattern Recognition
🎉 Festival Load Forecasting
🏢 Commercial Activity Modeling
🎓 Academic Session Impact Analysis
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, 'delhi_temporal_pattern_features_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Temporal features visualization saved: {output_path}")
    
    def generate_temporal_report(self):
        """Generate comprehensive temporal features report"""
        print(f"\n📋 DELHI TEMPORAL PATTERN FEATURES REPORT")
        print("=" * 70)
        
        print(f"Dataset: {len(self.df):,} records × {len(self.df.columns)} columns")
        print(f"Temporal analysis period: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        print(f"\nTemporal Feature Engineering Summary:")
        print(f"  Total new temporal features: {len(self.feature_log)}")
        
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
                        print(f"    {feature}: μ={info['mean']:.3f}")
                    else:
                        print(f"    {feature}: {info.get('count', 'N/A')} values")
            if len(features) > 5:
                print(f"    ... and {len(features) - 5} more")
        
        # Temporal insights
        if 'festival_intensity' in self.df.columns:
            max_festival = self.df['festival_intensity'].max()
            festival_hours = (self.df['festival_intensity'] > 1).sum()
            print(f"\n🎉 Temporal Insights:")
            print(f"  Maximum festival intensity: {max_festival:.1f}")
            print(f"  Festival/holiday hours: {festival_hours:,}")
        
        if 'academic_activity_intensity' in self.df.columns:
            avg_academic = self.df['academic_activity_intensity'].mean()
            print(f"  Average academic activity: {avg_academic:.3f}")
        
        if 'commercial_activity_index' in self.df.columns:
            avg_commercial = self.df['commercial_activity_index'].mean()
            print(f"  Average commercial activity: {avg_commercial:.3f}")
        
        print(f"\nDelhi-Specific Temporal Capabilities:")
        print(f"  ✅ Advanced cyclical encoding (hour, day, month, year)")
        print(f"  ✅ Festival season detection (Diwali, Holi, Wedding)")
        print(f"  ✅ Holiday proximity effects (±3 days)")
        print(f"  ✅ Commercial activity patterns (markets, business)")
        print(f"  ✅ Academic calendar impacts (sessions, vacations)")
        print(f"  ✅ Multi-factor temporal interactions")
        
        print(f"\nTemporal-Load Correlation Ready:")
        print(f"  🔄 Festival load surge predictions")
        print(f"  📊 Commercial peak demand modeling")
        print(f"  🎓 Academic session impact analysis")
        print(f"  🏢 Weekend vs weekday pattern recognition")
        print(f"  ⏰ Advanced time pattern forecasting")
        
        print(f"\n💾 Temporal-enhanced dataset ready for sophisticated modeling!")
    
    def save_temporal_enhanced_dataset(self):
        """Save the dataset with temporal pattern features"""
        output_path = os.path.join(self.output_dir, 'delhi_temporal_pattern_enhanced.csv')
        self.df.to_csv(output_path, index=False)
        print(f"\n✅ Temporal-enhanced dataset saved: {output_path}")
        return output_path
    
    def run_complete_temporal_analysis(self):
        """Execute the complete temporal pattern feature engineering"""
        try:
            # Load thermal-enhanced dataset from Phase 2 Step 2
            self.load_thermal_enhanced_dataset()
            
            # Create Delhi-specific temporal features
            print(f"\n🚀 EXECUTING DELHI TEMPORAL PATTERN ANALYSIS")
            print("=" * 50)
            
            # Step 1: Advanced Cyclical Encoding
            self.create_advanced_cyclical_encoding()
            
            # Step 2: Holiday and Festival Features
            self.create_delhi_holiday_features()
            
            # Step 3: Commercial Activity Patterns
            self.create_commercial_activity_patterns()
            
            # Step 4: Academic Calendar Features
            self.create_academic_calendar_features()
            
            # Step 5: Advanced Temporal Interactions
            self.create_advanced_temporal_interactions()
            
            # Create visualizations and reports
            self.create_temporal_visualization()
            self.generate_temporal_report()
            
            # Save temporal-enhanced dataset
            output_path = self.save_temporal_enhanced_dataset()
            
            print(f"\n🎉 DELHI TEMPORAL PATTERN ANALYSIS COMPLETED!")
            print(f"📁 Output file: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during temporal analysis: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Use the thermal-enhanced dataset from Phase 2 Step 2
    current_dir = os.path.dirname(__file__)
    input_path = os.path.join(current_dir, 'delhi_thermal_comfort_enhanced.csv')
    
    # Initialize temporal pattern analysis
    temporal_analysis = DelhiTemporalPatternFeatures(input_path)
    
    # Run complete temporal pattern analysis
    output_path = temporal_analysis.run_complete_temporal_analysis()
    
    return output_path


if __name__ == "__main__":
    output_file = main()
