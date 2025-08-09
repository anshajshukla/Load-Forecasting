"""
Delhi Load Forecasting - Comprehensive Missing Values Treatment
===============================================================
Phase 1 of data preprocessing pipeline.
Strategic missing value treatment based on feature importance and data patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for headless operation
plt.switch_backend('Agg')

class ComprehensiveMissingValueTreatment:
    """Comprehensive missing value analysis and treatment for Delhi Load Forecasting"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_treated = None
        self.treatment_log = {}
        self.output_dir = os.path.dirname(__file__)
        
    def load_data(self):
        """Load the dataset and perform initial analysis"""
        print("=" * 70)
        print("DELHI LOAD FORECASTING - MISSING VALUES TREATMENT")
        print("=" * 70)
        
        # Try different possible paths
        possible_paths = [
            self.data_path,
            os.path.join(os.path.dirname(__file__), '..', '..', 'Load-Forecasting', 'final_dataset_solar_treated.csv'),
            os.path.join(os.path.dirname(__file__), '..', 'final_dataset.csv'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'Load-Forecasting', 'final_dataset_with_authentic_load.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.data_path = path
                print(f"✅ Loading dataset: {path}")
                break
        else:
            raise FileNotFoundError("No valid dataset found in expected locations")
        
        self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
        self.df_treated = self.df.copy()
        
        print(f"📊 Dataset loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        
        # Add time features for imputation
        self.df_treated['hour'] = self.df_treated['datetime'].dt.hour
        self.df_treated['day_of_year'] = self.df_treated['datetime'].dt.dayofyear
        self.df_treated['month'] = self.df_treated['datetime'].dt.month
        self.df_treated['day_of_week'] = self.df_treated['datetime'].dt.dayofweek
        
    def analyze_missing_patterns(self):
        """Comprehensive missing value pattern analysis"""
        print(f"\n📈 MISSING VALUE PATTERN ANALYSIS")
        print("-" * 50)
        
        # Overall statistics
        total_missing = self.df.isnull().sum().sum()
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_percentage = (total_missing / total_cells) * 100
        
        print(f"Total missing values: {total_missing:,}")
        print(f"Overall missing percentage: {missing_percentage:.2f}%")
        
        # Missing by column
        missing_by_column = self.df.isnull().sum()
        missing_cols = missing_by_column[missing_by_column > 0].sort_values(ascending=False)
        
        print(f"\nColumns with missing values: {len(missing_cols)}")
        
        # Categorize features
        self.feature_categories = self.categorize_features(missing_cols)
        
        # Display missing by category
        for category, features in self.feature_categories.items():
            if features:
                print(f"\n{category.upper()}:")
                for feature in features[:5]:  # Show top 5
                    if feature in missing_cols.index:
                        count = missing_cols[feature]
                        pct = (count / len(self.df)) * 100
                        print(f"  {feature}: {count:,} ({pct:.2f}%)")
        
        return missing_cols
    
    def categorize_features(self, missing_cols):
        """Categorize features by type for targeted treatment"""
        categories = {
            'solar_features': [col for col in missing_cols.index if 'solar' in col.lower()],
            'weather_features': [col for col in missing_cols.index if any(x in col.lower() 
                               for x in ['temperature', 'humidity', 'wind', 'pressure', 'rain', 'precipitation', 'dew'])],
            'load_features': [col for col in missing_cols.index if 'load' in col.lower()],
            'calendar_features': [col for col in missing_cols.index if any(x in col.lower() 
                                for x in ['festival', 'holiday', 'event', 'diwali'])],
            'derived_features': [col for col in missing_cols.index if any(x in col.lower() 
                               for x in ['index', 'factor', 'penetration', 'ramp', 'stress'])],
            'other_features': [col for col in missing_cols.index if not any(
                category_check(col) for category_check in [
                    lambda x: 'solar' in x.lower(),
                    lambda x: any(w in x.lower() for w in ['temperature', 'humidity', 'wind', 'pressure', 'rain', 'precipitation', 'dew']),
                    lambda x: 'load' in x.lower(),
                    lambda x: any(w in x.lower() for w in ['festival', 'holiday', 'event', 'diwali']),
                    lambda x: any(w in x.lower() for w in ['index', 'factor', 'penetration', 'ramp', 'stress'])
                ]
            )]
        }
        return categories
    
    def calculate_solar_position_delhi(self, datetime_obj):
        """
        Calculate solar position for Delhi using proper solar physics
        Delhi coordinates: 28.6139°N, 77.2090°E
        Returns solar elevation angle and theoretical clear sky irradiance
        """
        # Delhi coordinates
        latitude = 28.6139  # degrees
        longitude = 77.2090  # degrees
        
        # Extract date components
        day_of_year = datetime_obj.timetuple().tm_yday
        hour = datetime_obj.hour + datetime_obj.minute/60.0
        
        # Solar declination angle (degrees)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle (degrees) - solar noon is when hour angle = 0
        # Convert local time to solar time (approximate for Delhi)
        solar_time = hour + (longitude - 82.5) / 15.0  # IST reference is 82.5°E
        hour_angle = 15 * (solar_time - 12)
        
        # Solar elevation angle (degrees above horizon)
        elevation_rad = np.arcsin(
            np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        elevation_deg = np.degrees(elevation_rad)
        
        # Solar azimuth angle (degrees from north)
        azimuth_rad = np.arctan2(
            np.sin(np.radians(hour_angle)),
            (np.cos(np.radians(hour_angle)) * np.sin(np.radians(latitude)) - 
             np.tan(np.radians(declination)) * np.cos(np.radians(latitude)))
        )
        azimuth_deg = np.degrees(azimuth_rad)
        
        # Extraterrestrial solar radiation (W/m²)
        solar_constant = 1367  # W/m²
        eccentricity_factor = 1 + 0.033 * np.cos(np.radians(360 * day_of_year / 365))
        extraterrestrial_radiation = solar_constant * eccentricity_factor
        
        # Clear sky global horizontal irradiance (simplified Ineichen-Perez model)
        if elevation_deg > 0:
            # Air mass calculation
            air_mass = 1 / (np.sin(elevation_rad) + 0.50572 * (elevation_deg + 6.07995)**(-1.6364))
            
            # Atmospheric transmission (simplified for Delhi conditions)
            # Delhi typical values: turbidity ≈ 3-5, altitude ≈ 216m
            turbidity = 4.0  # Typical for Delhi
            altitude = 216  # meters
            
            # Simplified clear sky model
            beam_transmission = np.exp(-0.09 * air_mass * (turbidity - 1))
            diffuse_transmission = 0.271 - 0.294 * beam_transmission
            
            # Global horizontal irradiance
            beam_horizontal = extraterrestrial_radiation * beam_transmission * np.sin(elevation_rad)
            diffuse_horizontal = extraterrestrial_radiation * diffuse_transmission
            clear_sky_ghi = beam_horizontal + diffuse_horizontal
            
            # Apply realistic limits for Delhi
            clear_sky_ghi = min(clear_sky_ghi, 1200)  # Maximum realistic GHI for Delhi
            clear_sky_ghi = max(clear_sky_ghi, 0)
        else:
            clear_sky_ghi = 0
        
        return {
            'elevation_angle': elevation_deg,
            'azimuth_angle': azimuth_deg,
            'clear_sky_ghi': clear_sky_ghi,
            'is_daytime': elevation_deg > 0
        }
    
    def treat_solar_features(self):
        """
        🌞 **PHYSICS-BASED SOLAR FEATURES TREATMENT**
        Advanced solar irradiance imputation using solar position calculations,
        clear sky models, and weather adjustments for Delhi location
        """
        print(f"\n🌞 PHYSICS-BASED SOLAR FEATURES TREATMENT")
        print("-" * 50)
        print("📍 Using Delhi coordinates: 28.6139°N, 77.2090°E")
        print("🔬 Applying solar physics and clear sky modeling")
        
        solar_features = self.feature_categories['solar_features']
        if not solar_features:
            print("No solar features found.")
            return
        
        self.treatment_log['solar'] = {}
        
        for feature in solar_features:
            if feature not in self.df_treated.columns:
                continue
                
            print(f"\n⚡ Treating {feature}:")
            initial_missing = self.df_treated[feature].isnull().sum()
            print(f"  Initial missing: {initial_missing:,}")
            
            # Counters for different treatment methods
            nighttime_filled = 0
            physics_based_filled = 0
            weather_adjusted_filled = 0
            similar_day_filled = 0
            
            # Process each missing value with physics-based approach
            missing_indices = self.df_treated[self.df_treated[feature].isnull()].index
            
            for idx in missing_indices:
                datetime_obj = self.df_treated.loc[idx, 'datetime']
                
                # Calculate solar position and clear sky irradiance
                solar_data = self.calculate_solar_position_delhi(datetime_obj)
                
                if not solar_data['is_daytime']:
                    # STEP 1: Nighttime - set to 0
                    self.df_treated.loc[idx, feature] = 0.0
                    nighttime_filled += 1
                    
                elif solar_data['elevation_angle'] < 10:
                    # STEP 2: Very low sun angle - minimal irradiance
                    self.df_treated.loc[idx, feature] = 0.0
                    nighttime_filled += 1
                    
                else:
                    # STEP 3: Daytime - apply physics-based imputation
                    clear_sky_ghi = solar_data['clear_sky_ghi']
                    
                    # Apply weather adjustments
                    weather_adjusted_ghi = self.apply_weather_adjustment(clear_sky_ghi, datetime_obj)
                    
                    # Scale to solar generation capacity (assuming this is solar generation data)
                    # Typical solar farm capacity factor in Delhi: 15-25%
                    if 'generation' in feature.lower() or 'mw' in feature.lower():
                        # Assume this is generation data - apply capacity factor
                        capacity_factor = 0.85  # Peak efficiency factor
                        solar_value = weather_adjusted_ghi * capacity_factor / 1000  # Convert to MW scale
                    else:
                        # Assume this is irradiance data
                        solar_value = weather_adjusted_ghi
                    
                    # Validate against existing data patterns
                    solar_value = self.validate_solar_value(feature, datetime_obj, solar_value)
                    
                    self.df_treated.loc[idx, feature] = max(0, solar_value)
                    physics_based_filled += 1
            
            # Apply final validation and smoothing
            self.apply_solar_smoothing(feature)
            
            final_missing = self.df_treated[feature].isnull().sum()
            
            print(f"  🌙 Nighttime filled (0): {nighttime_filled:,}")
            print(f"  ☀️ Physics-based filled: {physics_based_filled:,}")
            print(f"  🎯 Final missing: {final_missing:,}")
            
            # Calculate improvement
            improvement = ((initial_missing - final_missing) / initial_missing * 100) if initial_missing > 0 else 0
            print(f"  📈 Improvement: {improvement:.1f}%")
            
            self.treatment_log['solar'][feature] = {
                'initial': initial_missing,
                'nighttime_filled': nighttime_filled,
                'physics_based_filled': physics_based_filled,
                'weather_adjusted_filled': weather_adjusted_filled,
                'similar_day_filled': similar_day_filled,
                'final': final_missing,
                'improvement_percent': improvement
            }
        
        # Generate solar treatment summary
        total_solar_initial = sum([stats['initial'] for stats in self.treatment_log['solar'].values()])
        total_solar_final = sum([stats['final'] for stats in self.treatment_log['solar'].values()])
        overall_improvement = ((total_solar_initial - total_solar_final) / total_solar_initial * 100) if total_solar_initial > 0 else 0
        
        print(f"\n🎯 SOLAR TREATMENT SUMMARY:")
        print(f"  Total solar features: {len(solar_features)}")
        print(f"  Overall improvement: {overall_improvement:.1f}%")
        print(f"  Physics-based approach: ✅ Implemented")
        print(f"  Delhi-specific calculations: ✅ Applied")
        print(f"  Weather adjustments: ✅ Integrated")
        
        if total_solar_final == 0:
            print("  🏆 PERFECT: 0% missing values in solar data!")
        else:
            print(f"  ⚠️  {total_solar_final:,} values still missing - review needed")
    
    def validate_solar_value(self, feature, datetime_obj, proposed_value):
        """
        Validate proposed solar value against existing data patterns
        Ensures physical constraints and realistic ranges
        """
        # Get statistics from existing data for this hour and month
        hour = datetime_obj.hour
        month = datetime_obj.month
        
        # Find similar time conditions in existing data
        similar_mask = (
            (self.df_treated['hour'] == hour) &
            (self.df_treated['month'] == month) &
            (~self.df_treated[feature].isnull())
        )
        
        similar_values = self.df_treated[similar_mask][feature]
        
        if len(similar_values) > 0:
            # Calculate percentile bounds
            p25 = similar_values.quantile(0.25)
            p75 = similar_values.quantile(0.75)
            median = similar_values.median()
            iqr = p75 - p25
            
            # Define reasonable bounds (1.5 * IQR rule, but more conservative)
            lower_bound = max(0, p25 - 1.0 * iqr)
            upper_bound = p75 + 1.0 * iqr
            
            # Apply bounds
            if proposed_value < lower_bound:
                return max(lower_bound, median * 0.5)  # At least 50% of median
            elif proposed_value > upper_bound:
                return min(upper_bound, median * 1.5)  # At most 150% of median
            else:
                return proposed_value
        else:
            # No similar data available - apply general physical constraints
            # Maximum theoretical solar irradiance at surface level: ~1200 W/m²
            if 'generation' in feature.lower() or 'mw' in feature.lower():
                # For generation data, reasonable daily maximum varies by capacity
                return min(proposed_value, 1000)  # Arbitrary MW limit
            else:
                # For irradiance data
                return min(proposed_value, 1200)  # Physical maximum
    
    def apply_solar_smoothing(self, feature):
        """
        Apply temporal smoothing to solar data to remove unrealistic jumps
        Solar irradiance should change gradually except for weather events
        """
        # Sort by datetime for proper smoothing
        df_sorted = self.df_treated.sort_values('datetime').copy()
        
        # Calculate rate of change
        df_sorted['solar_diff'] = df_sorted[feature].diff()
        df_sorted['time_diff'] = df_sorted['datetime'].diff().dt.total_seconds() / 3600  # Convert to hours
        
        # Rate of change per hour
        df_sorted['rate_of_change'] = df_sorted['solar_diff'] / df_sorted['time_diff']
        
        # Identify unrealistic jumps (>500 W/m²/hour for irradiance, >100 MW/hour for generation)
        if 'generation' in feature.lower() or 'mw' in feature.lower():
            max_rate = 100  # MW/hour
        else:
            max_rate = 500  # W/m²/hour
        
        # Find and smooth excessive rate changes
        excessive_jumps = abs(df_sorted['rate_of_change']) > max_rate
        
        if excessive_jumps.sum() > 0:
            print(f"    Smoothing {excessive_jumps.sum()} excessive rate changes")
            
            for idx in df_sorted[excessive_jumps].index:
                if idx > 0 and idx < len(df_sorted) - 1:
                    # Linear interpolation between neighbors
                    prev_val = df_sorted.loc[idx-1, feature]
                    next_val = df_sorted.loc[idx+1, feature]
                    
                    if not pd.isnull(prev_val) and not pd.isnull(next_val):
                        smoothed_val = (prev_val + next_val) / 2
                        self.df_treated.loc[idx, feature] = smoothed_val
    
    def apply_weather_adjustment(self, clear_sky_ghi, datetime_obj):
        """
        Apply weather-based adjustments to clear sky irradiance
        Uses available weather data to estimate actual solar irradiance
        """
        # Get weather data for this timestamp
        weather_data = {}
        
        # Find weather columns
        temp_cols = [col for col in self.df_treated.columns if 'temperature' in col.lower()]
        humidity_cols = [col for col in self.df_treated.columns if 'humidity' in col.lower()]
        cloud_cols = [col for col in self.df_treated.columns if any(x in col.lower() for x in ['cloud', 'visibility', 'pressure'])]
        
        # Get current weather conditions
        current_idx = self.df_treated[self.df_treated['datetime'] == datetime_obj].index
        if len(current_idx) == 0:
            return clear_sky_ghi * 0.7  # Default reduction factor
        
        idx = current_idx[0]
        
        # Temperature factor (optimal around 25°C for solar panels)
        if temp_cols:
            temp = self.df_treated.loc[idx, temp_cols[0]]
            if not pd.isnull(temp):
                # Temperature coefficient for solar panels: -0.4%/°C above 25°C
                temp_factor = 1 - 0.004 * max(0, temp - 25)
                temp_factor = max(0.7, min(1.1, temp_factor))  # Limit range
            else:
                temp_factor = 0.85  # Default moderate factor
        else:
            temp_factor = 0.85
        
        # Humidity factor (high humidity often indicates clouds/haze)
        if humidity_cols:
            humidity = self.df_treated.loc[idx, humidity_cols[0]]
            if not pd.isnull(humidity):
                # High humidity reduces solar irradiance
                humidity_factor = 1 - 0.003 * max(0, humidity - 50)
                humidity_factor = max(0.6, min(1.0, humidity_factor))
            else:
                humidity_factor = 0.8
        else:
            humidity_factor = 0.8
        
        # Cloud/atmospheric factor (estimated from visibility/pressure if available)
        if cloud_cols:
            # Use first available cloud-related column
            cloud_data = self.df_treated.loc[idx, cloud_cols[0]]
            if not pd.isnull(cloud_data):
                # Normalize cloud factor (assuming 0-100 scale)
                if 'pressure' in cloud_cols[0].lower():
                    # Pressure: higher pressure often means clearer skies
                    normalized_pressure = (cloud_data - 950) / 50  # Typical range 950-1050 hPa
                    cloud_factor = 0.6 + 0.4 * max(0, min(1, normalized_pressure))
                else:
                    # Visibility or other cloud indicators
                    cloud_factor = 0.9  # Default good visibility
            else:
                cloud_factor = 0.75  # Default moderate conditions
        else:
            cloud_factor = 0.75
        
        # Seasonal atmospheric factor for Delhi
        month = datetime_obj.month
        if month in [12, 1, 2]:  # Winter - cleaner air but lower angle
            seasonal_factor = 0.85
        elif month in [3, 4, 5]:  # Pre-monsoon - dusty conditions
            seasonal_factor = 0.70
        elif month in [6, 7, 8, 9]:  # Monsoon - cloudy but clean air
            seasonal_factor = 0.65
        else:  # Post-monsoon - good conditions
            seasonal_factor = 0.90
        
        # Combine all factors
        total_factor = temp_factor * humidity_factor * cloud_factor * seasonal_factor
        total_factor = max(0.1, min(1.0, total_factor))  # Ensure reasonable range
        
        adjusted_ghi = clear_sky_ghi * total_factor
        
        return adjusted_ghi
    
    def treat_weather_features(self):
        """Treat weather-related missing values"""
        print(f"\n🌡️ WEATHER FEATURES TREATMENT")
        print("-" * 50)
        
        weather_features = self.feature_categories['weather_features']
        if not weather_features:
            print("No weather features found.")
            return
        
        self.treatment_log['weather'] = {}
        
        for feature in weather_features:
            if feature not in self.df_treated.columns:
                continue
                
            print(f"\nTreating {feature}:")
            initial_missing = self.df_treated[feature].isnull().sum()
            print(f"  Initial missing: {initial_missing:,}")
            
            if initial_missing == 0:
                continue
            
            # Method 1: Temporal interpolation for short gaps
            filled_interpolation = self.fill_temporal_interpolation(feature)
            
            # Method 2: Seasonal patterns for remaining gaps
            remaining_missing = self.df_treated[feature].isnull().sum()
            filled_seasonal = 0
            if remaining_missing > 0:
                filled_seasonal = self.fill_seasonal_patterns(feature)
            
            # Method 3: KNN imputation for any remaining
            final_remaining = self.df_treated[feature].isnull().sum()
            filled_knn = 0
            if final_remaining > 0:
                filled_knn = self.fill_knn_imputation(feature)
            
            final_missing = self.df_treated[feature].isnull().sum()
            print(f"  Temporal interpolation: {filled_interpolation:,}")
            print(f"  Seasonal patterns: {filled_seasonal:,}")
            print(f"  KNN imputation: {filled_knn:,}")
            print(f"  Final missing: {final_missing:,}")
            
            self.treatment_log['weather'][feature] = {
                'initial': initial_missing,
                'interpolation': filled_interpolation,
                'seasonal': filled_seasonal,
                'knn': filled_knn,
                'final': final_missing
            }
    
    def fill_temporal_interpolation(self, feature, max_gap_hours=3):
        """Fill short gaps using temporal interpolation"""
        df_sorted = self.df_treated.sort_values('datetime')
        filled_count = 0
        
        missing_mask = df_sorted[feature].isnull()
        
        # Find consecutive missing value groups
        missing_groups = []
        current_group = []
        
        for idx, is_missing in missing_mask.items():
            if is_missing:
                current_group.append(idx)
            else:
                if current_group:
                    missing_groups.append(current_group)
                    current_group = []
        
        if current_group:
            missing_groups.append(current_group)
        
        # Fill small gaps
        for group in missing_groups:
            if len(group) <= max_gap_hours:
                # Linear interpolation
                start_idx = group[0] - 1 if group[0] - 1 in df_sorted.index else None
                end_idx = group[-1] + 1 if group[-1] + 1 in df_sorted.index else None
                
                if start_idx is not None and end_idx is not None:
                    start_val = df_sorted.loc[start_idx, feature]
                    end_val = df_sorted.loc[end_idx, feature]
                    
                    if not pd.isnull(start_val) and not pd.isnull(end_val):
                        # Linear interpolation
                        for i, idx in enumerate(group):
                            ratio = (i + 1) / (len(group) + 1)
                            interpolated_val = start_val + ratio * (end_val - start_val)
                            self.df_treated.loc[idx, feature] = interpolated_val
                            filled_count += 1
        
        return filled_count
    
    def fill_seasonal_patterns(self, feature):
        """Fill using seasonal/hourly patterns"""
        missing_mask = self.df_treated[feature].isnull()
        filled_count = 0
        
        for idx in missing_mask[missing_mask].index:
            hour = self.df_treated.loc[idx, 'hour']
            month = self.df_treated.loc[idx, 'month']
            day_of_week = self.df_treated.loc[idx, 'day_of_week']
            
            # Find similar time conditions
            similar_mask = (
                (self.df_treated['hour'] == hour) &
                (self.df_treated['month'] == month) &
                (~self.df_treated[feature].isnull())
            )
            
            similar_values = self.df_treated[similar_mask][feature]
            
            if len(similar_values) > 0:
                # Use median of similar conditions
                imputed_value = similar_values.median()
                self.df_treated.loc[idx, feature] = imputed_value
                filled_count += 1
            else:
                # Broaden search to same hour across all months
                broader_mask = (
                    (self.df_treated['hour'] == hour) &
                    (~self.df_treated[feature].isnull())
                )
                broader_values = self.df_treated[broader_mask][feature]
                
                if len(broader_values) > 0:
                    imputed_value = broader_values.median()
                    self.df_treated.loc[idx, feature] = imputed_value
                    filled_count += 1
        
        return filled_count
    
    def fill_knn_imputation(self, feature):
        """Use KNN imputation for remaining missing values"""
        missing_count_before = self.df_treated[feature].isnull().sum()
        
        if missing_count_before == 0:
            return 0
        
        # Select related features for KNN
        numeric_features = self.df_treated.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target feature and datetime-related features that might cause data leakage
        exclude_features = [feature, 'datetime'] + [col for col in numeric_features if 'datetime' in col.lower()]
        knn_features = [col for col in numeric_features if col not in exclude_features][:20]  # Limit to 20 features
        
        if len(knn_features) < 3:
            # Fallback to forward/backward fill
            self.df_treated[feature] = self.df_treated[feature].fillna(method='ffill').fillna(method='bfill')
        else:
            # KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            features_for_imputation = [feature] + knn_features
            
            imputed_data = imputer.fit_transform(self.df_treated[features_for_imputation])
            self.df_treated[feature] = imputed_data[:, 0]
        
        missing_count_after = self.df_treated[feature].isnull().sum()
        return missing_count_before - missing_count_after
    
    def treat_calendar_features(self):
        """Treat calendar/festival related features"""
        print(f"\n📅 CALENDAR FEATURES TREATMENT")
        print("-" * 50)
        
        calendar_features = self.feature_categories['calendar_features']
        if not calendar_features:
            print("No calendar features found.")
            return
        
        self.treatment_log['calendar'] = {}
        
        for feature in calendar_features:
            if feature not in self.df_treated.columns:
                continue
                
            print(f"\nTreating {feature}:")
            initial_missing = self.df_treated[feature].isnull().sum()
            print(f"  Initial missing: {initial_missing:,}")
            
            if initial_missing == 0:
                continue
            
            # For festival/holiday features, fill with 0 (assume non-festival)
            # This is conservative but safer than guessing
            filled_count = initial_missing
            self.df_treated[feature] = self.df_treated[feature].fillna(0)
            
            final_missing = self.df_treated[feature].isnull().sum()
            print(f"  Filled with 0 (non-festival): {filled_count:,}")
            print(f"  Final missing: {final_missing:,}")
            
            self.treatment_log['calendar'][feature] = {
                'initial': initial_missing,
                'filled_with_zero': filled_count,
                'final': final_missing
            }
    
    def treat_derived_features(self):
        """Treat derived/calculated features"""
        print(f"\n🔢 DERIVED FEATURES TREATMENT")
        print("-" * 50)
        
        derived_features = self.feature_categories['derived_features']
        if not derived_features:
            print("No derived features found.")
            return
        
        self.treatment_log['derived'] = {}
        
        for feature in derived_features:
            if feature not in self.df_treated.columns:
                continue
                
            print(f"\nTreating {feature}:")
            initial_missing = self.df_treated[feature].isnull().sum()
            print(f"  Initial missing: {initial_missing:,}")
            
            if initial_missing == 0:
                continue
            
            # Use median imputation for derived features
            median_value = self.df_treated[feature].median()
            filled_count = initial_missing
            self.df_treated[feature] = self.df_treated[feature].fillna(median_value)
            
            final_missing = self.df_treated[feature].isnull().sum()
            print(f"  Filled with median ({median_value:.2f}): {filled_count:,}")
            print(f"  Final missing: {final_missing:,}")
            
            self.treatment_log['derived'][feature] = {
                'initial': initial_missing,
                'filled_with_median': filled_count,
                'median_value': median_value,
                'final': final_missing
            }
    
    def treat_load_features(self):
        """Treat load-related features with high priority"""
        print(f"\n⚡ LOAD FEATURES TREATMENT")
        print("-" * 50)
        
        load_features = self.feature_categories['load_features']
        if not load_features:
            print("No load features found.")
            return
        
        self.treatment_log['load'] = {}
        
        for feature in load_features:
            if feature not in self.df_treated.columns:
                continue
                
            print(f"\nTreating {feature}:")
            initial_missing = self.df_treated[feature].isnull().sum()
            print(f"  Initial missing: {initial_missing:,}")
            
            if initial_missing == 0:
                continue
            
            # Use sophisticated temporal patterns for load data
            filled_temporal = self.fill_load_temporal_patterns(feature)
            
            remaining_missing = self.df_treated[feature].isnull().sum()
            filled_knn = 0
            if remaining_missing > 0:
                filled_knn = self.fill_knn_imputation(feature)
            
            final_missing = self.df_treated[feature].isnull().sum()
            print(f"  Temporal patterns: {filled_temporal:,}")
            print(f"  KNN imputation: {filled_knn:,}")
            print(f"  Final missing: {final_missing:,}")
            
            self.treatment_log['load'][feature] = {
                'initial': initial_missing,
                'temporal': filled_temporal,
                'knn': filled_knn,
                'final': final_missing
            }
    
    def fill_load_temporal_patterns(self, feature):
        """Fill load features using sophisticated temporal patterns"""
        missing_mask = self.df_treated[feature].isnull()
        filled_count = 0
        
        for idx in missing_mask[missing_mask].index:
            hour = self.df_treated.loc[idx, 'hour']
            day_of_week = self.df_treated.loc[idx, 'day_of_week']
            month = self.df_treated.loc[idx, 'month']
            
            # Find same day type, hour, and month
            similar_mask = (
                (self.df_treated['hour'] == hour) &
                (self.df_treated['day_of_week'] == day_of_week) &
                (self.df_treated['month'] == month) &
                (~self.df_treated[feature].isnull())
            )
            
            similar_values = self.df_treated[similar_mask][feature]
            
            if len(similar_values) >= 3:  # Need at least 3 similar points
                # Use median with some randomness to avoid artificial patterns
                base_value = similar_values.median()
                std_value = similar_values.std()
                
                # Add small random variation (±5% of std)
                variation = np.random.normal(0, 0.05 * std_value)
                imputed_value = max(0, base_value + variation)
                
                self.df_treated.loc[idx, feature] = imputed_value
                filled_count += 1
            else:
                # Broaden search - same hour and day type across all months
                broader_mask = (
                    (self.df_treated['hour'] == hour) &
                    (self.df_treated['day_of_week'] == day_of_week) &
                    (~self.df_treated[feature].isnull())
                )
                broader_values = self.df_treated[broader_mask][feature]
                
                if len(broader_values) > 0:
                    imputed_value = broader_values.median()
                    self.df_treated.loc[idx, feature] = imputed_value
                    filled_count += 1
        
        return filled_count
    
    def create_treatment_visualization(self):
        """Create comprehensive treatment results visualization"""
        print(f"\n📊 CREATING TREATMENT VISUALIZATION")
        print("-" * 50)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Overall missing values before/after
        ax1 = axes[0]
        categories = list(self.treatment_log.keys())
        before_counts = []
        after_counts = []
        
        for category in categories:
            before = sum([stats['initial'] for stats in self.treatment_log[category].values()])
            after = sum([stats['final'] for stats in self.treatment_log[category].values()])
            before_counts.append(before)
            after_counts.append(after)
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax1.bar(x - width/2, before_counts, width, label='Before', color='red', alpha=0.7)
        ax1.bar(x + width/2, after_counts, width, label='After', color='green', alpha=0.7)
        ax1.set_xlabel('Feature Categories')
        ax1.set_ylabel('Missing Values Count')
        ax1.set_title('Missing Values: Before vs After Treatment')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reduction percentage by category
        ax2 = axes[1]
        reductions = []
        for i, category in enumerate(categories):
            if before_counts[i] > 0:
                reduction = ((before_counts[i] - after_counts[i]) / before_counts[i]) * 100
            else:
                reduction = 0
            reductions.append(reduction)
        
        bars = ax2.bar(categories, reductions, color='blue', alpha=0.7)
        ax2.set_xlabel('Feature Categories')
        ax2.set_ylabel('Reduction Percentage (%)')
        ax2.set_title('Missing Values Reduction by Category')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, reduction in zip(bars, reductions):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{reduction:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Treatment methods effectiveness (solar features)
        ax3 = axes[2]
        if 'solar' in self.treatment_log:
            methods = ['nighttime_filled', 'daytime_filled']
            method_counts = []
            for method in methods:
                count = sum([stats.get(method, 0) for stats in self.treatment_log['solar'].values()])
                method_counts.append(count)
            
            ax3.pie(method_counts, labels=['Nighttime Fill', 'Daytime Imputation'], 
                   autopct='%1.1f%%', startangle=90)
            ax3.set_title('Solar Treatment Methods')
        else:
            ax3.text(0.5, 0.5, 'No Solar Features', transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=14)
        
        # 4. Missing values distribution (remaining)
        ax4 = axes[3]
        final_missing = self.df_treated.isnull().sum()
        remaining_missing = final_missing[final_missing > 0].sort_values(ascending=False)[:10]
        
        if len(remaining_missing) > 0:
            ax4.barh(range(len(remaining_missing)), remaining_missing.values, color='orange', alpha=0.7)
            ax4.set_yticks(range(len(remaining_missing)))
            ax4.set_yticklabels([col[:20] + '...' if len(col) > 20 else col for col in remaining_missing.index])
            ax4.set_xlabel('Remaining Missing Values')
            ax4.set_title('Top 10 Features with Remaining Missing Values')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Remaining Missing Values!', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=14, color='green', fontweight='bold')
        
        # 5. Data completeness improvement
        ax5 = axes[4]
        original_completeness = ((len(self.df) * len(self.df.columns) - self.df.isnull().sum().sum()) / 
                               (len(self.df) * len(self.df.columns))) * 100
        final_completeness = ((len(self.df_treated) * len(self.df_treated.columns) - self.df_treated.isnull().sum().sum()) / 
                            (len(self.df_treated) * len(self.df_treated.columns))) * 100
        
        completeness_data = [original_completeness, final_completeness]
        colors = ['red', 'green']
        labels = ['Before Treatment', 'After Treatment']
        
        bars = ax5.bar(labels, completeness_data, color=colors, alpha=0.7)
        ax5.set_ylabel('Data Completeness (%)')
        ax5.set_title('Overall Data Completeness Improvement')
        ax5.set_ylim(90, 100)
        ax5.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, completeness in zip(bars, completeness_data):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{completeness:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Treatment summary text
        ax6 = axes[5]
        ax6.axis('off')
        
        summary_text = f"""
TREATMENT SUMMARY

Dataset: {len(self.df_treated):,} records
Total Features: {len(self.df_treated.columns)}

Completeness Improvement:
{original_completeness:.2f}% → {final_completeness:.2f}%

Features Treated:
• Solar: {len(self.feature_categories['solar_features'])} features
• Weather: {len(self.feature_categories['weather_features'])} features  
• Load: {len(self.feature_categories['load_features'])} features
• Calendar: {len(self.feature_categories['calendar_features'])} features

Methods Used:
• Nighttime solar filling
• Temporal interpolation
• Seasonal patterns
• KNN imputation
• Domain-specific rules
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'missing_values_treatment_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved: {output_path}")
        
    def generate_final_report(self):
        """Generate comprehensive treatment report"""
        print(f"\n📋 COMPREHENSIVE TREATMENT REPORT")
        print("=" * 70)
        
        # Overall statistics
        original_missing = self.df.isnull().sum().sum()
        final_missing = self.df_treated.isnull().sum().sum()
        improvement = ((original_missing - final_missing) / original_missing * 100) if original_missing > 0 else 0
        
        print(f"Dataset: {len(self.df_treated):,} records × {len(self.df_treated.columns)} features")
        print(f"Treatment Period: {self.df_treated['datetime'].min()} to {self.df_treated['datetime'].max()}")
        print(f"\nMissing Values Summary:")
        print(f"  Original missing: {original_missing:,}")
        print(f"  Final missing: {final_missing:,}")
        print(f"  Values imputed: {original_missing - final_missing:,}")
        print(f"  Improvement: {improvement:.1f}%")
        
        # Category-wise report
        print(f"\nTreatment by Category:")
        for category, category_log in self.treatment_log.items():
            category_original = sum([stats['initial'] for stats in category_log.values()])
            category_final = sum([stats['final'] for stats in category_log.values()])
            category_improvement = ((category_original - category_final) / category_original * 100) if category_original > 0 else 0
            
            print(f"  {category.upper()}:")
            print(f"    Features treated: {len(category_log)}")
            print(f"    Missing reduced: {category_original:,} → {category_final:,} ({category_improvement:.1f}%)")
        
        # Data quality assessment
        print(f"\nData Quality Assessment:")
        completeness = ((len(self.df_treated) * len(self.df_treated.columns) - final_missing) / 
                       (len(self.df_treated) * len(self.df_treated.columns))) * 100
        print(f"  Data completeness: {completeness:.2f}%")
        
        if final_missing == 0:
            print("  ✅ No remaining missing values - Dataset ready for modeling!")
        else:
            print(f"  ⚠️  {final_missing:,} values still missing - Review required")
        
        # Recommendations
        print(f"\nRecommendations:")
        if completeness >= 99.5:
            print("  ✅ Excellent data quality - Proceed to feature engineering")
        elif completeness >= 98.0:
            print("  ✅ Good data quality - Minor cleanup may be beneficial")
        else:
            print("  ⚠️  Consider additional imputation methods for remaining gaps")
        
        print(f"  📊 Review visualization: missing_values_treatment_results.png")
        print(f"  💾 Treated dataset ready for next pipeline phase")
    
    def save_treated_dataset(self):
        """Save the treated dataset"""
        # Remove temporary columns used for imputation
        temp_columns = ['hour', 'day_of_year', 'month', 'day_of_week']
        columns_to_drop = [col for col in temp_columns if col in self.df_treated.columns and col not in self.df.columns]
        
        if columns_to_drop:
            self.df_treated = self.df_treated.drop(columns=columns_to_drop)
        
        # Save the treated dataset
        output_path = os.path.join(self.output_dir, 'final_dataset_missing_values_treated.csv')
        self.df_treated.to_csv(output_path, index=False)
        print(f"\n✅ Treated dataset saved: {output_path}")
        
        return output_path
    
    def run_complete_treatment(self):
        """Execute the complete missing value treatment pipeline"""
        try:
            # Load data
            self.load_data()
            
            # Analyze missing patterns
            missing_cols = self.analyze_missing_patterns()
            
            if len(missing_cols) == 0:
                print("\n✅ No missing values found! Dataset is already complete.")
                return self.data_path
            
            # Execute treatment by category
            self.treat_solar_features()
            self.treat_weather_features()
            self.treat_load_features()
            self.treat_calendar_features()
            self.treat_derived_features()
            
            # Create visualization
            self.create_treatment_visualization()
            
            # Generate report
            self.generate_final_report()
            
            # Save treated dataset
            output_path = self.save_treated_dataset()
            
            print(f"\n🎉 MISSING VALUE TREATMENT COMPLETED SUCCESSFULLY!")
            print(f"📁 Output file: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error during treatment: {str(e)}")
            raise


def main():
    """Main execution function"""
    # Determine data path
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'final_dataset.csv')
    
    # Initialize treatment
    treatment = ComprehensiveMissingValueTreatment(data_path)
    
    # Run complete treatment
    output_path = treatment.run_complete_treatment()
    
    return output_path


if __name__ == "__main__":
    output_file = main()
