"""
Temporal Feature Engineering for Delhi Load Forecasting - SIH 2024
Generates time-based features for accurate load prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from typing import Dict, List, Tuple
import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TemporalFeatureEngine:
    """Temporal feature engineering for Delhi electrical load forecasting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Database connection
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # India holidays (Delhi specific) - Use existing system
        # Note: Database already has comprehensive holiday data populated
        self.use_existing_holidays = True
        
        # Delhi-specific festivals and important days (for additional analysis)
        self.delhi_festivals = {
            'Diwali': ['2022-10-24', '2023-11-12', '2024-11-01', '2025-10-20'],
            'Holi': ['2022-03-18', '2023-03-08', '2024-03-25', '2025-03-14'],
            'Dussehra': ['2022-10-05', '2023-10-24', '2024-10-12', '2025-10-02'],
            'Karva_Chauth': ['2022-10-13', '2023-11-01', '2024-10-20', '2025-10-09'],
            'Raksha_Bandhan': ['2022-08-11', '2023-08-30', '2024-08-19', '2025-08-09'],
            'Janmashtami': ['2022-08-18', '2023-09-07', '2024-08-26', '2025-08-16']
        }
        
        # Delhi seasonal patterns
        self.delhi_seasons = {
            'Winter': [12, 1, 2],      # High heating load
            'Spring': [3, 4],          # Moderate load
            'Summer': [5, 6],          # Very high cooling load
            'Monsoon': [7, 8, 9],      # High humidity, moderate cooling
            'Post_Monsoon': [10, 11]   # Pleasant weather, low load
        }
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def get_connection(self):
        """Get database connection."""
        try:
            conn = psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def add_temporal_columns_to_schema(self):
        """Add temporal feature columns to the load_data table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            print("🕐 Adding temporal feature columns to schema...")
            
            # Define temporal feature columns
            temporal_columns = [
                # Basic time features
                "hour INTEGER",
                "day_of_week INTEGER",
                "month INTEGER", 
                "day_of_year INTEGER",
                "week_of_year INTEGER",
                
                # Cyclical encodings
                "hour_sin NUMERIC(10,6)",
                "hour_cos NUMERIC(10,6)",
                "day_sin NUMERIC(10,6)",
                "day_cos NUMERIC(10,6)",
                "month_sin NUMERIC(10,6)",
                "month_cos NUMERIC(10,6)",
                
                # Working patterns
                "is_weekend BOOLEAN",
                "is_weekday BOOLEAN",
                "is_working_day BOOLEAN",
                
                # Delhi-specific peak hours
                "is_morning_peak BOOLEAN",
                "is_evening_peak BOOLEAN", 
                "is_office_hours BOOLEAN",
                "is_late_night BOOLEAN",
                
                # Holiday features (use existing data from database)
                # Note: Holiday columns already exist, so we skip them
                
                # Seasonal features
                "season VARCHAR(20)",
                "delhi_season VARCHAR(20)",
                "season_intensity NUMERIC(3,2)",
                
                # Load pattern features
                "load_pattern VARCHAR(20)",
                "peak_probability NUMERIC(3,2)",
                "demand_category VARCHAR(20)"
            ]
            
            # Add columns if they don't exist (skip holiday-related comments)
            added_columns = 0
            for column_def in temporal_columns:
                if column_def.startswith("--") or column_def.startswith("#"):
                    continue  # Skip comments
                    
                column_name = column_def.split()[0]
                try:
                    cursor.execute(f"ALTER TABLE load_data ADD COLUMN IF NOT EXISTS {column_def};")
                    added_columns += 1
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.warning(f"Could not add column {column_name}: {str(e)}")
            
            conn.commit()
            print(f"✅ Added {added_columns} temporal feature columns to schema")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to add temporal columns: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_existing_holiday_data(self, conn, timestamp: datetime) -> Dict:
        """Get existing holiday data from database for a timestamp using provided connection."""
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT is_holiday, holiday_name, holiday_type, 
                       is_festival_season, day_before_holiday, day_after_holiday
                FROM load_data 
                WHERE timestamp = %s 
                LIMIT 1;
            """, (timestamp,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'is_holiday': result['is_holiday'] or False,
                    'holiday_name': result['holiday_name'],
                    'holiday_type': result['holiday_type'],
                    'is_festival_season': result['is_festival_season'] or False,
                    'day_before_holiday': result['day_before_holiday'] or False,
                    'day_after_holiday': result['day_after_holiday'] or False
                }
            else:
                return {
                    'is_holiday': False,
                    'holiday_name': None,
                    'holiday_type': None,
                    'is_festival_season': False,
                    'day_before_holiday': False,
                    'day_after_holiday': False
                }
        except Exception as e:
            self.logger.warning(f"Could not get holiday data for {timestamp}: {e}")
            return {
                'is_holiday': False,
                'holiday_name': None,
                'holiday_type': None,
                'is_festival_season': False,
                'day_before_holiday': False,
                'day_after_holiday': False
            }
        finally:
            cursor.close()
    
    def create_temporal_features(self, timestamp: datetime, holiday_data: Dict = None) -> Dict:
        """Create comprehensive temporal features for a given timestamp."""
        
        # Basic time features
        hour = timestamp.hour
        day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
        month = timestamp.month
        day_of_year = timestamp.timetuple().tm_yday
        week_of_year = timestamp.isocalendar()[1]
        
        # Cyclical encodings (to capture circular nature of time)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Working patterns
        is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6
        is_weekday = day_of_week < 5
        
        # Get existing holiday data from database (already populated)
        if self.use_existing_holidays and holiday_data:
            is_holiday = holiday_data['is_holiday']
            holiday_name = holiday_data['holiday_name']
            holiday_type = holiday_data['holiday_type']
            is_festival_season = holiday_data['is_festival_season']
            day_before_holiday = holiday_data['day_before_holiday']
            day_after_holiday = holiday_data['day_after_holiday']
        else:
            # Fallback to calculating if needed
            is_holiday = False
            holiday_name = None
            holiday_type = None
            is_festival_season = False
            day_before_holiday = False
            day_after_holiday = False
        
        # Working day (not weekend and not holiday)
        is_working_day = is_weekday and not is_holiday
        
        # Delhi-specific peak hours
        is_morning_peak = 7 <= hour <= 10  # Morning rush + office start
        is_evening_peak = 18 <= hour <= 22  # Evening rush + dinner time
        is_office_hours = 9 <= hour <= 18 and is_working_day
        is_late_night = hour >= 23 or hour <= 5
        
        # Festival season detection (additional to existing data)
        # This adds extra festival analysis on top of existing holiday data
        
        # Seasonal classification
        season = self.get_delhi_season(month)
        delhi_season = self.get_detailed_delhi_season(month, timestamp)
        season_intensity = self.get_season_intensity(month, hour)
        
        # Load pattern prediction
        load_pattern = self.predict_load_pattern(hour, day_of_week, is_holiday, season)
        peak_probability = self.calculate_peak_probability(hour, day_of_week, month, is_holiday)
        demand_category = self.get_demand_category(hour, day_of_week, season, is_holiday)
        
        return {
            # Basic time features
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'day_of_year': day_of_year,
            'week_of_year': week_of_year,
            
            # Cyclical encodings
            'hour_sin': round(hour_sin, 6),
            'hour_cos': round(hour_cos, 6),
            'day_sin': round(day_sin, 6),
            'day_cos': round(day_cos, 6),
            'month_sin': round(month_sin, 6),
            'month_cos': round(month_cos, 6),
            
            # Working patterns
            'is_weekend': is_weekend,
            'is_weekday': is_weekday,
            'is_working_day': is_working_day,
            
            # Peak hours
            'is_morning_peak': is_morning_peak,
            'is_evening_peak': is_evening_peak,
            'is_office_hours': is_office_hours,
            'is_late_night': is_late_night,
            
            # Holiday features
            'is_holiday': is_holiday,
            'holiday_name': holiday_name,
            'holiday_type': holiday_type,
            'is_festival_season': is_festival_season,
            'day_before_holiday': day_before_holiday,
            'day_after_holiday': day_after_holiday,
            
            # Seasonal features
            'season': season,
            'delhi_season': delhi_season,
            'season_intensity': round(season_intensity, 2),
            
            # Load patterns
            'load_pattern': load_pattern,
            'peak_probability': round(peak_probability, 2),
            'demand_category': demand_category
        }
    
    def get_delhi_season(self, month: int) -> str:
        """Get Delhi season based on month."""
        for season, months in self.delhi_seasons.items():
            if month in months:
                return season
        return 'Unknown'
    
    def get_detailed_delhi_season(self, month: int, timestamp: datetime) -> str:
        """Get detailed Delhi seasonal classification."""
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4]:
            if month == 3:
                return 'Early_Spring'
            else:
                return 'Late_Spring'
        elif month in [5, 6]:
            if month == 5:
                return 'Pre_Summer'
            else:
                return 'Peak_Summer'
        elif month in [7, 8, 9]:
            if month == 7:
                return 'Early_Monsoon'
            elif month == 8:
                return 'Peak_Monsoon'
            else:
                return 'Late_Monsoon'
        elif month in [10, 11]:
            if month == 10:
                return 'Post_Monsoon'
            else:
                return 'Pre_Winter'
        return 'Unknown'
    
    def get_season_intensity(self, month: int, hour: int) -> float:
        """Calculate seasonal intensity factor (0.0 to 1.0)."""
        # Base intensity by month
        intensity_map = {
            1: 0.8, 2: 0.7, 3: 0.4, 4: 0.5, 5: 0.9, 6: 1.0,  # Winter to Summer
            7: 0.6, 8: 0.7, 9: 0.6, 10: 0.3, 11: 0.4, 12: 0.8  # Monsoon to Winter
        }
        
        base_intensity = intensity_map.get(month, 0.5)
        
        # Adjust for time of day
        if month in [5, 6]:  # Summer months
            if 12 <= hour <= 16:  # Peak afternoon heat
                base_intensity = min(1.0, base_intensity + 0.2)
        elif month in [12, 1, 2]:  # Winter months
            if 6 <= hour <= 9 or 19 <= hour <= 22:  # Cold mornings/evenings
                base_intensity = min(1.0, base_intensity + 0.1)
        
        return base_intensity
    
    def predict_load_pattern(self, hour: int, day_of_week: int, is_holiday: bool, season: str) -> str:
        """Predict load pattern category."""
        if is_holiday:
            return 'Holiday'
        
        if day_of_week >= 5:  # Weekend
            if season == 'Summer' and 12 <= hour <= 18:
                return 'Weekend_High'
            else:
                return 'Weekend_Normal'
        
        # Weekday patterns
        if 7 <= hour <= 10 or 18 <= hour <= 22:
            if season == 'Summer':
                return 'Weekday_Peak_Summer'
            else:
                return 'Weekday_Peak_Normal'
        elif 11 <= hour <= 17:
            return 'Weekday_Day'
        elif 23 <= hour or hour <= 6:
            return 'Weekday_Night'
        else:
            return 'Weekday_Normal'
    
    def calculate_peak_probability(self, hour: int, day_of_week: int, month: int, is_holiday: bool) -> float:
        """Calculate probability of peak demand (0.0 to 1.0)."""
        if is_holiday:
            return 0.3  # Lower probability on holidays
        
        base_prob = 0.2
        
        # Hour-based probability
        if 19 <= hour <= 21:  # Evening peak
            base_prob = 0.9
        elif 7 <= hour <= 9:  # Morning peak
            base_prob = 0.7
        elif 12 <= hour <= 16 and month in [5, 6]:  # Summer afternoon
            base_prob = 0.8
        elif 11 <= hour <= 17:  # Day time
            base_prob = 0.5
        elif 22 <= hour or hour <= 6:  # Night
            base_prob = 0.1
        
        # Day of week adjustment
        if day_of_week >= 5:  # Weekend
            base_prob *= 0.7
        
        # Season adjustment
        if month in [5, 6]:  # Peak summer
            base_prob = min(1.0, base_prob * 1.2)
        elif month in [12, 1]:  # Winter
            base_prob = min(1.0, base_prob * 1.1)
        
        return base_prob
    
    def get_demand_category(self, hour: int, day_of_week: int, season: str, is_holiday: bool) -> str:
        """Categorize expected demand level."""
        if is_holiday:
            return 'Low'
        
        if day_of_week >= 5:  # Weekend
            if season == 'Summer' and 12 <= hour <= 18:
                return 'Medium_High'
            elif 19 <= hour <= 22:
                return 'Medium'
            else:
                return 'Low'
        
        # Weekday
        if 19 <= hour <= 21:
            return 'Very_High'
        elif 7 <= hour <= 9:
            return 'High'
        elif season == 'Summer' and 12 <= hour <= 16:
            return 'Very_High'
        elif 10 <= hour <= 18:
            return 'Medium_High'
        elif 22 <= hour or hour <= 6:
            return 'Low'
        else:
            return 'Medium'
    
    def populate_temporal_features(self, batch_size: int = 1000):
        """Populate temporal features for all records in the database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            print("🕐 Populating temporal features for load data...")
            
            # Get total count for records missing basic temporal features (not holiday data)
            cursor.execute("SELECT COUNT(*) FROM load_data WHERE hour IS NULL;")
            total_records = cursor.fetchone()['count']
            
            if total_records == 0:
                print("✅ All records already have temporal features!")
                return
            
            print(f"📊 Processing {total_records:,} records without temporal features")
            
            # Process in batches
            processed = 0
            batch_count = 0
            
            while processed < total_records:
                # Get batch of records with holiday data in one query
                cursor.execute("""
                    SELECT id, timestamp, is_holiday, holiday_name, holiday_type, 
                           is_festival_season, day_before_holiday, day_after_holiday
                    FROM load_data 
                    WHERE hour IS NULL 
                    ORDER BY timestamp 
                    LIMIT %s;
                """, (batch_size,))
                
                batch_records = cursor.fetchall()
                if not batch_records:
                    break
                
                batch_count += 1
                print(f"Processing batch {batch_count} ({len(batch_records)} records)...")
                
                # Generate features for each record (excluding holiday data which exists)
                batch_updates = []
                for record in batch_records:
                    timestamp = record['timestamp']
                    
                    # Prepare holiday data
                    holiday_data = {
                        'is_holiday': record['is_holiday'] or False,
                        'holiday_name': record['holiday_name'],
                        'holiday_type': record['holiday_type'],
                        'is_festival_season': record['is_festival_season'] or False,
                        'day_before_holiday': record['day_before_holiday'] or False,
                        'day_after_holiday': record['day_after_holiday'] or False
                    }
                    
                    features = self.create_temporal_features(timestamp, holiday_data)
                    
                    update_tuple = (
                        features['hour'], features['day_of_week'], features['month'],
                        features['day_of_year'], features['week_of_year'],
                        features['hour_sin'], features['hour_cos'],
                        features['day_sin'], features['day_cos'],
                        features['month_sin'], features['month_cos'],
                        features['is_weekend'], features['is_weekday'], features['is_working_day'],
                        features['is_morning_peak'], features['is_evening_peak'],
                        features['is_office_hours'], features['is_late_night'],
                        features['season'], features['delhi_season'], features['season_intensity'],
                        features['load_pattern'], features['peak_probability'], features['demand_category'],
                        record['id']
                    )
                    batch_updates.append(update_tuple)
                
                # Update database (excluding holiday columns which already exist)
                update_query = """
                    UPDATE load_data SET
                        hour = %s, day_of_week = %s, month = %s,
                        day_of_year = %s, week_of_year = %s,
                        hour_sin = %s, hour_cos = %s,
                        day_sin = %s, day_cos = %s,
                        month_sin = %s, month_cos = %s,
                        is_weekend = %s, is_weekday = %s, is_working_day = %s,
                        is_morning_peak = %s, is_evening_peak = %s,
                        is_office_hours = %s, is_late_night = %s,
                        season = %s, delhi_season = %s, season_intensity = %s,
                        load_pattern = %s, peak_probability = %s, demand_category = %s
                    WHERE id = %s;
                """
                
                cursor.executemany(update_query, batch_updates)
                conn.commit()
                
                processed += len(batch_records)
                print(f"✅ Updated {processed:,}/{total_records:,} records ({processed/total_records*100:.1f}%)")
            
            print(f"🎉 Successfully populated temporal features for all {total_records:,} records!")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to populate temporal features: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_temporal_summary(self):
        """Get summary of temporal features."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            print("\n📊 TEMPORAL FEATURES SUMMARY")
            print("=" * 50)
            
            # Records with temporal features
            cursor.execute("SELECT COUNT(*) FROM load_data WHERE hour IS NOT NULL;")
            temporal_count = cursor.fetchone()['count']
            
            cursor.execute("SELECT COUNT(*) FROM load_data;")
            total_count = cursor.fetchone()['count']
            
            print(f"Records with temporal features: {temporal_count:,}/{total_count:,} ({temporal_count/total_count*100:.1f}%)")
            
            # Peak hours distribution
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN is_morning_peak THEN 1 ELSE 0 END) as morning_peak,
                    SUM(CASE WHEN is_evening_peak THEN 1 ELSE 0 END) as evening_peak,
                    SUM(CASE WHEN is_office_hours THEN 1 ELSE 0 END) as office_hours,
                    SUM(CASE WHEN is_weekend THEN 1 ELSE 0 END) as weekend,
                    SUM(CASE WHEN is_holiday THEN 1 ELSE 0 END) as holidays
                FROM load_data WHERE hour IS NOT NULL;
            """)
            stats = cursor.fetchone()
            
            print(f"\n🕐 Time Pattern Distribution:")
            print(f"   Morning Peak (7-10 AM): {stats['morning_peak']:,} records")
            print(f"   Evening Peak (6-10 PM): {stats['evening_peak']:,} records")
            print(f"   Office Hours: {stats['office_hours']:,} records")
            print(f"   Weekend: {stats['weekend']:,} records")
            print(f"   Holidays: {stats['holidays']:,} records")
            
            # Seasonal distribution
            cursor.execute("""
                SELECT delhi_season, COUNT(*) as count
                FROM load_data 
                WHERE delhi_season IS NOT NULL
                GROUP BY delhi_season
                ORDER BY count DESC;
            """)
            seasons = cursor.fetchall()
            
            print(f"\n🌤️  Seasonal Distribution:")
            for season in seasons:
                print(f"   {season['delhi_season']}: {season['count']:,} records")
            
            # Load patterns
            cursor.execute("""
                SELECT load_pattern, COUNT(*) as count
                FROM load_data 
                WHERE load_pattern IS NOT NULL
                GROUP BY load_pattern
                ORDER BY count DESC
                LIMIT 10;
            """)
            patterns = cursor.fetchall()
            
            print(f"\n📈 Load Pattern Distribution:")
            for pattern in patterns:
                print(f"   {pattern['load_pattern']}: {pattern['count']:,} records")
            
        except Exception as e:
            self.logger.error(f"Failed to get temporal summary: {str(e)}")
        finally:
            cursor.close()
            conn.close()
    
    def run_temporal_feature_engineering(self):
        """Run complete temporal feature engineering process."""
        try:
            print("=" * 60)
            print("🕐 TEMPORAL FEATURE ENGINEERING - SIH 2024")
            print("=" * 60)
            print("Generating comprehensive time-based features for Delhi load forecasting")
            print()
            
            # Step 1: Add schema columns
            self.add_temporal_columns_to_schema()
            print()
            
            # Step 2: Populate features
            self.populate_temporal_features()
            print()
            
            # Step 3: Get summary
            self.get_temporal_summary()
            
            print("\n" + "=" * 60)
            print("✅ TEMPORAL FEATURE ENGINEERING COMPLETE!")
            print("=" * 60)
            print("Generated features:")
            print("🕐 • Hour, day, month, cyclical encodings")
            print("📅 • Working days, weekends, holidays")
            print("⏰ • Peak hours (morning/evening)")
            print("🏢 • Office hours detection")
            print("🎉 • Festival seasons and special days")
            print("🌤️  • Delhi seasonal patterns")
            print("📊 • Load pattern predictions")
            print("📈 • Peak probability scores")
            print("🎯 • Demand category classification")
            print()
            print("Ready for advanced time-series forecasting! 🚀")
            print("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Temporal feature engineering failed: {str(e)}")
            print(f"ERROR: {str(e)}")

def main():
    """Run temporal feature engineering."""
    engine = TemporalFeatureEngine()
    engine.run_temporal_feature_engineering()

if __name__ == "__main__":
    main()
