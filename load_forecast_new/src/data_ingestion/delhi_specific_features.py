#!/usr/bin/env python3
"""
Delhi-Specific Features Integration
Critical temporal features for accurate Delhi load forecasting (2022-2025)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DelhiSpecificFeatures:
    """
    Delhi-specific temporal features that significantly impact electricity load patterns
    """
    
    def __init__(self):
        self.festival_calendar = self._build_festival_calendar()
        self.delhi_events = self._build_delhi_events()
        
    def _build_festival_calendar(self) -> Dict[str, List[Dict]]:
        """
        Build comprehensive festival calendar for Delhi (2022-2025)
        Returns dictionary with year as key and festival list as value
        """
        
        festivals = {
            # 2022 Festival Calendar
            "2022": [
                # Major Hindu Festivals
                {"name": "Diwali", "start": "2022-10-22", "end": "2022-10-26", "type": "religious", "intensity": 5},
                {"name": "Holi", "start": "2022-03-17", "end": "2022-03-18", "type": "religious", "intensity": 4},
                {"name": "Dussehra", "start": "2022-10-05", "end": "2022-10-05", "type": "religious", "intensity": 3},
                {"name": "Janmashtami", "start": "2022-08-19", "end": "2022-08-19", "type": "religious", "intensity": 3},
                {"name": "Raksha Bandhan", "start": "2022-08-11", "end": "2022-08-11", "type": "religious", "intensity": 2},
                
                # National Holidays
                {"name": "Independence Day", "start": "2022-08-15", "end": "2022-08-15", "type": "national", "intensity": 3},
                {"name": "Republic Day", "start": "2022-01-26", "end": "2022-01-26", "type": "national", "intensity": 3},
                {"name": "Gandhi Jayanti", "start": "2022-10-02", "end": "2022-10-02", "type": "national", "intensity": 2},
                
                # Islamic Festivals
                {"name": "Eid ul-Fitr", "start": "2022-05-03", "end": "2022-05-03", "type": "religious", "intensity": 4},
                {"name": "Eid ul-Adha", "start": "2022-07-10", "end": "2022-07-10", "type": "religious", "intensity": 3},
                
                # Sikh Festivals
                {"name": "Guru Nanak Jayanti", "start": "2022-11-08", "end": "2022-11-08", "type": "religious", "intensity": 3},
                
                # Christian Festivals
                {"name": "Christmas", "start": "2022-12-25", "end": "2022-12-25", "type": "religious", "intensity": 3},
            ],
            
            # 2023 Festival Calendar
            "2023": [
                # Major Hindu Festivals
                {"name": "Diwali", "start": "2023-11-12", "end": "2023-11-16", "type": "religious", "intensity": 5},
                {"name": "Holi", "start": "2023-03-08", "end": "2023-03-09", "type": "religious", "intensity": 4},
                {"name": "Dussehra", "start": "2023-10-24", "end": "2023-10-24", "type": "religious", "intensity": 3},
                {"name": "Janmashtami", "start": "2023-09-07", "end": "2023-09-07", "type": "religious", "intensity": 3},
                {"name": "Raksha Bandhan", "start": "2023-08-30", "end": "2023-08-30", "type": "religious", "intensity": 2},
                
                # National Holidays
                {"name": "Independence Day", "start": "2023-08-15", "end": "2023-08-15", "type": "national", "intensity": 3},
                {"name": "Republic Day", "start": "2023-01-26", "end": "2023-01-26", "type": "national", "intensity": 3},
                {"name": "Gandhi Jayanti", "start": "2023-10-02", "end": "2023-10-02", "type": "national", "intensity": 2},
                
                # Islamic Festivals
                {"name": "Eid ul-Fitr", "start": "2023-04-22", "end": "2023-04-22", "type": "religious", "intensity": 4},
                {"name": "Eid ul-Adha", "start": "2023-06-29", "end": "2023-06-29", "type": "religious", "intensity": 3},
                
                # Sikh Festivals
                {"name": "Guru Nanak Jayanti", "start": "2023-11-27", "end": "2023-11-27", "type": "religious", "intensity": 3},
                
                # Christian Festivals
                {"name": "Christmas", "start": "2023-12-25", "end": "2023-12-25", "type": "religious", "intensity": 3},
            ],
            
            # 2024 Festival Calendar
            "2024": [
                # Major Hindu Festivals
                {"name": "Diwali", "start": "2024-10-31", "end": "2024-11-04", "type": "religious", "intensity": 5},
                {"name": "Holi", "start": "2024-03-25", "end": "2024-03-26", "type": "religious", "intensity": 4},
                {"name": "Dussehra", "start": "2024-10-12", "end": "2024-10-12", "type": "religious", "intensity": 3},
                {"name": "Janmashtami", "start": "2024-08-26", "end": "2024-08-26", "type": "religious", "intensity": 3},
                {"name": "Raksha Bandhan", "start": "2024-08-19", "end": "2024-08-19", "type": "religious", "intensity": 2},
                
                # National Holidays
                {"name": "Independence Day", "start": "2024-08-15", "end": "2024-08-15", "type": "national", "intensity": 3},
                {"name": "Republic Day", "start": "2024-01-26", "end": "2024-01-26", "type": "national", "intensity": 3},
                {"name": "Gandhi Jayanti", "start": "2024-10-02", "end": "2024-10-02", "type": "national", "intensity": 2},
                
                # Islamic Festivals
                {"name": "Eid ul-Fitr", "start": "2024-04-11", "end": "2024-04-11", "type": "religious", "intensity": 4},
                {"name": "Eid ul-Adha", "start": "2024-06-17", "end": "2024-06-17", "type": "religious", "intensity": 3},
                
                # Sikh Festivals
                {"name": "Guru Nanak Jayanti", "start": "2024-11-15", "end": "2024-11-15", "type": "religious", "intensity": 3},
                
                # Christian Festivals
                {"name": "Christmas", "start": "2024-12-25", "end": "2024-12-25", "type": "religious", "intensity": 3},
            ],
            
            # 2025 Festival Calendar (Projected)
            "2025": [
                # Major Hindu Festivals
                {"name": "Diwali", "start": "2025-10-20", "end": "2025-10-24", "type": "religious", "intensity": 5},
                {"name": "Holi", "start": "2025-03-14", "end": "2025-03-15", "type": "religious", "intensity": 4},
                {"name": "Dussehra", "start": "2025-10-02", "end": "2025-10-02", "type": "religious", "intensity": 3},
                {"name": "Janmashtami", "start": "2025-08-16", "end": "2025-08-16", "type": "religious", "intensity": 3},
                {"name": "Raksha Bandhan", "start": "2025-08-09", "end": "2025-08-09", "type": "religious", "intensity": 2},
                
                # National Holidays
                {"name": "Independence Day", "start": "2025-08-15", "end": "2025-08-15", "type": "national", "intensity": 3},
                {"name": "Republic Day", "start": "2025-01-26", "end": "2025-01-26", "type": "national", "intensity": 3},
                {"name": "Gandhi Jayanti", "start": "2025-10-02", "end": "2025-10-02", "type": "national", "intensity": 2},
                
                # Islamic Festivals (Projected)
                {"name": "Eid ul-Fitr", "start": "2025-03-31", "end": "2025-03-31", "type": "religious", "intensity": 4},
                {"name": "Eid ul-Adha", "start": "2025-06-07", "end": "2025-06-07", "type": "religious", "intensity": 3},
                
                # Sikh Festivals
                {"name": "Guru Nanak Jayanti", "start": "2025-11-05", "end": "2025-11-05", "type": "religious", "intensity": 3},
                
                # Christian Festivals
                {"name": "Christmas", "start": "2025-12-25", "end": "2025-12-25", "type": "religious", "intensity": 3},
            ]
        }
        
        return festivals
    
    def _build_delhi_events(self) -> Dict[str, List[Dict]]:
        """
        Build Delhi-specific events calendar (2022-2025)
        """
        
        events = {
            "2022": [
                {"name": "Parliament Budget Session", "start": "2022-01-31", "end": "2022-04-08", "type": "political"},
                {"name": "Parliament Monsoon Session", "start": "2022-07-18", "end": "2022-08-12", "type": "political"},
                {"name": "Parliament Winter Session", "start": "2022-12-07", "end": "2022-12-29", "type": "political"},
                {"name": "Stubble Burning Period", "start": "2022-10-15", "end": "2022-11-15", "type": "environmental"},
            ],
            
            "2023": [
                {"name": "G20 Summit Delhi", "start": "2023-09-09", "end": "2023-09-10", "type": "political"},
                {"name": "Parliament Budget Session", "start": "2023-01-31", "end": "2023-04-06", "type": "political"},
                {"name": "Parliament Monsoon Session", "start": "2023-07-20", "end": "2023-08-11", "type": "political"},
                {"name": "Parliament Winter Session", "start": "2023-12-04", "end": "2023-12-22", "type": "political"},
                {"name": "Stubble Burning Period", "start": "2023-10-15", "end": "2023-11-15", "type": "environmental"},
                {"name": "Delhi Assembly Session", "start": "2023-03-13", "end": "2023-03-24", "type": "political"},
            ],
            
            "2024": [
                {"name": "Parliament Budget Session", "start": "2024-01-31", "end": "2024-04-05", "type": "political"},
                {"name": "Parliament Monsoon Session", "start": "2024-07-22", "end": "2024-08-12", "type": "political"},
                {"name": "Parliament Winter Session", "start": "2024-11-25", "end": "2024-12-20", "type": "political"},
                {"name": "Stubble Burning Period", "start": "2024-10-15", "end": "2024-11-15", "type": "environmental"},
                {"name": "Lok Sabha Elections", "start": "2024-04-19", "end": "2024-06-01", "type": "political"},
            ],
            
            "2025": [
                {"name": "Parliament Budget Session", "start": "2025-01-31", "end": "2025-04-04", "type": "political"},
                {"name": "Parliament Monsoon Session", "start": "2025-07-21", "end": "2025-08-15", "type": "political"},
                {"name": "Parliament Winter Session", "start": "2025-11-24", "end": "2025-12-20", "type": "political"},
                {"name": "Stubble Burning Period", "start": "2025-10-15", "end": "2025-11-15", "type": "environmental"},
                {"name": "Delhi Assembly Elections", "start": "2025-02-01", "end": "2025-02-28", "type": "political"},
            ]
        }
        
        return events
    
    def add_delhi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Delhi-specific temporal features to the dataset
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with additional Delhi-specific features
        """
        
        logger.info("Adding Delhi-specific temporal features...")
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Initialize new feature columns
        df['is_diwali_period'] = False
        df['is_major_festival'] = False
        df['is_national_holiday'] = False
        df['is_religious_festival'] = False
        df['festival_intensity'] = 0
        df['pre_festival_day'] = False
        df['post_festival_day'] = False
        df['festival_type'] = 'none'
        
        df['is_political_event'] = False
        df['is_pollution_emergency'] = False
        df['is_stubble_burning_period'] = False
        df['is_odd_even_scheme'] = False
        
        # Add festival features
        for year, festivals in self.festival_calendar.items():
            year_mask = df['timestamp'].dt.year == int(year)
            
            for festival in festivals:
                start_date = pd.to_datetime(festival['start'])
                end_date = pd.to_datetime(festival['end'])
                
                # Festival period mask
                festival_mask = year_mask & (df['timestamp'].dt.date >= start_date.date()) & (df['timestamp'].dt.date <= end_date.date())
                
                # Set festival features
                if festival['name'] == 'Diwali':
                    df.loc[festival_mask, 'is_diwali_period'] = True
                
                df.loc[festival_mask, 'is_major_festival'] = True
                df.loc[festival_mask, 'festival_intensity'] = festival['intensity']
                df.loc[festival_mask, 'festival_type'] = festival['type']
                
                if festival['type'] == 'national':
                    df.loc[festival_mask, 'is_national_holiday'] = True
                elif festival['type'] == 'religious':
                    df.loc[festival_mask, 'is_religious_festival'] = True
                
                # Pre and post festival days
                pre_date = start_date - timedelta(days=1)
                post_date = end_date + timedelta(days=1)
                
                pre_mask = year_mask & (df['timestamp'].dt.date == pre_date.date())
                post_mask = year_mask & (df['timestamp'].dt.date == post_date.date())
                
                df.loc[pre_mask, 'pre_festival_day'] = True
                df.loc[post_mask, 'post_festival_day'] = True
        
        # Add Delhi-specific events
        for year, events in self.delhi_events.items():
            year_mask = df['timestamp'].dt.year == int(year)
            
            for event in events:
                start_date = pd.to_datetime(event['start'])
                end_date = pd.to_datetime(event['end'])
                
                event_mask = year_mask & (df['timestamp'].dt.date >= start_date.date()) & (df['timestamp'].dt.date <= end_date.date())
                
                if event['type'] == 'political':
                    df.loc[event_mask, 'is_political_event'] = True
                elif event['type'] == 'environmental':
                    if 'Stubble Burning' in event['name']:
                        df.loc[event_mask, 'is_stubble_burning_period'] = True
                        # During stubble burning, pollution increases -> AC usage changes
                        df.loc[event_mask, 'is_pollution_emergency'] = True
        
        # Add derived features
        df['festival_season'] = 'none'
        df.loc[df['is_diwali_period'], 'festival_season'] = 'diwali'
        df.loc[df['is_major_festival'] & (df['timestamp'].dt.month == 3), 'festival_season'] = 'holi'
        df.loc[df['is_major_festival'] & (df['timestamp'].dt.month.isin([10, 11])), 'festival_season'] = 'autumn_festivals'
        
        logger.info(f"Added 12 Delhi-specific features to {len(df)} records")
        
        # Summary statistics
        logger.info("Delhi Features Summary:")
        logger.info(f"- Diwali periods: {df['is_diwali_period'].sum()} hours")
        logger.info(f"- Major festivals: {df['is_major_festival'].sum()} hours")
        logger.info(f"- National holidays: {df['is_national_holiday'].sum()} hours")
        logger.info(f"- Political events: {df['is_political_event'].sum()} hours")
        logger.info(f"- Stubble burning periods: {df['is_stubble_burning_period'].sum()} hours")
        
        return df
    
    def validate_delhi_features(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Validate Delhi-specific features coverage
        """
        
        validation_results = {
            'total_records': len(df),
            'diwali_periods': df['is_diwali_period'].sum(),
            'major_festivals': df['is_major_festival'].sum(),
            'national_holidays': df['is_national_holiday'].sum(),
            'political_events': df['is_political_event'].sum(),
            'stubble_burning_periods': df['is_stubble_burning_period'].sum(),
            'high_intensity_festivals': (df['festival_intensity'] >= 4).sum(),
            'years_covered': df['timestamp'].dt.year.nunique()
        }
        
        return validation_results

def main():
    """
    Demo function to show Delhi-specific features integration
    """
    
    # Create sample timestamp data
    start_date = '2022-01-01'
    end_date = '2025-12-31'
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    
    df = pd.DataFrame({'timestamp': timestamps})
    
    # Initialize Delhi features
    delhi_features = DelhiSpecificFeatures()
    
    # Add Delhi-specific features
    df_enhanced = delhi_features.add_delhi_features(df)
    
    # Validate features
    validation = delhi_features.validate_delhi_features(df_enhanced)
    
    print("\n🎯 DELHI-SPECIFIC FEATURES VALIDATION")
    print("=" * 50)
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    # Show sample of enhanced data
    print("\n📊 SAMPLE DATA WITH DELHI FEATURES")
    print("=" * 50)
    sample_cols = ['timestamp', 'is_diwali_period', 'is_major_festival', 'festival_intensity', 'is_political_event']
    print(df_enhanced[sample_cols].head(10))
    
    return df_enhanced

if __name__ == "__main__":
    enhanced_df = main()
