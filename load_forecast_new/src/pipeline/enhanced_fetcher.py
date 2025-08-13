"""
Enhanced Data Fetcher for Delhi SLDC Load Forecasting Pipeline.
Handles 5-year historical data collection and real-time data updates.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import time
import os
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import schedule
from pathlib import Path

# Import our database manager
from database.db_manager import DatabaseManager, create_database_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FetchStatus:
    """Data class to track fetching status."""
    total_days: int = 0
    completed_days: int = 0
    failed_days: int = 0
    current_date: str = ""
    start_time: datetime.datetime = None
    estimated_completion: datetime.datetime = None

class EnhancedDataFetcher:
    """Enhanced data fetcher with database integration and parallel processing."""
    
    def __init__(self, db_manager: DatabaseManager = None, max_workers: int = 3):
        """
        Initialize the enhanced data fetcher.
        
        Args:
            db_manager: Database manager instance
            max_workers: Maximum number of parallel workers for data fetching
        """
        self.base_url = "https://www.delhisldc.org/Loaddata.aspx?mode="
        self.realtime_url = "https://www.delhisldc.org/Redirect.aspx?Loc=0805"
        self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        self.max_workers = max_workers
        
        # Database setup
        self.db_manager = db_manager or create_database_manager()
        
        # Status tracking
        self.fetch_status = FetchStatus()
        self.is_fetching = False
        self.stop_requested = False
        
        # Rate limiting
        self.request_delay = 2  # seconds between requests
        self.last_request_time = 0
        
        # Weather API setup
        self.weather_apis = {
            'weatherapi': "http://api.weatherapi.com/v1/current.json",
            'openweather': "https://api.openweathermap.org/data/2.5/weather"
        }
        
        # Create output directories
        self.output_dir = Path('data_pipeline/raw_data')
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_retry_session(self):
        """Create a requests session with retry functionality."""
        session = requests.Session()
        retry = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=2
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        return session
    
    def rate_limit(self):
        """Implement rate limiting to be respectful to the server."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_daily_data(self, date_str: str, include_weather: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch hourly load data for a specific date with enhanced error handling.
        
        Args:
            date_str: Date in DD/MM/YYYY format
            include_weather: Whether to fetch weather data
            
        Returns:
            DataFrame with hourly data for that date
        """
        try:
            self.rate_limit()
            
            url = f"{self.base_url}{date_str}"
            logger.debug(f"Fetching data for {date_str}...")
            
            response = self.create_retry_session().get(url, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract hourly data from tables
            hourly_data = []
            
            # Look for tables containing hourly load data
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                # Check if this table has the proper structure (TIMESLOT header)
                if len(rows) > 1:
                    header_row = rows[0]
                    header_cells = [cell.get_text().strip() for cell in header_row.find_all(['td', 'th'])]
                    
                    # Look for the table with TIMESLOT header
                    if 'TIMESLOT' in header_cells:
                        logger.debug(f"Found data table with {len(rows)} rows for {date_str}")
                        
                        # Process data rows - HOURLY OPTIMIZATION
                        for row in rows[1:]:  # Skip header row
                            cells = row.find_all(['td', 'th'])
                            
                            if len(cells) < 6:  # Need at least TIMESLOT + 5 load values
                                continue
                                
                            cell_texts = [cell.get_text().strip() for cell in cells]
                            
                            # First cell should be time (HH:MM)
                            time_str = cell_texts[0]
                            
                            # Validate time format
                            import re
                            time_match = re.match(r'^(\d{1,2}):(\d{2})$', time_str)
                            
                            if time_match:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2))
                                
                                # HYBRID APPROACH: Collect both hourly and high-frequency data
                                # For spike detection: collect 15-minute intervals (minute = 0, 15, 30, 45)
                                # For general forecasting: can aggregate to hourly later
                                if minute not in [0, 15, 30, 45]:
                                    continue  # Keep 15-minute resolution for spike detection
                                
                                if 0 <= hour <= 23 and 0 <= minute <= 59:
                                    # Extract load values based on column headers
                                    load_values = {
                                        'hour': hour,
                                        'minute': minute,
                                        'time_str': time_str
                                    }
                                    
                                    # Map columns to targets based on header
                                    target_mapping = {
                                        'DELHI': 'DELHI',
                                        'BRPL': 'BRPL', 
                                        'BYPL': 'BYPL',
                                        'NDPL': 'NDMC',  # NDPL maps to NDMC
                                        'NDMC': 'NDMC',
                                        'MES': 'MES'
                                    }
                                    
                                    # Extract values based on column positions
                                    for i, header in enumerate(header_cells[1:], 1):  # Skip TIMESLOT
                                        if header in target_mapping and i < len(cell_texts):
                                            target = target_mapping[header]
                                            try:
                                                value = float(cell_texts[i])
                                                
                                                # Validate the value
                                                if self._validate_load_value(target, value):
                                                    load_values[target] = value
                                                    
                                            except (ValueError, IndexError):
                                                continue
                                    
                                    # Only add if we found data for most targets
                                    found_targets = [k for k in self.targets if k in load_values]
                                    if len(found_targets) >= 4:  # At least 4 out of 5 targets
                                        hourly_data.append(load_values)
                        
                        # If we found a good data table, we can stop looking
                        if hourly_data:
                            break
            
            if hourly_data:
                # Convert to DataFrame
                df = pd.DataFrame(hourly_data)
                
                # Add date information
                date_obj = datetime.datetime.strptime(date_str, '%d/%m/%Y')
                df['date'] = date_str
                df['datetime'] = df.apply(lambda row: 
                    date_obj.replace(hour=int(row['hour']), minute=int(row['minute'])), axis=1)
                df['weekday'] = df['datetime'].dt.day_name()
                
                # Add weather data if requested
                if include_weather:
                    weather_data = self._fetch_weather_for_date(date_obj)
                    for key, value in weather_data.items():
                        df[key] = value
                
                # Sort by hour and minute
                df = df.sort_values(['hour', 'minute']).reset_index(drop=True)
                
                logger.info(f"Successfully extracted {len(df)} records for {date_str}")
                return df
            else:
                logger.warning(f"No valid hourly data found for {date_str}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {date_str}: {e}")
            return None
    
    def _validate_load_value(self, target: str, value: float) -> bool:
        """Validate if a load value is reasonable for the target."""
        if value is None or value < 0:
            return False
            
        ranges = {
            'DELHI': (4000, 8000),   
            'BRPL': (1800, 3500),    
            'BYPL': (900, 1800),     
            'NDMC': (150, 500),      
            'MES': (20, 100)         
        }
        
        if target in ranges:
            min_val, max_val = ranges[target]
            return min_val <= value <= max_val
        
        return 10 <= value <= 10000
    
    def _fetch_weather_for_date(self, date_obj: datetime.datetime) -> Dict:
        """Fetch weather data for a specific date (or use defaults for historical dates)."""
        # For historical dates, we'll use seasonal averages
        # For current date, we'll try to fetch real weather data
        
        if date_obj.date() == datetime.date.today():
            return self._fetch_current_weather()
        else:
            return self._generate_seasonal_weather(date_obj)
    
    def _fetch_current_weather(self) -> Dict:
        """Fetch current weather data from APIs."""
        try:
            # Try WeatherAPI first
            import os
            api_key = os.getenv("WEATHERAPI_KEY")
            if api_key:
                url = self.weather_apis['weatherapi']
                params = {
                    "key": api_key,
                    "q": "Delhi,India",
                    "aqi": "no"
                }
                
                response = self.create_retry_session().get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    current = data["current"]
                    
                    return {
                        "temperature": round(current["temp_c"], 2),
                        "humidity": current["humidity"],
                        "wind_speed": round(current["wind_kph"], 2),
                        "precipitation": round(current["precip_mm"], 2)
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch current weather: {e}")
        
        # Fallback to seasonal averages
        return self._generate_seasonal_weather(datetime.datetime.now())
    
    def _generate_seasonal_weather(self, date_obj: datetime.datetime) -> Dict:
        """Generate seasonal weather data based on Delhi's climate patterns."""
        month = date_obj.month
        
        # Delhi seasonal weather patterns
        if month in [12, 1, 2]:  # Winter
            temp_range = (10, 25)
            humidity_range = (40, 70)
            wind_range = (5, 15)
            precip_range = (0, 2)
        elif month in [3, 4, 5]:  # Summer
            temp_range = (25, 45)
            humidity_range = (20, 60)
            wind_range = (10, 25)
            precip_range = (0, 5)
        elif month in [6, 7, 8, 9]:  # Monsoon
            temp_range = (25, 35)
            humidity_range = (70, 95)
            wind_range = (15, 30)
            precip_range = (2, 15)
        else:  # Post-monsoon
            temp_range = (15, 30)
            humidity_range = (50, 80)
            wind_range = (5, 20)
            precip_range = (0, 3)
        
        return {
            "temperature": round(np.random.uniform(*temp_range), 2),
            "humidity": round(np.random.uniform(*humidity_range), 0),
            "wind_speed": round(np.random.uniform(*wind_range), 2),
            "precipitation": round(np.random.uniform(*precip_range), 2)
        }
    
    def fetch_date_range_parallel(self, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        Fetch data for a range of dates using parallel processing.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Combined DataFrame with all data
        """
        self.is_fetching = True
        self.stop_requested = False
        
        # Calculate date range
        date_list = []
        current_date = start_date
        while current_date <= end_date:
            date_list.append(current_date.strftime('%d/%m/%Y'))
            current_date += datetime.timedelta(days=1)
        
        # Update status
        self.fetch_status.total_days = len(date_list)
        self.fetch_status.completed_days = 0
        self.fetch_status.failed_days = 0
        self.fetch_status.start_time = datetime.datetime.now()
        
        logger.info(f"Starting parallel fetch for {len(date_list)} days from {start_date} to {end_date}")
        
        all_data = []
        
        # Process in batches to avoid overwhelming the server
        batch_size = 10
        
        for i in range(0, len(date_list), batch_size):
            if self.stop_requested:
                logger.info("Fetch process stopped by user request")
                break
                
            batch = date_list[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                # Submit tasks
                future_to_date = {
                    executor.submit(self.fetch_daily_data, date_str): date_str 
                    for date_str in batch
                }
                
                # Collect results
                for future in as_completed(future_to_date):
                    date_str = future_to_date[future]
                    
                    try:
                        result = future.result()
                        
                        if result is not None and not result.empty:
                            all_data.append(result)
                            self.fetch_status.completed_days += 1
                            
                            # Save to database immediately
                            try:
                                self.db_manager.insert_historical_data(result)
                            except Exception as e:
                                logger.error(f"Failed to insert data for {date_str}: {e}")
                        else:
                            self.fetch_status.failed_days += 1
                            logger.warning(f"No data retrieved for {date_str}")
                        
                        self.fetch_status.current_date = date_str
                        
                        # Update estimated completion
                        if self.fetch_status.completed_days > 0:
                            elapsed = datetime.datetime.now() - self.fetch_status.start_time
                            avg_time_per_day = elapsed.total_seconds() / self.fetch_status.completed_days
                            remaining_days = self.fetch_status.total_days - self.fetch_status.completed_days
                            self.fetch_status.estimated_completion = datetime.datetime.now() + datetime.timedelta(seconds=avg_time_per_day * remaining_days)
                        
                        # Progress logging
                        if self.fetch_status.completed_days % 50 == 0:
                            logger.info(f"Progress: {self.fetch_status.completed_days}/{self.fetch_status.total_days} days completed")
                        
                    except Exception as e:
                        logger.error(f"Error processing {date_str}: {e}")
                        self.fetch_status.failed_days += 1
            
            # Small delay between batches
            if not self.stop_requested and i + batch_size < len(date_list):
                time.sleep(5)
        
        self.is_fetching = False
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total records collected: {len(combined_df)}")
            
            # Save final dataset
            output_file = self.output_dir / f"historical_data_{start_date}_{end_date}.csv"
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Data saved to {output_file}")
            
            return combined_df
        else:
            logger.warning("No data collected")
            return pd.DataFrame()
    
    def fetch_last_n_years(self, n_years: int = 5) -> pd.DataFrame:
        """
        Fetch data for the last N years.
        
        Args:
            n_years: Number of years to fetch (default 5)
            
        Returns:
            DataFrame with historical data
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=n_years * 365)
        
        logger.info(f"Fetching {n_years} years of data from {start_date} to {end_date}")
        
        return self.fetch_date_range_parallel(start_date, end_date)
    
    def get_fetch_status(self) -> Dict:
        """Get current fetch status for monitoring."""
        return {
            'is_fetching': self.is_fetching,
            'total_days': self.fetch_status.total_days,
            'completed_days': self.fetch_status.completed_days,
            'failed_days': self.fetch_status.failed_days,
            'current_date': self.fetch_status.current_date,
            'progress_percentage': (self.fetch_status.completed_days / max(self.fetch_status.total_days, 1)) * 100,
            'estimated_completion': self.fetch_status.estimated_completion.isoformat() if self.fetch_status.estimated_completion else None,
            'start_time': self.fetch_status.start_time.isoformat() if self.fetch_status.start_time else None
        }
    
    def stop_fetch(self):
        """Stop the current fetch process."""
        self.stop_requested = True
        logger.info("Stop request sent to fetch process")
    
    def setup_scheduled_updates(self):
        """Setup scheduled updates for real-time data."""
        # Schedule real-time data fetch every 5 minutes
        schedule.every(5).minutes.do(self._fetch_realtime_update)
        
        # Schedule daily historical data backup
        schedule.every().day.at("02:00").do(self._daily_backup)
        
        logger.info("Scheduled updates configured")
    
    def _fetch_realtime_update(self):
        """Fetch real-time data update."""
        try:
            # Implementation for real-time data fetching
            # This would use the existing fetch_live_data logic
            logger.info("Fetching real-time data update")
            # TODO: Implement real-time fetch and validation
        except Exception as e:
            logger.error(f"Real-time update failed: {e}")
    
    def _daily_backup(self):
        """Perform daily data backup."""
        try:
            # Fetch yesterday's data to ensure completeness
            yesterday = datetime.date.today() - datetime.timedelta(days=1)
            date_str = yesterday.strftime('%d/%m/%Y')
            
            df = self.fetch_daily_data(date_str)
            if df is not None:
                self.db_manager.insert_historical_data(df)
                logger.info(f"Daily backup completed for {date_str}")
        except Exception as e:
            logger.error(f"Daily backup failed: {e}")
    
    def run_scheduler(self):
        """Run the scheduler for automated updates."""
        logger.info("Starting scheduled task runner")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    # Example usage
    fetcher = EnhancedDataFetcher()
    
    # Test with a small date range first
    test_start = datetime.date.today() - datetime.timedelta(days=2)
    test_end = datetime.date.today()
    
    print("Testing data fetcher with recent dates...")
    test_df = fetcher.fetch_date_range_parallel(test_start, test_end)
    
    if not test_df.empty:
        print(f"Test successful! Collected {len(test_df)} records")
        print("\nSample data:")
        print(test_df.head())
        
        print("\nReady to fetch 5 years of historical data.")
        print("This will take several hours...")
        
        response = input("Proceed with full 5-year fetch? (y/n): ")
        
        if response.lower() == 'y':
            print("Starting 5-year historical data collection...")
            historical_df = fetcher.fetch_last_n_years(5)
            print(f"Historical data collection completed! Total records: {len(historical_df)}")
        else:
            print("Full fetch cancelled")
    else:
        print("Test failed - check network connection and website availability")
