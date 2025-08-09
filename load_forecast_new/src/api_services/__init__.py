"""
SIH 2024 API Services
Streamlined real-time data fetching and database operations
"""

from .live_data_fetcher import DelhiSLDCDataFetcher
from .weather_data_fetcher import DelhiSLDCDataFetcher as WeatherDataFetcher
from .database_manager import DatabaseSchemaManager
from .data_validator import DataQualityValidator
from .weather_api_client import APIManager

__all__ = [
    'DelhiSLDCDataFetcher',
    'WeatherDataFetcher',
    'DatabaseSchemaManager', 
    'DataQualityValidator',
    'APIManager'
]

class APIServicesManager:
    """
    Manages real-time data collection and database operations.
    Focused on live data fetching and updates only.
    """
    
    def __init__(self):
        self.load_fetcher = None
        self.weather_fetcher = None
        self.db_manager = None
        self.validator = None
        
    def initialize_services(self):
        """Initialize all API services."""
        self.load_fetcher = DelhiSLDCDataFetcher()
        self.weather_fetcher = WeatherDataFetcher()
        self.db_manager = DatabaseSchemaManager()
        self.validator = DataQualityValidator()
        self.api_manager = APIManager()
        
    def fetch_and_store_latest_data(self):
        """Fetch latest data from all sources and store in database."""
        try:
            # Fetch latest load data
            load_data = self.load_fetcher.fetch_realtime_data()
            
            # Fetch latest weather data
            weather_data = self.weather_fetcher.fetch_current_weather()
            
            # Validate data
            if self.validator.validate_load_data(load_data) and                self.validator.validate_weather_data(weather_data):
                
                # Store in database
                self.db_manager.insert_latest_data(load_data, weather_data)
                return True
            else:
                print("⚠️  Data validation failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in data fetching: {e}")
            return False
