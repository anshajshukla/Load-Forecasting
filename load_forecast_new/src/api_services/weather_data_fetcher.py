"""
Enhanced data fetcher for Delhi SLDC with real-time weather integration
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime
import time
import os
import pickle
import json
from sklearn.preprocessing import StandardScaler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from typing import Optional, Dict, List

# Load environment variables
load_dotenv()

class DelhiSLDCDataFetcher:
    """Class to fetch and process data from Delhi SLDC website."""
    
    def __init__(self, model_path='../../models/saved_models/gru_forecast_model.h5',
                 scalers_dir='../../models/saved_models',
                 historical_data_path='../../data/final_data.csv',
                 cache_file='../../data/live_data_cache.csv'):
        """
        Initialize the data fetcher.
        
        Args:
            model_path: Path to the trained model
            scalers_dir: Directory containing trained scalers
            historical_data_path: Path to historical data for feature creation
            cache_file: File to cache live data
        """
        self.url = "https://www.delhisldc.org/Redirect.aspx?Loc=0805"  # For real-time data
        self.historical_url = "https://www.delhisldc.org/Loaddata.aspx?mode="  # For historical data
        self.model_path = model_path
        self.scalers_dir = scalers_dir
        self.historical_data_path = historical_data_path
        self.cache_file = cache_file
        self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        # Weather API configuration
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.weather_base_url = "https://api.openweathermap.org/data/2.5"
        
        # Setup session with retry strategy
        self.setup_session()
        
        # Load feature scaler and target scalers
        self.feature_scaler = self.load_feature_scaler()
        self.target_scalers = self.load_target_scalers()
        
        # Initialize cache
        self.live_data_cache = []
        
        print(f"✅ DelhiSLDCDataFetcher initialized successfully")
        print(f"📁 Model path: {self.model_path}")
        print(f"📁 Scalers directory: {self.scalers_dir}")
        print(f"🌦️ Weather API: {'Configured' if self.weather_api_key else 'Not configured'}")
    
    def setup_session(self):
        """Setup requests session with retry strategy."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def load_feature_scaler(self):
        """Load the feature scaler used during training."""
        scaler_path = os.path.join(self.scalers_dir, 'feature_scaler.pkl')
        try:
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"✅ Feature scaler loaded from {scaler_path}")
                return scaler
            else:
                print(f"⚠️ Feature scaler not found at {scaler_path}")
                return StandardScaler()
        except Exception as e:
            print(f"❌ Error loading feature scaler: {e}")
            return StandardScaler()
    
    def load_target_scalers(self):
        """Load target scalers for each target variable."""
        scalers = {}
        for target in self.targets:
            scaler_path = os.path.join(self.scalers_dir, f'{target}_scaler.pkl')
            try:
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        scalers[target] = pickle.load(f)
                    print(f"✅ {target} scaler loaded")
                else:
                    print(f"⚠️ {target} scaler not found at {scaler_path}")
                    scalers[target] = StandardScaler()
            except Exception as e:
                print(f"❌ Error loading {target} scaler: {e}")
                scalers[target] = StandardScaler()
        
        return scalers
    
    def get_weather_data(self, lat=28.6139, lon=77.2090):
        """
        Fetch current weather data for Delhi.
        
        Args:
            lat: Latitude of Delhi
            lon: Longitude of Delhi
            
        Returns:
            Dictionary with weather data
        """
        if not self.weather_api_key:
            print("⚠️ Weather API key not configured")
            return self.get_default_weather()
        
        try:
            # Current weather
            current_url = f"{self.weather_base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = self.session.get(current_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind'].get('speed', 0),
                'wind_direction': data['wind'].get('deg', 0),
                'visibility': data.get('visibility', 10000) / 1000,  # Convert to km
                'weather_condition': data['weather'][0]['main']
            }
            
            print(f"🌦️ Weather data fetched: {weather_data['temperature']}°C, {weather_data['humidity']}% humidity")
            return weather_data
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {e}")
            return self.get_default_weather()
    
    def get_default_weather(self):
        """Return default weather values when API is unavailable."""
        # Use seasonal averages for Delhi
        now = datetime.datetime.now()
        month = now.month
        
        # Delhi seasonal temperature patterns
        if month in [12, 1, 2]:  # Winter
            temp = 15.0
            humidity = 60
        elif month in [3, 4, 5]:  # Spring/Summer
            temp = 30.0
            humidity = 40
        elif month in [6, 7, 8, 9]:  # Monsoon
            temp = 28.0
            humidity = 80
        else:  # Autumn
            temp = 25.0
            humidity = 50
        
        return {
            'temperature': temp,
            'humidity': humidity,
            'pressure': 1013.25,
            'wind_speed': 5.0,
            'wind_direction': 270,
            'visibility': 10.0,
            'weather_condition': 'Clear'
        }
    
    def fetch_sldc_data(self):
        """
        Fetch current load data from Delhi SLDC website.
        
        Returns:
            Dictionary with current load data or None if failed
        """
        try:
            print("🔄 Fetching data from Delhi SLDC...")
            response = self.session.get(self.url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table with load data
            table = soup.find('table', {'class': 'tableborder'})
            if not table:
                # Try alternative selectors
                table = soup.find('table', {'border': '1'})
                if not table:
                    print("❌ Could not find data table on webpage")
                    return None
            
            # Extract data from table
            rows = table.find_all('tr')
            data = {}
            timestamp = None
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    cell_text = [cell.get_text(strip=True) for cell in cells]
                    
                    # Look for timestamp
                    if 'time' in cell_text[0].lower() or 'updated' in cell_text[0].lower():
                        timestamp = cell_text[1]
                        continue
                    
                    # Look for load data
                    for target in self.targets:
                        if target in cell_text[0]:
                            try:
                                value = float(cell_text[1].replace(',', ''))
                                data[target] = value
                                print(f"📊 {target}: {value} MW")
                            except (ValueError, IndexError):
                                continue
            
            if data:
                data['timestamp'] = timestamp or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data['fetch_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"✅ Successfully fetched data for {len(data)} targets")
                return data
            else:
                print("❌ No valid load data found")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error fetching SLDC data: {e}")
            return None
        except Exception as e:
            print(f"❌ Error parsing SLDC data: {e}")
            return None
    
    def create_time_features(self, timestamp):
        """
        Create time-based features from timestamp.
        
        Args:
            timestamp: datetime object or string
            
        Returns:
            Dictionary with time features
        """
        if isinstance(timestamp, str):
            try:
                dt = pd.to_datetime(timestamp)
            except:
                dt = datetime.datetime.now()
        else:
            dt = timestamp
        
        features = {
            'hour': dt.hour,
            'day_of_week': dt.weekday(),
            'month': dt.month,
            'year': dt.year,
            'day_of_year': dt.timetuple().tm_yday,
            'is_weekend': 1 if dt.weekday() >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
            'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
            'day_sin': np.sin(2 * np.pi * dt.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * dt.weekday() / 7),
            'month_sin': np.sin(2 * np.pi * dt.month / 12),
            'month_cos': np.cos(2 * np.pi * dt.month / 12)
        }
        
        return features
    
    def fetch_and_preprocess_live_data(self, sequence_length=24):
        """
        Fetch live data and preprocess it for model prediction.
        
        Args:
            sequence_length: Length of input sequence required by model
            
        Returns:
            Preprocessed data ready for model prediction
        """
        print(f"\n🚀 Starting live data fetch and preprocessing...")
        
        # Step 1: Fetch current SLDC data
        sldc_data = self.fetch_sldc_data()
        if not sldc_data:
            print("❌ Failed to fetch SLDC data")
            return None
        
        # Step 2: Fetch weather data
        weather_data = self.get_weather_data()
        
        # Step 3: Create current data point
        current_time = datetime.datetime.now()
        time_features = self.create_time_features(current_time)
        
        # Combine all features
        current_data = {
            **sldc_data,
            **weather_data,
            **time_features
        }
        
        # Step 4: Create sequence by combining with historical data
        sequence_data = self.create_input_sequence(current_data, sequence_length)
        
        if sequence_data is None:
            print("❌ Failed to create input sequence")
            return None
        
        print(f"✅ Preprocessed data shape: {sequence_data.shape}")
        return sequence_data
    
    def create_input_sequence(self, current_data, sequence_length):
        """
        Create input sequence by combining current data with historical data.
        
        Args:
            current_data: Current data point
            sequence_length: Required sequence length
            
        Returns:
            Preprocessed sequence ready for model
        """
        try:
            # Load historical data to create sequence
            if os.path.exists(self.historical_data_path):
                historical_df = pd.read_csv(self.historical_data_path)
                print(f"📊 Loaded historical data: {len(historical_df)} records")
            else:
                print("⚠️ Historical data not found, using simulated data")
                historical_df = self.create_simulated_historical_data()
            
            # Prepare feature columns (matching training)
            feature_columns = [
                'DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES',
                'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'visibility',
                'hour', 'day_of_week', 'month', 'is_weekend',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
            ]
            
            # Get last sequence_length-1 rows from historical data
            if len(historical_df) >= sequence_length - 1:
                last_rows = historical_df[feature_columns].tail(sequence_length - 1)
            else:
                # Pad with last available data if not enough history
                last_row = historical_df[feature_columns].iloc[-1:] if len(historical_df) > 0 else None
                if last_row is not None:
                    last_rows = pd.concat([last_row] * (sequence_length - 1), ignore_index=True)
                else:
                    print("❌ No historical data available")
                    return None
            
            # Create current data row
            current_row = {}
            for col in feature_columns:
                if col in current_data:
                    current_row[col] = current_data[col]
                else:
                    # Use average from historical data if feature missing
                    if col in last_rows.columns:
                        current_row[col] = last_rows[col].mean()
                    else:
                        current_row[col] = 0
            
            current_df = pd.DataFrame([current_row])
            
            # Combine historical and current data
            sequence_df = pd.concat([last_rows, current_df], ignore_index=True)
            
            # Ensure we have the right sequence length
            if len(sequence_df) > sequence_length:
                sequence_df = sequence_df.tail(sequence_length)
            elif len(sequence_df) < sequence_length:
                # Pad with last row if needed
                last_row_df = sequence_df.iloc[-1:].copy()
                while len(sequence_df) < sequence_length:
                    sequence_df = pd.concat([sequence_df, last_row_df], ignore_index=True)
            
            # Scale features
            if self.feature_scaler:
                try:
                    scaled_features = self.feature_scaler.transform(sequence_df[feature_columns])
                except Exception as e:
                    print(f"⚠️ Error scaling features, using unscaled data: {e}")
                    scaled_features = sequence_df[feature_columns].values
            else:
                print("⚠️ Feature scaler not available, using unscaled data")
                scaled_features = sequence_df[feature_columns].values
            
            # Reshape for model input (1 sample, sequence_length, features)
            input_data = scaled_features.reshape(1, sequence_length, len(feature_columns))
            
            print(f"✅ Created input sequence: {input_data.shape}")
            return input_data
            
        except Exception as e:
            print(f"❌ Error creating input sequence: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_simulated_historical_data(self, days=7):
        """
        Create simulated historical data when real data is not available.
        
        Args:
            days: Number of days of data to simulate
            
        Returns:
            DataFrame with simulated historical data
        """
        print(f"🔄 Creating {days} days of simulated historical data...")
        
        # Create timestamps
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='H')
        
        data = []
        for ts in timestamps:
            # Simulate load patterns based on time of day and day of week
            hour = ts.hour
            day_of_week = ts.weekday()
            
            # Base load with daily and weekly patterns
            base_load = 3000  # MW
            daily_factor = 0.8 + 0.4 * np.sin(2 * np.pi * (hour - 6) / 24)
            weekly_factor = 0.9 if day_of_week < 5 else 0.7  # Lower on weekends
            
            delhi_load = base_load * daily_factor * weekly_factor
            
            # Distribute among different areas
            row = {
                'DELHI': delhi_load,
                'BRPL': delhi_load * 0.25,
                'BYPL': delhi_load * 0.20,
                'NDMC': delhi_load * 0.15,
                'MES': delhi_load * 0.10,
                'temperature': 25 + 10 * np.sin(2 * np.pi * (hour - 6) / 24),
                'humidity': 50 + 20 * np.random.normal(0, 0.1),
                'pressure': 1013.25,
                'wind_speed': 5 + 3 * np.random.normal(0, 0.1),
                'wind_direction': 270,
                'visibility': 10,
                'hour': hour,
                'day_of_week': day_of_week,
                'month': ts.month,
                'is_weekend': 1 if day_of_week >= 5 else 0,
                'hour_sin': np.sin(2 * np.pi * hour / 24),
                'hour_cos': np.cos(2 * np.pi * hour / 24),
                'day_sin': np.sin(2 * np.pi * day_of_week / 7),
                'day_cos': np.cos(2 * np.pi * day_of_week / 7),
                'month_sin': np.sin(2 * np.pi * ts.month / 12),
                'month_cos': np.cos(2 * np.pi * ts.month / 12)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        print(f"✅ Created simulated data: {len(df)} records")
        return df


def main():
    """Test the data fetcher."""
    print("🧪 Testing Delhi SLDC Data Fetcher")
    print("=" * 50)
    
    # Initialize fetcher
    fetcher = DelhiSLDCDataFetcher()
    
    # Test live data fetch
    input_data = fetcher.fetch_and_preprocess_live_data()
    
    if input_data is not None:
        print(f"✅ Successfully prepared input data: {input_data.shape}")
    else:
        print("❌ Failed to prepare input data")


if __name__ == "__main__":
    main()
