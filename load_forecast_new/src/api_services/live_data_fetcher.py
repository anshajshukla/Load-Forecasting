"""
Script to fetch live data from Delhi SLDC website and process it for load forecasting.
Updated with real-time weather API integration.
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
    
    def __init__(self, model_path='models/saved_models/gru_forecast_model.h5',
                 scalers_dir='models/saved_models',
                 historical_data_path='data/final_data.csv',
                 cache_file='data/live_data_cache.csv'):
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
        
        # Initialize cache
        if not os.path.exists(self.cache_file):
            self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the cache file with historical data."""
        if os.path.exists(self.historical_data_path):
            # Use the last few records from historical data
            hist_df = pd.read_csv(self.historical_data_path)
            hist_df = hist_df.tail(24)  # Last 24 hours as initial context
            hist_df.to_csv(self.cache_file, index=False)
            print(f"Cache initialized with {len(hist_df)} records from historical data")
        else:
            # Create an empty cache
            columns = ['datetime', 'weekday', 'DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES', 
                      'temperature', 'humidity', 'wind_speed', 'precipitation']
            pd.DataFrame(columns=columns).to_csv(self.cache_file, index=False)
            print("Empty cache file created")
    
    def create_retry_session(self):
        """Create a requests session with retry functionality."""
        session = requests.Session()
        retry = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            backoff_factor=1
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        return session
    
    def fetch_weather_data_openweathermap(self, city="Delhi,IN") -> Optional[Dict]:
        """
        Fetch weather data from OpenWeatherMap API.
        
        Args:
            city: City name for weather data
            
        Returns:
            dict: Weather data or None if failed
        """
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            print("OPENWEATHERMAP_API_KEY not found in environment variables")
            return None
        
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        
        try:
            response = self.create_retry_session().get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = {
                "temperature": round(data["main"]["temp"], 2),
                "humidity": data["main"]["humidity"],
                "wind_speed": round(data["wind"]["speed"] * 3.6, 2),  # Convert m/s to km/h
                "precipitation": data.get("rain", {}).get("1h", 0.0)
            }
            
            print("Weather data fetched successfully from OpenWeatherMap")
            return weather_data
            
        except Exception as e:
            print(f"OpenWeatherMap API error: {e}")
            return None
    
    def fetch_weather_data_weatherapi(self, city="Delhi,India") -> Optional[Dict]:
        """
        Fetch weather data from WeatherAPI.com as backup.
        
        Args:
            city: City name for weather data
            
        Returns:
            dict: Weather data or None if failed
        """
        api_key = os.getenv("WEATHERAPI_KEY")
        if not api_key:
            return None
        
        url = "http://api.weatherapi.com/v1/current.json"
        params = {
            "key": api_key,
            "q": city,
            "aqi": "no"
        }
        
        try:
            response = self.create_retry_session().get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data["current"]
            
            weather_data = {
                "temperature": round(current["temp_c"], 2),
                "humidity": current["humidity"],
                "wind_speed": round(current["wind_kph"], 2),
                "precipitation": round(current["precip_mm"], 2)
            }
            
            print("Weather data fetched successfully from WeatherAPI")
            return weather_data
            
        except Exception as e:
            print(f"WeatherAPI error: {e}")
            return None
    
    def fetch_weather_data(self) -> Dict:
        """
        Fetch weather data with fallback strategy.
        
        Returns:
            dict: Weather data (real or fallback random values)
        """
        # Try OpenWeatherMap first
        weather_data = self.fetch_weather_data_openweathermap()
        
        if weather_data is None:
            print("OpenWeatherMap failed, trying WeatherAPI...")
            weather_data = self.fetch_weather_data_weatherapi()
        
        if weather_data is None:
            print("All weather APIs failed, using random fallback values")
            weather_data = {
                "temperature": np.random.uniform(20, 35),  # °C
                "humidity": np.random.uniform(30, 90),     # %
                "wind_speed": np.random.uniform(0, 15),   # km/h
                "precipitation": np.random.uniform(0, 5)  # mm
            }
        
        return weather_data
    
    def fetch_todays_historical_data(self) -> Optional[Dict]:
        """
        Fetch today's hourly data from the historical URL for better accuracy.
        
        Returns:
            dict: Latest hourly data from today's historical records
        """
        try:
            today = datetime.datetime.now()
            date_str = today.strftime('%d/%m/%Y')
            url = f"{self.historical_url}{date_str}"
            
            print(f"Fetching today's historical data from {url}...")
            response = self.create_retry_session().get(url, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract hourly data
            hourly_records = []
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    
                    if len(cells) < 5:
                        continue
                        
                    cell_texts = [cell.get_text().strip() for cell in cells]
                    first_cell = cell_texts[0]
                    
                    # Match time patterns
                    import re
                    time_match = re.match(r'^(\d{1,2}):?(\d{0,2})$', first_cell)
                    
                    if time_match:
                        hour = int(time_match.group(1))
                        
                        if 0 <= hour <= 23:
                            # Extract numeric values
                            numeric_values = []
                            for cell_text in cell_texts[1:]:
                                try:
                                    clean_text = re.sub(r'[^\d.]', '', cell_text)
                                    if clean_text and clean_text.count('.') <= 1:
                                        value = float(clean_text)
                                        if 10 <= value <= 10000:
                                            numeric_values.append(value)
                                except:
                                    continue
                            
                            if len(numeric_values) >= 3:
                                # Map values to targets
                                mapped_values = self._map_historical_values(numeric_values)
                                if len(mapped_values) >= 3:
                                    mapped_values['hour'] = hour
                                    hourly_records.append(mapped_values)
            
            if hourly_records:
                # Return the most recent hour's data
                latest_record = max(hourly_records, key=lambda x: x['hour'])
                
                # Add metadata
                result = {
                    'datetime': today.strftime('%d-%m-%Y %H:%M'),
                    'weekday': today.strftime('%A')
                }
                
                # Add load values
                for target in self.targets:
                    result[target] = latest_record.get(target)
                
                print(f"Found historical data for hour {latest_record['hour']}: {[f'{k}={v}' for k,v in latest_record.items() if k in self.targets]}")
                return result
            else:
                print("No valid historical data found for today")
                return None
                
        except Exception as e:
            print(f"Error fetching today's historical data: {e}")
            return None
    
    def _map_historical_values(self, values: List[float]) -> Dict[str, float]:
        """Map extracted values to targets for historical data."""
        mapped = {}
        used_values = set()
        
        ranges = {
            'DELHI': (4000, 8000),
            'BRPL': (1800, 3500), 
            'BYPL': (900, 1800),
            'NDMC': (150, 500),
            'MES': (20, 100)
        }
        
        # Sort targets by expected value (highest first)
        targets_by_range = sorted(self.targets, key=lambda t: ranges[t][1], reverse=True)
        sorted_values = sorted(values, reverse=True)
        
        for target in targets_by_range:
            min_val, max_val = ranges[target]
            for value in sorted_values:
                if value not in used_values and min_val <= value <= max_val:
                    mapped[target] = value
                    used_values.add(value)
                    break
        
        return mapped

    def fetch_live_data(self):
        """
        Fetch live data from Delhi SLDC website.
        Tries historical URL first for better accuracy, then falls back to real-time.
        
        Returns:
            dict: Dictionary containing the fetched data
        """
        try:
            # First try to get today's historical data (more accurate)
            print("Attempting to fetch today's historical data...")
            historical_data = self.fetch_todays_historical_data()
            
            if historical_data and len([k for k in self.targets if historical_data.get(k) is not None]) >= 3:
                print("Successfully fetched historical data")
                data = historical_data
            else:
                print("Historical data insufficient, trying real-time data...")
                # Fall back to real-time data fetching
                response = requests.get(self.url)
                response.raise_for_status()
                
                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Get current datetime
                now = datetime.datetime.now()
                data = {
                    'datetime': now.strftime('%d-%m-%Y %H:%M'),
                    'weekday': now.strftime('%A')
                }
                
                # Extract load values with DICOM-aware parsing
                print("Attempting DICOM data extraction...")
                dicom_data = self._extract_dicom_data(soup)
                
                # Merge the data
                for target in self.targets:
                    data[target] = dicom_data.get(target)
                
                # If DICOM extraction didn't get enough data, fall back to general parsing
                if len([k for k in self.targets if data.get(k) is not None]) < 3:
                    print("DICOM extraction incomplete, trying general table parsing...")
                    
                    # Look for table structures that contain the load data
                    tables = soup.find_all('table')
                    
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 6:  # Should have TIMESLOT + 5 load values
                                first_cell = cells[0].get_text().strip()
                                if ':' in first_cell or 'TIMESLOT' in first_cell.upper():
                                    if 'TIMESLOT' in first_cell.upper():
                                        continue
                                    
                                    try:
                                        for i, target in enumerate(['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']):
                                            if i + 1 < len(cells):
                                                cell_text = cells[i + 1].get_text().strip()
                                                try:
                                                    value = float(cell_text.replace(',', ''))
                                                    if self._validate_load_value(target, value):
                                                        data[target] = value
                                                        print(f"Found {target}: {value} MW from table")
                                                except:
                                                    continue
                                        
                                        if len([k for k in self.targets if data.get(k) is not None]) >= 3:
                                            break
                                    except Exception as e:
                                        continue
                
                # Set None for targets we couldn't find
                for target in self.targets:
                    if target not in data or data[target] is None:
                        data[target] = None
                        print(f"Could not find data for {target}")
            
            # Print summary of extracted data
            found_targets = [k for k in self.targets if data.get(k) is not None]
            print(f"Successfully extracted data for {len(found_targets)}/{len(self.targets)} targets: {found_targets}")
            
            # Fetch real weather data with fallback
            weather_data = self.fetch_weather_data()
            data.update(weather_data)
            
            print("Data fetched successfully")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def _extract_dicom_data(self, soup):
        """
        Extract data from DICOM interface with 5-minute load values
        """
        data = {}
        
        try:
            # Look for main data table that shows current values
            main_tables = soup.find_all('table')
            
            for table in main_tables:
                rows = table.find_all('tr')
                
                # Look for data rows with proper structure
                for i, row in enumerate(rows):
                    cells = row.find_all(['td', 'th'])
                    
                    # Skip rows with too few cells
                    if len(cells) < 5:
                        continue
                        
                    cell_texts = [cell.get_text().strip() for cell in cells]
                    
                    # Check if this looks like a data row with timestamp
                    first_cell = cell_texts[0] if cells else ""
                    
                    # Look for time pattern (HH:MM) in the first cell
                    import re
                    if re.match(r'\d{1,2}:\d{2}', first_cell):
                        print(f"Found data row with time {first_cell}: analyzing {len(cells)} cells")
                        
                        # Try to extract load values from this row
                        # Look for values that match our expected ranges
                        extracted_values = {}
                        
                        for j, cell_text in enumerate(cell_texts[1:], 1):  # Skip first cell (time)
                            try:
                                # Clean and convert to float
                                clean_text = cell_text.replace(',', '').strip()
                                if not clean_text or not re.match(r'^\d+\.?\d*$', clean_text):
                                    continue
                                    
                                value = float(clean_text)
                                
                                # Check which target this value might belong to
                                for target in self.targets:
                                    if target not in extracted_values and self._validate_load_value(target, value):
                                        extracted_values[target] = value
                                        print(f"Matched {target}: {value} MW from cell {j}")
                                        break
                                        
                            except (ValueError, AttributeError):
                                continue
                        
                        # If we found good values, use them
                        if len(extracted_values) >= 3:
                            data.update(extracted_values)
                            print(f"Extracted {len(extracted_values)} values from row: {extracted_values}")
                            break
                
                # If we found good data in this table, stop looking
                if len(data) >= 3:
                    break
            
            # If no good data found with timestamp matching, try alternative approach
            if len(data) < 3:
                print("Time-based extraction failed, trying value range matching...")
                
                for table in main_tables:
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        cell_texts = [cell.get_text().strip() for cell in cells]
                        
                        # Look for numeric values in expected ranges
                        found_in_row = {}
                        
                        for cell_text in cell_texts:
                            try:
                                clean_text = cell_text.replace(',', '').strip()
                                if not re.match(r'^\d+\.?\d*$', clean_text):
                                    continue
                                    
                                value = float(clean_text)
                                
                                # Check against each target's expected range
                                for target in self.targets:
                                    if target not in found_in_row and self._validate_load_value(target, value):
                                        found_in_row[target] = value
                                        break
                                        
                            except (ValueError, AttributeError):
                                continue
                        
                        # If this row has multiple valid values, it's likely our data row
                        if len(found_in_row) >= 3:
                            data.update(found_in_row)
                            print(f"Found valid data row with {len(found_in_row)} targets: {found_in_row}")
                            break
                    
                    if len(data) >= 3:
                        break
                        
        except Exception as e:
            print(f"Error in DICOM extraction: {e}")
            
        return data

    def _validate_load_value(self, target: str, value: float) -> bool:
        """Validate if a load value is reasonable for the target based on real data patterns."""
        ranges = {
            'DELHI': (4000, 7000),    # Updated based on real data showing 4000-6000 MW
            'BRPL': (1800, 3000),     # Updated based on real data showing 2000-2500 MW  
            'BYPL': (900, 1500),      # Updated based on real data showing 1100-1300 MW
            'NDMC': (150, 400),       # Based on real data showing 300-350 MW
            'MES': (25, 50)           # Based on real data showing 30-40 MW
        }
        
        if target not in ranges:
            return False
        
        min_val, max_val = ranges[target]
        return min_val <= value <= max_val
    
    def update_cache(self, new_data):
        """
        Update the cache with new data.
        
        Args:
            new_data: Dictionary containing new data to add to cache
        """
        try:
            # Validate the new data before adding to cache
            valid_data = True
            for target in self.targets:
                if new_data.get(target) is not None:
                    if not self._validate_load_value(target, new_data[target]):
                        print(f"Invalid {target} value: {new_data[target]}, skipping cache update")
                        valid_data = False
                        break
            
            # Only update cache if data is valid
            if not valid_data:
                print("Skipping cache update due to invalid data")
                return pd.read_csv(self.cache_file) if os.path.exists(self.cache_file) else None
            
            # Read existing cache
            if os.path.exists(self.cache_file):
                cache_df = pd.read_csv(self.cache_file)
            else:
                # Create empty cache with proper columns
                columns = ['datetime', 'weekday'] + self.targets + ['temperature', 'humidity', 'wind_speed', 'precipitation']
                cache_df = pd.DataFrame(columns=columns)
            
            # Create new record, filtering out None values for targets
            filtered_data = {}
            for key, value in new_data.items():
                if key in self.targets and value is None:
                    # Skip None values for targets
                    continue
                filtered_data[key] = value
            
            new_record = pd.DataFrame([filtered_data])
            
            # Append to cache
            updated_df = pd.concat([cache_df, new_record], ignore_index=True)
            
            # Keep only the most recent records (last 72 hours)
            if len(updated_df) > 72:
                updated_df = updated_df.tail(72)
            
            # Save updated cache
            updated_df.to_csv(self.cache_file, index=False)
            print(f"Cache updated, now contains {len(updated_df)} records")
            
            return updated_df
        
        except Exception as e:
            print(f"Error updating cache: {e}")
            return None
    
    def prepare_data_for_prediction(self):
        """
        Prepare cached data for prediction.
        
        Returns:
            dict: Dictionary containing processed features ready for prediction
        """
        try:
            # Read cache
            df = pd.read_csv(self.cache_file)
            
            # Convert datetime and set as index
            df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
            
            # Convert weekday to numerical
            weekday_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            df['weekday_num'] = df['weekday'].map(weekday_map)
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Create time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # Create lag features for targets
            for target in self.targets:
                df[f'{target}_lag_1'] = df[target].shift(1)
                df[f'{target}_lag_24'] = df[target].shift(24)
            
            # Fill NaN values with forward fill, then backward fill
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Define features (same as in training)
            categorical_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'weekday_num']
            numerical_features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
            
            # Add lag features
            for target in self.targets:
                numerical_features.extend([f'{target}_lag_1', f'{target}_lag_24'])
            
            # Filter features that exist in the dataframe
            categorical_features = [f for f in categorical_features if f in df.columns]
            numerical_features = [f for f in numerical_features if f in df.columns]
            
            # Combine all features
            features = numerical_features + categorical_features
            
            # Get the latest row for prediction
            latest_data = df.iloc[-1:][features].values
            
            # Load scalers
            feature_scaler = self.load_feature_scaler()
            
            # Scale features
            if feature_scaler is not None:
                try:
                    # If it's a scikit-learn pipeline
                    scaled_features = feature_scaler.transform(df.iloc[-1:][features])
                except:
                    # If it's a simple scaler
                    scaled_features = feature_scaler.transform(latest_data)
            else:
                # If no scaler, use normalized data
                scaled_features = latest_data
            
            # Reshape for RNN (1, timesteps, features)
            X = scaled_features.reshape(1, 1, scaled_features.shape[1])
            
            result = {
                'X': X,
                'latest_data': df.iloc[-1:],
                'df': df,
                'features': features
            }
            
            return result
        
        except Exception as e:
            print(f"Error preparing data for prediction: {e}")
            return None
    
    def load_feature_scaler(self):
        """
        Load the feature scaler from disk.
        
        Returns:
            scaler: Loaded feature scaler
        """
        try:
            scaler_path = os.path.join(self.scalers_dir, 'feature_scaler.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        except Exception as e:
            print(f"Error loading feature scaler: {e}")
            return None
    
    def load_target_scalers(self):
        """
        Load the target scalers from disk.
        
        Returns:
            dict: Dictionary of target scalers
        """
        try:
            scalers = {}
            for target in self.targets:
                scaler_path = os.path.join(self.scalers_dir, f'{target}_scaler.pkl')
                with open(scaler_path, 'rb') as f:
                    scalers[target] = pickle.load(f)
            return scalers
        except Exception as e:
            print(f"Error loading target scalers: {e}")
            return None

    def export_latest_data(self, output_file='data/latest_data.json'):
        """
        Export the latest data to a JSON file for the dashboard.
        
        Args:
            output_file: Path to save the JSON file
        """
        try:
            # Read cache
            df = pd.read_csv(self.cache_file)
            
            # Get the last 24 hours of data
            recent_data = df.tail(24).copy()
            
            # Convert datetime for JSON serialization with dayfirst=True to avoid warning
            recent_data['datetime'] = pd.to_datetime(recent_data['datetime'], dayfirst=True)
            recent_data['datetime_str'] = recent_data['datetime'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Prepare data for JSON
            data_dict = {
                'timestamps': recent_data['datetime_str'].tolist(),
                'targets': {}
            }
            
            # Add data for each target
            for target in self.targets:
                if target in recent_data.columns:
                    data_dict['targets'][target] = recent_data[target].tolist()
            
            # Add weather data
            for weather_var in ['temperature', 'humidity', 'wind_speed', 'precipitation']:
                if weather_var in recent_data.columns:
                    data_dict[weather_var] = recent_data[weather_var].tolist()
            
            # Save to JSON
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(data_dict, f)
            
            print(f"Latest data exported to {output_file}")
            
        except Exception as e:
            print(f"Error exporting latest data: {e}")

# Example usage
if __name__ == "__main__":
    fetcher = DelhiSLDCDataFetcher()
    
    # Fetch live data
    live_data = fetcher.fetch_live_data()
    
    if live_data:
        # Update cache
        fetcher.update_cache(live_data)
        
        # Export latest data for dashboard
        fetcher.export_latest_data()
        
        # Prepare data for prediction
        processed_data = fetcher.prepare_data_for_prediction()
        
        if processed_data:
            print("Data prepared for prediction")
            print(f"Feature shape: {processed_data['X'].shape}")
            print(f"Latest datetime: {processed_data['latest_data'].index[0]}")
        else:
            print("Failed to prepare data for prediction")