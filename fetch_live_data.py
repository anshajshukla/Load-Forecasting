"""
Script to fetch live data from Delhi SLDC website and process it for load forecasting.
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
        self.url = "https://www.delhisldc.org/Redirect.aspx?Loc=0804"
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
    
    def fetch_live_data(self):
        """
        Fetch live data from Delhi SLDC website.
        
        Returns:
            dict: Dictionary containing the fetched data
        """
        try:
            print("Fetching data from Delhi SLDC website...")
            response = requests.get(self.url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract data
            # The data structure may vary, so we'll need to inspect the page and adjust accordingly
            # This is a placeholder based on common HTML structures
            data = {}
            
            # Get current datetime
            now = datetime.datetime.now()
            data['datetime'] = now.strftime('%d-%m-%Y %H:%M')
            data['weekday'] = now.strftime('%A')
            
            # Extract load values (these selectors need to be adjusted based on actual page structure)
            try:
                # Find tables with load data
                tables = soup.find_all('table')
                
                # Example extraction - this needs to be tailored to the actual page structure
                for table in tables:
                    # Look for headers or rows containing our target regions
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) > 1:
                            # Check if this row contains one of our targets
                            text = cells[0].get_text().strip().upper()
                            for target in self.targets:
                                if target in text:
                                    # Try to extract a numeric value from the second cell
                                    try:
                                        value_text = cells[1].get_text().strip()
                                        value = float(''.join(c for c in value_text if c.isdigit() or c == '.'))
                                        data[target] = value
                                    except:
                                        print(f"Could not extract value for {target}")
            except Exception as e:
                print(f"Error extracting load values: {e}")
            
            # If we couldn't find values for some targets, use None
            for target in self.targets:
                if target not in data:
                    data[target] = None
            
            # Fetch weather data from a free API like OpenWeatherMap
            # (you would need an API key for an actual implementation)
            # This is a placeholder with random values for demonstration
            data['temperature'] = np.random.uniform(20, 35)  # °C
            data['humidity'] = np.random.uniform(30, 90)  # %
            data['wind_speed'] = np.random.uniform(0, 15)  # km/h
            data['precipitation'] = np.random.uniform(0, 5)  # mm
            
            print("Data fetched successfully")
            return data
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return None or a default structure
            return None
    
    def update_cache(self, new_data):
        """
        Update the cache with new data.
        
        Args:
            new_data: Dictionary containing new data to add to cache
        """
        try:
            # Read existing cache
            cache_df = pd.read_csv(self.cache_file)
            
            # Create new record
            new_record = pd.DataFrame([new_data])
            
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
            
            # Convert datetime for JSON serialization
            recent_data['datetime'] = pd.to_datetime(recent_data['datetime'])
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
