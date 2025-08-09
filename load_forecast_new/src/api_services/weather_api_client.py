"""
API Manager for Weather Data Collection
Handles multiple weather API integrations.
"""

import os
import requests
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Union

class APIManager:
    """Manages weather API integrations and responses."""
    
    def __init__(self):
        """Initialize API manager with credentials."""
        self.logger = logging.getLogger(__name__)
        
        # Weather API configuration - Multiple APIs for better coverage
        self.weather_apis = {
            'openweather': os.getenv('OPENWEATHER_API_KEY'),
            'visual_crossing': os.getenv('VISUAL_CROSSING_API_KEY'),
            'weatherapi': os.getenv('WEATHERAPI_KEY'),
            'weatherunion': os.getenv('WEATHERUNION_API_KEY')
        }
        self.delhi_coords = {"lat": 28.6139, "lon": 77.2090}
        
        # Check which APIs are available
        available_apis = [api for api, key in self.weather_apis.items() if key]
        if available_apis:
            self.logger.info(f"📡 Available weather APIs: {', '.join(available_apis)}")
        else:
            self.logger.warning("⚠️  No weather API keys configured - using simulation mode")
    
    def fetch_historical_weather_api(self, date: datetime, retries: int = 3) -> Optional[Dict]:
        """Fetch historical weather data from multiple weather APIs."""
        if not any(self.weather_apis.values()):
            return None
        
        try:
            # Convert date to timestamp
            timestamp = int(date.timestamp())
            
            # Try OpenWeatherMap Historical API first
            if self.weather_apis['openweather']:
                url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
                params = {
                    'lat': self.delhi_coords['lat'],
                    'lon': self.delhi_coords['lon'],
                    'dt': timestamp,
                    'appid': self.weather_apis['openweather'],
                    'units': 'metric'
                }
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    return self.parse_openweather_response(response.json())
                else:
                    self.logger.warning(f"OpenWeatherMap API failed: {response.status_code}")
                    
        except Exception as e:
            if retries > 0:
                self.logger.warning(f"API request failed, retrying... ({retries} left)")
                return self.fetch_historical_weather_api(date, retries-1)
            else:
                self.logger.error(f"Failed to fetch weather data: {str(e)}")
        
        # Try alternative APIs if OpenWeatherMap fails
        return self.fetch_alternative_weather_apis(date)
    
    def fetch_alternative_weather_apis(self, date: datetime) -> Optional[Dict]:
        """Try alternative weather APIs for historical data."""
        try:
            # Try Visual Crossing Weather API (free tier available)
            if self.weather_apis['visual_crossing']:
                result = self.fetch_visual_crossing_weather(date, self.weather_apis['visual_crossing'])
                if result:
                    return result
            
            # Try WeatherAPI.com (free tier available)
            if self.weather_apis['weatherapi']:
                result = self.fetch_weatherapi_com(date, self.weather_apis['weatherapi'])
                if result:
                    return result
            
            # Try WeatherUnion (Indian weather service - good for Delhi)
            if self.weather_apis['weatherunion']:
                result = self.fetch_weatherunion_api(date, self.weather_apis['weatherunion'])
                if result:
                    return result
            
        except Exception as e:
            self.logger.warning(f"Alternative weather APIs failed: {str(e)}")
        
        return None
    
    def fetch_visual_crossing_weather(self, date: datetime, api_key: str) -> Optional[Dict]:
        """Fetch weather data from Visual Crossing Weather API."""
        try:
            date_str = date.strftime('%Y-%m-%d')
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/28.6139,77.2090/{date_str}"
            
            params = {
                'key': api_key,
                'unitGroup': 'metric',
                'include': 'hours',
                'contentType': 'json'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                return self.parse_visual_crossing_response(response.json(), date)
            else:
                self.logger.warning(f"Visual Crossing API failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Visual Crossing API error: {str(e)}")
        
        return None
    
    def fetch_weatherapi_com(self, date: datetime, api_key: str) -> Optional[Dict]:
        """Fetch weather data from WeatherAPI.com."""
        try:
            date_str = date.strftime('%Y-%m-%d')
            url = f"http://api.weatherapi.com/v1/history.json"
            
            params = {
                'key': api_key,
                'q': f"{self.delhi_coords['lat']},{self.delhi_coords['lon']}",
                'dt': date_str
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                return self.parse_weatherapi_response(response.json(), date)
            else:
                self.logger.warning(f"WeatherAPI.com failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"WeatherAPI.com error: {str(e)}")
        
        return None
    
    def fetch_weatherunion_api(self, date: datetime, api_key: str) -> Optional[Dict]:
        """Fetch weather data from WeatherUnion API (Indian service)."""
        try:
            # WeatherUnion API for historical data (Indian service)
            url = f"https://weatherunion.com/gw/weather/external/v0/get_weatherdata_rg"
            
            headers = {
                'X-Zomato-Api-Key': api_key
            }
            
            params = {
                'locality_id': 'ZWL005764',  # Delhi locality ID
                'device_type': 'android'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                return self.parse_weatherunion_response(response.json(), date)
            else:
                self.logger.warning(f"WeatherUnion API failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"WeatherUnion API error: {str(e)}")
        
        return None
    
    def parse_openweather_response(self, data: Dict) -> Dict:
        """Parse OpenWeatherMap API response."""
        try:
            current = data.get('current', {})
            
            return {
                'temperature': current.get('temp', 25),
                'humidity': current.get('humidity', 60),
                'precipitation_mm': current.get('rain', {}).get('1h', 0),
                'wind_speed_kmh': current.get('wind_speed', 0) * 3.6,  # Convert m/s to km/h
                'wind_direction_deg': current.get('wind_deg', 0),
                'atmospheric_pressure_hpa': current.get('pressure', 1013),
                'cloud_cover_percent': current.get('clouds', 0),
                'visibility_km': current.get('visibility', 10000) / 1000,  # Convert m to km
                'weather_condition': current.get('weather', [{}])[0].get('main', 'Clear'),
                'uv_index': current.get('uvi', 0),
                'dew_point': current.get('dew_point', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to parse OpenWeather response: {str(e)}")
            return {}
    
    def parse_visual_crossing_response(self, data: Dict, target_date: datetime) -> Dict:
        """Parse Visual Crossing Weather API response."""
        try:
            day_data = data.get('days', [{}])[0]
            
            # Get hourly data for the specific hour if available
            hours = day_data.get('hours', [])
            hour_data = day_data  # Fallback to daily data
            
            if hours:
                target_hour = target_date.hour
                for hour in hours:
                    if datetime.strptime(hour.get('datetime', '00:00:00'), '%H:%M:%S').hour == target_hour:
                        hour_data = hour
                        break
            
            return {
                'temperature': hour_data.get('temp', 25),
                'humidity': hour_data.get('humidity', 60),
                'precipitation_mm': hour_data.get('precip', 0),
                'wind_speed_kmh': hour_data.get('windspeed', 0),
                'wind_direction_deg': hour_data.get('winddir', 0),
                'atmospheric_pressure_hpa': hour_data.get('pressure', 1013),
                'cloud_cover_percent': hour_data.get('cloudcover', 0),
                'visibility_km': hour_data.get('visibility', 10),
                'weather_condition': hour_data.get('conditions', 'Clear'),
                'uv_index': hour_data.get('uvindex', 0),
                'dew_point': hour_data.get('dew', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to parse Visual Crossing response: {str(e)}")
            return {}
    
    def parse_weatherapi_response(self, data: Dict, target_date: datetime) -> Dict:
        """Parse WeatherAPI.com response."""
        try:
            forecast_day = data.get('forecast', {}).get('forecastday', [{}])[0]
            day_data = forecast_day.get('day', {})
            
            # Get hourly data for the specific hour if available
            hours = forecast_day.get('hour', [])
            hour_data = day_data  # Fallback to daily data
            
            if hours:
                target_hour = target_date.hour
                for hour in hours:
                    hour_time = datetime.strptime(hour.get('time', ''), '%Y-%m-%d %H:%M')
                    if hour_time.hour == target_hour:
                        hour_data = hour
                        break
            
            return {
                'temperature': hour_data.get('temp_c', 25),
                'humidity': hour_data.get('humidity', 60),
                'precipitation_mm': hour_data.get('precip_mm', 0),
                'wind_speed_kmh': hour_data.get('wind_kph', 0),
                'wind_direction_deg': hour_data.get('wind_degree', 0),
                'atmospheric_pressure_hpa': hour_data.get('pressure_mb', 1013),
                'cloud_cover_percent': hour_data.get('cloud', 0),
                'visibility_km': hour_data.get('vis_km', 10),
                'weather_condition': hour_data.get('condition', {}).get('text', 'Clear'),
                'uv_index': hour_data.get('uv', 0),
                'dew_point': hour_data.get('dewpoint_c', 0)
            }
        except Exception as e:
            self.logger.error(f"Failed to parse WeatherAPI response: {str(e)}")
            return {}
    
    def parse_weatherunion_response(self, data: Dict, target_date: datetime) -> Dict:
        """Parse WeatherUnion API response."""
        try:
            locality_weather_data = data.get('locality_weather_data', {})
            
            return {
                'temperature': locality_weather_data.get('temperature', 25),
                'humidity': locality_weather_data.get('humidity', 60),
                'precipitation_mm': 0,  # Not provided by WeatherUnion
                'wind_speed_kmh': locality_weather_data.get('wind_speed', 0),
                'wind_direction_deg': locality_weather_data.get('wind_direction', 0),
                'atmospheric_pressure_hpa': 1013,  # Default value
                'cloud_cover_percent': 50,  # Default value
                'visibility_km': 10,  # Default value
                'weather_condition': locality_weather_data.get('weather_description', 'Clear'),
                'uv_index': 5,  # Default value
                'dew_point': locality_weather_data.get('temperature', 25) - 5  # Estimate
            }
        except Exception as e:
            self.logger.error(f"Failed to parse WeatherUnion response: {str(e)}")
            return {}
