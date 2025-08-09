"""
Database Operations Manager
Handles all database connections, schema operations, and data persistence.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd

class DatabaseOperationsManager:
    """Manages database operations for weather and solar data."""
    
    def __init__(self):
        """Initialize database operations manager."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("🗃️ Database Operations Manager initialized")
    
    def connect_database(self):
        """Connect to the database."""
        try:
            conn = psycopg2.connect(self.database_url)
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def add_weather_columns_to_schema(self):
        """Add additional weather columns to load_data table."""
        conn = self.connect_database()
        cursor = conn.cursor()
        
        try:
            self.logger.info("Adding enhanced weather columns to database schema...")
            
            # Additional weather columns
            weather_columns = [
                # Precipitation
                "ADD COLUMN IF NOT EXISTS precipitation_mm DECIMAL(6,2) DEFAULT 0",
                "ADD COLUMN IF NOT EXISTS rain_intensity VARCHAR(20) DEFAULT 'None'",  # None, Light, Moderate, Heavy
                "ADD COLUMN IF NOT EXISTS is_raining BOOLEAN DEFAULT FALSE",
                
                # Wind
                "ADD COLUMN IF NOT EXISTS wind_speed_kmh DECIMAL(5,2) DEFAULT 0",
                "ADD COLUMN IF NOT EXISTS wind_direction_deg INTEGER DEFAULT 0",
                "ADD COLUMN IF NOT EXISTS wind_category VARCHAR(20) DEFAULT 'Calm'",  # Calm, Light, Moderate, Strong
                
                # Atmospheric
                "ADD COLUMN IF NOT EXISTS atmospheric_pressure_hpa DECIMAL(7,2) DEFAULT 1013.25",
                "ADD COLUMN IF NOT EXISTS cloud_cover_percent INTEGER DEFAULT 0",
                "ADD COLUMN IF NOT EXISTS visibility_km DECIMAL(4,1) DEFAULT 10.0",
                
                # Weather conditions
                "ADD COLUMN IF NOT EXISTS weather_condition VARCHAR(50) DEFAULT 'Clear'",
                "ADD COLUMN IF NOT EXISTS uv_index DECIMAL(3,1) DEFAULT 0",
                "ADD COLUMN IF NOT EXISTS dew_point DECIMAL(5,2) DEFAULT 0",
                
                # Comfort indices
                "ADD COLUMN IF NOT EXISTS wind_chill DECIMAL(5,2) DEFAULT NULL",
                "ADD COLUMN IF NOT EXISTS apparent_temperature DECIMAL(5,2) DEFAULT NULL",
                "ADD COLUMN IF NOT EXISTS comfort_index DECIMAL(3,2) DEFAULT 0.5",  # 0=very uncomfortable, 1=very comfortable
                
                # Weather impact on load
                "ADD COLUMN IF NOT EXISTS weather_load_factor DECIMAL(3,2) DEFAULT 1.0"  # Multiplier for expected load impact
            ]
            
            for column_sql in weather_columns:
                cursor.execute(f"ALTER TABLE load_data {column_sql};")
            
            conn.commit()
            self.logger.info(f"SUCCESS: Added {len(weather_columns)} enhanced weather columns")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to add weather columns: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def add_solar_columns_to_schema(self):
        """Add solar energy columns to the database schema."""
        conn = self.connect_database()
        cursor = conn.cursor()
        
        try:
            self.logger.info("☀️ Adding solar energy columns to database schema...")
            
            solar_columns = [
                # Solar Irradiance
                "ADD COLUMN IF NOT EXISTS solar_ghi DECIMAL(7,2) DEFAULT 0",  # Global Horizontal Irradiance (W/m²)
                "ADD COLUMN IF NOT EXISTS solar_dni DECIMAL(7,2) DEFAULT 0",  # Direct Normal Irradiance (W/m²)
                "ADD COLUMN IF NOT EXISTS solar_dhi DECIMAL(7,2) DEFAULT 0",  # Diffuse Horizontal Irradiance (W/m²)
                "ADD COLUMN IF NOT EXISTS solar_zenith_angle DECIMAL(5,2) DEFAULT 90",  # Solar zenith angle (degrees)
                
                # Solar Potential and Generation
                "ADD COLUMN IF NOT EXISTS solar_potential_kwh DECIMAL(8,3) DEFAULT 0",  # Daily solar potential (kWh)
                "ADD COLUMN IF NOT EXISTS delhi_solar_capacity_factor DECIMAL(4,3) DEFAULT 0",  # Solar capacity factor (0-1)
                
                # Grid Impact
                "ADD COLUMN IF NOT EXISTS duck_curve_impact DECIMAL(4,3) DEFAULT 0",  # Duck curve effect (0-1)
                "ADD COLUMN IF NOT EXISTS net_load_factor DECIMAL(4,3) DEFAULT 1.0",  # Net load after solar (factor)
                "ADD COLUMN IF NOT EXISTS solar_grid_contribution_mw DECIMAL(8,2) DEFAULT 0",  # Solar contribution to grid (MW)
                
                # Delhi-Specific Solar Metrics
                "ADD COLUMN IF NOT EXISTS delhi_rooftop_solar_mw DECIMAL(8,2) DEFAULT 0",  # Rooftop solar generation
                "ADD COLUMN IF NOT EXISTS delhi_utility_solar_mw DECIMAL(8,2) DEFAULT 0",  # Utility-scale solar
                "ADD COLUMN IF NOT EXISTS solar_load_offset_percent DECIMAL(5,2) DEFAULT 0"  # % of load offset by solar
            ]
            
            for column_sql in solar_columns:
                cursor.execute(f"ALTER TABLE load_data {column_sql};")
            
            conn.commit()
            self.logger.info(f"SUCCESS: Added {len(solar_columns)} solar energy columns")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to add solar columns: {str(e)}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def get_load_data_summary(self) -> Dict:
        """Get summary of load data in the database."""
        conn = self.connect_database()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get basic statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_date,
                    MAX(timestamp) as latest_date,
                    AVG(load_mw) as avg_load,
                    MIN(load_mw) as min_load,
                    MAX(load_mw) as max_load
                FROM load_data
            """)
            summary = dict(cursor.fetchone())
            
            # Get weather data completeness
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_weather_records,
                    COUNT(*) FILTER (WHERE temperature IS NOT NULL) as has_temperature,
                    COUNT(*) FILTER (WHERE humidity IS NOT NULL) as has_humidity,
                    COUNT(*) FILTER (WHERE precipitation_mm IS NOT NULL) as has_precipitation,
                    COUNT(*) FILTER (WHERE wind_speed_kmh IS NOT NULL) as has_wind_speed,
                    COUNT(*) FILTER (WHERE solar_ghi IS NOT NULL) as has_solar_ghi
                FROM load_data
            """)
            weather_summary = dict(cursor.fetchone())
            summary.update(weather_summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get load data summary: {str(e)}")
            return {}
        finally:
            cursor.close()
            conn.close()
    
    def get_weather_data_summary(self) -> Dict:
        """Get comprehensive weather data summary."""
        conn = self.connect_database()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            # Get detailed weather statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    MIN(timestamp) as earliest_timestamp,
                    MAX(timestamp) as latest_timestamp,
                    
                    -- Temperature stats
                    AVG(temperature) as avg_temperature,
                    MIN(temperature) as min_temperature,
                    MAX(temperature) as max_temperature,
                    
                    -- Humidity stats
                    AVG(humidity) as avg_humidity,
                    MIN(humidity) as min_humidity,
                    MAX(humidity) as max_humidity,
                    
                    -- Weather completeness
                    COUNT(*) FILTER (WHERE temperature IS NOT NULL) as has_temperature,
                    COUNT(*) FILTER (WHERE humidity IS NOT NULL) as has_humidity,
                    COUNT(*) FILTER (WHERE precipitation_mm IS NOT NULL) as has_precipitation,
                    COUNT(*) FILTER (WHERE wind_speed_kmh IS NOT NULL) as has_wind_speed,
                    COUNT(*) FILTER (WHERE atmospheric_pressure_hpa IS NOT NULL) as has_pressure,
                    COUNT(*) FILTER (WHERE weather_condition IS NOT NULL) as has_weather_condition,
                    
                    -- Solar data completeness
                    COUNT(*) FILTER (WHERE solar_ghi IS NOT NULL) as has_solar_ghi,
                    COUNT(*) FILTER (WHERE solar_dni IS NOT NULL) as has_solar_dni,
                    COUNT(*) FILTER (WHERE delhi_solar_capacity_factor IS NOT NULL) as has_solar_capacity,
                    
                    -- Average solar values
                    AVG(solar_ghi) as avg_solar_ghi,
                    AVG(delhi_solar_capacity_factor) as avg_solar_capacity_factor,
                    AVG(duck_curve_impact) as avg_duck_curve_impact
                FROM load_data
                WHERE timestamp IS NOT NULL
            """)
            
            summary = dict(cursor.fetchone())
            
            # Calculate completeness percentages
            total = summary['total_records'] or 1  # Avoid division by zero
            for key in summary:
                if key.startswith('has_'):
                    percentage_key = key.replace('has_', '') + '_completeness_percent'
                    summary[percentage_key] = round((summary[key] / total) * 100, 2)
            
            # Get recent data sample
            cursor.execute("""
                SELECT timestamp, load_mw, temperature, humidity, solar_ghi, weather_condition
                FROM load_data 
                WHERE timestamp IS NOT NULL 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_records = [dict(row) for row in cursor.fetchall()]
            summary['recent_records'] = recent_records
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get weather data summary: {str(e)}")
            return {}
        finally:
            cursor.close()
            conn.close()
    
    def update_weather_record(self, timestamp: datetime, weather_data: Dict) -> bool:
        """Update a single weather record in the database."""
        conn = self.connect_database()
        cursor = conn.cursor()
        
        try:
            # Build UPDATE query dynamically based on available data
            set_clauses = []
            values = []
            
            # Weather fields mapping
            field_mapping = {
                'temperature': 'temperature',
                'humidity': 'humidity',
                'precipitation_mm': 'precipitation_mm',
                'wind_speed_kmh': 'wind_speed_kmh',
                'wind_direction_deg': 'wind_direction_deg',
                'atmospheric_pressure_hpa': 'atmospheric_pressure_hpa',
                'cloud_cover_percent': 'cloud_cover_percent',
                'visibility_km': 'visibility_km',
                'weather_condition': 'weather_condition',
                'uv_index': 'uv_index',
                'dew_point': 'dew_point',
                'solar_ghi': 'solar_ghi',
                'solar_dni': 'solar_dni',
                'solar_dhi': 'solar_dhi',
                'solar_zenith_angle': 'solar_zenith_angle',
                'solar_potential_kwh': 'solar_potential_kwh',
                'delhi_solar_capacity_factor': 'delhi_solar_capacity_factor',
                'duck_curve_impact': 'duck_curve_impact',
                'net_load_factor': 'net_load_factor'
            }
            
            for data_key, db_column in field_mapping.items():
                if data_key in weather_data and weather_data[data_key] is not None:
                    set_clauses.append(f"{db_column} = %s")
                    values.append(weather_data[data_key])
            
            if not set_clauses:
                self.logger.warning("No valid weather data to update")
                return False
            
            # Add timestamp to values for WHERE clause
            values.append(timestamp)
            
            query = f"""
                UPDATE load_data 
                SET {', '.join(set_clauses)}
                WHERE timestamp = %s
            """
            
            cursor.execute(query, values)
            conn.commit()
            
            if cursor.rowcount > 0:
                self.logger.debug(f"Updated weather data for {timestamp}")
                return True
            else:
                self.logger.warning(f"No record found for timestamp {timestamp}")
                return False
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to update weather record: {str(e)}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def bulk_update_weather_data(self, weather_records: List[Dict]) -> Dict:
        """Bulk update multiple weather records."""
        conn = self.connect_database()
        cursor = conn.cursor()
        
        results = {
            'updated': 0,
            'failed': 0,
            'errors': []
        }
        
        try:
            for record in weather_records:
                try:
                    timestamp = record.get('timestamp')
                    if not timestamp:
                        continue
                    
                    # Remove timestamp from data dict for update
                    weather_data = {k: v for k, v in record.items() if k != 'timestamp'}
                    
                    if self.update_weather_record(timestamp, weather_data):
                        results['updated'] += 1
                    else:
                        results['failed'] += 1
                        
                except Exception as e:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to update record {record.get('timestamp', 'unknown')}: {str(e)}")
            
            self.logger.info(f"Bulk update completed: {results['updated']} updated, {results['failed']} failed")
            return results
            
        except Exception as e:
            self.logger.error(f"Bulk update failed: {str(e)}")
            results['errors'].append(str(e))
            return results
        finally:
            cursor.close()
            conn.close()
    
    def get_missing_weather_timestamps(self, start_date: datetime = None, end_date: datetime = None) -> List[datetime]:
        """Get timestamps that are missing weather data."""
        conn = self.connect_database()
        cursor = conn.cursor()
        
        try:
            where_clause = "WHERE timestamp IS NOT NULL"
            params = []
            
            if start_date:
                where_clause += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                where_clause += " AND timestamp <= %s"
                params.append(end_date)
            
            # Find records missing key weather data
            query = f"""
                SELECT timestamp 
                FROM load_data 
                {where_clause}
                AND (
                    temperature IS NULL 
                    OR humidity IS NULL 
                    OR weather_condition IS NULL 
                    OR weather_condition = 'Unknown'
                )
                ORDER BY timestamp
            """
            
            cursor.execute(query, params)
            missing_timestamps = [row[0] for row in cursor.fetchall()]
            
            self.logger.info(f"Found {len(missing_timestamps)} timestamps missing weather data")
            return missing_timestamps
            
        except Exception as e:
            self.logger.error(f"Failed to get missing weather timestamps: {str(e)}")
            return []
        finally:
            cursor.close()
            conn.close()
