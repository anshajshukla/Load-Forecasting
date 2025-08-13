"""
Supabase Client Utilities for SIH 2024 Load Forecasting
Provides easy-to-use interfaces for Supabase database operations.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SupabaseClient:
    """Enhanced Supabase client for load forecasting operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Supabase configuration
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_anon_key = os.getenv('SUPABASE_ANON_KEY')
        self.supabase_service_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
        self.database_url = os.getenv('DATABASE_URL')
        
        self.validate_configuration()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
    def validate_configuration(self):
        """Validate that all required Supabase credentials are available."""
        required_vars = {
            'SUPABASE_URL': self.supabase_url,
            'SUPABASE_ANON_KEY': self.supabase_anon_key,
            'DATABASE_URL': self.database_url
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        self.logger.info("✅ Supabase configuration validated")
        
    def get_connection(self, use_dict_cursor=True):
        """Get a direct PostgreSQL connection to Supabase."""
        try:
            if use_dict_cursor:
                conn = psycopg2.connect(
                    self.database_url,
                    cursor_factory=RealDictCursor
                )
            else:
                conn = psycopg2.connect(self.database_url)
            
            self.logger.info("✅ Connected to Supabase PostgreSQL")
            return conn
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Supabase: {str(e)}")
            raise
    
    def test_connection(self):
        """Test the Supabase database connection."""
        try:
            print("🔍 Testing Supabase Connection...")
            print("=" * 50)
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"📍 Database: {version['version'] if isinstance(version, dict) else version[0]}")
            
            # Test current time
            cursor.execute("SELECT NOW();")
            current_time = cursor.fetchone()
            print(f"🕐 Current Time: {current_time['now'] if isinstance(current_time, dict) else current_time[0]}")
            
            # Check if load_data table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'load_data'
                );
            """)
            table_exists = cursor.fetchone()
            table_exists_bool = table_exists['exists'] if isinstance(table_exists, dict) else table_exists[0]
            print(f"📋 load_data table exists: {'✅ Yes' if table_exists_bool else '❌ No'}")
            
            if table_exists_bool:
                # Get table info
                cursor.execute("SELECT COUNT(*) FROM load_data;")
                row_count = cursor.fetchone()
                count = row_count['count'] if isinstance(row_count, dict) else row_count[0]
                print(f"📊 Total records: {count:,}")
                
                # Get date range
                cursor.execute("""
                    SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date 
                    FROM load_data;
                """)
                date_range = cursor.fetchone()
                min_date = date_range['min_date'] if isinstance(date_range, dict) else date_range[0]
                max_date = date_range['max_date'] if isinstance(date_range, dict) else date_range[1]
                if min_date:
                    print(f"📅 Date range: {min_date} to {max_date}")
            
            cursor.close()
            conn.close()
            
            print("=" * 50)
            print("✅ Supabase connection test successful!")
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False
    
    def create_supabase_js_client_example(self):
        """Create example JavaScript code for Supabase client."""
        js_code = f"""
// Supabase JavaScript Client Setup
import {{ createClient }} from '@supabase/supabase-js'

const supabaseUrl = '{self.supabase_url}'
const supabaseKey = '{self.supabase_anon_key}'
const supabase = createClient(supabaseUrl, supabaseKey)

// Example: Fetch load data
async function fetchLoadData() {{
    try {{
        const {{ data, error }} = await supabase
            .from('load_data')
            .select('*')
            .order('timestamp', {{ ascending: false }})
            .limit(100);
        
        if (error) throw error;
        console.log('Load data:', data);
        return data;
    }} catch (error) {{
        console.error('Error fetching data:', error);
    }}
}}

// Example: Insert new load data
async function insertLoadData(loadRecord) {{
    try {{
        const {{ data, error }} = await supabase
            .from('load_data')
            .insert([loadRecord]);
        
        if (error) throw error;
        console.log('Inserted:', data);
        return data;
    }} catch (error) {{
        console.error('Error inserting data:', error);
    }}
}}

// Example: Real-time subscription
const subscription = supabase
    .channel('load_data_changes')
    .on('postgres_changes', {{ 
        event: '*', 
        schema: 'public', 
        table: 'load_data' 
    }}, (payload) => {{
        console.log('Database change:', payload);
    }})
    .subscribe();

export {{ supabase, fetchLoadData, insertLoadData }};
"""
        return js_code
    
    def get_load_data(self, start_date=None, end_date=None, limit=1000):
        """Fetch load data from Supabase with optional date filtering."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            query = "SELECT * FROM load_data"
            params = []
            
            if start_date or end_date:
                query += " WHERE"
                conditions = []
                
                if start_date:
                    conditions.append(" timestamp >= %s")
                    params.append(start_date)
                
                if end_date:
                    conditions.append(" timestamp <= %s")
                    params.append(end_date)
                
                query += " AND".join(conditions)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, params)
            data = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            # Convert to pandas DataFrame
            if data:
                df = pd.DataFrame(data)
                self.logger.info(f"✅ Fetched {len(df)} records from Supabase")
                return df
            else:
                self.logger.warning("⚠️  No data found")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching data: {str(e)}")
            raise
    
    def insert_load_data(self, data_records):
        """Insert load data records into Supabase."""
        try:
            conn = self.get_connection(use_dict_cursor=False)
            cursor = conn.cursor()
            
            # Prepare the insert query
            insert_query = """
                INSERT INTO load_data (
                    timestamp, load_mw, temperature, humidity, 
                    precipitation_mm, wind_speed_kmh, weather_condition
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp) DO UPDATE SET
                    load_mw = EXCLUDED.load_mw,
                    temperature = EXCLUDED.temperature,
                    humidity = EXCLUDED.humidity,
                    precipitation_mm = EXCLUDED.precipitation_mm,
                    wind_speed_kmh = EXCLUDED.wind_speed_kmh,
                    weather_condition = EXCLUDED.weather_condition;
            """
            
            # Execute batch insert
            cursor.executemany(insert_query, data_records)
            conn.commit()
            
            cursor.close()
            conn.close()
            
            self.logger.info(f"✅ Successfully inserted {len(data_records)} records")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error inserting data: {str(e)}")
            raise
    
    def get_table_schema(self, table_name='load_data'):
        """Get the schema information for a table."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            
            schema = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return schema
            
        except Exception as e:
            self.logger.error(f"❌ Error getting schema: {str(e)}")
            raise
    
    def get_connection_info(self):
        """Get detailed connection information."""
        info = {
            'supabase_url': self.supabase_url,
            'database_url': self.database_url,
            'has_anon_key': bool(self.supabase_anon_key),
            'has_service_key': bool(self.supabase_service_key)
        }
        return info

def main():
    """Test the Supabase client."""
    try:
        print("🚀 SIH 2024 - Supabase Client Test")
        print("=" * 60)
        
        # Initialize client
        client = SupabaseClient()
        
        # Test connection
        success = client.test_connection()
        
        if success:
            print("\n📋 Table Schema:")
            schema = client.get_table_schema()
            for col in schema:
                print(f"   {col['column_name']}: {col['data_type']} ({'NULL' if col['is_nullable'] == 'YES' else 'NOT NULL'})")
            
            print("\n📊 Sample Data:")
            df = client.get_load_data(limit=5)
            if not df.empty:
                print(df.head())
            
            print("\n🔗 Connection Info:")
            info = client.get_connection_info()
            for key, value in info.items():
                if 'url' in key:
                    print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {'✅' if value else '❌'}")
            
            print("\n💻 JavaScript Client Code:")
            js_code = client.create_supabase_js_client_example()
            print("   (Saved to supabase_client_example.js)")
            
            # Save JS example
            with open('supabase_client_example.js', 'w') as f:
                f.write(js_code)
        
        print("\n" + "=" * 60)
        print("✅ Supabase client test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()
