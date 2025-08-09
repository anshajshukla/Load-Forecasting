"""
Phase 1.1 - Database Schema Manager
Creates and manages optimized schema for 6-column load data
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load environment variables
load_dotenv()

class DatabaseSchemaManager:
    """Manage database schema for Phase 1.1 with 6-column load data."""
    
    def __init__(self):
        """Initialize schema manager."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print("🗄️ Database Schema Manager for Phase 1.1")
        print("📊 Supporting 6-column load data: DELHI, BRPL, BYPL, NDPL, NDMC, MES")
    
    def create_optimized_schema(self):
        """Create optimized database schema for load forecasting."""
        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            
            # Drop existing tables if they exist
            self.logger.info("🧹 Cleaning existing schema...")
            cur.execute("DROP TABLE IF EXISTS predictions CASCADE")
            cur.execute("DROP TABLE IF EXISTS weather_data CASCADE")
            cur.execute("DROP TABLE IF EXISTS load_data CASCADE")
            
            # Create main load data table with 6 columns
            load_data_schema = """
                CREATE TABLE load_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL UNIQUE,
                    delhi_load FLOAT NOT NULL DEFAULT 0,
                    brpl_load FLOAT NOT NULL DEFAULT 0,
                    bypl_load FLOAT NOT NULL DEFAULT 0,
                    ndpl_load FLOAT NOT NULL DEFAULT 0,
                    ndmc_load FLOAT NOT NULL DEFAULT 0,
                    mes_load FLOAT NOT NULL DEFAULT 0,
                    data_source VARCHAR(50) DEFAULT 'unknown',
                    quality_score FLOAT DEFAULT 1.0,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """
            
            cur.execute(load_data_schema)
            self.logger.info("✅ Created load_data table with 6 load columns")
            
            # Create weather data table
            weather_data_schema = """
                CREATE TABLE weather_data (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL UNIQUE,
                    temperature FLOAT,
                    humidity FLOAT,
                    wind_speed FLOAT,
                    precipitation FLOAT,
                    pressure FLOAT,
                    visibility FLOAT,
                    weather_condition VARCHAR(100),
                    data_source VARCHAR(50) DEFAULT 'openweather',
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
            """
            
            cur.execute(weather_data_schema)
            self.logger.info("✅ Created weather_data table")
            
            # Create predictions table
            predictions_schema = """
                CREATE TABLE predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL,
                    prediction_timestamp TIMESTAMPTZ NOT NULL,
                    delhi_pred FLOAT NOT NULL,
                    brpl_pred FLOAT NOT NULL,
                    bypl_pred FLOAT NOT NULL,
                    ndpl_pred FLOAT NOT NULL,
                    ndmc_pred FLOAT NOT NULL,
                    mes_pred FLOAT NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    confidence_score FLOAT DEFAULT 0.5,
                    horizon_minutes INTEGER NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(prediction_timestamp, model_name, horizon_minutes)
                );
            """
            
            cur.execute(predictions_schema)
            self.logger.info("✅ Created predictions table")
            
            # Create performance indexes
            self.create_indexes(cur)
            
            # Create update triggers
            self.create_triggers(cur)
            
            # Create helpful views
            self.create_views(cur)
            
            # Commit all changes
            conn.commit()
            cur.close()
            conn.close()
            
            self.logger.info("🎉 Database schema created successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error creating schema: {e}")
            return False
    
    def create_indexes(self, cur):
        """Create performance indexes."""
        self.logger.info("🚀 Creating performance indexes...")
        
        indexes = [
            "CREATE INDEX idx_load_data_timestamp ON load_data(timestamp)",
            "CREATE INDEX idx_load_data_source ON load_data(data_source)",
            "CREATE INDEX idx_load_data_delhi ON load_data(delhi_load)",
            "CREATE INDEX idx_weather_timestamp ON weather_data(timestamp)",
            "CREATE INDEX idx_predictions_timestamp ON predictions(prediction_timestamp)",
            "CREATE INDEX idx_predictions_model ON predictions(model_name)",
            "CREATE INDEX idx_predictions_horizon ON predictions(horizon_minutes)"
        ]
        
        for idx_query in indexes:
            cur.execute(idx_query)
        
        self.logger.info("✅ Created performance indexes")
    
    def create_triggers(self, cur):
        """Create update triggers."""
        # Create update trigger for load_data
        trigger_function = """
            CREATE OR REPLACE FUNCTION update_load_data_timestamp()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """
        
        trigger_create = """
            CREATE TRIGGER trigger_update_load_data_timestamp
                BEFORE UPDATE ON load_data
                FOR EACH ROW
                EXECUTE FUNCTION update_load_data_timestamp();
        """
        
        cur.execute(trigger_function)
        cur.execute(trigger_create)
        self.logger.info("✅ Created update timestamp triggers")
    
    def create_views(self, cur):
        """Create helpful views for data analysis."""
        
        # Latest data view
        latest_data_view = """
            CREATE OR REPLACE VIEW latest_load_data AS
            SELECT *
            FROM load_data
            WHERE timestamp = (SELECT MAX(timestamp) FROM load_data);
        """
        
        # Daily summary view
        daily_summary_view = """
            CREATE OR REPLACE VIEW daily_load_summary AS
            SELECT 
                DATE(timestamp) as date,
                AVG(delhi_load) as avg_delhi,
                MAX(delhi_load) as max_delhi,
                MIN(delhi_load) as min_delhi,
                AVG(brpl_load) as avg_brpl,
                AVG(bypl_load) as avg_bypl,
                AVG(ndpl_load) as avg_ndpl,
                AVG(ndmc_load) as avg_ndmc,
                AVG(mes_load) as avg_mes,
                COUNT(*) as data_points
            FROM load_data
            GROUP BY DATE(timestamp)
            ORDER BY date DESC;
        """
        
        # Combined data view (load + weather)
        combined_data_view = """
            CREATE OR REPLACE VIEW combined_load_weather AS
            SELECT 
                l.timestamp,
                l.delhi_load, l.brpl_load, l.bypl_load,
                l.ndpl_load, l.ndmc_load, l.mes_load,
                w.temperature, w.humidity, w.wind_speed, w.precipitation
            FROM load_data l
            LEFT JOIN weather_data w ON DATE_TRUNC('hour', l.timestamp) = DATE_TRUNC('hour', w.timestamp)
            ORDER BY l.timestamp;
        """
        
        cur.execute(latest_data_view)
        cur.execute(daily_summary_view)
        cur.execute(combined_data_view)
        
        self.logger.info("✅ Created helpful views")
    
    def insert_sample_data(self):
        """Insert some sample data for testing."""
        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            
            # Sample load data
            sample_data = """
                INSERT INTO load_data (timestamp, delhi_load, brpl_load, bypl_load, ndpl_load, ndmc_load, mes_load, data_source)
                VALUES 
                    ('2024-01-01 00:00:00+00', 4500.5, 1260.0, 990.0, 810.0, 540.0, 360.0, 'sample'),
                    ('2024-01-01 00:30:00+00', 4200.0, 1176.0, 924.0, 756.0, 504.0, 336.0, 'sample'),
                    ('2024-01-01 01:00:00+00', 3900.5, 1092.0, 858.0, 702.0, 468.0, 312.0, 'sample')
                ON CONFLICT (timestamp) DO NOTHING;
            """
            
            cur.execute(sample_data)
            
            # Sample weather data
            sample_weather = """
                INSERT INTO weather_data (timestamp, temperature, humidity, wind_speed, precipitation)
                VALUES 
                    ('2024-01-01 00:00:00+00', 15.5, 65.0, 8.2, 0.0),
                    ('2024-01-01 01:00:00+00', 14.8, 68.0, 7.5, 0.0),
                    ('2024-01-01 02:00:00+00', 14.2, 70.0, 6.8, 0.0)
                ON CONFLICT (timestamp) DO NOTHING;
            """
            
            cur.execute(sample_weather)
            
            conn.commit()
            cur.close()
            conn.close()
            
            self.logger.info("✅ Sample data inserted")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error inserting sample data: {e}")
            return False
    
    def verify_schema(self):
        """Verify that the schema was created correctly."""
        try:
            conn = psycopg2.connect(self.database_url)
            cur = conn.cursor()
            
            # Check tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            expected_tables = ['load_data', 'weather_data', 'predictions']
            
            print("\n📊 SCHEMA VERIFICATION")
            print("=" * 40)
            
            for table in expected_tables:
                if table in tables:
                    print(f"✅ {table} table exists")
                    
                    # Get column info
                    cur.execute(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                    """)
                    columns = cur.fetchall()
                    print(f"   Columns: {len(columns)}")
                    
                    if table == 'load_data':
                        load_columns = [col[0] for col in columns]
                        expected_load_cols = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
                        for col in expected_load_cols:
                            status = "✅" if col in load_columns else "❌"
                            print(f"   {status} {col}")
                else:
                    print(f"❌ {table} table missing")
            
            cur.close()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying schema: {e}")
            return False


def main():
    """Main function to set up database schema."""
    print("=" * 70)
    print("🚀 PHASE 1.1 - DATABASE SCHEMA SETUP")
    print("=" * 70)
    print("📊 Creating optimized schema for 6-column load data")
    print("🗄️ Target: Supabase PostgreSQL")
    print("=" * 70)
    
    try:
        # Initialize schema manager
        schema_manager = DatabaseSchemaManager()
        
        # Create schema
        print("\n🔨 Creating database schema...")
        if schema_manager.create_optimized_schema():
            print("✅ Schema created successfully!")
            
            # Insert sample data
            print("\n📊 Inserting sample data...")
            schema_manager.insert_sample_data()
            
            # Verify schema
            schema_manager.verify_schema()
            
            print("\n🎉 Phase 1.1 database setup completed!")
            print("✅ Ready for historical data migration")
            
        else:
            print("❌ Schema creation failed!")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
