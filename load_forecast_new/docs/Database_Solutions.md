# SIH 2024 Cloud Database Solutions
## Free Production-Level Database Options for Load Forecasting

### 🎯 Recommended Solution: **PostgreSQL on Supabase**

#### Why Supabase for SIH 2024?
- **500MB Free Database** + **2GB Additional Storage**
- **Production-ready PostgreSQL** with real-time capabilities
- **Built-in APIs** for easy integration
- **Real-time subscriptions** perfect for spike alerts
- **Dashboard & Analytics** built-in
- **No credit card required** for free tier

#### Setup Process:
1. **Sign up**: https://supabase.com
2. **Create project**: Select closest region (Asia - Mumbai for Delhi)
3. **Get connection details**: Database URL, API keys
4. **Configure**: Update your pipeline to use Supabase

---

## 🔄 Alternative Options Comparison

### 1. **Supabase (PostgreSQL)** ⭐⭐⭐⭐⭐
```
✅ Free Tier: 500MB DB + 2GB storage + 2GB bandwidth
✅ Real-time capabilities (perfect for spike detection)
✅ Built-in dashboard and analytics
✅ RESTful API auto-generated
✅ Production-grade PostgreSQL
✅ Easy scaling path
✅ No credit card required
```

### 2. **MongoDB Atlas** ⭐⭐⭐⭐
```
✅ Free Tier: 512MB storage
✅ Global clusters
✅ Built-in analytics
✅ JSON document storage
❌ Requires credit card verification
❌ Lower storage limit
```

### 3. **PlanetScale (MySQL)** ⭐⭐⭐⭐
```
✅ Free Tier: 5GB storage + 1 billion row reads/month
✅ Serverless MySQL
✅ Branching for database schema changes
❌ MySQL limitations for time-series data
❌ Requires credit card
```

### 4. **CockroachDB Serverless** ⭐⭐⭐⭐
```
✅ Free Tier: 5GB storage + 250M Request Units/month
✅ Distributed SQL database
✅ PostgreSQL compatibility
❌ Complex for simple use cases
❌ Requires credit card
```

### 5. **Firebase Firestore** ⭐⭐⭐
```
✅ Free Tier: 1GB storage + 50K reads/day
✅ Real-time updates
✅ Google Cloud integration
❌ Document-based (not ideal for time-series)
❌ Limited querying capabilities
❌ Expensive for large datasets
```

---

## 🏆 Final Recommendation: **Supabase Setup**

### Immediate Implementation Steps

#### Step 1: Create Supabase Project
```bash
# 1. Go to https://supabase.com
# 2. Click "Start your project"
# 3. Create account (GitHub/Google login recommended)
# 4. Create new project:
#    - Name: "delhi-load-forecasting-sih2024"
#    - Database Password: [strong password]
#    - Region: "Asia Pacific (Mumbai)" [closest to Delhi]
```

#### Step 2: Configure Database Schema
```sql
-- Historical Load Data (Main table)
CREATE TABLE historical_load_data (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMPTZ NOT NULL,
    weekday VARCHAR(10),
    delhi FLOAT,
    brpl FLOAT,
    bypl FLOAT,
    ndmc FLOAT,
    mes FLOAT,
    temperature FLOAT,
    humidity FLOAT,
    wind_speed FLOAT,
    precipitation FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for time-based queries
CREATE INDEX idx_historical_datetime ON historical_load_data(datetime);
CREATE INDEX idx_historical_date ON historical_load_data(DATE(datetime));

-- SIH 2024 Enhanced Tables
CREATE TABLE spike_alerts (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    target VARCHAR(10) NOT NULL,
    magnitude FLOAT NOT NULL,
    spike_type VARCHAR(20) NOT NULL,
    severity VARCHAR(10) DEFAULT 'medium',
    description TEXT,
    weather_factor FLOAT DEFAULT 1.0,
    holiday_factor FLOAT DEFAULT 1.0
);

CREATE TABLE peak_forecasts (
    id BIGSERIAL PRIMARY KEY,
    forecast_time TIMESTAMPTZ DEFAULT NOW(),
    target VARCHAR(10) NOT NULL,
    predicted_peak FLOAT NOT NULL,
    confidence FLOAT DEFAULT 0.8,
    horizon_hours INTEGER DEFAULT 24,
    weather_conditions JSONB,
    holiday_impact FLOAT DEFAULT 1.0
);

CREATE TABLE holiday_impacts (
    id BIGSERIAL PRIMARY KEY,
    holiday_date DATE NOT NULL,
    holiday_name VARCHAR(100) NOT NULL,
    impact_type VARCHAR(20) NOT NULL,
    load_reduction_percent FLOAT DEFAULT 0.0,
    affected_targets TEXT[],
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enhanced features table for ML
CREATE TABLE enhanced_features (
    id BIGSERIAL PRIMARY KEY,
    datetime TIMESTAMPTZ NOT NULL,
    is_holiday BOOLEAN DEFAULT FALSE,
    holiday_type VARCHAR(20),
    days_to_holiday INTEGER DEFAULT 0,
    is_peak_hour BOOLEAN DEFAULT FALSE,
    season VARCHAR(15),
    heat_index FLOAT,
    weather_stress_factor FLOAT,
    spike_probability FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Real-time monitoring table
CREATE TABLE system_health (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    component VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    accuracy FLOAT,
    response_time_ms INTEGER,
    error_count INTEGER DEFAULT 0,
    metadata JSONB
);
```

#### Step 3: Update Database Configuration
```python
# Create: config/supabase_config.py
import os
from supabase import create_client, Client

class SupabaseConfig:
    """Supabase configuration for SIH 2024 production deployment."""
    
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")  # From Supabase dashboard
        self.key = os.getenv("SUPABASE_ANON_KEY")  # From Supabase dashboard
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # For admin operations
        
    def get_client(self) -> Client:
        """Get Supabase client for data operations."""
        return create_client(self.url, self.key)
    
    def get_admin_client(self) -> Client:
        """Get admin client for schema operations."""
        return create_client(self.url, self.service_role_key)

# Database connection settings
DATABASE_CONFIG = {
    'postgresql': {
        'host': 'db.[your-project-ref].supabase.co',
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres',
        'password': '[your-password]',
        'sslmode': 'require'
    }
}
```

---

## 🔧 Implementation Changes Required

### 1. Update Database Manager
**File:** `data_pipeline/database/db_manager.py`

```python
import psycopg2
import pandas as pd
from supabase import create_client
import os
from typing import Optional, Dict, Any

class ProductionDatabaseManager:
    """Production database manager using Supabase PostgreSQL."""
    
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY")
        self.db_url = os.getenv("DATABASE_URL")  # PostgreSQL connection string
        
        # Initialize Supabase client
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
    def get_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(self.db_url)
    
    def execute_query(self, query: str, params=None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def insert_data(self, table_name: str, data: Dict[str, Any]) -> bool:
        """Insert data using Supabase client for real-time updates."""
        try:
            result = self.supabase.table(table_name).insert(data).execute()
            return len(result.data) > 0
        except Exception as e:
            print(f"Insert error: {e}")
            return False
    
    def bulk_insert(self, table_name: str, data_list: list) -> bool:
        """Bulk insert for historical data migration."""
        try:
            # Use PostgreSQL COPY for large datasets
            df = pd.DataFrame(data_list)
            with self.get_connection() as conn:
                df.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
            return True
        except Exception as e:
            print(f"Bulk insert error: {e}")
            return False
```

### 2. Environment Variables Setup
**File:** `.env`
```bash
# Supabase Configuration (Get from Supabase Dashboard)
SUPABASE_URL=https://[your-project-ref].supabase.co
SUPABASE_ANON_KEY=[your-anon-key]
SUPABASE_SERVICE_ROLE_KEY=[your-service-role-key]
DATABASE_URL=postgresql://postgres:[password]@db.[your-project-ref].supabase.co:5432/postgres

# Weather API Keys (existing)
OPENWEATHERMAP_API_KEY=[your-key]
WEATHERAPI_KEY=[your-key]

# SIH 2024 Configuration
SIH_ENVIRONMENT=production
DATA_COLLECTION_INTERVAL=15
SPIKE_DETECTION_THRESHOLD=1.5
```

### 3. Data Migration Script
**File:** `scripts/migrate_to_supabase.py`
```python
"""
Script to migrate existing SQLite data to Supabase PostgreSQL.
"""
import sqlite3
import pandas as pd
from data_pipeline.database.db_manager import ProductionDatabaseManager
import os
from tqdm import tqdm

def migrate_historical_data():
    """Migrate historical data from SQLite to Supabase."""
    
    # Connect to existing SQLite database
    sqlite_path = 'data_pipeline/database/load_forecasting.db'
    if not os.path.exists(sqlite_path):
        print("SQLite database not found. Starting fresh.")
        return
    
    # Initialize production database
    prod_db = ProductionDatabaseManager()
    
    # Read from SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    
    try:
        # Get historical data
        print("Reading historical data from SQLite...")
        df = pd.read_sql_query("SELECT * FROM historical_load_data", sqlite_conn)
        print(f"Found {len(df)} historical records")
        
        # Migrate in batches to avoid memory issues
        batch_size = 1000
        total_batches = len(df) // batch_size + 1
        
        print("Migrating to Supabase PostgreSQL...")
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i+batch_size]
            batch_data = batch.to_dict('records')
            
            success = prod_db.bulk_insert('historical_load_data', batch_data)
            if not success:
                print(f"Failed to migrate batch {i//batch_size + 1}")
                break
        
        print("✅ Historical data migration completed!")
        
    except Exception as e:
        print(f"Migration error: {e}")
    finally:
        sqlite_conn.close()

if __name__ == "__main__":
    migrate_historical_data()
```

---

## 📊 Storage Calculation & Optimization

### Data Size Estimation (15-minute intervals)
```
Daily records: 96 (every 15 minutes)
Monthly records: 2,880
Yearly records: 35,040
5-year records: 175,200

Record size estimate:
- Basic fields: ~200 bytes per record
- Enhanced features: ~300 bytes per record
- Total per record: ~500 bytes

5-year storage: 175,200 × 500 bytes = ~87.6 MB
With indexes and metadata: ~150 MB
```

### Storage Optimization Strategies
1. **Data Compression**: Use PostgreSQL's built-in compression
2. **Partitioning**: Partition tables by month/year
3. **Archival**: Move old data to cheaper storage after 2 years
4. **Efficient Indexing**: Only essential indexes

### Supabase Usage Monitoring
```python
# Add to your monitoring dashboard
def get_database_usage():
    """Monitor Supabase usage for SIH 2024."""
    return {
        'storage_used_mb': get_storage_usage(),
        'requests_today': get_request_count(),
        'bandwidth_used_mb': get_bandwidth_usage(),
        'remaining_quota': calculate_remaining_quota()
    }
```

---

## 🚀 Implementation Priority for Database Migration

### Week 1: Database Setup
- [ ] **Day 1**: Create Supabase account and project
- [ ] **Day 2**: Configure database schema
- [ ] **Day 3**: Update database manager code
- [ ] **Day 4**: Test connection and basic operations
- [ ] **Day 5**: Migrate existing data (if any)
- [ ] **Day 6**: Update all pipeline components
- [ ] **Day 7**: Test end-to-end pipeline

### Production Benefits
✅ **Scalability**: Handle 2-4GB easily, can scale to 8GB on free tier  
✅ **Performance**: Real-time queries and updates  
✅ **Reliability**: 99.9% uptime guarantee  
✅ **Monitoring**: Built-in dashboard and metrics  
✅ **Security**: Row-level security and authentication  
✅ **APIs**: Auto-generated REST and GraphQL APIs  

---

## 🎯 SIH 2024 Competition Advantages

### Technical Advantages
1. **Production Database**: Real production-grade PostgreSQL
2. **Real-time Capabilities**: Live spike detection and alerts
3. **Scalability**: Can handle increased load during demo
4. **Professional Setup**: Shows understanding of production requirements

### Demo Benefits
1. **Live Dashboard**: Real-time updates during presentation
2. **Cloud Access**: Accessible from anywhere for judges
3. **Performance**: Fast queries and responses
4. **Reliability**: No "it works on my machine" issues

---

**Recommendation**: Start with **Supabase** immediately. It's the perfect balance of features, reliability, and cost for SIH 2024, and you can have it running within a few hours!

Would you like me to help you set up the Supabase configuration and migration scripts right away?
