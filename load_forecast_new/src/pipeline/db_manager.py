"""
Database configuration and schema for Delhi SLDC Load Forecasting System.
Supports both SQLite (for development) and PostgreSQL (for production).
"""

import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Optional, List, Dict, Tuple
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations for load forecasting data."""
    
    def __init__(self, db_type='sqlite', db_config=None):
        """
        Initialize database manager.
        
        Args:
            db_type: 'sqlite' or 'postgresql'
            db_config: Database configuration dict
        """
        self.db_type = db_type
        self.db_config = db_config or {}
        self.connection = None
        
        # Default SQLite config
        if db_type == 'sqlite':
            self.db_path = self.db_config.get('db_path', 'database/delhi_sldc.db')
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self._create_tables()
    
    def connect(self):
        """Establish database connection."""
        try:
            if self.db_type == 'sqlite':
                self.connection = sqlite3.connect(self.db_path)
                self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            elif self.db_type == 'postgresql':
                self.connection = psycopg2.connect(
                    host=self.db_config.get('host', 'localhost'),
                    database=self.db_config.get('database', 'delhi_sldc'),
                    user=self.db_config.get('user', 'postgres'),
                    password=self.db_config.get('password', ''),
                    port=self.db_config.get('port', 5432),
                    cursor_factory=RealDictCursor
                )
            logger.info(f"Connected to {self.db_type} database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def _create_tables(self):
        """Create necessary tables for the system."""
        self.connect()
        
        cursor = self.connection.cursor()
        
        # Historical load data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_load_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TIMESTAMP UNIQUE NOT NULL,
                date_str VARCHAR(20) NOT NULL,
                time_str VARCHAR(10) NOT NULL,
                weekday VARCHAR(15) NOT NULL,
                hour INTEGER NOT NULL,
                minute INTEGER NOT NULL,
                delhi_load REAL,
                brpl_load REAL,
                bypl_load REAL,
                ndmc_load REAL,
                mes_load REAL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                precipitation REAL,
                data_source VARCHAR(50) DEFAULT 'delhi_sldc',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model training sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name VARCHAR(100) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                training_start TIMESTAMP NOT NULL,
                training_end TIMESTAMP,
                data_start_date DATE NOT NULL,
                data_end_date DATE NOT NULL,
                total_records INTEGER,
                train_accuracy REAL,
                validation_accuracy REAL,
                test_accuracy REAL,
                model_params TEXT,  -- JSON string
                model_path VARCHAR(255),
                status VARCHAR(20) DEFAULT 'training',  -- training, completed, failed
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER REFERENCES training_sessions(id),
                prediction_datetime TIMESTAMP NOT NULL,
                actual_datetime TIMESTAMP NOT NULL,
                delhi_predicted REAL,
                brpl_predicted REAL,
                bypl_predicted REAL,
                ndmc_predicted REAL,
                mes_predicted REAL,
                delhi_actual REAL,
                brpl_actual REAL,
                bypl_actual REAL,
                ndmc_actual REAL,
                mes_actual REAL,
                delhi_accuracy REAL,
                brpl_accuracy REAL,
                bypl_accuracy REAL,
                ndmc_accuracy REAL,
                mes_accuracy REAL,
                overall_mape REAL,
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Data validation logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                datetime TIMESTAMP NOT NULL,
                validation_type VARCHAR(50) NOT NULL,  -- range_check, trend_check, model_validation
                target_region VARCHAR(20) NOT NULL,
                expected_value REAL,
                actual_value REAL,
                validation_result VARCHAR(20) NOT NULL,  -- passed, failed, warning
                error_message TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # System metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name VARCHAR(100) NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit VARCHAR(20),
                measurement_time TIMESTAMP NOT NULL,
                context_data TEXT,  -- JSON string for additional context
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
        logger.info("Database tables created successfully")
        
        # Create indexes for better performance
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for better query performance."""
        cursor = self.connection.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_historical_datetime ON historical_load_data(datetime)",
            "CREATE INDEX IF NOT EXISTS idx_historical_date ON historical_load_data(date_str)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_session ON model_predictions(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_predictions_datetime ON model_predictions(prediction_datetime)",
            "CREATE INDEX IF NOT EXISTS idx_validation_datetime ON validation_logs(datetime)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_time ON system_metrics(measurement_time)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        self.connection.commit()
        logger.info("Database indexes created successfully")
    
    def insert_historical_data(self, df: pd.DataFrame) -> int:
        """
        Insert historical load data into database.
        
        Args:
            df: DataFrame containing historical data
            
        Returns:
            Number of records inserted
        """
        self.connect()
        cursor = self.connection.cursor()
        
        inserted_count = 0
        
        for _, row in df.iterrows():
            try:
                if self.db_type == 'sqlite':
                    cursor.execute("""
                        INSERT OR REPLACE INTO historical_load_data 
                        (datetime, date_str, time_str, weekday, hour, minute,
                         delhi_load, brpl_load, bypl_load, ndmc_load, mes_load,
                         temperature, humidity, wind_speed, precipitation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('datetime'),
                        row.get('date', ''),
                        row.get('time_str', ''),
                        row.get('weekday', ''),
                        row.get('hour', 0),
                        row.get('minute', 0),
                        row.get('DELHI'),
                        row.get('BRPL'),
                        row.get('BYPL'),
                        row.get('NDMC'),
                        row.get('MES'),
                        row.get('temperature'),
                        row.get('humidity'),
                        row.get('wind_speed'),
                        row.get('precipitation')
                    ))
                
                inserted_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to insert record {row.get('datetime')}: {e}")
                continue
        
        self.connection.commit()
        logger.info(f"Inserted {inserted_count} historical records")
        return inserted_count
    
    def get_training_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Retrieve training data for specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing training data
        """
        self.connect()
        
        query = """
            SELECT * FROM historical_load_data 
            WHERE DATE(datetime) BETWEEN ? AND ?
            ORDER BY datetime
        """
        
        df = pd.read_sql_query(query, self.connection, params=(start_date, end_date))
        logger.info(f"Retrieved {len(df)} training records from {start_date} to {end_date}")
        
        return df
    
    def create_training_session(self, session_name: str, model_type: str, 
                               data_start: str, data_end: str, 
                               model_params: dict) -> int:
        """
        Create a new training session record.
        
        Args:
            session_name: Name of the training session
            model_type: Type of model (e.g., 'GRU', 'LSTM', 'CNN')
            data_start: Start date of training data
            data_end: End date of training data
            model_params: Model parameters dictionary
            
        Returns:
            Session ID
        """
        self.connect()
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO training_sessions 
            (session_name, model_type, training_start, data_start_date, data_end_date, model_params)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_name,
            model_type,
            datetime.now(),
            data_start,
            data_end,
            json.dumps(model_params)
        ))
        
        session_id = cursor.lastrowid
        self.connection.commit()
        
        logger.info(f"Created training session {session_id}: {session_name}")
        return session_id
    
    def update_training_session(self, session_id: int, **kwargs):
        """Update training session with results."""
        self.connect()
        cursor = self.connection.cursor()
        
        # Build dynamic update query
        update_fields = []
        values = []
        
        for key, value in kwargs.items():
            update_fields.append(f"{key} = ?")
            values.append(value)
        
        if update_fields:
            update_fields.append("updated_at = ?")
            values.append(datetime.now())
            values.append(session_id)
            
            query = f"UPDATE training_sessions SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)
            self.connection.commit()
            
            logger.info(f"Updated training session {session_id}")
    
    def insert_predictions(self, session_id: int, predictions_df: pd.DataFrame):
        """Insert model predictions into database."""
        self.connect()
        cursor = self.connection.cursor()
        
        for _, row in predictions_df.iterrows():
            cursor.execute("""
                INSERT INTO model_predictions 
                (session_id, prediction_datetime, actual_datetime,
                 delhi_predicted, brpl_predicted, bypl_predicted, ndmc_predicted, mes_predicted,
                 delhi_actual, brpl_actual, bypl_actual, ndmc_actual, mes_actual,
                 delhi_accuracy, brpl_accuracy, bypl_accuracy, ndmc_accuracy, mes_accuracy,
                 overall_mape, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                row.get('prediction_datetime'),
                row.get('actual_datetime'),
                row.get('delhi_predicted'),
                row.get('brpl_predicted'),
                row.get('bypl_predicted'),
                row.get('ndmc_predicted'),
                row.get('mes_predicted'),
                row.get('delhi_actual'),
                row.get('brpl_actual'),
                row.get('bypl_actual'),
                row.get('ndmc_actual'),
                row.get('mes_actual'),
                row.get('delhi_accuracy'),
                row.get('brpl_accuracy'),
                row.get('bypl_accuracy'),
                row.get('ndmc_accuracy'),
                row.get('mes_accuracy'),
                row.get('overall_mape'),
                row.get('model_version', '1.0')
            ))
        
        self.connection.commit()
        logger.info(f"Inserted {len(predictions_df)} prediction records for session {session_id}")
    
    def log_validation(self, datetime_val: datetime, validation_type: str, 
                      target_region: str, expected: float, actual: float, 
                      result: str, error_msg: str = None, confidence: float = None):
        """Log validation results."""
        self.connect()
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO validation_logs 
            (datetime, validation_type, target_region, expected_value, actual_value,
             validation_result, error_message, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime_val, validation_type, target_region, expected, actual,
            result, error_msg, confidence
        ))
        
        self.connection.commit()
    
    def get_recent_accuracy_trend(self, days: int = 30) -> pd.DataFrame:
        """Get recent accuracy trends for analysis."""
        self.connect()
        
        query = """
            SELECT 
                DATE(prediction_datetime) as date,
                AVG(overall_mape) as avg_mape,
                AVG(delhi_accuracy) as avg_delhi_acc,
                AVG(brpl_accuracy) as avg_brpl_acc,
                AVG(bypl_accuracy) as avg_bypl_acc,
                AVG(ndmc_accuracy) as avg_ndmc_acc,
                AVG(mes_accuracy) as avg_mes_acc,
                COUNT(*) as prediction_count
            FROM model_predictions 
            WHERE prediction_datetime >= DATE('now', '-{} days')
            GROUP BY DATE(prediction_datetime)
            ORDER BY date DESC
        """.format(days)
        
        return pd.read_sql_query(query, self.connection)
    
    def cleanup_old_data(self, keep_days: int = 365):
        """Clean up old data to maintain performance."""
        self.connect()
        cursor = self.connection.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        # Keep historical load data (it's valuable for retraining)
        # But clean up old validation logs and system metrics
        cursor.execute("""
            DELETE FROM validation_logs 
            WHERE created_at < ?
        """, (cutoff_date,))
        
        cursor.execute("""
            DELETE FROM system_metrics 
            WHERE created_at < ?
        """, (cutoff_date,))
        
        self.connection.commit()
        logger.info(f"Cleaned up data older than {keep_days} days")

# Database configuration examples
SQLITE_CONFIG = {
    'db_path': 'database/delhi_sldc.db'
}

POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'database': 'delhi_sldc',
    'user': 'postgres',
    'password': 'your_password',
    'port': 5432
}

# Factory function for easy database creation
def create_database_manager(use_postgresql: bool = False) -> DatabaseManager:
    """Create appropriate database manager based on configuration."""
    if use_postgresql:
        return DatabaseManager(db_type='postgresql', db_config=POSTGRESQL_CONFIG)
    else:
        return DatabaseManager(db_type='sqlite', db_config=SQLITE_CONFIG)

if __name__ == "__main__":
    # Test database setup
    db = create_database_manager(use_postgresql=False)
    print("Database setup completed successfully!")
