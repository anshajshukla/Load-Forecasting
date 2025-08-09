"""
Application Configuration
Main configuration settings for the SIH 2024 Load Forecasting System
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models' / 'saved_models'
LOGS_DIR = PROJECT_ROOT / 'logs'
SIH_DIR = PROJECT_ROOT / 'sih_2024'

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Application settings
APP_CONFIG = {
    'name': 'SIH 2024 Load Forecasting System',
    'version': '1.0.0',
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
    'host': os.getenv('HOST', '0.0.0.0'),
    'port': int(os.getenv('PORT', 5000)),
    'environment': os.getenv('ENVIRONMENT', 'development'),
    'secret_key': os.getenv('SECRET_KEY', 'sih-2024-load-forecasting'),
    'timezone': 'Asia/Kolkata'
}

# Database configuration - Optimized for Supabase Transaction Pooler
DATABASE_CONFIG = {
    'type': 'postgresql',  # Using Supabase PostgreSQL
    'url': os.getenv('DATABASE_URL', ''),
    'supabase_url': os.getenv('SUPABASE_URL', ''),
    'supabase_anon_key': os.getenv('SUPABASE_ANON_KEY', ''),
    'supabase_service_key': os.getenv('SUPABASE_SERVICE_ROLE_KEY', ''),
    
    # Connection pool settings optimized for Transaction Pooler
    'pool_size': 10,  # Smaller pool for transaction pooler
    'max_overflow': 15,  # Conservative overflow
    'pool_timeout': 20,  # Shorter timeout for stateless operations
    'pool_recycle': 3600,  # 1 hour recycle for transaction pooler
    'pool_pre_ping': True,  # Verify connections before use
    
    # Transaction pooler specific settings
    'connect_args': {
        'sslmode': 'require',
        'options': '-c default_transaction_isolation=read_committed'
    },
    
    'echo': APP_CONFIG['debug']
}

# Model configuration
MODEL_CONFIG = {
    'default_models': ['gru', 'lstm'],
    'ensemble_enabled': True,
    'auto_retrain': True,
    'retrain_interval_hours': 24,
    'forecast_horizon': 24,
    'sequence_length': 24,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'test_split': 0.1,
    'model_save_format': 'h5',
    'backup_models': True
}

# Data processing configuration
DATA_CONFIG = {
    'target_column': 'load_demand',
    'timestamp_column': 'timestamp',
    'required_columns': ['timestamp', 'load_demand'],
    'optional_columns': ['temperature', 'humidity', 'wind_speed', 'solar_irradiance'],
    'missing_value_threshold': 0.1,
    'outlier_detection': True,
    'feature_engineering': True,
    'normalization': True,
    'data_validation': True,
    'cache_processed_data': True
}

# API configuration
API_CONFIG = {
    'version': 'v1',
    'title': 'SIH 2024 Load Forecasting API',
    'description': 'Advanced Load Forecasting API for Smart Grid Management',
    'rate_limiting': True,
    'requests_per_minute': 100,
    'cors_enabled': True,
    'cors_origins': ['*'],
    'authentication': {
        'enabled': True,
        'type': 'api_key',
        'header_name': 'X-API-Key'
    },
    'response_format': 'json',
    'cache_responses': True,
    'cache_timeout': 300  # 5 minutes
}

# Weather integration configuration
WEATHER_CONFIG = {
    'enabled': True,
    'api_key': os.getenv('WEATHER_API_KEY', ''),
    'provider': 'openweathermap',
    'base_url': 'https://api.openweathermap.org/data/2.5',
    'locations': {
        'delhi': {'lat': 28.6139, 'lon': 77.2090}
    },
    'update_interval_minutes': 60,
    'forecast_days': 2,
    'features': ['temperature', 'humidity', 'wind_speed', 'pressure', 'visibility']
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'app.log'),
            'mode': 'a'
        },
        'error_file': {
            'class': 'logging.FileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'error.log'),
            'mode': 'a'
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG' if APP_CONFIG['debug'] else 'INFO',
            'propagate': False
        },
        'werkzeug': {
            'handlers': ['console'],
            'level': 'WARNING',
            'propagate': False
        },
        'tensorflow': {
            'handlers': ['file'],
            'level': 'WARNING',
            'propagate': False
        }
    }
}

# Security configuration
SECURITY_CONFIG = {
    'csrf_protection': True,
    'secure_headers': True,
    'rate_limiting': True,
    'input_validation': True,
    'sql_injection_protection': True,
    'xss_protection': True,
    'api_key_required': True,
    'session_timeout_minutes': 30,
    'max_request_size_mb': 10
}

# Monitoring and alerts configuration
MONITORING_CONFIG = {
    'health_checks_enabled': True,
    'metrics_collection': True,
    'performance_monitoring': True,
    'error_tracking': True,
    'alert_thresholds': {
        'prediction_error_threshold': 0.15,  # 15% MAPE
        'system_load_threshold': 0.8,  # 80% CPU/Memory
        'response_time_threshold': 5.0,  # 5 seconds
        'error_rate_threshold': 0.05  # 5% error rate
    },
    'notification_channels': {
        'email': os.getenv('ALERT_EMAIL', ''),
        'webhook': os.getenv('ALERT_WEBHOOK', ''),
        'slack': os.getenv('SLACK_WEBHOOK', '')
    }
}

# SIH 2024 specific configuration
SIH_CONFIG = {
    'competition_mode': True,
    'innovation_features': {
        'real_time_processing': True,
        'advanced_analytics': True,
        'ai_explanability': True,
        'mobile_optimization': True,
        'cloud_scalability': True
    },
    'demo_mode': {
        'enabled': os.getenv('DEMO_MODE', 'False').lower() == 'true',
        'sample_data_only': True,
        'limited_features': False
    },
    'evaluation_metrics': [
        'mae', 'mse', 'rmse', 'mape', 'r2',
        'peak_accuracy', 'load_factor_prediction',
        'renewable_integration_score'
    ],
    'presentation_assets': {
        'slides_path': SIH_DIR / 'docs' / 'presentation',
        'demo_videos': SIH_DIR / 'docs' / 'videos',
        'technical_documentation': SIH_DIR / 'docs'
    }
}

# Feature flags
FEATURE_FLAGS = {
    'weather_integration': True,
    'holiday_calendar': True,
    'spike_detection': True,
    'ensemble_models': True,
    'real_time_updates': True,
    'mobile_api': True,
    'advanced_visualizations': True,
    'automated_reports': True,
    'multi_region_support': True,
    'renewable_energy_integration': True
}

# Aggregated configuration
CONFIG = {
    'app': APP_CONFIG,
    'database': DATABASE_CONFIG,
    'model': MODEL_CONFIG,
    'data': DATA_CONFIG,
    'api': API_CONFIG,
    'weather': WEATHER_CONFIG,
    'logging': LOGGING_CONFIG,
    'security': SECURITY_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'sih': SIH_CONFIG,
    'features': FEATURE_FLAGS,
    'paths': {
        'project_root': PROJECT_ROOT,
        'data_dir': DATA_DIR,
        'models_dir': MODELS_DIR,
        'logs_dir': LOGS_DIR,
        'sih_dir': SIH_DIR
    }
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary
    
    Returns:
        Complete configuration dictionary
    """
    return CONFIG

def get_database_url() -> str:
    """
    Get the database URL for SQLAlchemy
    
    Returns:
        Database connection URL
    """
    if DATABASE_CONFIG['url']:
        return DATABASE_CONFIG['url']
    
    return (f"postgresql://{DATABASE_CONFIG['username']}:"
           f"{DATABASE_CONFIG['password']}@"
           f"{DATABASE_CONFIG['host']}:"
           f"{DATABASE_CONFIG['port']}/"
           f"{DATABASE_CONFIG['database']}")

def validate_config() -> bool:
    """
    Validate configuration settings
    
    Returns:
        True if configuration is valid
    """
    required_env_vars = []
    
    # Check database configuration
    if not DATABASE_CONFIG['url'] and not DATABASE_CONFIG['password']:
        required_env_vars.append('DATABASE_URL or DB_PASSWORD')
    
    # Check weather API key if weather integration is enabled
    if WEATHER_CONFIG['enabled'] and not WEATHER_CONFIG['api_key']:
        required_env_vars.append('WEATHER_API_KEY')
    
    if required_env_vars:
        print("⚠️  Missing required environment variables:")
        for var in required_env_vars:
            print(f"   - {var}")
        return False
    
    print("✅ Configuration validated successfully!")
    return True

def setup_logging():
    """Setup logging configuration"""
    import logging.config
    logging.config.dictConfig(LOGGING_CONFIG)

if __name__ == "__main__":
    print(f"🚀 {APP_CONFIG['name']} v{APP_CONFIG['version']}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    
    # Validate configuration
    validate_config()
    
    # Setup logging
    setup_logging()
    
    print("⚙️ Configuration loaded successfully!")
