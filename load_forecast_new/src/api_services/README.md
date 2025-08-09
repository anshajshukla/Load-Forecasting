# API Services

Streamlined real-time data collection for SIH 2024 Load Forecasting.

## Components

### Live Data Fetchers
- `live_data_fetcher.py` - Delhi SLDC real-time load data
- `weather_data_fetcher.py` - Weather API data collection

### Database Operations  
- `database_manager.py` - Schema management and data storage
- `db_operations.py` - Core database operations

### Data Validation
- `data_validator.py` - Data quality validation
- `weather_validator.py` - Weather data validation

### API Clients
- `weather_api_client.py` - Weather API client management

## Usage

```python
from src.api_services import APIServicesManager

# Initialize services
api_manager = APIServicesManager()
api_manager.initialize_services()

# Fetch and store latest data
success = api_manager.fetch_and_store_latest_data()
```

## Purpose

This module handles only real-time data operations:
- Fetching current load data from Delhi SLDC
- Fetching current weather data from APIs
- Validating incoming data
- Storing new data in database
- No historical data migration (already completed)
