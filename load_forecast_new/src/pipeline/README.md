# Delhi SLDC Load Forecasting Data Pipeline

This comprehensive data pipeline automatically collects 5 years of historical data from Delhi SLDC, trains deep learning models, and provides real-time validation and accuracy monitoring.

## 🏗️ Architecture Overview

The pipeline consists of four main components:

### 1. **Enhanced Data Fetcher** (`enhanced_fetcher.py`)
- **Purpose**: Automated historical data collection from Delhi SLDC website
- **Features**:
  - Parallel processing for faster data collection
  - 5-year historical data fetching
  - Real-time data updates every 5 minutes
  - Rate limiting and retry mechanisms
  - Weather data integration
  - Data quality validation

### 2. **Training Pipeline** (`training_pipeline.py`)
- **Purpose**: Automated model training with deep learning
- **Features**:
  - GRU-based neural networks for each target (DELHI, BRPL, BYPL, NDMC, MES)
  - Advanced feature engineering (time features, lag features, weather)
  - Automatic hyperparameter optimization
  - Model checkpointing and early stopping
  - Performance visualization and metrics

### 3. **Validation Pipeline** (`validation_pipeline.py`)
- **Purpose**: Real-time accuracy monitoring and validation
- **Features**:
  - Continuous accuracy tracking
  - Data quality assessment
  - Alert system for performance degradation
  - Model drift detection
  - Automated reporting

### 4. **Database Manager** (`database/db_manager.py`)
- **Purpose**: Centralized data storage and management
- **Features**:
  - SQLite for development, PostgreSQL for production
  - Structured schema for training sessions and predictions
  - Validation logging and performance tracking
  - Data backup and recovery

## 🚀 Quick Start

### Prerequisites
```bash
# Install required packages
pip install tensorflow pandas numpy scikit-learn beautifulsoup4 requests matplotlib seaborn schedule joblib
```

### 1. Initialize the Pipeline
```bash
cd data_pipeline
python main_orchestrator.py --action start
```

This will:
- ✅ Create database tables
- ✅ Test data fetching with recent dates
- ✅ Collect 5 years of historical data (this takes several hours)
- ✅ Train initial models for all targets
- ✅ Setup automated monitoring and validation

### 2. Monitor Pipeline Status
```bash
python main_orchestrator.py --action status
```

### 3. Manual Operations
```bash
# Manual retraining
python main_orchestrator.py --action retrain

# Manual data fetch (last 7 days)
python main_orchestrator.py --action fetch --days 7
```

## 📊 Data Collection Details

### Historical Data Sources
- **Primary URL**: `https://www.delhisldc.org/Loaddata.aspx?mode=DD/MM/YYYY`
- **Real-time URL**: `https://www.delhisldc.org/Redirect.aspx?Loc=0805`
- **Data Points**: 179 hourly records per day
- **Targets**: DELHI, BRPL, BYPL, NDMC, MES (load in MW)
- **Weather**: Temperature, Humidity, Wind Speed, Precipitation

### Data Pipeline Flow
```
Delhi SLDC Website → Enhanced Fetcher → Database → Training Pipeline → Trained Models
                                                ↓
Real-time Data → Validation Pipeline → Accuracy Monitoring → Alerts/Reports
```

## 🤖 Model Architecture

### GRU-Based Neural Networks
- **Input**: 24-hour sequences with weather and time features
- **Output**: 24-hour load predictions
- **Architecture**: 
  - GRU layers: [128, 64, 32] units
  - Dropout: 0.2 for regularization
  - Dense layers: [64, 32] → output
  - Optimizer: Adam with learning rate scheduling

### Feature Engineering
- **Time Features**: Hour, day of week, month, season (cyclical encoding)
- **Lag Features**: 1-hour, 24-hour, 168-hour (weekly) lags
- **Weather Integration**: Real-time and seasonal patterns
- **Load Aggregations**: Sum, mean, standard deviation across targets

## 📈 Validation & Monitoring

### Real-time Validation
- **Frequency**: Every hour
- **Metrics**: Accuracy (%), MAE, MSE, RMSE, R², MAPE
- **Data Quality**: Missing data detection, outlier identification
- **Alerts**: Low accuracy, high error, data quality issues

### Performance Thresholds
- ✅ **Target Accuracy**: ≥85%
- ✅ **Max Error**: ≤500 MW
- ✅ **Data Quality**: ≥80%
- ⚠️ **Retraining Trigger**: <80% accuracy

### Automated Schedules
- **Data Updates**: Every 5 minutes
- **Validation**: Every hour
- **Reports**: Daily at 6:00 AM
- **Retraining**: Weekly (configurable)

## 📁 File Structure

```
data_pipeline/
├── main_orchestrator.py      # Main pipeline coordinator
├── enhanced_fetcher.py       # Data collection module
├── training_pipeline.py      # Model training module
├── validation_pipeline.py    # Validation and monitoring
├── README.md                 # This file
├── config.json              # Configuration settings
├── database/
│   └── db_manager.py         # Database management
├── models/
│   ├── trained_models/       # Saved H5 model files
│   ├── scalers/             # Preprocessing scalers
│   ├── metrics/             # Training metrics and history
│   └── plots/               # Visualization outputs
├── raw_data/                # Raw CSV data files
├── validation/              # Validation reports
└── logs/                    # System logs
```

## 🔧 Configuration

### Pipeline Settings (`config.json`)
```json
{
  "fetch_historical_years": 5,
  "fetch_parallel_workers": 3,
  "retrain_schedule": "weekly",
  "validation_interval_minutes": 60,
  "min_accuracy_threshold": 85.0,
  "max_error_threshold": 500.0,
  "enable_real_time_monitoring": true,
  "enable_alerts": true
}
```

### Training Configuration
```python
TrainingConfig(
    sequence_length=24,        # Hours to look back
    prediction_horizon=24,     # Hours to predict ahead
    validation_split=0.2,      # 20% for validation
    test_split=0.1,           # 10% for testing
    batch_size=32,
    epochs=100,
    learning_rate=0.001,
    gru_units=[128, 64, 32],
    dropout_rate=0.2,
    patience=15                # Early stopping
)
```

## 📊 Expected Performance

### Historical Training Results
Based on 5 years of Delhi SLDC data:

| Target | Expected Accuracy | Typical MAE | Use Case |
|--------|------------------|-------------|-----------|
| DELHI | 90-95% | 200-400 MW | Main grid load |
| BRPL | 88-93% | 100-200 MW | BSES Rajdhani |
| BYPL | 87-92% | 80-150 MW | BSES Yamuna |
| NDMC | 85-90% | 20-50 MW | New Delhi Municipal |
| MES | 83-88% | 5-15 MW | Military stations |

### Data Collection Timeline
- **5 Years of Data**: ~6,500 days × 179 records = 1.16M+ records
- **Collection Time**: 8-12 hours (with rate limiting)
- **Storage Size**: ~500MB for complete dataset
- **Update Frequency**: Real-time every 5 minutes

## 🚨 Monitoring & Alerts

### Alert Types
1. **Low Accuracy**: Model performance below 85%
2. **High Error**: MAE exceeds 500 MW
3. **Data Quality**: Missing or corrupted data
4. **Model Drift**: Performance degradation over time
5. **System Failure**: Multiple target failures

### Health Checks
- Database connectivity
- Model file availability
- Recent data availability
- Validation pipeline status

## 🔄 Automated Operations

### Daily Tasks
- ✅ Fetch yesterday's complete data
- ✅ Validate current models
- ✅ Generate performance reports
- ✅ Check data quality metrics

### Weekly Tasks
- ✅ Retrain models with new data
- ✅ Performance comparison analysis
- ✅ Update baseline performance metrics
- ✅ Clean old log files

### Real-time Tasks
- ✅ Fetch current load data every 5 minutes
- ✅ Validate predictions every hour
- ✅ Monitor system health continuously
- ✅ Alert on performance issues immediately

## 🎯 Use Cases

### 1. **Grid Operations Center**
- Real-time load forecasting dashboard
- 24-hour ahead planning
- Peak demand predictions
- Emergency response planning

### 2. **Research & Development**
- Historical trend analysis
- Weather impact studies
- Load pattern recognition
- Model performance benchmarking

### 3. **Energy Trading**
- Price forecasting support
- Supply-demand balancing
- Market operation planning
- Risk management

## 📝 Troubleshooting

### Common Issues

**1. Data Collection Fails**
```bash
# Check network connectivity
python -c "import requests; print(requests.get('https://www.delhisldc.org').status_code)"

# Test with single date
cd data_pipeline
python enhanced_fetcher.py
```

**2. Training Fails**
```bash
# Check data availability
python -c "from database.db_manager import create_database_manager; print(create_database_manager().execute_query('SELECT COUNT(*) FROM historical_load_data'))"

# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

**3. Low Accuracy Issues**
- Check data quality scores
- Verify website structure hasn't changed
- Review training logs for overfitting
- Consider retraining with more data

### Logs Location
- **Pipeline Logs**: `data_pipeline/logs/pipeline.log`
- **Training Logs**: `data_pipeline/models/metrics/`
- **Validation Logs**: Database `validation_logs` table

## 🚀 Next Steps

After successful setup:

1. **Integration**: Connect to your existing dashboard
2. **Customization**: Adjust thresholds and schedules
3. **Expansion**: Add more weather data sources
4. **Optimization**: Fine-tune model architectures
5. **Deployment**: Scale to production environment

## 📞 Support

For issues or questions:
1. Check the logs first: `data_pipeline/logs/pipeline.log`
2. Verify configuration: `config.json`
3. Test individual components separately
4. Review database contents for data issues

---

**Status**: ✅ Ready for 5-year data collection and automated forecasting!

The pipeline is designed to run continuously, automatically improving accuracy as more data becomes available. Initial setup takes 8-12 hours for complete historical data collection, after which the system operates autonomously with minimal maintenance required.
