"""
Main Flask Application - Load Forecasting Dashboard
Centralized entry point for SIH 2024 Load Forecasting System
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Import configuration and core modules
try:
    from config.app_config import CONFIG, setup_logging, validate_config
    from src.core.forecaster import ForecastingEngine
    from src.core.data_processor import DataProcessor
    from src.data.data_fetcher import DelhiSLDCDataFetcher
    from src.core.live_predictor import LivePredictor
    config_loaded = True
except ImportError as e:
    print(f"Warning: Could not import new modules: {e}")
    print("Falling back to legacy implementation...")
    config_loaded = False

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Setup logging
if config_loaded:
    setup_logging()
    validate_config()
    logger = logging.getLogger(__name__)
else:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Global variables for core components
forecasting_engine = None
data_processor = None
live_predictor = None


def initialize_components():
    """Initialize core application components."""
    global forecasting_engine, data_processor, live_predictor
    
    try:
        if config_loaded:
            logger.info("Initializing components with new configuration...")
            
            # Initialize data processor
            data_processor = DataProcessor()
            
            # Initialize forecasting engine
            forecasting_engine = ForecastingEngine()
            
            # Initialize live predictor
            live_predictor = LivePredictor()
            
            logger.info("✅ All components initialized successfully")
        else:
            logger.info("Using legacy components...")
            # Legacy initialization would go here
            
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")
        # Fallback to basic functionality


@app.route('/')
def index():
    """Serve the main dashboard."""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        return jsonify({'error': 'Dashboard temporarily unavailable'}), 500


@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'config_loaded': config_loaded,
            'forecasting_engine': forecasting_engine is not None,
            'data_processor': data_processor is not None,
            'live_predictor': live_predictor is not None
        }
    })


@app.route('/api/live-prediction')
def get_live_prediction():
    """Get live load forecast predictions."""
    try:
        if not live_predictor:
            return jsonify({'error': 'Live predictor not available'}), 503
        
        # Run prediction cycle
        results = live_predictor.run_prediction_cycle()
        
        if results:
            return jsonify({
                'status': 'success',
                'data': results
            })
        else:
            return jsonify({'error': 'Failed to generate predictions'}), 500
            
    except Exception as e:
        logger.error(f"Error in live prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical-forecast', methods=['POST'])
def get_historical_forecast():
    """Generate forecast for historical data."""
    try:
        if not forecasting_engine:
            return jsonify({'error': 'Forecasting engine not available'}), 503
        
        data = request.get_json()
        
        # Extract parameters
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        forecast_horizon = data.get('forecast_horizon', 24)
        
        # Generate forecast
        forecast_results = forecasting_engine.generate_forecast(
            start_date=start_date,
            end_date=end_date,
            forecast_horizon=forecast_horizon
        )
        
        return jsonify({
            'status': 'success',
            'data': forecast_results
        })
        
    except Exception as e:
        logger.error(f"Error in historical forecast: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-status')
def get_model_status():
    """Get status of trained models."""
    try:
        status = {
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        if forecasting_engine:
            status['models'] = forecasting_engine.get_model_status()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/data-stats')
def get_data_stats():
    """Get data statistics and quality metrics."""
    try:
        if not data_processor:
            return jsonify({'error': 'Data processor not available'}), 503
        
        stats = data_processor.get_data_statistics()
        
        return jsonify({
            'status': 'success',
            'data': stats
        })
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/train-model', methods=['POST'])
def train_model():
    """Trigger model training."""
    try:
        if not forecasting_engine:
            return jsonify({'error': 'Forecasting engine not available'}), 503
        
        data = request.get_json()
        model_type = data.get('model_type', 'gru')
        
        # Start training (this could be async in production)
        training_results = forecasting_engine.train_forecasting_models(
            model_type=model_type
        )
        
        return jsonify({
            'status': 'success',
            'data': training_results
        })
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/latest-data')
def get_latest_data():
    """Get latest live data from Delhi SLDC."""
    try:
        # Create temporary data fetcher
        fetcher = DelhiSLDCDataFetcher()
        
        # Fetch current data
        sldc_data = fetcher.fetch_sldc_data()
        weather_data = fetcher.get_weather_data()
        
        if sldc_data:
            combined_data = {
                **sldc_data,
                **weather_data,
                'fetch_timestamp': datetime.now().isoformat()
            }
            
            return jsonify({
                'status': 'success',
                'data': combined_data
            })
        else:
            return jsonify({'error': 'Failed to fetch data'}), 500
            
    except Exception as e:
        logger.error(f"Error fetching latest data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files."""
    return send_from_directory('static', filename)


@app.route('/api/config')
def get_config():
    """Get application configuration (non-sensitive parts)."""
    try:
        if config_loaded:
            public_config = {
                'app': {
                    'name': CONFIG['app']['name'],
                    'version': CONFIG['app']['version'],
                    'debug': CONFIG['app']['debug']
                },
                'model': {
                    'default_model_type': CONFIG['model']['default_model_type'],
                    'sequence_length': CONFIG['model']['sequence_length'],
                    'prediction_horizon': CONFIG['model']['prediction_horizon']
                }
            }
        else:
            public_config = {
                'app': {
                    'name': 'Load Forecasting System',
                    'version': '1.0.0',
                    'debug': False
                }
            }
        
        return jsonify(public_config)
        
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize components
    initialize_components()
    
    # Get configuration
    if config_loaded:
        host = CONFIG['app']['host']
        port = CONFIG['app']['port']
        debug = CONFIG['app']['debug']
    else:
        host = '0.0.0.0'
        port = 5000
        debug = True
    
    logger.info(f"🚀 Starting Load Forecasting Dashboard")
    logger.info(f"🌐 Server: http://{host}:{port}")
    logger.info(f"🔧 Debug mode: {debug}")
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug
    )
