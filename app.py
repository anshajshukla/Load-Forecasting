"""
Flask web application for Delhi load forecasting dashboard.
"""
from flask import Flask, render_template, jsonify
import json
import os
import pandas as pd
import threading
import time
from datetime import datetime
import traceback

# Import our prediction modules
from live_prediction import LivePredictor

app = Flask(__name__)

# Global variables
predictor = None
last_update_time = None
update_interval = 3600  # Update interval in seconds (1 hour)
is_updating = False

def load_json_data(file_path, default=None):
    """Load data from a JSON file."""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        print(f"Error loading JSON data from {file_path}: {e}")
        return default

def background_update():
    """Background task to update predictions periodically."""
    global is_updating, last_update_time, predictor
    
    if is_updating:
        return
    
    try:
        is_updating = True
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting background update...")
        
        # Initialize predictor if not already initialized
        if predictor is None:
            predictor = LivePredictor()
        
        # Run prediction cycle
        results = predictor.run_prediction_cycle()
        
        if results:
            last_update_time = datetime.now()
            print(f"Background update completed successfully at {last_update_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("Background update failed")
    except Exception as e:
        print(f"Error in background update: {e}")
        traceback.print_exc()
    finally:
        is_updating = False

def start_background_thread():
    """Start the background update thread."""
    thread = threading.Thread(target=background_update)
    thread.daemon = True
    thread.start()
    return thread

@app.route('/')
def index():
    """Render the dashboard."""
    # Check if an update is needed
    global last_update_time
    
    current_time = datetime.now()
    
    if last_update_time is None or (current_time - last_update_time).total_seconds() >= update_interval:
        start_background_thread()
    
    # Get the last update time string
    update_time_str = "Never" if last_update_time is None else last_update_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template('index.html', last_update=update_time_str)

@app.route('/api/current')
def get_current_data():
    """API endpoint to get current load data."""
    data = load_json_data('data/latest_data.json', {'targets': {}})
    return jsonify(data)

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get load predictions."""
    data = load_json_data('data/predictions.json', {'timestamps': [], 'targets': {}})
    return jsonify(data)

@app.route('/api/update')
def trigger_update():
    """API endpoint to trigger a manual update."""
    start_background_thread()
    return jsonify({'status': 'update started'})

@app.route('/api/status')
def get_status():
    """API endpoint to get update status."""
    global last_update_time, is_updating
    
    return jsonify({
        'is_updating': is_updating,
        'last_update': None if last_update_time is None else last_update_time.strftime('%Y-%m-%d %H:%M:%S')
    })

def create_directories():
    """Create necessary directories."""
    directories = ['data', 'models/saved_models', 'templates', 'static/css', 'static/js']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == '__main__':
    # Create directories
    create_directories()
    
    # Start initial update
    print("Starting initial data update...")
    start_background_thread()
    
    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
