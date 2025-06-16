# Load Forecasting System

A machine learning-based system for forecasting electricity load demand in Delhi's power distribution companies.

## Features

- Real-time load forecasting for multiple distribution companies (DELHI, BRPL, BYPL, NDMC, MES)
- Live data fetching from Delhi SLDC
- Interactive dashboard for visualization
- Historical data analysis
- Model training and evaluation tools

## Project Structure

```
├── app.py                 # Main application entry point
├── fetch_live_data.py     # Live data fetching module
├── live_prediction.py     # Real-time prediction module
├── main.py               # Core functionality
├── run_dashboard.py      # Dashboard server
├── requirements.txt      # Project dependencies
├── static/              # Static assets for dashboard
├── templates/           # HTML templates
└── src/                 # Source code modules
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/anshajshukla/Load-Forecasting.git
cd Load-Forecasting
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the dashboard:
```bash
python run_dashboard.py
```

2. Access the dashboard at `http://localhost:5000`

3. For live predictions:
```bash
python live_prediction.py
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Pandas
- NumPy
- Flask
- Other dependencies listed in requirements.txt

## License

MIT License

## Author

Anshaj Shukla 