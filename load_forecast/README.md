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

## Model Explanation

The load forecasting model employs a Long Short-Term Memory (LSTM) neural network architecture, which is particularly well-suited for sequence prediction problems like time series forecasting. The model processes historical load data, along with relevant features such as temperature, humidity, and time-based indicators (e.g., hour of day, day of week), to learn complex patterns and trends in electricity consumption.

**Key Components:**
- **Data Preprocessing:** Raw data is cleaned, normalized, and engineered into features suitable for the LSTM model. This includes handling missing values, scaling numerical features, and creating lagged features and rolling statistics.
- **LSTM Layers:** Multiple LSTM layers are stacked to capture both short-term and long-term dependencies within the time series data. Dropout layers are often included to prevent overfitting.
- **Output Layer:** A dense layer with a linear activation function is used as the output layer to produce the forecasted load values.
- **Training:** The model is trained using historical data, minimizing a loss function (e.g., Mean Squared Error) through an optimization algorithm (e.g., Adam).

## Results Explanation

The forecasting system provides predictions for electricity load demand, which are visualized on the Streamlit dashboard.

**Interpretation of Results:**
- **Forecasted Load:** The dashboard displays the predicted load for upcoming time intervals. This helps power distribution companies anticipate demand and manage resources effectively.
- **Historical vs. Predicted:** The dashboard also allows for comparison of historical load data with the model's predictions, enabling users to assess the accuracy and performance of the model over time.
- **Error Metrics:** (If implemented) You would typically evaluate the model's performance using metrics like Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), or Mean Absolute Percentage Error (MAPE). These metrics quantify the difference between predicted and actual values, indicating the reliability of the forecasts.
- **Impact:** Accurate load forecasting helps in optimizing power generation, reducing operational costs, minimizing transmission losses, and ensuring grid stability. The system aims to provide timely and precise forecasts to aid in better decision-making for energy management. 