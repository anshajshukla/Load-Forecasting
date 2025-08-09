# SIH 2024 Load Forecasting System - Documentation

## 📋 Project Overview
This project provides a comprehensive load forecasting system for Delhi's power grid using machine learning models. It includes historical data migration, real-time data processing, model training, and prediction visualization.

## 📁 Documentation Structure

### 📂 Main Documentation
- **System_Architecture.md** - Overall system design and architecture
- **Implementation_Roadmap.md** - Project roadmap and phase planning
- **API_Documentation.md** - API endpoints and usage guides

### 📂 Phases Documentation
- **phases/** - Phase-specific setup guides and technical documentation
  - `README.md` - Phase overview and progress tracking
  - `phase_1_1_setup_guide.md` - Historical data migration setup
  - Future phase guides as development progresses

### 📂 Learning Documentation
- **../learning/** - Comprehensive learning documentation
  - `phases/` - Phase-specific learning and insights
  - Implementation summaries and lessons learned

## 🚀 Quick Start

### Current Phase: 1.1 - Historical Data Migration

1. **Setup Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Database**:
   ```bash
   # Create .env file with your Supabase credentials
   DATABASE_URL=postgresql://[user]:[password]@[host]:[port]/[database]
   ```

3. **Run Phase 1.1 Setup**:
   ```bash
   python scripts/run_phase_1_1.py
   ```

## 📊 System Components
- **`run_dashboard.py`**: Starts the Flask web server and serves the dashboard.
- **`fetch_live_data.py`**: Fetches live data from the Delhi SLDC website and updates the data cache.
- **`generate_prediction_graphs.py`**: Generates prediction graphs for each model and saves them.
- **`test_model.py`**: Tests and compares different model architectures, providing performance metrics.

## Usage Instructions
- **Run the Dashboard**:
  ```bash
  python run_dashboard.py
  ```
  Access the dashboard at `http://127.0.0.1:5000`.

- **Generate Prediction Graphs**:
  Visit `/generate-graphs` on the dashboard to generate graphs for all models.

- **Calculate Model Accuracy**:
  Run `test_model.py` to evaluate models and obtain accuracy metrics:
  ```bash
  python test_model.py --compare
  ```

## Checking Individual Model Accuracies
To check the accuracy of individual models and compare them, follow these steps:

1. **Run the Model Comparison Script**:
   Use the `model_compare.py` script to evaluate models and generate a comparison report.
   ```bash
   python model_compare.py
   ```
   This will output the results to `model_comparison_results.txt`, including metrics like MSE, MAPE, R², and accuracy percentages.

2. **Interpret the Results**:
   Open `model_comparison_results.txt` to view the performance metrics for each model. The file will list:
   - **MSE**: Mean Squared Error
   - **MAPE**: Mean Absolute Percentage Error
   - **R²**: R-squared score
   - **Accuracy (Validation)**: Accuracy on the validation dataset
   - **Accuracy (New Data)**: Accuracy on new data

3. **Identify the Best Model**:
   The script identifies the best model based on the lowest MAPE. Review the metrics to determine which model best fits your needs.

## Troubleshooting
- **Missing Packages**: Ensure all dependencies are installed by running `pip install -r requirements.txt`.
- **Tcl/Tk Errors**: Use the `Agg` backend for `matplotlib` to avoid display issues.

For further assistance, please consult the code comments or contact the project maintainer.

## Directory Structure Update

All files and scripts related to model training, evaluation, and learning have been moved to a new `learning/` folder for better organization. Please refer to the `learning/` directory for all learning-related code, models, and results.
