# Delhi Load Forecasting Project

## Quick Start Guide

This guide will help you execute the Delhi load forecasting project pipeline, starting with Phase 3: Model Development & Training.

## Project Structure

The project is organized into several phases:
- Phase 1: Data Integration
- Phase 2: Feature Engineering & Validation
- Phase 3: Model Development & Training (current phase)
- Phase 4: Model Evaluation & Selection
- Phase 5: Deployment

## Prerequisites

1. Python 3.8 or higher
2. Required packages (install using `pip install -r requirements.txt` in each phase directory)
3. Prepared dataset from Phase 2.5 Feature Validation

## Running Phase 3: Model Development & Training

Phase 3 is divided into four weeks:
1. Week 1: Baseline Establishment (Ridge, Lasso, Random Forest, XGBoost)
2. Week 2: Neural Networks (LSTM, GRU)
3. Week 3: Advanced Architectures (Hybrid models, Attention mechanisms)
4. Week 4: Optimization & Deployment preparation

### Step 1: Execute Week 1 (Baseline Establishment)

Navigate to the project directory and run:

```bash
cd load_forecast_new/phase_3_week_1_model_development
python scripts/run_week1_pipeline.py
```

This script will:
1. Automatically locate the required dataset from Phase 2.5
2. Execute the complete Week 1 pipeline in sequence:
   - Day 1-2: Data Preparation Pipeline
   - Day 3-4: Linear & Tree-Based Baselines
   - Day 3-4: Gradient Boosting & Time Series Baselines
   - Day 7: Baseline Evaluation & Documentation
3. Generate a summary report and execution log

### Step 2: Review Week 1 Results

After completion, review the following files:
- `WEEK_1_EXECUTION_SUMMARY.json`: Overview of execution results
- `results/`: Directory containing model performance metrics
- `reports/WEEK1_EXECUTIVE_SUMMARY.md`: Detailed findings and recommendations

### Step 3: Proceed to Week 2 (Neural Networks)

Once Week 1 is complete, proceed to Week 2:

```bash
cd ../phase_3_week_2_neural_networks
python scripts/00_week2_fast_implementation.py
```

This will execute the neural network implementation pipeline.

### Step 4: Execute Week 3 (Advanced Architectures)

After Week 2 completion:

```bash
cd ../phase_3_week_3_advanced_architectures
python scripts/00_week3_advanced_architectures_pipeline.py
```

### Step 5: Complete Phase 3 with Week 4 (Optimization & Deployment)

Finally, execute Week 4:

```bash
cd ../phase_3_week_4_optimization_deployment
python scripts/00_week4_optimization_deployment_pipeline.py
```

## Troubleshooting

### Missing Dataset
If the script cannot find the dataset automatically:
1. Locate the `delhi_selected_features.csv` file from Phase 2.5
2. Provide the full path when prompted by the script

### Package Installation Issues
If you encounter package installation issues:

```bash
# Install core dependencies
pip install pandas numpy scikit-learn xgboost

# For Prophet installation issues
pip install pystan==2.19.1.1
pip install prophet

# For TensorFlow
pip install tensorflow
```

### Memory Issues
For memory-intensive operations:
- Reduce `n_jobs` parameter in code
- Close other applications
- Consider using a machine with at least 16GB RAM

## Dashboard Visualization

After Phase 3 completion, you can run the dashboard:

```bash
cd ../delhi_forecasting_dashboard
python main.py
```

## Next Steps After Phase 3

After completing Phase 3, proceed to Phase 4 (Model Evaluation & Selection):

```bash
cd ../phase_4_model_evaluation_selection
python scripts/01_comprehensive_evaluation.py
```

## Project Documentation

For more detailed information about each phase, refer to:
- `docs/`: Contains detailed documentation for each phase
- `PROJECT_FLOW.txt`: Complete project roadmap and timeline