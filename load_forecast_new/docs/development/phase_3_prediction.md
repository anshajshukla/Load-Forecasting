# Phase 3: Prediction

This phase describes how trained models are used to make predictions on new or future data. The logic is implemented in several functions for preparing data, making predictions, and saving results.

## Steps and Techniques

### 1. Preparing Prediction Data
- **Feature Extraction:** Extracts the required features from the input DataFrame.
- **Scaling:** Applies the saved feature scaler to transform the features to the scale used during training.
- **Reshaping:** Reshapes the data to (samples, timesteps=1, features) for RNN model compatibility.

### 2. Making Predictions
- **Model Inference:**
  - Uses the trained model to predict target values.
  - Supports different model types (e.g., GRU, LSTM) with appropriate batch handling.
- **Inverse Scaling:**
  - Applies the saved target scalers to convert predictions back to the original scale.
- **Result Formatting:**
  - Returns predictions as a pandas DataFrame for easy analysis and saving.

### 3. Iterative Future Forecasting
- **Step-by-Step Prediction:**
  - For multi-step forecasting (e.g., next 24 hours), iteratively predicts each step using the last available data.
  - Updates lag features and time-based features for each new prediction.
  - Handles updating of lag_1 and lag_24 features as predictions progress.
- **Timestamp Management:**
  - Calculates the next timestamp for each prediction step, handling hourly and 2-hour intervals.
- **Result Aggregation:**
  - Collects all future predictions and timestamps into a DataFrame.

### 4. Saving and Visualizing Results
- **CSV Output:**
  - Saves predictions to CSV files, including model type and timestamp in filenames.
- **Plotting:**
  - Generates and saves plots of predicted load for each target over time.
  - Uses matplotlib for high-quality visualizations.
- **Forecast Plot Customization:**
  - Customizes colors, labels, and layout for clarity.
  - Adds generation timestamp to plots for traceability.

### 5. GPU/Resource Management
- **TensorFlow GPU Memory Growth:**
  - Attempts to enable memory growth for TensorFlow if a GPU is available, to avoid memory allocation issues.

### 6. Output
- Prediction CSV files for future load values
- Visualizations of predicted load over time (PNG)

---

**Key Libraries Used:**
- pandas, numpy, matplotlib, tensorflow, datetime

**Functions:**
- `prepare_prediction_data`, `predict_load`, `predict_future`, `save_predictions`

**Best Practices:**
- Ensures predictions are on the same scale as training
- Handles iterative forecasting with lag and time feature updates
- Provides clear, timestamped outputs for traceability
- Modular and reusable for different models and forecasting horizons 