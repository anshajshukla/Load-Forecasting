# Phase 4: Evaluation

This phase covers the evaluation of model performance using a comprehensive set of metrics and visualizations. The logic is implemented in the `evaluate_model` function and related utilities.

## Steps and Techniques

### 1. Making Predictions on Test Data
- **Model Inference:** Uses the trained model to predict on the test set.
- **Inverse Scaling:** Applies the saved target scalers to convert predictions and true values back to the original scale.

### 2. Calculating Metrics
- **Per-Target Metrics:** For each target (e.g., DELHI, BRPL, BYPL, NDMC, MES), computes:
  - **RMSE (Root Mean Squared Error):** Measures average prediction error magnitude.
  - **MAPE (Mean Absolute Percentage Error):** Measures average percentage error, clipped to avoid outliers.
  - **R² (R-squared):** Proportion of variance explained by the model.
  - **MAE (Mean Absolute Error):** Average absolute error.
  - **Max Error:** Largest absolute error.
  - **MedAE (Median Absolute Error):** Median of absolute errors.
  - **Explained Variance:** How much of the variance is explained by the model.
  - **MSLE (Mean Squared Log Error):** Penalizes underestimation more than overestimation.
  - **Poisson Deviance:** For count data, measures deviance from Poisson distribution.
  - **Gamma Deviance:** For positive continuous data, measures deviance from Gamma distribution.
- **Overall Metrics:** Averages of RMSE, MAPE, and R² across all targets.

### 3. Saving Metrics
- **CSV Output:** Saves all metrics to CSV files for each model type.
- **Pickle Output:** Saves metrics as pickle files for later comparison.
- **Bar Charts:** Generates bar charts (e.g., MAPE by region) and saves as PNG.
- **Heatmaps:** Generates annotated heatmaps (e.g., MAPE by region) for visual comparison.

### 4. Visualizing Predictions
- **Prediction Plots:** Plots and saves comparison graphs of true vs. predicted values for each target.
- **MAPE Comparison:** Bar chart and heatmap for MAPE across regions.
- **All-Metric Visualization:** Subplots for each metric, sorted for clarity.

### 5. Output
- Evaluation metrics in CSV and pickle format
- Visualizations of model performance (PNG)
- Summary of model strengths and weaknesses

---

**Key Libraries Used:**
- numpy, pandas, matplotlib, seaborn, scikit-learn (metrics), pickle

**Functions:**
- `evaluate_model`, `save_metrics`, `save_evaluation_metrics`, `visualize_predictions`, `mean_absolute_percentage_error`

**Best Practices:**
- Uses a wide range of metrics for robust evaluation
- Visualizes results for both technical and non-technical audiences
- Saves all results for reproducibility and comparison
- Modular and reusable for different models and datasets 