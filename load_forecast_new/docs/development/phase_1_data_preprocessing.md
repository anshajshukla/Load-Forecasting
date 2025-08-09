# Phase 1: Data Preprocessing

This phase prepares the raw data for model training and evaluation. The process is implemented in the `DataProcessor` class and uses libraries such as pandas, numpy, and scikit-learn.

## Steps and Techniques

### 1. Loading Data
- **File Reading:** Loads the dataset from a CSV file using `pandas.read_csv`.
- **Datetime Parsing:** Converts the 'datetime' column to pandas datetime objects with a specific format (`%d-%m-%Y %H:%M`).
- **Indexing:** Sets the 'datetime' column as the DataFrame index for time-based operations.

### 2. Feature Engineering
- **Categorical to Numerical:** Converts the 'weekday' column (if present) from string (e.g., 'Monday') to integer (0=Monday, ..., 6=Sunday) using a mapping.
- **Time-based Features:** Extracts features from the datetime index:
  - `hour`: Hour of the day
  - `day_of_week`: Day of the week (0=Monday)
  - `month`: Month of the year
  - `is_weekend`: Boolean indicating if the day is Saturday or Sunday

### 3. Lag Features
- **Lag Creation:** For each target variable (`DELHI`, `BRPL`, `BYPL`, `NDMC`, `MES`), creates lag features:
  - `{target}_lag_1`: Value from the previous time step
  - `{target}_lag_24`: Value from 24 time steps ago
- **NaN Handling:** Drops rows with NaN values introduced by lagging.

### 4. Feature Selection
- **Categorical Features:** `hour`, `day_of_week`, `month`, `is_weekend`, `weekday_num` (if present)
- **Numerical Features:** `temperature`, `humidity`, `wind_speed`, `precipitation`, and all lag features
- **Dynamic Filtering:** Only includes features present in the DataFrame.

### 5. Handling Missing Values
- **Forward Fill:** Fills missing values using forward fill (`fillna(method='ffill')`).
- **Imputation:** Uses `SimpleImputer` (median for numerical, most frequent for categorical) in scikit-learn pipelines.

### 6. Data Splitting
- **Train/Test Split:** Uses `train_test_split` from scikit-learn to split the data, ensuring no data leakage.
- **Target Scaling:** Scales each target variable using `StandardScaler` and stores the scalers for later use.

### 7. Feature Scaling and Pipeline
- **Numerical Pipeline:** Imputation + StandardScaler for numerical features.
- **Categorical Pipeline:** Imputation for categorical features.
- **ColumnTransformer:** Combines both pipelines for robust preprocessing.
- **Fit/Transform:** Fits the preprocessor on training data and transforms both train and test sets.

### 8. Reshaping for RNNs
- **Shape:** Reshapes features to (samples, timesteps=1, features) for compatibility with RNN models.

### 9. Saving and Loading Scalers
- **Saving:** Saves the feature scaler and each target scaler as `.pkl` files using `pickle`.
- **Loading:** Loads saved scalers for use during prediction or evaluation.

### 10. Outputs
- Preprocessed training and testing data arrays
- List of features and targets
- Saved scalers for features and targets

---

**Key Libraries Used:**
- pandas, numpy, scikit-learn (StandardScaler, SimpleImputer, Pipeline, ColumnTransformer, train_test_split), pickle

**Class:** `DataProcessor`
- Methods: `load_data`, `create_features`, `create_lags`, `prepare_features`, `split_data`, `process_data`, `save_scalers`, `load_scalers`

**Best Practices:**
- Avoids data leakage by splitting before scaling
- Handles missing data robustly
- Modular and reusable for different datasets 