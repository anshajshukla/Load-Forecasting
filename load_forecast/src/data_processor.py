import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import os
import pickle
import warnings
from datetime import datetime, timedelta
import holidays

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DataProcessor:
    """Class for loading and preprocessing the load forecasting data."""
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        """
        Initialize the data processor.
        
        Args:
            data_path (str): Path to the CSV data file
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.targets = ['DELHI', 'BRPL', 'BYPL', 'NDMC', 'MES']
        self.scalers_y = {}
        self.feature_scaler = None
        self.indian_holidays = holidays.India()
    
    def load_data(self):
        """Load the dataset and perform initial preprocessing."""
        # Load the CSV file
        self.df = pd.read_csv(self.data_path)
        
        # Convert datetime and set as index
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], format='%d-%m-%Y %H:%M')
        
        # Convert categorical weekday to numerical (0=Monday, 6=Sunday)
        if 'weekday' in self.df.columns:
            weekday_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            self.df['weekday_num'] = self.df['weekday'].map(weekday_map)
            print("Converted weekday strings to numerical values")
        
        # Set datetime as index
        self.df.set_index('datetime', inplace=True)
        
        print(f"Data loaded successfully with shape: {self.df.shape}")
        return self.df
    
    def validate_data(self, df):
        """Validate and clean the input data."""
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("\nMissing values found:")
            print(missing_values[missing_values > 0])
        
        # Check for infinite values
        inf_values = np.isinf(df.select_dtypes(include=np.number)).sum()
        if inf_values.any():
            print("\nInfinite values found:")
            print(inf_values[inf_values > 0])
        
        # Check for outliers in numerical columns
        numerical_cols = df.select_dtypes(include=np.number).columns
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col]
            if len(outliers) > 0:
                print(f"\nOutliers found in {col}: {len(outliers)} points")
        
        return df
    
    def create_advanced_features(self):
        """Create advanced time-based and interaction features."""
        # Basic time features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        # Advanced time features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour']/24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour']/24)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month']/12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month']/12)
        
        # Holiday features
        self.df['is_holiday'] = self.df.index.map(lambda x: x in self.indian_holidays).astype(int)
        
        # Rolling statistics
        for target in self.targets:
            # Rolling mean and std
            self.df[f'{target}_rolling_mean_24h'] = self.df[target].rolling(window=24).mean()
            self.df[f'{target}_rolling_std_24h'] = self.df[target].rolling(window=24).std()
            
            # Rolling mean and std for 7 days
            self.df[f'{target}_rolling_mean_7d'] = self.df[target].rolling(window=7*24).mean()
            self.df[f'{target}_rolling_std_7d'] = self.df[target].rolling(window=7*24).std()
        
        # Weather interaction features
        if 'temperature' in self.df.columns and 'humidity' in self.df.columns:
            self.df['temp_humidity'] = self.df['temperature'] * self.df['humidity']
            self.df['temp_squared'] = self.df['temperature'] ** 2
        
        # Time-based interaction features
        self.df['hour_month'] = self.df['hour'] * self.df['month']
        self.df['hour_weekend'] = self.df['hour'] * self.df['is_weekend']
        
        print("Advanced features created")
        return self.df
    
    def create_lags(self):
        """Create enhanced lag features for each target variable."""
        # Create lag features for each target
        for target in self.targets:
            # Short-term lags
            for lag in [1, 2, 3, 4, 5, 6, 12, 24]:
                self.df[f'{target}_lag_{lag}'] = self.df[target].shift(lag)
            
            # Weekly lags
            for lag in [7, 14, 21]:
                self.df[f'{target}_lag_{lag*24}'] = self.df[target].shift(lag*24)
            
            # Monthly lag
            self.df[f'{target}_lag_720'] = self.df[target].shift(30*24)
        
        # Create difference features
        for target in self.targets:
            self.df[f'{target}_diff_1'] = self.df[target].diff()
            self.df[f'{target}_diff_24'] = self.df[target].diff(24)
        
        # Drop rows with NaN values after creating lag features
        self.df.dropna(inplace=True)
        
        print(f"Enhanced lag features created. Data shape after dropping NaNs: {self.df.shape}")
        return self.df
    
    def prepare_features(self):
        """Prepare feature and target variables."""
        # Define categorical and numerical features, replacing 'weekday' with 'weekday_num'
        categorical_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'weekday_num']
        numerical_features = ['temperature', 'humidity', 'wind_speed', 'precipitation'] 
        
        # Add lag features
        for target in self.targets:
            numerical_features.extend([f'{target}_lag_1', f'{target}_lag_24'])
        
        # Filter out non-existent features
        categorical_features = [f for f in categorical_features if f in self.df.columns]
        numerical_features = [f for f in numerical_features if f in self.df.columns]
        
        self.features = numerical_features + categorical_features
        
        print(f"Features prepared: {len(self.features)} features selected")
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Store feature types for preprocessing
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        return self.features, self.targets
    
    def split_data(self):
        """Split data into features and targets, then into train and test sets."""
        features, targets = self.prepare_features()
        
        # Handle missing values
        self.df = self.df.fillna(method='ffill')
        
        # Split data first to avoid data leakage
        train_df, test_df = train_test_split(
            self.df, test_size=self.test_size, random_state=self.random_state
        )
        
        # Prepare y (targets)
        y_train = np.zeros((len(train_df), len(targets)))
        y_test = np.zeros((len(test_df), len(targets)))
        
        for i, target in enumerate(targets):
            scaler = StandardScaler()
            y_train[:, i] = scaler.fit_transform(train_df[[target]]).ravel()
            y_test[:, i] = scaler.transform(test_df[[target]]).ravel()
            self.scalers_y[target] = scaler
        
        # Create preprocessing pipeline
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # No preprocessing for categorical features except imputation
        # (we'll convert them to numerical in the data loading step)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Fit the preprocessor on the training data and transform both train and test
        self.feature_scaler = preprocessor
        X_train = self.feature_scaler.fit_transform(train_df[features])
        X_test = self.feature_scaler.transform(test_df[features])
        
        print(f"Data split complete. Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Data split complete. Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def select_features(self, X, y):
        """Select the most important features using statistical tests."""
        selector = SelectKBest(score_func=f_regression, k='all')
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'Feature': self.features,
            'Score': selector.scores_
        })
        scores = scores.sort_values('Score', ascending=False)
        
        # Select top features
        top_features = scores['Feature'].head(30).tolist()
        print("\nTop 30 features selected:")
        print(scores.head(30))
        
        return top_features
    
    def process_data(self, sequence_length=24):
        """Process the data for model training with enhanced preprocessing."""
        # Load and validate data
        self.load_data()
        self.df = self.validate_data(self.df)
        
        # Create features
        self.create_advanced_features()
        self.create_lags()
        
        # Print column info before processing
        print("\nDataFrame columns:")
        print(self.df.columns.tolist())
        print("\nDataFrame info:")
        print(self.df.dtypes)
        
        # Process the data
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Select important features
        self.features = self.select_features(X_train, y_train)
        
        # Create sequences with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        def create_sequences(X, y, seq_length):
            Xs, ys = [], []
            for i in range(len(X) - seq_length):
                Xs.append(X[i:(i + seq_length)])
                ys.append(y[i + seq_length])
            return np.array(Xs), np.array(ys)
        
        # Create sequences for training and testing
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
        
        # Ensure shapes match
        y_train_seq = y_train_seq.reshape(-1, len(self.targets))
        y_test_seq = y_test_seq.reshape(-1, len(self.targets))
        
        print("\nData processing complete")
        print(f"X_train shape: {X_train_seq.shape}")
        print(f"X_test shape: {X_test_seq.shape}")
        print(f"y_train shape: {y_train_seq.shape}")
        print(f"y_test shape: {y_test_seq.shape}")
        
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq
    
    def save_scalers(self, output_dir):
        """Save scalers for later use in predictions."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save feature scaler
        with open(os.path.join(output_dir, 'feature_scaler.pkl'), 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        # Save target scalers
        for target, scaler in self.scalers_y.items():
            with open(os.path.join(output_dir, f'{target}_scaler.pkl'), 'wb') as f:
                pickle.dump(scaler, f)
        
        print(f"Scalers saved to {output_dir}")
    
    def load_scalers(self, input_dir):
        """Load saved scalers."""
        # Load feature scaler
        with open(os.path.join(input_dir, 'feature_scaler.pkl'), 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        # Load target scalers
        self.scalers_y = {}
        for target in self.targets:
            with open(os.path.join(input_dir, f'{target}_scaler.pkl'), 'rb') as f:
                self.scalers_y[target] = pickle.load(f)
        
        print(f"Scalers loaded from {input_dir}")
        return self.feature_scaler, self.scalers_y
