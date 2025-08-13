import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
import pickle
import warnings

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
    
    def create_features(self):
        """Create time-based features from datetime index."""
        # Extract time-based features
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        
        print("Time-based features created")
        return self.df
    
    def create_lags(self):
        """Create lag features for each target variable."""
        # Create lag features for each target
        for target in self.targets:
            self.df[f'{target}_lag_1'] = self.df[target].shift(1)
            self.df[f'{target}_lag_24'] = self.df[target].shift(24)
        
        # Drop rows with NaN values after creating lag features
        self.df.dropna(inplace=True)
        
        print(f"Lag features created. Data shape after dropping NaNs: {self.df.shape}")
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
    
    def process_data(self):
        """Process the data for model training."""
        self.load_data()
        self.create_features()
        self.create_lags()
        
        # Print column info before processing
        print("\nDataFrame columns:")
        print(self.df.columns.tolist())
        print("\nDataFrame info:")
        print(self.df.dtypes)
        
        # Process the data
        X_train, X_test, y_train, y_test = self.split_data()
        
        # Reshape data for RNN (samples, timesteps, features)
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        print("\nData processing complete")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
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
