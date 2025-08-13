"""
Delhi Load Forecasting - Phase 3 Week 2
Day 1-2: Neural Network Data Preparation

This script prepares time series sequences for LSTM, GRU, and Transformer models.
Creates sequential data with lookback windows and handles multi-step forecasting.

Target: Prepare sequential data for neural network training
Timeline: Days 1-2 of Week 2 neural network development
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
import joblib
import json
from datetime import datetime
import os

class NeuralNetworkDataPreparation:
    """
    Data preparation pipeline for neural network models
    
    Features:
    - Time series sequence creation with lookback windows
    - Multi-step forecasting preparation
    - Proper train/validation/test splits for time series
    - Feature engineering for temporal patterns
    """
    
    def __init__(self, week1_dir, output_dir):
        """Initialize neural network data preparation"""
        self.week1_dir = week1_dir
        self.output_dir = output_dir
        self.create_output_directories()
        
        # Data containers
        self.data = None
        self.features = None
        self.target_columns = ['delhi_load', 'brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']
        
        # Neural network specific parameters
        self.sequence_length = 24  # 24 hours lookback
        self.prediction_horizon = 1  # 1 hour ahead prediction
        self.feature_scaler = None
        self.target_scaler = None
        
        # Sequential data containers
        self.X_train_seq = None
        self.X_val_seq = None
        self.X_test_seq = None
        self.y_train_seq = None
        self.y_val_seq = None
        self.y_test_seq = None
        
        # Metadata
        self.preparation_metadata = {}
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            os.path.join(self.output_dir, 'data'),
            os.path.join(self.output_dir, 'scalers'),
            os.path.join(self.output_dir, 'metadata'),
            os.path.join(self.output_dir, 'models'),
            os.path.join(self.output_dir, 'results'),
            os.path.join(self.output_dir, 'visualizations')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print("[OK] Output directories created successfully")
    
    def load_week1_data(self):
        """Load prepared data from Week 1"""
        print("\n[LOADING] Week 1 prepared data...")
        
        try:
            # Load the original time series data (not scaled) for proper sequence creation
            X_train = pd.read_csv(
                os.path.join(self.week1_dir, 'data', 'X_train.csv'),
                index_col=0, parse_dates=True
            )
            X_val = pd.read_csv(
                os.path.join(self.week1_dir, 'data', 'X_val.csv'),
                index_col=0, parse_dates=True
            )
            X_test = pd.read_csv(
                os.path.join(self.week1_dir, 'data', 'X_test.csv'),
                index_col=0, parse_dates=True
            )
            
            y_train = pd.read_csv(
                os.path.join(self.week1_dir, 'data', 'y_train.csv'),
                index_col=0, parse_dates=True
            )
            y_val = pd.read_csv(
                os.path.join(self.week1_dir, 'data', 'y_val.csv'),
                index_col=0, parse_dates=True
            )
            y_test = pd.read_csv(
                os.path.join(self.week1_dir, 'data', 'y_test.csv'),
                index_col=0, parse_dates=True
            )
            
            # Combine all data for proper sequential processing
            self.data = pd.concat([
                pd.concat([X_train, y_train], axis=1),
                pd.concat([X_val, y_val], axis=1),
                pd.concat([X_test, y_test], axis=1)
            ]).sort_index()
            
            # Store split indices for later use
            self.train_end_idx = len(X_train)
            self.val_end_idx = self.train_end_idx + len(X_val)
            
            self.features = X_train.columns.tolist()
            
            print(f"[OK] Week 1 data loaded successfully")
            print(f"   [INFO] Total samples: {len(self.data)}")
            print(f"   [INFO] Features: {len(self.features)}")
            print(f"   [INFO] Target variables: {len(self.target_columns)}")
            print(f"   [INFO] Date range: {self.data.index.min()} to {self.data.index.max()}")
            
            # Handle missing values
            self.handle_missing_values()
            
        except Exception as e:
            print(f"[ERROR] Failed to load Week 1 data: {str(e)}")
            raise
    
    def handle_missing_values(self):
        """Handle missing values in the data"""
        print("\n[PROCESSING] Handling missing values...")
        
        missing_before = self.data.isnull().sum().sum()
        if missing_before > 0:
            print(f"   [INFO] Found {missing_before} missing values")
            
            # Forward fill then backward fill for time series
            self.data = self.data.ffill().bfill()
            
            # Fill any remaining with median
            self.data = self.data.fillna(self.data.median())
            
            missing_after = self.data.isnull().sum().sum()
            print(f"   [OK] Missing values after treatment: {missing_after}")
        else:
            print("   [OK] No missing values found")
        
        # Handle infinite values
        inf_count = np.isinf(self.data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            print(f"   [INFO] Found {inf_count} infinite values")
            self.data = self.data.replace([np.inf, -np.inf], [self.data.max().max(), self.data.min().min()])
            print("   [OK] Infinite values handled")
    
    def create_sequences(self):
        """Create sequences for neural network training"""
        print(f"\n[CREATING] Sequential data with {self.sequence_length}h lookback...")
        
        # Separate features and targets
        X_data = self.data[self.features].values
        y_data = self.data[self.target_columns].values
        
        # Scale the data
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        X_scaled = self.feature_scaler.fit_transform(X_data)
        y_scaled = self.target_scaler.fit_transform(y_data)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled) - self.prediction_horizon + 1):
            # Input sequence (features for past sequence_length hours)
            X_seq = X_scaled[i - self.sequence_length:i]
            
            # Target (load values for next prediction_horizon hours)
            y_seq = y_scaled[i:i + self.prediction_horizon]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq.flatten() if self.prediction_horizon > 1 else y_seq[0])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        print(f"[OK] Created {len(X_sequences)} sequences")
        print(f"   [INFO] X shape: {X_sequences.shape}")
        print(f"   [INFO] y shape: {y_sequences.shape}")
        
        # Split sequences based on original time splits
        # Account for the sequence_length offset
        train_seq_end = self.train_end_idx - self.sequence_length
        val_seq_end = self.val_end_idx - self.sequence_length
        
        # Ensure we don't exceed available sequences
        train_seq_end = min(train_seq_end, len(X_sequences))
        val_seq_end = min(val_seq_end, len(X_sequences))
        
        if train_seq_end <= 0:
            raise ValueError("Not enough data to create training sequences")
        
        self.X_train_seq = X_sequences[:train_seq_end]
        self.y_train_seq = y_sequences[:train_seq_end]
        
        if val_seq_end > train_seq_end:
            self.X_val_seq = X_sequences[train_seq_end:val_seq_end]
            self.y_val_seq = y_sequences[train_seq_end:val_seq_end]
        else:
            # Use last 20% of training data as validation if original split doesn't work
            split_idx = int(0.8 * len(self.X_train_seq))
            self.X_val_seq = self.X_train_seq[split_idx:]
            self.y_val_seq = self.y_train_seq[split_idx:]
            self.X_train_seq = self.X_train_seq[:split_idx]
            self.y_train_seq = self.y_train_seq[:split_idx]
        
        if val_seq_end < len(X_sequences):
            self.X_test_seq = X_sequences[val_seq_end:]
            self.y_test_seq = y_sequences[val_seq_end:]
        else:
            # Use last 10% of validation data as test
            split_idx = int(0.9 * len(self.X_val_seq))
            self.X_test_seq = self.X_val_seq[split_idx:]
            self.y_test_seq = self.y_val_seq[split_idx:]
            self.X_val_seq = self.X_val_seq[:split_idx]
            self.y_val_seq = self.y_val_seq[:split_idx]
        
        print(f"[OK] Sequential data splits created:")
        print(f"   [INFO] Training sequences: {len(self.X_train_seq)}")
        print(f"   [INFO] Validation sequences: {len(self.X_val_seq)}")
        print(f"   [INFO] Test sequences: {len(self.X_test_seq)}")
    
    def save_neural_network_data(self):
        """Save prepared neural network data"""
        print("\n[SAVING] Neural network prepared data...")
        
        data_dir = os.path.join(self.output_dir, 'data')
        scalers_dir = os.path.join(self.output_dir, 'scalers')
        metadata_dir = os.path.join(self.output_dir, 'metadata')
        
        # Save sequential data as numpy arrays (more efficient for neural networks)
        np.save(os.path.join(data_dir, 'X_train_seq.npy'), self.X_train_seq)
        np.save(os.path.join(data_dir, 'X_val_seq.npy'), self.X_val_seq)
        np.save(os.path.join(data_dir, 'X_test_seq.npy'), self.X_test_seq)
        np.save(os.path.join(data_dir, 'y_train_seq.npy'), self.y_train_seq)
        np.save(os.path.join(data_dir, 'y_val_seq.npy'), self.y_val_seq)
        np.save(os.path.join(data_dir, 'y_test_seq.npy'), self.y_test_seq)
        
        print("   [SAVED] Sequential data arrays")
        
        # Save scalers
        joblib.dump(self.feature_scaler, os.path.join(scalers_dir, 'nn_feature_scaler.pkl'))
        joblib.dump(self.target_scaler, os.path.join(scalers_dir, 'nn_target_scaler.pkl'))
        
        print("   [SAVED] Neural network scalers")
        
        # Save metadata
        self.preparation_metadata = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'n_features': len(self.features),
            'n_targets': len(self.target_columns),
            'feature_names': self.features,
            'target_names': self.target_columns,
            'data_shapes': {
                'X_train': list(self.X_train_seq.shape),
                'X_val': list(self.X_val_seq.shape),
                'X_test': list(self.X_test_seq.shape),
                'y_train': list(self.y_train_seq.shape),
                'y_val': list(self.y_val_seq.shape),
                'y_test': list(self.y_test_seq.shape)
            },
            'preparation_timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(metadata_dir, 'neural_network_metadata.json'), 'w') as f:
            json.dump(self.preparation_metadata, f, indent=2, default=str)
        
        print("   [SAVED] Neural network metadata")
        print("[OK] All neural network data saved successfully")
    
    def run_complete_pipeline(self):
        """Run the complete neural network data preparation pipeline"""
        print("[STARTING] Neural Network Data Preparation Pipeline")
        print("=" * 80)
        
        try:
            # Step 1: Load Week 1 data
            self.load_week1_data()
            
            # Step 2: Create sequences
            self.create_sequences()
            
            # Step 3: Save prepared data
            self.save_neural_network_data()
            
            print("\n[SUCCESS] Neural Network Data Preparation Completed!")
            print("=" * 80)
            print(f"[INFO] Sequence length: {self.sequence_length} hours")
            print(f"[INFO] Prediction horizon: {self.prediction_horizon} hour(s)")
            print(f"[INFO] Training sequences: {len(self.X_train_seq)}")
            print(f"[INFO] Features per sequence: {self.X_train_seq.shape[2]}")
            print(f"[INFO] Target variables: {len(self.target_columns)}")
            print(f"[INFO] Output directory: {self.output_dir}")
            print("\n[READY] Ready for neural network model training!")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    # Configuration
    week1_dir = r"C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_1_model_development"
    output_dir = r"C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_2_neural_networks"
    
    # Initialize and run pipeline
    pipeline = NeuralNetworkDataPreparation(week1_dir, output_dir)
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n[NEXT STEPS]")
        print("   1. Proceed to LSTM/GRU model development")
        print("   2. Use prepared sequential data in 'data/' directory")
        print("   3. Use neural network scalers for consistent preprocessing")
        print("   4. Target neural network MAPE <6% for improvement over baselines")

if __name__ == "__main__":
    main()
