"""
Delhi Load Forecasting - Phase 3 Week 3
Day 3-4: Attention Mechanisms

This script implements attention-based models to improve forecasting accuracy
through advanced feature attention and temporal attention mechanisms.

Approaches:
- Self-attention LSTM
- Multi-head attention models
- Temporal attention with feature selection
- Transformer-inspired architectures

Target: Achieve MAPE <6% through attention mechanisms
Timeline: Days 3-4 of Week 3 advanced architecture development
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import joblib
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionMechanisms:
    """
    Advanced attention-based forecasting models
    
    Features:
    - Self-attention layers
    - Multi-head attention
    - Temporal feature attention
    - Transformer architectures
    """
    
    def __init__(self, project_dir, week1_dir, week2_dir):
        """Initialize attention mechanisms development"""
        self.project_dir = project_dir
        self.week1_dir = week1_dir
        self.week2_dir = week2_dir
        
        # Directories
        self.models_dir = os.path.join(project_dir, 'models')
        self.results_dir = os.path.join(project_dir, 'results')
        self.viz_dir = os.path.join(project_dir, 'visualizations')
        
        # Create directories
        for directory in [self.models_dir, self.results_dir, self.viz_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Model configurations
        self.attention_configs = {
            'self_attention_lstm': {
                'attention_units': 64,
                'lstm_units': 128,
                'dropout': 0.3,
                'epochs': 25,
                'batch_size': 64
            },
            'multi_head_attention': {
                'num_heads': 8,
                'key_dim': 64,
                'ff_dim': 256,
                'dropout': 0.2,
                'epochs': 30,
                'batch_size': 32
            },
            'temporal_attention': {
                'attention_units': 96,
                'temporal_steps': 24,
                'feature_dim': 128,
                'dropout': 0.25,
                'epochs': 25,
                'batch_size': 64
            },
            'transformer_mini': {
                'num_layers': 4,
                'd_model': 128,
                'num_heads': 8,
                'dff': 512,
                'dropout': 0.1,
                'epochs': 35,
                'batch_size': 32
            }
        }
        
        # Training configuration
        self.training_config = {
            'validation_split': 0.2,
            'early_stopping_patience': 8,
            'lr_patience': 5,
            'min_lr': 1e-6,
            'learning_rate': 0.001,
            'optimizer': 'adam'
        }
        
        print("[OK] Attention mechanisms initialization completed")
    
    def load_data(self):
        """Load and prepare data from previous weeks"""
        print("[LOADING] Data from previous weeks...")
        
        # Load sequential data for attention models
        try:
            week2_data_path = os.path.join(self.week2_dir, 'data', 'sequential_data.npz')
            if os.path.exists(week2_data_path):
                with np.load(week2_data_path) as data:
                    self.X_train_seq = data['X_train']
                    self.X_test_seq = data['X_test']
                    self.y_train_seq = data['y_train']
                    self.y_test_seq = data['y_test']
                print("   [OK] Sequential data loaded from Week 2")
            else:
                # Create sequential data from main dataset
                dataset_path = os.path.join(os.path.dirname(self.week2_dir), 'final_dataset.csv')
                df = pd.read_csv(dataset_path)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                
                # Select numeric columns only
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                target_col = 'delhi_load'
                feature_cols = [col for col in numeric_cols if col != target_col]
                
                print(f"   [INFO] Using {len(feature_cols)} numeric features")
                
                # Create sequences for attention models
                sequence_length = 24
                
                sequences = []
                targets = []
                
                for i in range(sequence_length, len(df)):
                    seq_features = df[feature_cols].iloc[i-sequence_length:i].values.astype(np.float32)
                    target = float(df[target_col].iloc[i])
                    sequences.append(seq_features)
                    targets.append(target)
                
                X = np.array(sequences, dtype=np.float32)
                y = np.array(targets, dtype=np.float32)
                
                # Normalize the data
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()
                
                # Reshape for scaling
                X_reshaped = X.reshape(-1, X.shape[-1])
                X_scaled = scaler_X.fit_transform(X_reshaped)
                X = X_scaled.reshape(X.shape)
                
                y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
                
                # Store scalers
                self.scaler_X = scaler_X
                self.scaler_y = scaler_y
                
                # Split
                split_idx = int(0.8 * len(X))
                self.X_train_seq = X[:split_idx]
                self.X_test_seq = X[split_idx:]
                self.y_train_seq = y[:split_idx]
                self.y_test_seq = y[split_idx:]
                
                print(f"   [OK] Created sequential data: {X.shape}")
            
            print(f"   [INFO] Train sequences: {self.X_train_seq.shape}")
            print(f"   [INFO] Test sequences: {self.X_test_seq.shape}")
            
            return True
            
        except Exception as e:
            print(f"   [ERROR] Data loading failed: {str(e)}")
            return False
    
    def create_self_attention_layer(self, units):
        """Create a self-attention layer"""
        class SelfAttention(layers.Layer):
            def __init__(self, units):
                super(SelfAttention, self).__init__()
                self.units = units
                self.W_q = layers.Dense(units)
                self.W_k = layers.Dense(units)
                self.W_v = layers.Dense(units)
                
            def call(self, inputs):
                # inputs shape: (batch_size, time_steps, features)
                Q = self.W_q(inputs)
                K = self.W_k(inputs)
                V = self.W_v(inputs)
                
                # Attention scores
                scores = tf.matmul(Q, K, transpose_b=True)
                scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
                attention_weights = tf.nn.softmax(scores, axis=-1)
                
                # Apply attention
                context = tf.matmul(attention_weights, V)
                return context
        
        return SelfAttention(units)
    
    def build_self_attention_lstm(self, input_shape):
        """Build self-attention LSTM model"""
        config = self.attention_configs['self_attention_lstm']
        
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = layers.LSTM(config['lstm_units'], return_sequences=True)(inputs)
        lstm_out = layers.Dropout(config['dropout'])(lstm_out)
        
        # Self-attention
        attention_out = self.create_self_attention_layer(config['attention_units'])(lstm_out)
        
        # Global average pooling
        pooled = layers.GlobalAveragePooling1D()(attention_out)
        
        # Dense layers
        dense = layers.Dense(64, activation='relu')(pooled)
        dense = layers.Dropout(config['dropout'])(dense)
        dense = layers.Dense(32, activation='relu')(dense)
        outputs = layers.Dense(1)(dense)
        
        model = models.Model(inputs, outputs, name='self_attention_lstm')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_multi_head_attention(self, input_shape):
        """Build multi-head attention model"""
        config = self.attention_configs['multi_head_attention']
        
        inputs = layers.Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=config['num_heads'],
            key_dim=config['key_dim'],
            dropout=config['dropout']
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = layers.Add()([inputs, attention_output])
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)
        
        # Feed forward
        ffn_output = layers.Dense(config['ff_dim'], activation='relu')(attention_output)
        ffn_output = layers.Dropout(config['dropout'])(ffn_output)
        ffn_output = layers.Dense(input_shape[-1])(ffn_output)
        
        # Add & Norm
        ffn_output = layers.Add()([attention_output, ffn_output])
        ffn_output = layers.LayerNormalization(epsilon=1e-6)(ffn_output)
        
        # Global pooling and output
        pooled = layers.GlobalAveragePooling1D()(ffn_output)
        dense = layers.Dense(128, activation='relu')(pooled)
        dense = layers.Dropout(config['dropout'])(dense)
        dense = layers.Dense(64, activation='relu')(dense)
        outputs = layers.Dense(1)(dense)
        
        model = models.Model(inputs, outputs, name='multi_head_attention')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_temporal_attention(self, input_shape):
        """Build temporal attention model"""
        config = self.attention_configs['temporal_attention']
        
        inputs = layers.Input(shape=input_shape)
        
        # Feature extraction
        conv1d = layers.Conv1D(config['feature_dim'], 3, padding='same', activation='relu')(inputs)
        conv1d = layers.Dropout(config['dropout'])(conv1d)
        
        # Temporal attention mechanism
        attention_weights = layers.Dense(1, activation='sigmoid')(conv1d)
        attention_weights = layers.Softmax(axis=1)(attention_weights)
        
        # Apply attention
        attended_features = layers.Multiply()([conv1d, attention_weights])
        
        # LSTM processing
        lstm_out = layers.LSTM(config['attention_units'], return_sequences=True)(attended_features)
        lstm_out = layers.Dropout(config['dropout'])(lstm_out)
        
        # Final attention
        final_attention = layers.Dense(1, activation='sigmoid')(lstm_out)
        final_attention = layers.Softmax(axis=1)(final_attention)
        final_output = layers.Multiply()([lstm_out, final_attention])
        
        # Output
        pooled = layers.GlobalAveragePooling1D()(final_output)
        dense = layers.Dense(64, activation='relu')(pooled)
        outputs = layers.Dense(1)(dense)
        
        model = models.Model(inputs, outputs, name='temporal_attention')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_transformer_mini(self, input_shape):
        """Build mini transformer model"""
        config = self.attention_configs['transformer_mini']
        
        inputs = layers.Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        positions = layers.Embedding(input_shape[0], config['d_model'])(positions)
        
        # Input projection
        x = layers.Dense(config['d_model'])(inputs)
        x = x + positions
        
        # Transformer blocks
        for i in range(config['num_layers']):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=config['num_heads'],
                key_dim=config['d_model'] // config['num_heads'],
                dropout=config['dropout']
            )(x, x)
            
            # Add & Norm
            x = layers.Add()([x, attention_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            
            # Feed forward
            ffn_output = layers.Dense(config['dff'], activation='relu')(x)
            ffn_output = layers.Dropout(config['dropout'])(ffn_output)
            ffn_output = layers.Dense(config['d_model'])(ffn_output)
            
            # Add & Norm
            x = layers.Add()([x, ffn_output])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Global pooling and output
        pooled = layers.GlobalAveragePooling1D()(x)
        dense = layers.Dense(128, activation='relu')(pooled)
        dense = layers.Dropout(config['dropout'])(dense)
        dense = layers.Dense(64, activation='relu')(dense)
        outputs = layers.Dense(1)(dense)
        
        model = models.Model(inputs, outputs, name='transformer_mini')
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.training_config['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_attention_model(self, model, model_name):
        """Train an attention model"""
        print(f"   [TRAINING] {model_name}...")
        
        config = self.attention_configs[model_name]
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.training_config['lr_patience'],
                min_lr=self.training_config['min_lr'],
                verbose=0
            ),
            ModelCheckpoint(
                os.path.join(self.models_dir, f'{model_name}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Training
        history = model.fit(
            self.X_train_seq, self.y_train_seq,
            validation_split=self.training_config['validation_split'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluation
        y_pred = model.predict(self.X_test_seq, verbose=0)
        y_pred = y_pred.flatten()
        
        # Denormalize predictions and targets
        y_pred_orig = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test_orig = self.scaler_y.inverse_transform(self.y_test_seq.reshape(-1, 1)).flatten()
        
        mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        
        results = {
            'mape': mape,
            'rmse': rmse,
            'mae': mae,
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'mae': history.history['mae'],
                'val_mae': history.history['val_mae']
            }
        }
        
        print(f"   [SUCCESS] {model_name} - MAPE: {mape:.2f}%, RMSE: {rmse:.3f}")
        
        return results
    
    def run_attention_development(self):
        """Run complete attention mechanisms development"""
        print("\\n[CREATING] Attention-based models...")
        
        if not self.load_data():
            return False
        
        input_shape = (self.X_train_seq.shape[1], self.X_train_seq.shape[2])
        print(f"   [INFO] Input shape: {input_shape}")
        
        results = {}
        
        # Build and train attention models
        attention_models = [
            ('self_attention_lstm', self.build_self_attention_lstm),
            ('multi_head_attention', self.build_multi_head_attention),
            ('temporal_attention', self.build_temporal_attention),
            ('transformer_mini', self.build_transformer_mini)
        ]
        
        for model_name, build_func in attention_models:
            try:
                print(f"\\n[BUILDING] {model_name.replace('_', ' ').title()}...")
                model = build_func(input_shape)
                print(f"   [INFO] Model parameters: {model.count_params():,}")
                
                # Train model
                model_results = self.train_attention_model(model, model_name)
                results[model_name] = model_results
                
            except Exception as e:
                print(f"   [ERROR] {model_name} failed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_model = min(valid_results.keys(), key=lambda x: valid_results[x]['mape'])
            best_mape = valid_results[best_model]['mape']
            print(f"\\n[BEST ATTENTION MODEL] {best_model} with MAPE: {best_mape:.2f}%")
        else:
            print("\\n[ERROR] No attention models trained successfully")
            return False
        
        # Save results
        results_path = os.path.join(self.results_dir, 'attention_models_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\\n[SAVED] Results saved to {results_path}")
        
        return True

def main():
    """Main execution function"""
    print("[STARTING] Attention Mechanisms Development Pipeline")
    print("="*80)
    
    # Configuration
    project_dir = r"C:\\Users\\ansha\\Desktop\\SIH_new\\load_forecast\\phase_3_week_3_advanced_architectures"
    week1_dir = r"C:\\Users\\ansha\\Desktop\\SIH_new\\load_forecast\\phase_3_week_1_model_development"
    week2_dir = r"C:\\Users\\ansha\\Desktop\\SIH_new\\load_forecast\\phase_3_week_2_neural_networks"
    
    # Initialize development
    attention_dev = AttentionMechanisms(project_dir, week1_dir, week2_dir)
    
    # Run attention development
    success = attention_dev.run_attention_development()
    
    if success:
        print("\\n[SUCCESS] Attention Mechanisms Development Pipeline Completed!")
        print("="*80)
        print("[TARGET] ✓ Advanced attention models developed")
        print("\\n[READY] Ready for Day 5-6: Multi-scale Forecasting")
        print("\\n[NEXT STEPS]")
        print("   1. Review attention model results in 'results/' directory")
        print("   2. Analyze attention weights and interpretability")
        print("   3. Proceed to multi-scale forecasting development")
        print("   4. Consider attention mechanisms for final ensemble")
    else:
        print("\\n[FAILED] Attention development encountered errors")
    
    return success

if __name__ == "__main__":
    main()
