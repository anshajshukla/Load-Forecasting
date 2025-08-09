"""
Advanced LSTM/GRU Models - Optimized Implementation
Week 2 Neural Networks with Enhanced Features and Efficiency

This script implements state-of-the-art LSTM and GRU models with:
- Advanced architectural features (attention, residual connections)
- Efficient training with fewer epochs
- Better preprocessing and feature engineering
- Target: Achieve MAPE <15% (significant improvement from 45%)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.regularizers import l2, l1_l2
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import joblib
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

class AdvancedLSTMGRU:
    """
    Advanced LSTM/GRU implementation with enhanced features and efficiency
    """
    
    def __init__(self, data_dir, output_dir):
        """Initialize with optimized configuration"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Optimized model configurations
        self.model_configs = {
            'efficient_lstm': {
                'type': 'lstm_with_attention',
                'lstm_units': [64, 32],
                'dense_units': [32, 16],
                'dropout': 0.2,
                'recurrent_dropout': 0.1,
                'l2_reg': 0.001,
                'use_attention': True,
                'use_residual': True
            },
            'efficient_gru': {
                'type': 'gru_with_attention',
                'gru_units': [64, 32],
                'dense_units': [32, 16],
                'dropout': 0.2,
                'recurrent_dropout': 0.1,
                'l2_reg': 0.001,
                'use_attention': True,
                'use_residual': True
            },
            'hybrid_lstm_gru': {
                'type': 'hybrid',
                'lstm_units': [32],
                'gru_units': [32],
                'dense_units': [24, 12],
                'dropout': 0.15,
                'recurrent_dropout': 0.1,
                'l2_reg': 0.0005,
                'use_attention': True
            }
        }
        
        # Efficient training configuration
        self.training_config = {
            'epochs': 25,  # Reduced from 100 for efficiency
            'batch_size': 64,  # Larger batch for faster training
            'patience': 8,  # Reduced patience
            'learning_rate': 0.002,  # Slightly higher LR
            'min_lr': 1e-6,
            'lr_factor': 0.8,
            'lr_patience': 4,
            'validation_split': 0.2
        }
        
        # Enhanced preprocessing configuration
        self.preprocessing_config = {
            'sequence_length': 24,  # 24 hours lookback
            'target_steps': 1,  # Predict next hour
            'feature_scaling': 'robust',  # RobustScaler for outliers
            'target_scaling': 'minmax',  # MinMax for target
            'rolling_features': True,
            'lag_features': True
        }
        
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Enhanced data loading and preprocessing"""
        print("🔄 Loading and preprocessing data...")
        
        # Load data
        data_path = os.path.join(self.data_dir, 'final_dataset.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")
            
        df = pd.read_csv(data_path, parse_dates=['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"📊 Dataset shape: {df.shape}")
        print(f"📅 Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        # Feature engineering
        df = self.enhance_features(df)
        
        # Select features intelligently
        feature_cols = self.select_best_features(df)
        
        # Prepare sequences
        X, y, feature_names = self.create_sequences(df, feature_cols)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Training sequences: {self.X_train.shape}")
        print(f"✅ Test sequences: {self.X_test.shape}")
        
        return feature_names
        
    def enhance_features(self, df):
        """Add time-based and rolling features"""
        # Time features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Rolling features for load
        target_col = 'Load(MW)' if 'Load(MW)' in df.columns else 'load'
        if target_col in df.columns:
            df['load_ma_3'] = df[target_col].rolling(3).mean()
            df['load_ma_6'] = df[target_col].rolling(6).mean()
            df['load_ma_24'] = df[target_col].rolling(24).mean()
            df['load_std_24'] = df[target_col].rolling(24).std()
            
            # Lag features
            df['load_lag_1'] = df[target_col].shift(1)
            df['load_lag_24'] = df[target_col].shift(24)
            df['load_lag_168'] = df[target_col].shift(168)  # Week lag
            
        # Fill missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
        
    def select_best_features(self, df):
        """Intelligent feature selection"""
        # Priority features
        essential_features = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_weekend', 'month'
        ]
        
        # Load-related features
        load_features = [col for col in df.columns if 'load' in col.lower() and 'Load(MW)' not in col]
        
        # Weather features (if available)
        weather_features = [col for col in df.columns if any(w in col.lower() 
                          for w in ['temp', 'humidity', 'pressure', 'wind'])]
        
        # Combine features
        feature_cols = essential_features + load_features + weather_features
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        print(f"🎯 Selected {len(feature_cols)} features: {feature_cols[:10]}...")
        return feature_cols
        
    def create_sequences(self, df, feature_cols):
        """Create sequences for time series prediction"""
        target_col = 'Load(MW)' if 'Load(MW)' in df.columns else 'load'
        
        # Scale features and target separately
        feature_scaler = RobustScaler()
        target_scaler = MinMaxScaler()
        
        # Prepare data
        features = df[feature_cols].values
        target = df[target_col].values.reshape(-1, 1)
        
        # Scale data
        features_scaled = feature_scaler.fit_transform(features)
        target_scaled = target_scaler.fit_transform(target)
        
        # Save scalers
        joblib.dump(feature_scaler, os.path.join(self.output_dir, 'feature_scaler.pkl'))
        joblib.dump(target_scaler, os.path.join(self.output_dir, 'target_scaler.pkl'))
        
        # Create sequences
        seq_length = self.preprocessing_config['sequence_length']
        X, y = [], []
        
        for i in range(seq_length, len(features_scaled)):
            X.append(features_scaled[i-seq_length:i])
            y.append(target_scaled[i, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        return X, y, feature_cols
        
    def create_attention_layer(self, sequence_input):
        """Create attention mechanism"""
        attention = layers.MultiHeadAttention(
            num_heads=4, 
            key_dim=16,
            name='attention'
        )(sequence_input, sequence_input)
        
        # Add & Norm
        attention = layers.Add()([sequence_input, attention])
        attention = layers.LayerNormalization()(attention)
        
        return attention
        
    def build_advanced_lstm(self, config, input_shape):
        """Build LSTM with advanced features"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(config['lstm_units']):
            return_sequences = (i < len(config['lstm_units']) - 1) or config.get('use_attention', False)
            
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=config['dropout'],
                recurrent_dropout=config['recurrent_dropout'],
                kernel_regularizer=l2(config['l2_reg']),
                name=f'lstm_{i+1}'
            )(x)
            
            if config.get('use_residual', False) and i > 0:
                # Simple residual connection
                x = layers.Add()([x, x])
                
        # Attention mechanism
        if config.get('use_attention', False):
            x = self.create_attention_layer(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        # Dense layers
        for i, units in enumerate(config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=l2(config['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(config['dropout'])(x)
            
        # Output layer
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = models.Model(inputs, outputs, name='advanced_lstm')
        return model
        
    def build_advanced_gru(self, config, input_shape):
        """Build GRU with advanced features"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        x = inputs
        
        # GRU layers
        for i, units in enumerate(config['gru_units']):
            return_sequences = (i < len(config['gru_units']) - 1) or config.get('use_attention', False)
            
            x = layers.GRU(
                units,
                return_sequences=return_sequences,
                dropout=config['dropout'],
                recurrent_dropout=config['recurrent_dropout'],
                kernel_regularizer=l2(config['l2_reg']),
                name=f'gru_{i+1}'
            )(x)
            
            if config.get('use_residual', False) and i > 0:
                x = layers.Add()([x, x])
                
        # Attention mechanism
        if config.get('use_attention', False):
            x = self.create_attention_layer(x)
            x = layers.GlobalAveragePooling1D()(x)
            
        # Dense layers
        for i, units in enumerate(config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=l2(config['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(config['dropout'])(x)
            
        # Output layer
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = models.Model(inputs, outputs, name='advanced_gru')
        return model
        
    def build_hybrid_model(self, config, input_shape):
        """Build hybrid LSTM-GRU model"""
        inputs = layers.Input(shape=input_shape, name='input')
        
        # LSTM branch
        lstm_out = layers.LSTM(
            config['lstm_units'][0],
            return_sequences=True,
            dropout=config['dropout'],
            recurrent_dropout=config['recurrent_dropout'],
            kernel_regularizer=l2(config['l2_reg']),
            name='lstm_branch'
        )(inputs)
        
        # GRU branch
        gru_out = layers.GRU(
            config['gru_units'][0],
            return_sequences=True,
            dropout=config['dropout'],
            recurrent_dropout=config['recurrent_dropout'],
            kernel_regularizer=l2(config['l2_reg']),
            name='gru_branch'
        )(inputs)
        
        # Combine branches
        combined = layers.Concatenate(axis=-1)([lstm_out, gru_out])
        
        # Attention
        if config.get('use_attention', False):
            combined = self.create_attention_layer(combined)
            
        x = layers.GlobalAveragePooling1D()(combined)
        
        # Dense layers
        for i, units in enumerate(config['dense_units']):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=l2(config['l2_reg']),
                name=f'dense_{i+1}'
            )(x)
            x = layers.Dropout(config['dropout'])(x)
            
        # Output
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = models.Model(inputs, outputs, name='hybrid_lstm_gru')
        return model
        
    def train_model(self, model_name, config):
        """Train individual model with optimized settings"""
        print(f"\\n🚀 Training {model_name}...")
        
        # Build model
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        if config['type'] == 'lstm_with_attention':
            model = self.build_advanced_lstm(config, input_shape)
        elif config['type'] == 'gru_with_attention':
            model = self.build_advanced_gru(config, input_shape)
        elif config['type'] == 'hybrid':
            model = self.build_hybrid_model(config, input_shape)
        else:
            raise ValueError(f"Unknown model type: {config['type']}")
            
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=self.training_config['learning_rate'],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE
            metrics=['mae']
        )
        
        print(f"📋 Model Summary for {model_name}:")
        print(f"   Parameters: {model.count_params():,}")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=self.training_config['lr_factor'],
                patience=self.training_config['lr_patience'],
                min_lr=self.training_config['min_lr'],
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'models', f'{model_name}_best.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        start_time = datetime.now()
        
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=self.training_config['validation_split'],
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            callbacks=callbacks_list,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Load best model
        model.load_weights(os.path.join(self.output_dir, 'models', f'{model_name}_best.h5'))
        
        # Evaluate
        metrics = self.evaluate_model(model, model_name, training_time)
        
        # Save model
        model.save(os.path.join(self.output_dir, 'models', f'{model_name}_final.h5'))
        
        return model, metrics, history
        
    def evaluate_model(self, model, model_name, training_time):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred_scaled = model.predict(self.X_test, verbose=0)
        
        # Inverse transform
        target_scaler = joblib.load(os.path.join(self.output_dir, 'target_scaler.pkl'))
        y_test_original = target_scaler.inverse_transform(self.y_test.reshape(-1, 1)).flatten()
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled).flatten()
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        
        metrics = {
            'model_name': model_name,
            'mape': mape,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'training_time': training_time,
            'week': 2,
            'model_type': model_name.replace('_', ' ').title()
        }
        
        print(f"\\n📊 {model_name} Results:")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   MAE: {mae:.2f} MW")
        print(f"   RMSE: {rmse:.2f} MW")
        print(f"   Training Time: {training_time:.1f} seconds")
        
        self.results[model_name] = metrics
        return metrics
        
    def run_all_models(self):
        """Train and evaluate all models"""
        print("🎯 Starting Advanced LSTM/GRU Model Training")
        print("=" * 60)
        
        # Load data
        feature_names = self.load_and_preprocess_data()
        
        # Train all models
        all_models = {}
        all_histories = {}
        
        for model_name, config in self.model_configs.items():
            try:
                model, metrics, history = self.train_model(model_name, config)
                all_models[model_name] = model
                all_histories[model_name] = history
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {str(e)}")
                continue
                
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations(all_histories)
        
        print("\\n✅ Advanced LSTM/GRU Training Complete!")
        return self.results
        
    def save_results(self):
        """Save training results"""
        results_path = os.path.join(self.output_dir, 'advanced_lstm_gru_results.json')
        
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, metrics in self.results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
            
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
            
        print(f"📄 Results saved to: {results_path}")
        
    def create_visualizations(self, histories):
        """Create training visualizations"""
        if not histories:
            return
            
        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced LSTM/GRU Training Progress', fontsize=16)
        
        for i, (model_name, history) in enumerate(histories.items()):
            row = i // 2
            col = i % 2
            
            if row < 2 and col < 2:
                ax = axes[row, col]
                
                # Plot loss
                ax.plot(history.history['loss'], label='Training Loss', color='blue')
                ax.plot(history.history['val_loss'], label='Validation Loss', color='red')
                ax.set_title(f'{model_name} - Training Progress')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'advanced_training_progress.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Results comparison
        self.plot_results_comparison()
        
    def plot_results_comparison(self):
        """Plot model comparison"""
        if not self.results:
            return
            
        models = list(self.results.keys())
        mapes = [self.results[m]['mape'] for m in models]
        training_times = [self.results[m]['training_time'] for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAPE comparison
        bars1 = ax1.bar(models, mapes, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax1.set_title('Model Performance Comparison (MAPE)', fontsize=14)
        ax1.set_ylabel('MAPE (%)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, mape in zip(bars1, mapes):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{mape:.1f}%', ha='center', va='bottom', fontweight='bold')
                    
        # Training time comparison
        bars2 = ax2.bar(models, training_times, color=['#FFA07A', '#98D8C8', '#87CEEB'])
        ax2.set_title('Training Time Comparison', fontsize=14)
        ax2.set_ylabel('Training Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars2, training_times):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
                    
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'advanced_model_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("📊 Visualizations saved to plots directory")

def main():
    """Main execution function"""
    # Paths
    data_dir = "C:/Users/ansha/Desktop/SIH_new/load_forecast/dataset"
    output_dir = "C:/Users/ansha/Desktop/SIH_new/load_forecast/phase_3_week_2_neural_networks/advanced_results"
    
    # Initialize and run
    advanced_models = AdvancedLSTMGRU(data_dir, output_dir)
    results = advanced_models.run_all_models()
    
    # Print summary
    print("\\n🏆 FINAL RESULTS SUMMARY")
    print("=" * 50)
    
    best_model = min(results.items(), key=lambda x: x[1]['mape'])
    
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['mape']):
        print(f"{model_name:20} | MAPE: {metrics['mape']:6.2f}% | Time: {metrics['training_time']:5.1f}s")
        
    print(f"\\n🥇 Best Model: {best_model[0]} with {best_model[1]['mape']:.2f}% MAPE")
    
    return results

if __name__ == "__main__":
    results = main()
