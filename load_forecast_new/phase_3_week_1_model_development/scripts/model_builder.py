"""
Model Architecture Builder for Delhi Load Forecasting
Phase 3 Week 1: Baseline Model Development
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow verbosity

import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Dropout, BatchNormalization, 
    Conv1D, MaxPooling1D, Flatten, Input, Concatenate,
    Attention, MultiHeadAttention, LayerNormalization, Reshape
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l1, l2, l1_l2
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DelhiModelBuilder:
    """
    Comprehensive model builder for Delhi Load Forecasting with multiple architectures.
    """
    
    def __init__(self, config_path: str = "../config/model_config.yaml"):
        """
        Initialize the model builder with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model_registry = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Create directories
        self._create_directories()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _create_directories(self):
        """Create necessary directories for models."""
        directories = [
            'models/checkpoints', 'models/saved_models', 'models/scalers',
            'evaluation/figures', 'logs', 'models/tensorboard'
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def build_gru_model(self, input_shape: Tuple, model_config: Dict) -> Sequential:
        """
        Build GRU-based model for load forecasting.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            model_config: Model configuration dictionary
            
        Returns:
            Compiled Keras model
        """
        model = Sequential(name="GRU_LoadForecaster")
        
        # Add GRU layers
        gru_layers = model_config['layers']
        for i, layer_config in enumerate(gru_layers):
            is_first = (i == 0)
            
            model.add(GRU(
                units=layer_config['units'],
                return_sequences=layer_config.get('return_sequences', True),
                dropout=layer_config.get('dropout', 0.0),
                recurrent_dropout=layer_config.get('recurrent_dropout', 0.0),
                input_shape=input_shape if is_first else None,
                name=f'gru_layer_{i+1}'
            ))
            
            # Add batch normalization if specified
            if layer_config.get('batch_normalization', False):
                model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Add dropout if specified
            if layer_config.get('dropout', 0.0) > 0:
                model.add(Dropout(layer_config['dropout'], name=f'dropout_{i+1}'))
        
        # Add dense layers
        dense_layers = model_config.get('dense_layers', [])
        for i, dense_config in enumerate(dense_layers):
            model.add(Dense(
                units=dense_config['units'],
                activation=dense_config.get('activation', 'relu'),
                name=f'dense_{i+1}'
            ))
            
            # Add dropout if specified
            if dense_config.get('dropout', 0.0) > 0:
                model.add(Dropout(dense_config['dropout'], name=f'dense_dropout_{i+1}'))
        
        # Add reshape layer to convert from (batch_size, 120) to (batch_size, 24, 5)
        model.add(Reshape((24, 5), name='output_reshape'))
        
        logger.info(f"GRU model built with {len(gru_layers)} GRU layers and {len(dense_layers)} dense layers")
        return model
    
    def build_lstm_model(self, input_shape: Tuple, model_config: Dict) -> Sequential:
        """
        Build LSTM-based model for load forecasting.
        
        Args:
            input_shape: Shape of input data
            model_config: Model configuration dictionary
            
        Returns:
            Compiled Keras model
        """
        model = Sequential(name="LSTM_LoadForecaster")
        
        # Add LSTM layers
        lstm_layers = model_config['layers']
        for i, layer_config in enumerate(lstm_layers):
            is_first = (i == 0)
            
            model.add(LSTM(
                units=layer_config['units'],
                return_sequences=layer_config.get('return_sequences', True),
                dropout=layer_config.get('dropout', 0.0),
                recurrent_dropout=layer_config.get('recurrent_dropout', 0.0),
                input_shape=input_shape if is_first else None,
                name=f'lstm_layer_{i+1}'
            ))
            
            # Add batch normalization if specified
            if layer_config.get('batch_normalization', False):
                model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
            
            # Add dropout if specified
            if layer_config.get('dropout', 0.0) > 0:
                model.add(Dropout(layer_config['dropout'], name=f'dropout_{i+1}'))
        
        # Add dense layers
        dense_layers = model_config.get('dense_layers', [])
        for i, dense_config in enumerate(dense_layers):
            model.add(Dense(
                units=dense_config['units'],
                activation=dense_config.get('activation', 'relu'),
                name=f'dense_{i+1}'
            ))
            
            # Add dropout if specified
            if dense_config.get('dropout', 0.0) > 0:
                model.add(Dropout(dense_config['dropout'], name=f'dense_dropout_{i+1}'))
        
        # Add reshape layer to convert from (batch_size, 120) to (batch_size, 24, 5)
        model.add(Reshape((24, 5), name='output_reshape'))
        
        logger.info(f"LSTM model built with {len(lstm_layers)} LSTM layers and {len(dense_layers)} dense layers")
        return model
    
    def build_cnn_gru_hybrid(self, input_shape: Tuple, model_config: Dict) -> Sequential:
        """
        Build CNN-GRU hybrid model for load forecasting.
        
        Args:
            input_shape: Shape of input data
            model_config: Model configuration dictionary
            
        Returns:
            Compiled Keras model
        """
        model = Sequential(name="CNN_GRU_Hybrid")
        
        # Add CNN layers
        cnn_layers = model_config.get('cnn_layers', [])
        for i, layer_config in enumerate(cnn_layers):
            is_first = (i == 0)
            
            model.add(Conv1D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation=layer_config.get('activation', 'relu'),
                padding='same',
                input_shape=input_shape if is_first else None,
                name=f'conv1d_{i+1}'
            ))
            
            # Add dropout if specified
            if layer_config.get('dropout', 0.0) > 0:
                model.add(Dropout(layer_config['dropout'], name=f'conv_dropout_{i+1}'))
        
        # Add GRU layers
        gru_layers = model_config.get('gru_layers', [])
        for i, layer_config in enumerate(gru_layers):
            model.add(GRU(
                units=layer_config['units'],
                return_sequences=layer_config.get('return_sequences', False),
                dropout=layer_config.get('dropout', 0.0),
                name=f'gru_layer_{i+1}'
            ))
            
            # Add batch normalization if specified
            if layer_config.get('batch_normalization', False):
                model.add(BatchNormalization(name=f'gru_batch_norm_{i+1}'))
        
        # Add dense layers
        dense_layers = model_config.get('dense_layers', [])
        for i, dense_config in enumerate(dense_layers):
            model.add(Dense(
                units=dense_config['units'],
                activation=dense_config.get('activation', 'relu'),
                name=f'dense_{i+1}'
            ))
            
            # Add dropout if specified
            if dense_config.get('dropout', 0.0) > 0:
                model.add(Dropout(dense_config['dropout'], name=f'dense_dropout_{i+1}'))
        
        # Add reshape layer to convert from (batch_size, 120) to (batch_size, 24, 5)
        model.add(Reshape((24, 5), name='output_reshape'))
        
        logger.info(f"CNN-GRU hybrid model built")
        return model
    
    def build_attention_model(self, input_shape: Tuple) -> Model:
        """
        Build attention-based model for load forecasting.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled Keras model with attention mechanism
        """
        # Input layer
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # GRU layers with return_sequences=True for attention
        x = GRU(64, return_sequences=True, name='gru_1')(inputs)
        x = BatchNormalization(name='batch_norm_1')(x)
        x = Dropout(0.2, name='dropout_1')(x)
        
        x = GRU(32, return_sequences=True, name='gru_2')(x)
        x = BatchNormalization(name='batch_norm_2')(x)
        
        # Multi-head attention layer
        attention_output = MultiHeadAttention(
            num_heads=4, 
            key_dim=32,
            name='multi_head_attention'
        )(x, x)
        
        # Add residual connection and layer normalization
        x = LayerNormalization(name='layer_norm')(x + attention_output)
        
        # Global average pooling (alternative to flattening)
        x = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(x)
        
        # Dense layers
        x = Dense(16, activation='relu', name='dense_1')(x)
        x = Dropout(0.1, name='final_dropout')(x)
        
        # Output layer (prediction_horizon * n_targets: 24 * 5 = 120)
        x = Dense(120, activation='linear', name='output')(x)
        
        # Reshape to (batch_size, 24, 5)
        outputs = Reshape((24, 5), name='output_reshape')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='Attention_LoadForecaster')
        
        logger.info("Attention-based model built successfully")
        return model
    
    def build_ensemble_model(self, input_shape: Tuple) -> Model:
        """
        Build ensemble model combining multiple architectures.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled ensemble model
        """
        # Input layer
        inputs = Input(shape=input_shape, name='ensemble_input')
        
        # GRU branch
        gru_branch = GRU(64, return_sequences=False, name='ensemble_gru')(inputs)
        gru_branch = BatchNormalization(name='gru_bn')(gru_branch)
        gru_branch = Dropout(0.2, name='gru_dropout')(gru_branch)
        gru_output = Dense(16, activation='relu', name='gru_dense')(gru_branch)
        
        # CNN branch
        cnn_branch = Conv1D(32, 3, activation='relu', padding='same', name='ensemble_conv')(inputs)
        cnn_branch = MaxPooling1D(2, name='ensemble_pool')(cnn_branch)
        cnn_branch = Flatten(name='ensemble_flatten')(cnn_branch)
        cnn_branch = Dense(32, activation='relu', name='cnn_dense_1')(cnn_branch)
        cnn_output = Dense(16, activation='relu', name='cnn_dense_2')(cnn_branch)
        
        # Combine branches
        combined = Concatenate(name='ensemble_concat')([gru_output, cnn_output])
        combined = Dense(32, activation='relu', name='combined_dense_1')(combined)
        combined = Dropout(0.2, name='combined_dropout')(combined)
        combined = Dense(16, activation='relu', name='combined_dense_2')(combined)
        
        # Output layer (prediction_horizon * n_targets: 24 * 5 = 120)
        combined = Dense(120, activation='linear', name='ensemble_output')(combined)
        
        # Reshape to (batch_size, 24, 5)
        outputs = Reshape((24, 5), name='output_reshape')(combined)
        
        model = Model(inputs=inputs, outputs=outputs, name='Ensemble_LoadForecaster')
        
        logger.info("Ensemble model built successfully")
        return model
    
    def compile_model(self, model: Model, model_type: str = 'gru') -> Model:
        """
        Compile model with appropriate optimizer and loss function.
        
        Args:
            model: Keras model to compile
            model_type: Type of model for logging
            
        Returns:
            Compiled model
        """
        training_config = self.config['training']
        
        # Choose optimizer
        optimizer_name = training_config.get('optimizer', 'adam').lower()
        learning_rate = training_config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=training_config.get('loss_function', 'mse'),
            metrics=['mae', 'mse']
        )
        
        logger.info(f"{model_type.upper()} model compiled with {optimizer_name} optimizer")
        return model
    
    def create_callbacks(self, model_name: str) -> List:
        """
        Create training callbacks for model.
        
        Args:
            model_name: Name of the model for file naming
            
        Returns:
            List of callbacks
        """
        callbacks_config = self.config['training']['callbacks']
        callbacks = []
        
        # Early stopping
        if 'early_stopping' in callbacks_config:
            es_config = callbacks_config['early_stopping']
            callbacks.append(EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 15),
                restore_best_weights=es_config.get('restore_best_weights', True),
                verbose=1
            ))
        
        # Learning rate reduction
        if 'reduce_lr' in callbacks_config:
            lr_config = callbacks_config['reduce_lr']
            callbacks.append(ReduceLROnPlateau(
                monitor=lr_config.get('monitor', 'val_loss'),
                factor=lr_config.get('factor', 0.5),
                patience=lr_config.get('patience', 8),
                min_lr=lr_config.get('min_lr', 1e-6),
                verbose=1
            ))
        
        # Model checkpoint
        if 'model_checkpoint' in callbacks_config:
            checkpoint_path = f"models/checkpoints/{model_name}_best_model.h5"
            callbacks.append(ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ))
        
        # TensorBoard logging
        log_dir = f"models/tensorboard/{model_name}"
        callbacks.append(TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ))
        
        logger.info(f"Created {len(callbacks)} callbacks for {model_name}")
        return callbacks
    
    def build_model(self, model_type: str, input_shape: Tuple) -> Model:
        """
        Build and compile model based on type.
        
        Args:
            model_type: Type of model to build ('gru', 'lstm', 'cnn_gru', 'attention', 'ensemble')
            input_shape: Shape of input data
            
        Returns:
            Compiled Keras model
        """
        model_type = model_type.lower()
        
        if model_type == 'gru':
            model_config = self.config['models']['gru_baseline']
            model = self.build_gru_model(input_shape, model_config)
        elif model_type == 'lstm':
            model_config = self.config['models']['lstm_baseline']
            model = self.build_lstm_model(input_shape, model_config)
        elif model_type == 'cnn_gru':
            model_config = self.config['models']['cnn_gru_hybrid']
            model = self.build_cnn_gru_hybrid(input_shape, model_config)
        elif model_type == 'attention':
            model = self.build_attention_model(input_shape)
        elif model_type == 'ensemble':
            model = self.build_ensemble_model(input_shape)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Compile model
        model = self.compile_model(model, model_type)
        
        # Store in registry
        self.model_registry[model_type] = model
        
        # Print model summary
        print(f"\\n{model_type.upper()} Model Summary:")
        print("=" * 50)
        model.summary()
        
        return model
    
    def save_model_architecture(self, model: Model, model_name: str):
        """
        Save model architecture and configuration.
        
        Args:
            model: Keras model to save
            model_name: Name for saving files
        """
        # Save model architecture as JSON
        architecture_path = f"models/saved_models/{model_name}_architecture.json"
        with open(architecture_path, 'w') as f:
            f.write(model.to_json())
        
        # Create architecture visualization
        plot_path = f"evaluation/figures/{model_name}_architecture.png"
        tf.keras.utils.plot_model(
            model,
            to_file=plot_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=150
        )
        
        logger.info(f"Model architecture saved: {architecture_path}")
        logger.info(f"Model plot saved: {plot_path}")
    
    def get_model_info(self, model: Model) -> Dict:
        """
        Get detailed information about the model.
        
        Args:
            model: Keras model
            
        Returns:
            Dictionary with model information
        """
        return {
            'name': model.name,
            'total_params': model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
            'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
            'layers': len(model.layers),
            'input_shape': model.input_shape,
            'output_shape': model.output_shape
        }

def create_all_models(input_shape: Tuple) -> Dict[str, Model]:
    """
    Create all model architectures for comparison.
    
    Args:
        input_shape: Shape of input data
        
    Returns:
        Dictionary of model name -> compiled model
    """
    builder = DelhiModelBuilder()
    models = {}
    
    model_types = ['gru', 'lstm', 'cnn_gru', 'attention', 'ensemble']
    
    print("Building all model architectures for Delhi Load Forecasting...")
    print("=" * 60)
    
    for model_type in model_types:
        try:
            print(f"\\nBuilding {model_type.upper()} model...")
            model = builder.build_model(model_type, input_shape)
            models[model_type] = model
            
            # Save architecture
            builder.save_model_architecture(model, model_type)
            
            # Print model info
            info = builder.get_model_info(model)
            print(f"Model Info: {info['total_params']:,} total params, {info['layers']} layers")
            
        except Exception as e:
            logger.error(f"Error building {model_type} model: {e}")
            continue
    
    print(f"\\nSuccessfully built {len(models)} models")
    return models

if __name__ == "__main__":
    # Example usage
    # Assume input shape: (sequence_length=24, n_features=30)
    input_shape = (24, 30)
    
    # Create all models
    models = create_all_models(input_shape)
    
    # Display model information
    for model_name, model in models.items():
        print(f"\\n{model_name.upper()} Model:")
        builder = DelhiModelBuilder()
        info = builder.get_model_info(model)
        for key, value in info.items():
            print(f"  {key}: {value}")
