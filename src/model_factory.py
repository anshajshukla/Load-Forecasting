import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class ModelFactory:
    """Factory for creating different types of forecasting models."""
    
    @staticmethod
    def create_model(model_type, input_shape, output_dim, l2_reg=0.001):
        """
        Create a model of the specified type.
        
        Args:
            model_type (str): Type of model ('gru', 'lstm', 'bilstm', 'cnn_lstm')
            input_shape (tuple): Shape of input data (time_steps, features)
            output_dim (int): Number of output dimensions (number of regions)
            l2_reg (float): L2 regularization value
            
        Returns:
            model: The created model
        """
        if model_type == 'gru':
            return GRUModel(input_shape, output_dim, l2_reg)
        elif model_type == 'enhanced_gru':
            return EnhancedGRUModel(input_shape, output_dim, l2_reg)
        elif model_type == 'lstm':
            return LSTMModel(input_shape, output_dim, l2_reg)
        elif model_type == 'bilstm':
            return BiLSTMModel(input_shape, output_dim, l2_reg)
        elif model_type == 'cnn_lstm':
            return CNNLSTMModel(input_shape, output_dim, l2_reg)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class BaseModel:
    """Base class for all forecasting models."""
    
    def __init__(self, input_shape, output_dim, l2_reg=0.001):
        """
        Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            output_dim (int): Number of output dimensions (number of regions)
            l2_reg (float): L2 regularization value
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.l2_reg = l2_reg
        self.model = self._build_model()
        self.model_type = "base"
    
    def _build_model(self):
        """Build the model. To be implemented by subclasses."""
        raise NotImplementedError
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate optimizer and loss function."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        print(f"{self.model_type} model compiled with learning rate: {learning_rate}")
        return self.model
    
    def save_model(self, filepath):
        """Save the model to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath):
        """Load a model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        model = load_model(filepath)
        print(f"Model loaded from: {filepath}")
        return model


class GRUModel(BaseModel):
    """Original GRU-based model for load forecasting."""
    
    def _build_model(self):
        """Build the original GRU model."""
        self.model_type = "GRU"
        model = Sequential([
            Input(shape=self.input_shape),
            GRU(256, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(128, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(32, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.output_dim)
        ])
        
        print("GRU model built successfully")
        return model


class EnhancedGRUModel(BaseModel):
    """Enhanced GRU model with additional layers and features."""
    
    def _build_model(self):
        """Build an enhanced GRU model with better performance."""
        self.model_type = "Enhanced GRU"
        model = Sequential([
            Input(shape=self.input_shape),
            GRU(512, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(256, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(128, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            GRU(64, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dense(self.output_dim)
        ])
        
        print("Enhanced GRU model built successfully")
        return model


class LSTMModel(BaseModel):
    """LSTM-based model for load forecasting."""
    
    def _build_model(self):
        """Build the LSTM model."""
        self.model_type = "LSTM"
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(256, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(self.output_dim)
        ])
        
        print("LSTM model built successfully")
        return model


class BiLSTMModel(BaseModel):
    """Bidirectional LSTM model for load forecasting."""
    
    def _build_model(self):
        """Build the bidirectional LSTM model."""
        self.model_type = "Bidirectional LSTM"
        model = Sequential([
            Input(shape=self.input_shape),
            Bidirectional(LSTM(256, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg))),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=l2(self.l2_reg))),
            BatchNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(64, activation='relu', kernel_regularizer=l2(self.l2_reg))),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(self.output_dim)
        ])
        
        print("Bidirectional LSTM model built successfully")
        return model


class CNNLSTMModel(BaseModel):
    """CNN-LSTM hybrid model for load forecasting."""
    
    def _build_model(self):
        """Build the CNN-LSTM hybrid model."""
        self.model_type = "CNN-LSTM"
        model = Sequential([
            Input(shape=self.input_shape),
            # CNN layers for feature extraction
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            # LSTM layers for temporal dynamics
            LSTM(128, return_sequences=True, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, kernel_regularizer=l2(self.l2_reg)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(self.output_dim)
        ])
        
        print("CNN-LSTM hybrid model built successfully")
        return model
