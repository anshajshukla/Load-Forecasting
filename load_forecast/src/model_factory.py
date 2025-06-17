import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, Dropout, BatchNormalization, Bidirectional, Conv1D, MaxPooling1D, Flatten, Layer, Add, MultiHeadAttention, LayerNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class ModelFactory:
    """Factory for creating different types of forecasting models."""
    
    @staticmethod
    def create_model(model_type, input_shape, output_dim, l2_reg=0.001):
        """
        Create a model of the specified type.
        
        Args:
            model_type (str): Type of model ('gru', 'enhanced_gru', 'lstm', 'bilstm', 'cnn_lstm', 'improved')
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
        elif model_type == 'improved':
            return ImprovedModel(input_shape, output_dim, l2_reg)
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
            loss='mse',
            metrics=['mae']
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
            GRU(64, input_shape=self.input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)),
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
            GRU(128, input_shape=self.input_shape, return_sequences=True),
            Dropout(0.2),
            GRU(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)),
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
            LSTM(64, input_shape=self.input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)),
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
            Bidirectional(LSTM(64, return_sequences=False), input_shape=self.input_shape),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)),
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
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(self.l2_reg)),
            Dense(self.output_dim)
        ])
        
        print("CNN-LSTM hybrid model built successfully")
        return model


class AttentionLayer(Layer):
    """Custom attention layer for time series data."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        e = np.tanh(np.dot(x, self.W) + self.b)
        a = np.softmax(e, axis=1)
        output = x * a
        return output, a
    
    def compute_output_shape(self, input_shape):
        return input_shape, (input_shape[0], input_shape[1], 1)

class ImprovedModel(BaseModel):
    """Enhanced model with attention mechanism and residual connections."""
    
    def _build_model(self):
        """Build an improved model with attention and residual connections."""
        self.model_type = "Improved"
        
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # First GRU block with residual connection
        x = GRU(128, return_sequences=True)(inputs)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Second GRU block with residual connection
        y = GRU(128, return_sequences=True)(x)
        y = LayerNormalization()(y)
        y = Dropout(0.2)(y)
        x = Add()([x, y])  # Residual connection
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=32)(x, x)
        x = Add()([x, attention_output])  # Residual connection
        x = LayerNormalization()(x)
        
        # Final GRU layer
        x = GRU(64, return_sequences=False)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers with residual connection
        y = Dense(64, activation='relu', kernel_regularizer=l2(self.l2_reg))(x)
        y = LayerNormalization()(y)
        y = Dropout(0.2)(y)
        x = Add()([x, y])  # Residual connection
        
        # Output layer
        outputs = Dense(self.output_dim)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        print("Improved model built successfully")
        return model
