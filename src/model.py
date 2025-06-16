from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import os

class LoadForecastModel:
    """GRU-based model for multi-region load forecasting."""
    
    def __init__(self, input_shape, output_dim, l2_reg=0.001):
        """
        Initialize the model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            output_dim (int): Number of output dimensions (number of regions)
            l2_reg (float): L2 regularization value
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.l2_reg = l2_reg
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the GRU model."""
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
        
        print("GRU model built successfully with the following architecture:")
        model.summary()
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with appropriate optimizer and loss function."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        print(f"Model compiled with learning rate: {learning_rate}")
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
