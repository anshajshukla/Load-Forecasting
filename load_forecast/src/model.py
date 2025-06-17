import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class LoadForecastModel(nn.Module):
    """GRU-based model for multi-region load forecasting."""
    
    def __init__(self, input_shape, output_dim):
        """
        Initialize the model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
            output_dim (int): Number of output dimensions (number of regions)
        """
        super(LoadForecastModel, self).__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # GRU layers
        self.gru1 = nn.GRU(
            input_size=input_shape[1],
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.gru2 = nn.GRU(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True
        )
        
        # Dense layers
        self.dense1 = nn.Linear(32, 16)
        self.dense2 = nn.Linear(16, output_dim)
        
        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(32)
        
    def forward(self, x):
        """Forward pass of the model."""
        # GRU layers
        gru1_out, _ = self.gru1(x)
        gru1_out = self.batch_norm1(gru1_out.transpose(1, 2)).transpose(1, 2)
        gru1_out = self.dropout(gru1_out)
        
        gru2_out, _ = self.gru2(gru1_out)
        gru2_out = self.batch_norm2(gru2_out.transpose(1, 2)).transpose(1, 2)
        gru2_out = self.dropout(gru2_out)
        
        # Take the last time step's output
        last_time_step = gru2_out[:, -1, :]
        
        # Dense layers
        dense1_out = self.relu(self.dense1(last_time_step))
        output = self.dense2(dense1_out)
        
        return output
    
    def compile_model(self):
        """Initialize optimizer and loss function."""
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def save_model(self, filepath):
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_shape': self.input_shape,
            'output_dim': self.output_dim
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath):
        """Load model from disk."""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at {filepath}")
        
        checkpoint = torch.load(filepath)
        model = cls(checkpoint['input_shape'], checkpoint['output_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()  # Set to evaluation mode
        return model
    
    def predict(self, x):
        """Make predictions."""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            return self(x).numpy()
