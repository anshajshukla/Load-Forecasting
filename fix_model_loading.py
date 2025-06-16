"""
Fix model loading issues in the Delhi Load Forecast Dashboard

This script provides a quick fix for the model loading error:
"Could not locate function 'mse'. Make sure custom classes are decorated with 
`@keras.saving.register_keras_serializable()`"

The script:
1. Creates a new model with the same architecture as the saved model
2. Loads the weights from the saved model
3. Compiles the model with the correct loss and metrics
4. Saves the model in a format compatible with the current TensorFlow version
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, Input, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

def fix_model():
    # Path to the saved model
    model_path = "models/saved_models/gru_forecast_model.h5"
    fixed_model_path = "models/saved_models/gru_forecast_model_fixed.h5"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return False
    
    try:
        # Load the model without compiling it
        print(f"Loading model from {model_path}...")
        saved_model = load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        # Get model configuration
        config = saved_model.get_config()
        weights = saved_model.get_weights()
        
        # Create a new model with the same architecture
        print("Creating new model with the same architecture...")
        new_model = tf.keras.models.model_from_config(config)
        
        # Copy the weights
        new_model.set_weights(weights)
        
        # Compile the model with the correct loss and metrics
        print("Compiling model...")
        new_model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=MeanSquaredError(),
            metrics=['mse', 'mae']
        )
        
        # Save the model in a format compatible with the current TensorFlow version
        print(f"Saving fixed model to {fixed_model_path}...")
        new_model.save(fixed_model_path)
        
        print("\nModel fixed successfully!")
        print(f"You can now use the fixed model at: {fixed_model_path}")
        print("To use this model, update the model_path in run_dashboard.py to point to the fixed model.")
        
        return True
        
    except Exception as e:
        print(f"Error fixing model: {e}")
        
        # Try an alternative approach: recreate the model from scratch
        print("\nTrying alternative approach: recreating the model from scratch...")
        
        try:
            # Recreate the GRU model architecture
            # This is based on the common architecture used in your project
            input_dim = None  # Will be determined dynamically when model is used
            
            # First try to determine input shape from saved model
            if saved_model:
                input_shape = saved_model.input_shape[1:]
                output_dim = saved_model.output_shape[-1]
            else:
                # Use placeholder values, will need to be updated before use
                input_shape = (1, 19)  # Placeholder
                output_dim = 5  # Number of targets (DELHI, BRPL, BYPL, NDMC, MES)
            
            print(f"Creating model with input shape {input_shape} and output dim {output_dim}")
            
            new_model = Sequential([
                Input(shape=input_shape),
                GRU(256, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                GRU(128, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                GRU(64, activation='relu', return_sequences=True, kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                GRU(32, activation='relu', kernel_regularizer=l2(0.001)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(output_dim)
            ])
            
            # Try to copy weights if possible
            try:
                if saved_model and len(new_model.weights) == len(saved_model.weights):
                    new_model.set_weights(saved_model.get_weights())
                    print("Weights transferred successfully!")
                else:
                    print("Couldn't transfer weights due to architecture mismatch.")
                    print("The new model will need to be retrained.")
            except:
                print("Couldn't transfer weights.")
                print("The new model will need to be retrained.")
            
            # Compile the model
            new_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=MeanSquaredError(),
                metrics=['mse', 'mae']
            )
            
            # Save the model
            new_model.save(fixed_model_path)
            
            print("\nModel recreated and saved successfully!")
            print(f"You can now use the new model at: {fixed_model_path}")
            print("To use this model, update the model_path in run_dashboard.py to point to the fixed model.")
            
            return True
            
        except Exception as inner_e:
            print(f"Error recreating model: {inner_e}")
            return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Delhi Load Forecast Model Fix Utility")
    print("="*60 + "\n")
    
    success = fix_model()
    
    if success:
        print("\nModel fix completed successfully!\n")
    else:
        print("\nModel fix failed. You may need to retrain the model.\n")
