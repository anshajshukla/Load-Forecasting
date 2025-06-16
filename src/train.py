import os
import numpy as np
# Set matplotlib to use non-interactive backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import datetime
import time

def train_model(model, X_train, y_train, batch_size=64, epochs=150, patience=20, validation_split=0.15, 
               save_dir='models/saved_models', model_filename='forecast_model.h5'):
    """
    Train the load forecast model.
    
    Args:
        model: The model to train
        X_train: Training features
        y_train: Training targets
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        patience: Patience for early stopping
        validation_split: Proportion of training data to use for validation
        save_dir: Directory to save the model
        model_filename: Filename for the saved model
        
    Returns:
        history: Training history
        model: The trained model
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        restore_best_weights=True,
        verbose=1
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=patience//2,  # Half the early stopping patience
        min_lr=1e-6,
        verbose=1
    )
    
    checkpoint_path = os.path.join(save_dir, f'best_{model_filename}')
    model_checkpoint = ModelCheckpoint(
        checkpoint_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    # Add TensorBoard callback for better visualization
    log_dir = os.path.join(save_dir, 'logs', f'{model.model_type}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
    tensorboard_callback = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    callbacks = [early_stopping, lr_reducer, model_checkpoint, tensorboard_callback]
    
    # Train the model
    print(f"\nTraining {model.model_type} model with {epochs} epochs, batch size {batch_size}")
    print(f"Early stopping patience: {patience}, Learning rate reduction patience: {patience//2}")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}\n")
    
    start_time = time.time()
    history = model.model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Save the final model
    model_path = os.path.join(save_dir, model_filename)
    model.save_model(model_path)
    
    # Plot training history
    plot_training_history(history, save_dir, model.model_type)
    
    print(f"Model training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return history, model

def plot_training_history(history, save_dir, model_type='unknown'):
    """
    Plot the training history.
    
    Args:
        history: Training history from model.fit()
        save_dir: Directory to save the plots
        model_type: Type of model being trained (for filename)
    """
    # Create figures directory
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot training & validation loss values
    plt.figure(figsize=(15, 7))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], linewidth=2)
    plt.plot(history.history['val_loss'], linewidth=2)
    plt.title(f'{model_type.upper()} Model Loss', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['Train', 'Validation'], loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Learning rate plot
    plt.subplot(1, 2, 2)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], linewidth=2, color='green')
        plt.title(f'{model_type.upper()} Learning Rate', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        # If learning rate is not available, show convergence rate
        # (% improvement in loss per epoch)
        loss = history.history['loss']
        epochs = range(1, len(loss))
        convergence = [(loss[i-1] - loss[i])/loss[i-1]*100 for i in epochs]
        
        plt.plot(epochs, convergence, linewidth=2, color='purple')
        plt.title(f'{model_type.upper()} Convergence Rate', fontsize=14)
        plt.ylabel('Improvement per Epoch (%)', fontsize=12)
        plt.xlabel('Epoch', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{model_type}_training_history.png'), dpi=300)
    
    # Also save the training metrics to a text file
    with open(os.path.join(figures_dir, f'{model_type}_training_metrics.txt'), 'w') as f:
        f.write(f"Training Summary for {model_type.upper()} Model\n")
        f.write(f"{'='*50}\n\n")
        
        # Calculate best epoch and best values
        best_epoch = np.argmin(history.history['val_loss']) + 1
        best_val_loss = min(history.history['val_loss'])
        best_train_loss = history.history['loss'][best_epoch-1]
        
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Training loss at best epoch: {best_train_loss:.6f}\n")
        
        if 'lr' in history.history:
            f.write(f"Final learning rate: {history.history['lr'][-1]:.8f}\n")
        
        f.write(f"\nTotal epochs trained: {len(history.history['loss'])}\n")
        f.write(f"Initial training loss: {history.history['loss'][0]:.6f}\n")
        f.write(f"Initial validation loss: {history.history['val_loss'][0]:.6f}\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n")
    
    plt.close()

def save_training_history(history, save_dir):
    """
    Save training history as plots.
    
    Args:
        history: Training history object
        save_dir: Directory to save plots
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot training & validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.grid(True)
    
    # Save the plot
    plt_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plt_path)
    plt.close()
    print(f"Training history plot saved to {plt_path}")
    
    # Save learning rate changes
    if 'lr' in history.history or 'learning_rate' in history.history:
        plt.figure(figsize=(12, 6))
        lr_key = 'lr' if 'lr' in history.history else 'learning_rate'
        plt.plot(history.history[lr_key])
        plt.title('Learning Rate Adjustments')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.grid(True)
        
        # Save the plot
        lr_plt_path = os.path.join(save_dir, 'learning_rate_history.png')
        plt.savefig(lr_plt_path)
        plt.close()
        print(f"Learning rate history plot saved to {lr_plt_path}")
