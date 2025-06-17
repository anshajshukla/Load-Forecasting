import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import time

def train_model(model, X_train, y_train, batch_size=64, epochs=150, patience=20, validation_split=0.15, 
               save_dir='models/saved_models', model_filename='forecast_model.pt'):
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
    
    # Convert data to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    
    # Split data into training and validation sets
    val_size = int(len(X_train) * validation_split)
    train_size = len(X_train) - val_size
    X_train, X_val = torch.split(X_train, [train_size, val_size])
    y_train, y_val = torch.split(y_train, [train_size, val_size])
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Initialize early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }
    
    print(f"\nTraining model with {epochs} epochs, batch size {batch_size}")
    print(f"Early stopping patience: {patience}")
    print(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}\n")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.6f} - '
              f'Val Loss: {val_loss:.6f} - '
              f'LR: {current_lr:.6f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            
            # Save best model
            model_path = os.path.join(save_dir, f'best_{model_filename}')
            torch.save({
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs')
                break
    
    training_time = time.time() - start_time
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save final model
    model_path = os.path.join(save_dir, model_filename)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }, model_path)
    
    # Plot training history
    plot_training_history(history, save_dir)
    
    print(f"Model training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    return history, model

def plot_training_history(history, save_dir):
    """
    Plot the training history.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save the plots
    """
    # Create figures directory
    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Plot training & validation loss values
    plt.figure(figsize=(15, 7))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], linewidth=2)
    plt.plot(history['val_loss'], linewidth=2)
    plt.title('Model Loss', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend(['Train', 'Validation'], loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Learning rate plot
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'], linewidth=2, color='green')
    plt.title('Learning Rate', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'training_history.png'), dpi=300)
    plt.close()  # Close the figure to free memory
    
    # Save training metrics to a text file
    with open(os.path.join(figures_dir, 'training_metrics.txt'), 'w') as f:
        f.write("Training Summary\n")
        f.write(f"{'='*50}\n\n")
        
        # Calculate best epoch and best values
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val_loss = min(history['val_loss'])
        best_train_loss = history['train_loss'][best_epoch-1]
        
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"Training loss at best epoch: {best_train_loss:.6f}\n")
        f.write(f"Final learning rate: {history['lr'][-1]:.8f}\n")
        
        f.write(f"\nTotal epochs trained: {len(history['train_loss'])}\n")
        f.write(f"Initial training loss: {history['train_loss'][0]:.6f}\n")
        f.write(f"Initial validation loss: {history['val_loss'][0]:.6f}\n")
        f.write(f"Final training loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final validation loss: {history['val_loss'][-1]:.6f}\n")

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
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
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
    if 'lr' in history or 'learning_rate' in history:
        plt.figure(figsize=(12, 6))
        lr_key = 'lr' if 'lr' in history else 'learning_rate'
        plt.plot(history[lr_key])
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
