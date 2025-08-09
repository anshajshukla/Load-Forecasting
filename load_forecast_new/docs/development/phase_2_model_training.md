# Phase 2: Model Training

This phase covers the process of training deep learning models for load forecasting using TensorFlow/Keras. The main logic is implemented in the `train_model` function and related utilities.

## Steps and Techniques

### 1. Model Initialization
- The model (e.g., GRU, LSTM, BiLSTM) is defined and passed as an argument.
- The model should have a `.model_type` attribute for logging and saving.

### 2. Callback Setup
- **EarlyStopping:**
  - Monitors validation loss (`val_loss`).
  - Stops training if no improvement after a set number of epochs (`patience`).
  - Restores the best weights.
- **ReduceLROnPlateau:**
  - Reduces learning rate by a factor (default 0.5) if validation loss plateaus.
  - Minimum learning rate is set to `1e-6`.
- **ModelCheckpoint:**
  - Saves the best model (lowest validation loss) to disk.
  - File is named `best_{model_filename}`.
- **TensorBoard:**
  - Logs training metrics for visualization in TensorBoard.
  - Log directory includes model type and timestamp.

### 3. Training Loop
- Uses `model.model.fit()` with:
  - Training data (`X_train`, `y_train`)
  - Validation split (default 0.15)
  - Batch size (default 64)
  - Number of epochs (default 150)
  - All callbacks above
  - Verbose output for progress
- Tracks training time for reporting.

### 4. Saving the Model
- Saves the final trained model to disk as `{model_filename}`.
- The best model (from checkpoint) is also saved.

### 5. Training History Visualization
- **plot_training_history:**
  - Plots training and validation loss curves.
  - Plots learning rate or convergence rate per epoch.
  - Saves plots as PNG files in a `figures` directory.
  - Saves a text summary of training metrics (best epoch, losses, learning rate, etc.).
- **save_training_history:**
  - Additional utility to save loss and learning rate plots.

### 6. Output
- Trained model files (`.h5`)
- Best model checkpoint
- Training history plots and metrics (PNG, TXT)
- TensorBoard logs for further analysis

---

**Key Libraries Used:**
- TensorFlow/Keras (Model, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard)
- matplotlib (for plotting)
- numpy, os, datetime, time

**Functions:**
- `train_model`, `plot_training_history`, `save_training_history`

**Best Practices:**
- Uses callbacks for robust, efficient training
- Saves both best and final models
- Provides detailed visual and textual feedback on training progress
- Modular and reusable for different model architectures 