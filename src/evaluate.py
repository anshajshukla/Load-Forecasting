import os
import numpy as np
import pandas as pd
# Set matplotlib to use non-interactive backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import pickle
from datetime import datetime

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate mean absolute percentage error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        mape: Mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mape_values = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    # Clip extremely high values to 1000% to avoid skewing averages
    mape_values = np.clip(mape_values, 0, 1000)
    return np.mean(mape_values)

def evaluate_model(model, X_test, y_test, scalers_y, targets, save_dir='../models/saved_models', model_type=None):
    """
    Evaluate the model on test data.
    
    Args:
        model: The trained model
        X_test: Test features
        y_test: Test targets
        scalers_y: Dictionary of target scalers
        targets: List of target names
        save_dir: Directory to save evaluation results
        model_type: Type of model being evaluated
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # If model_type is not provided, try to get it from the model
    if model_type is None and hasattr(model, 'model_type'):
        model_type = model.model_type
    elif model_type is None:
        model_type = 'unknown'
        
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    y_test_inv = np.zeros_like(y_test)
    y_pred_inv = np.zeros_like(y_pred)
    
    for i, target in enumerate(targets):
        y_test_inv[:, i] = scalers_y[target].inverse_transform(y_test[:, i].reshape(-1, 1)).ravel()
        y_pred_inv[:, i] = scalers_y[target].inverse_transform(y_pred[:, i].reshape(-1, 1)).ravel()
    
    # Calculate metrics
    metrics = {}
    overall_rmse = []
    overall_mape = []
    overall_r2 = []
    
    for i, target in enumerate(targets):
        rmse = np.sqrt(mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i]))
        mape = mean_absolute_percentage_error(y_test_inv[:, i], y_pred_inv[:, i])
        r2 = r2_score(y_test_inv[:, i], y_pred_inv[:, i])
        
        # Additional metrics
        mae = np.mean(np.abs(y_test_inv[:, i] - y_pred_inv[:, i]))
        max_error = np.max(np.abs(y_test_inv[:, i] - y_pred_inv[:, i]))
        
        # Store metrics
        metrics[target] = {
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'MAE': mae,
            'MAX_ERROR': max_error
        }
        
        # Track for overall metrics
        overall_rmse.append(rmse)
        overall_mape.append(mape)
        overall_r2.append(r2)
        
        print(f"\n{target} Metrics:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  R²: {r2:.4f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  Max Error: {max_error:.2f}")
        print()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create save directory if it doesn't exist
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Calculate overall metrics
    print("\nOverall Model Performance:")
    print(f"  Average RMSE: {np.mean(overall_rmse):.2f}")
    print(f"  Average MAPE: {np.mean(overall_mape):.2f}%")
    print(f"  Average R²: {np.mean(overall_r2):.4f}")
    
    # Visualize predictions
    from predict import visualize_predictions
    visualize_predictions(y_test_inv, y_pred_inv, targets, save_dir, model_type)
    
    # Save metrics to file
    save_evaluation_metrics(metrics, save_dir, model_type)
    
    return metrics

def save_metrics(metrics, save_dir):
    """
    Save evaluation metrics to CSV file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save metrics
    """
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame()
    
    for target, values in metrics.items():
        for metric, value in values.items():
            metrics_df.loc[target, metric] = value
    
    # Save to CSV
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path)
    print(f"Evaluation metrics saved to {metrics_path}")
    
    # Create a bar chart for MAPE
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df.index, metrics_df['MAPE'], color='skyblue')
    plt.title('Mean Absolute Percentage Error by Region')
    plt.ylabel('MAPE (%)')
    plt.xlabel('Region')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add MAPE values on top of bars
    for i, v in enumerate(metrics_df['MAPE']):
        plt.text(i, v + 0.1, f"{v:.2f}%", ha='center')
    
    # Save the plot
    mape_path = os.path.join(save_dir, 'mape_comparison.png')
    plt.savefig(mape_path)
    plt.close()
    print(f"MAPE comparison chart saved to {mape_path}")

def save_evaluation_metrics(metrics, save_dir, model_type='unknown'):
    """
    Save evaluation metrics to files and create visualizations.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save results
        model_type: Type of model being evaluated
    """
    # Create directory if it doesn't exist
    metrics_dir = os.path.join(save_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame()
    
    for target, target_metrics in metrics.items():
        for metric_name, value in target_metrics.items():
            metrics_df.loc[target, metric_name] = value
    
    # Save to CSV
    csv_path = os.path.join(metrics_dir, f'{model_type}_metrics.csv')
    metrics_df.to_csv(csv_path)
    print(f"Metrics saved to {csv_path}")
    
    # Save metrics as pickle for later comparison
    pickle_path = os.path.join(metrics_dir, f'{model_type}_metrics.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(metrics, f)
    
    # Create heatmap of metrics
    plt.figure(figsize=(12, 8))
    
    # Plot MAPE heatmap (most intuitive metric for stakeholders)
    mape_values = metrics_df['MAPE'].values.reshape(-1, 1)  # Convert to 2D array for heatmap
    
    # Create a new DataFrame for the heatmap
    heatmap_df = pd.DataFrame(mape_values, index=metrics_df.index, columns=['MAPE (%)'])
    
    # Create heatmap with annotated values
    ax = sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlGnBu_r')
    plt.title(f'{model_type.upper()} Model - MAPE by Region (%)', fontsize=14)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(metrics_dir, f'{model_type}_mape_heatmap.png')
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    # Create bar chart of all metrics
    plt.figure(figsize=(15, 10))
    
    # Get unique metrics (columns)
    metric_names = metrics_df.columns.tolist()
    
    # Create subplots for each metric
    for i, metric in enumerate(metric_names):
        plt.subplot(len(metric_names), 1, i+1)
        
        # Sort for better visualization
        sorted_df = metrics_df.sort_values(by=metric)
        
        # Define colors based on metric (lower is better for RMSE, MAPE, MAE, MAX_ERROR; higher is better for R2)
        colors = ['#2ca02c' if metric == 'R2' else '#d62728' for _ in range(len(sorted_df))]
        
        # Create bar chart
        bars = plt.barh(sorted_df.index, sorted_df[metric], color=colors)
        
        # Add value annotations
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width * 1.01
            plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                     va='center', fontsize=8)
        
        plt.title(f'{metric}', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust x-axis label based on metric
        if metric == 'MAPE':
            plt.xlabel('Percentage (%)', fontsize=10)
        elif metric == 'R2':
            plt.xlabel('R² Score (higher is better)', fontsize=10)
        else:
            plt.xlabel('Value', fontsize=10)
    
    plt.suptitle(f'{model_type.upper()} Model - Performance Metrics by Region', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save bar chart
    bar_path = os.path.join(metrics_dir, f'{model_type}_metrics_bars.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()
    
    print(f"Metric visualizations saved to {metrics_dir}")
    
    return metrics_df


def visualize_predictions(y_true, y_pred, targets, save_dir, num_samples=100):
    """
    Visualize actual vs predicted values for each target.
    
    Args:
        y_true: Actual values (inverse transformed)
        y_pred: Predicted values (inverse transformed)
        targets: List of target names
        save_dir: Directory to save visualizations
        num_samples: Number of samples to plot
    """
    # Create visualizations directory
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Use a subset of data for visualization
    n = min(num_samples, y_true.shape[0])
    indices = np.random.choice(y_true.shape[0], n, replace=False)
    
    # Create individual plots for each target
    for i, target in enumerate(targets):
        plt.figure(figsize=(12, 6))
        
        # Sort by actual values for better visualization
        sorted_indices = np.argsort(y_true[indices, i])
        x = np.arange(len(sorted_indices))
        
        # Plot actual and predicted values
        plt.plot(x, y_true[indices[sorted_indices], i], 'b-', label='Actual')
        plt.plot(x, y_pred[indices[sorted_indices], i], 'r-', label='Predicted')
        
        plt.title(f'Actual vs Predicted: {target}')
        plt.xlabel('Sample Index')
        plt.ylabel('Load')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(vis_dir, f'{target}_predictions.png'))
        plt.close()
    
    # Create a scatter plot of actual vs predicted for all targets
    for i, target in enumerate(targets):
        plt.figure(figsize=(10, 10))
        
        # Plot the scatter plot
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        
        # Add a perfect prediction line
        max_val = max(np.max(y_true[:, i]), np.max(y_pred[:, i]))
        min_val = min(np.min(y_true[:, i]), np.min(y_pred[:, i]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f'Actual vs Predicted Scatter Plot: {target}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(vis_dir, f'{target}_scatter.png'))
        plt.close()
    
    print(f"Prediction visualizations saved to {vis_dir}")
