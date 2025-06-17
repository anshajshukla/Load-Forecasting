import argparse
import os
import sys

# Set matplotlib to use non-interactive backend before any other matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from model_factory import ModelFactory
from train import train_model
from evaluate import evaluate_model
from predict import predict_load, predict_future


def main():
    """Main function to run the load forecasting system."""
    parser = argparse.ArgumentParser(description='Regional Load Forecasting System')
    
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'evaluate', 'predict', 'compare_models'],
                       help='Mode of operation: train, evaluate, predict, or compare_models')
    
    parser.add_argument('--model_type', type=str, default='gru',
                       choices=['gru', 'enhanced_gru', 'lstm', 'bilstm', 'cnn_lstm'],
                       help='Type of model to train/evaluate/predict')
    
    parser.add_argument('--data_path', type=str, default='data/final_data.csv',
                       help='Path to the input data CSV file')
    
    parser.add_argument('--model_path', type=str, default='models/saved_models/forecasting_model.h5',
                       help='Path to save/load the model')
    
    parser.add_argument('--scalers_dir', type=str, default='models/saved_models',
                       help='Directory to save/load scalers')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of epochs for training')
    
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Patience for early stopping')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data to use for testing')
    
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate for training')
    
    parser.add_argument('--prediction_steps', type=int, default=24,
                       help='Number of future steps to predict')
    
    args = parser.parse_args()
    
    # Ensure data path is relative to script location
    data_path = os.path.join(os.path.dirname(__file__), args.data_path)
    model_path = os.path.join(os.path.dirname(__file__), args.model_path)
    scalers_dir = os.path.join(os.path.dirname(__file__), args.scalers_dir)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)
    
    # Print execution details
    print(f"\n{'-'*50}")
    print(f"Regional Load Forecasting System")
    print(f"Mode: {args.mode}")
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'-'*50}\n")
    
    if args.mode == 'train':
        print("Starting training mode...")
        
        # Process data
        data_processor = DataProcessor(data_path, test_size=args.test_size)
        X_train, X_test, y_train, y_test = data_processor.process_data()
        
        # Save scalers for later use
        data_processor.save_scalers(scalers_dir)
        
        # Create and compile model
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]
        
        # Create model path with model type in filename
        model_dir = os.path.dirname(model_path)
        model_filename = f"{args.model_type}_forecast_model.h5"
        model_path_with_type = os.path.join(model_dir, model_filename)
        
        # Create model using factory
        model = ModelFactory.create_model(args.model_type, input_shape, output_dim)
        model.compile_model(learning_rate=args.learning_rate)
        
        # Train model
        history, trained_model = train_model(
            model, 
            X_train, 
            y_train, 
            batch_size=args.batch_size, 
            epochs=args.epochs,
            patience=args.early_stopping,
            validation_split=0.15,
            save_dir=model_dir,
            model_filename=model_filename
        )
        
        # Evaluate on test data
        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            data_processor.scalers_y,
            data_processor.targets,
            save_dir=model_dir
        )
        
        print("\nTraining completed successfully!")
        print(f"Model saved to {model_path_with_type}")
        
    elif args.mode == 'evaluate':
        print("Starting evaluation mode...")
        
        # Update model path with model type
        model_dir = os.path.dirname(model_path)
        model_filename = f"{args.model_type}_forecast_model.h5"
        model_path_with_type = os.path.join(model_dir, model_filename)
        
        # Check if model exists
        if not os.path.exists(model_path_with_type):
            print(f"Error: Model not found at {model_path_with_type}")
            print(f"Please train the {args.model_type} model first using --mode train --model_type {args.model_type}")
            return
        
        # Create a base model instance to hold the loaded model
        base_model_instance = ModelFactory.create_model(args.model_type, (1, 1), 1)
        base_model_instance.model = base_model_instance.load_from_file(model_path_with_type)
        
        # Process data
        data_processor = DataProcessor(data_path, test_size=args.test_size)
        X_train, X_test, y_train, y_test = data_processor.process_data()
        
        # Evaluate the model
        metrics = evaluate_model(
            base_model_instance,
            X_test,
            y_test,
            data_processor.scalers_y,
            data_processor.targets,
            save_dir=model_dir
        )
        
        print("\nEvaluation completed successfully!")
        
        # Print summary of metrics
        print("\nModel Performance Summary:")
        for target, values in metrics.items():
            print(f"  {target}:")
            for metric, value in values.items():
                print(f"    {metric}: {value:.4f}")
            print()
        
    elif args.mode == 'predict':
        print("Starting prediction mode...")
        
        # Update model path with model type
        model_dir = os.path.dirname(model_path)
        model_filename = f"{args.model_type}_forecast_model.h5"
        model_path_with_type = os.path.join(model_dir, model_filename)
        
        # Check if model exists
        if not os.path.exists(model_path_with_type):
            print(f"Error: Model not found at {model_path_with_type}")
            print(f"Please train the {args.model_type} model first using --mode train --model_type {args.model_type}")
            return
        
        # Create a base model instance to hold the loaded model
        base_model_instance = ModelFactory.create_model(args.model_type, (1, 1), 1)
        base_model_instance.model = base_model_instance.load_from_file(model_path_with_type)
        
        # Load data for prediction
        data_processor = DataProcessor(data_path)
        df = data_processor.load_data()
        
        # Create features
        df = data_processor.create_features()
        df = data_processor.create_lags()
        features, targets = data_processor.prepare_features()
        
        # Load scalers
        data_processor.load_scalers(scalers_dir)
        
        # Make predictions on test data
        X = data_processor.df[features].values
        X = data_processor.feature_scaler.transform(X)
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        
        predictions = predict_load(base_model_instance, X, data_processor.scalers_y, targets)
        
        # Add datetime index from original data
        predictions.index = data_processor.df.index
        
        # Save predictions
        save_dir = model_dir
        os.makedirs(save_dir, exist_ok=True)
        predictions_path = os.path.join(save_dir, f'{args.model_type}_historical_predictions.csv')
        predictions.to_csv(predictions_path)
        print(f"Historical predictions saved to {predictions_path}")
        
        # Predict future
        last_data = data_processor.df.iloc[-24:].copy()  # Use last 24 hours as context
        future_predictions = predict_future(
            base_model_instance,
            last_data,
            data_processor.feature_scaler,
            data_processor.scalers_y,
            targets,
            features,
            steps=args.prediction_steps,
            save_dir=save_dir,
            model_type=args.model_type
        )
        
        print("\nPrediction completed successfully!")
    
    elif args.mode == 'compare_models':
        print("Starting model comparison mode...")
        
        # Process data
        data_processor = DataProcessor(data_path, test_size=args.test_size)
        X_train, X_test, y_train, y_test = data_processor.process_data()
        
        # Save scalers for later use
        data_processor.save_scalers(scalers_dir)
        
        # Set up model parameters
        input_shape = (X_train.shape[1], X_train.shape[2])
        output_dim = y_train.shape[1]
        
        # Create model directory
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)
        
        # Compare different model architectures
        model_types = ['gru', 'enhanced_gru', 'lstm', 'bilstm', 'cnn_lstm']
        results = {}
        
        for model_type in model_types:
            print(f"\n{'-'*50}")
            print(f"Training and evaluating {model_type.upper()} model")
            print(f"{'-'*50}")
            
            # Create model path
            model_filename = f"{model_type}_forecast_model.h5"
            model_path_with_type = os.path.join(model_dir, model_filename)
            
            # Create and compile model
            model = ModelFactory.create_model(model_type, input_shape, output_dim)
            model.compile_model(learning_rate=args.learning_rate)
            
            # Train model with reduced epochs for comparison
            comparison_epochs = min(args.epochs, 100)  # Limit for comparison
            history, trained_model = train_model(
                model, 
                X_train, 
                y_train, 
                batch_size=args.batch_size, 
                epochs=comparison_epochs,
                patience=args.early_stopping,
                validation_split=0.15,
                save_dir=model_dir,
                model_filename=model_filename
            )
            
            # Evaluate on test data
            metrics = evaluate_model(
                model,
                X_test,
                y_test,
                data_processor.scalers_y,
                data_processor.targets,
                save_dir=os.path.join(model_dir, model_type)
            )
            
            # Store results
            results[model_type] = metrics
        
        # Compare models
        compare_model_results(results, model_dir)
        
        print("\nModel comparison completed successfully!")
    
    else:
        print(f"Invalid mode: {args.mode}")
        print("Please use one of: train, evaluate, predict, compare_models")


def compare_model_results(results, save_dir):
    """
    Compare and visualize the performance of different models.
    
    Args:
        results: Dictionary with model results
        save_dir: Directory to save comparison visualizations
    """
    # Create comparison directory
    comparison_dir = os.path.join(save_dir, 'model_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Prepare data for visualization
    models = list(results.keys())
    targets = list(results[models[0]].keys())
    metrics = list(results[models[0]][targets[0]].keys())
    
    # Create a DataFrame to store all metrics
    comparison_df = pd.DataFrame(index=pd.MultiIndex.from_product([models, targets], 
                                                                 names=['Model', 'Target']))
    
    for model in models:
        for target in targets:
            for metric in metrics:
                comparison_df.loc[(model, target), metric] = results[model][target][metric]
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(comparison_dir, 'model_comparison.csv'))
    
    # Create bar chart for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Pivot data for plotting
        plot_data = comparison_df.reset_index().pivot(index='Target', columns='Model', values=metric)
        
        # Plot
        ax = plot_data.plot(kind='bar', width=0.8)
        
        plt.title(f'Model Comparison - {metric}')
        plt.ylabel(metric)
        plt.xlabel('Target')
        plt.legend(title='Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(comparison_dir, f'comparison_{metric}.png'))
        plt.close()
    
    # Create heatmap for overall performance
    plt.figure(figsize=(15, 8))
    
    # Use MAPE for heatmap (lower is better)
    heatmap_data = comparison_df.reset_index().pivot(index='Target', columns='Model', values='MAPE')
    
    # Plot heatmap
    import seaborn as sns
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlGnBu_r')
    
    plt.title('Model Comparison - MAPE (Lower is Better)')
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(os.path.join(comparison_dir, 'comparison_heatmap.png'))
    plt.close()
    
    # Print summary
    print("\nModel Comparison Summary (MAPE):")
    print(heatmap_data)
    
    # Find best model for each target
    best_models = {}
    for target in targets:
        target_results = {model: results[model][target]['MAPE'] for model in models}
        best_model = min(target_results, key=target_results.get)
        best_models[target] = (best_model, target_results[best_model])
    
    print("\nBest Models by Target (lowest MAPE):")
    for target, (model, mape) in best_models.items():
        print(f"  {target}: {model} (MAPE: {mape:.2f}%)")
    
    # Find overall best model
    avg_mape = {model: np.mean([results[model][target]['MAPE'] for target in targets]) 
                for model in models}
    overall_best = min(avg_mape, key=avg_mape.get)
    
    print(f"\nOverall Best Model: {overall_best} (Average MAPE: {avg_mape[overall_best]:.2f}%)")


if __name__ == '__main__':
    main()
