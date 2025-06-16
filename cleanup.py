"""
Cleanup script to remove unnecessary files and organize the project structure.

This script:
1. Creates necessary directories if they don't exist
2. Moves important files to their appropriate locations
3. Removes old/unnecessary files
"""

import os
import shutil
from pathlib import Path
import argparse

def create_directories(base_dir):
    """Create necessary directories for the project."""
    # Main directories
    directories = [
        'data',
        'models/saved_models',
        'models/saved_models/logs',
        'models/saved_models/figures',
        'models/saved_models/metrics',
    ]
    
    # Create each directory
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def move_data_files(base_dir, source_dir):
    """Move data files to the data directory."""
    # Check if final_data.csv exists in source directory
    source_data = os.path.join(source_dir, 'final_data.csv')
    if os.path.exists(source_data):
        dest_data = os.path.join(base_dir, 'data', 'final_data.csv')
        # Copy instead of move to keep the original
        shutil.copy2(source_data, dest_data)
        print(f"Copied {source_data} to {dest_data}")
    else:
        print(f"Warning: {source_data} not found")
    
    # Keep InnovationReport.pdf in the base directory
    source_report = os.path.join(source_dir, 'InnovationReport.pdf')
    if os.path.exists(source_report):
        dest_report = os.path.join(base_dir, 'InnovationReport.pdf')
        # Only copy if it doesn't already exist in the destination
        if not os.path.exists(dest_report):
            shutil.copy2(source_report, dest_report)
            print(f"Copied {source_report} to {dest_report}")
    else:
        print(f"Warning: {source_report} not found")

def cleanup_files(base_dir, source_dir, remove_jupyter=False):
    """Remove unnecessary files."""
    # Files to definitely keep
    essential_files = [
        'final_data.csv',
        'InnovationReport.pdf',
    ]
    
    # Only remove Jupyter notebooks if specified
    if remove_jupyter:
        # Find and remove .ipynb files if they've been modularized
        jupyter_files = [f for f in os.listdir(source_dir) if f.endswith('.ipynb')]
        for file in jupyter_files:
            file_path = os.path.join(source_dir, file)
            if os.path.basename(file_path) not in essential_files:
                try:
                    os.rename(file_path, file_path + '.bak')
                    print(f"Renamed {file_path} to {file_path}.bak (backup)")
                except Exception as e:
                    print(f"Error backing up {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Cleanup and organize the load forecasting project.')
    parser.add_argument('--source_dir', type=str, default='..',
                        help='Source directory containing original files (default: parent directory)')
    parser.add_argument('--remove_jupyter', action='store_true',
                        help='Backup Jupyter notebooks that have been modularized (creates .bak files)')
    
    args = parser.parse_args()
    
    # Get base directory (the directory containing this script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.abspath(args.source_dir)
    
    print(f"Base directory: {base_dir}")
    print(f"Source directory: {source_dir}")
    
    # Create necessary directories
    create_directories(base_dir)
    
    # Move data files
    move_data_files(base_dir, source_dir)
    
    # Cleanup unnecessary files
    cleanup_files(base_dir, source_dir, args.remove_jupyter)
    
    print("\nCleanup complete!")
    print("\nRecommended next steps:")
    print("1. Run the model comparison:")
    print("   python main.py --mode compare_models --data_path data/final_data.csv")
    print("\n2. Train the best performing model:")
    print("   python main.py --mode train --model_type [best_model] --data_path data/final_data.csv")
    print("\n3. Make predictions:")
    print("   python main.py --mode predict --model_type [best_model] --data_path data/final_data.csv")

if __name__ == '__main__':
    main()
