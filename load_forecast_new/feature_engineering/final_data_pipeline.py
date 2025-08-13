"""
FINAL DATASET PIPELINE
=====================
Work with the final optimized Delhi load forecasting dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_final_data():
    """Load final optimized dataset"""
    print("Loading final Delhi load dataset...")
    
    try:
        # Load final dataset
        df = pd.read_csv('../final_dataset.csv')
        print(f"Final dataset loaded: {df.shape[0]:,} records × {df.shape[1]} features")
        
        # Basic info
        print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
    except FileNotFoundError:
        print("Final dataset not found")
        return None

def prepare_for_modeling(df):
    """Prepare final data for model training"""
    print("Preparing final data for modeling...")
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Basic data quality checks
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Data types: {df.dtypes.value_counts().to_dict()}")
    
    print(f"Data prepared for model training")
    return df

if __name__ == "__main__":
    # Load and prepare final data
    df = load_final_data()
    if df is not None:
        clean_df = prepare_for_modeling(df)
        print("Ready for model development")
