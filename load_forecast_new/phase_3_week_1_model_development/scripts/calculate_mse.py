#!/usr/bin/env python3
"""
Week 1 MSE Calculator
====================
Calculate MSE (Mean Squared Error) from RMSE values for all Week 1 models.
"""

import json
import pandas as pd
import numpy as np

def calculate_mse_from_rmse(rmse):
    """Calculate MSE from RMSE"""
    return rmse ** 2

def load_and_calculate_mse():
    """Load Week 1 results and calculate MSE values"""
    
    print("🔢 Week 1 Model Performance - MSE Calculation")
    print("=" * 60)
    
    # Load the evaluation results
    eval_file = "../evaluation/reports/week1_final_evaluation.json"
    
    try:
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        print(f"📊 Loaded evaluation data for {data['week_1_summary']['total_models_evaluated']} models")
        print(f"🎯 Best MAPE achieved: {data['week_1_summary']['best_mape']:.2f}%")
        print()
        
        # Calculate MSE for each model
        results = []
        for model in data['model_performance']:
            rmse = model['validation_rmse']
            mse = calculate_mse_from_rmse(rmse)
            
            results.append({
                'Model': model['model_name'],
                'MAPE (%)': f"{model['validation_mape']:.2f}",
                'RMSE': f"{rmse:.2f}",
                'MSE': f"{mse:.2f}",
                'Category': model['category']
            })
        
        # Sort by MSE (ascending - lower is better)
        results.sort(key=lambda x: float(x['MSE']))
        
        print("📈 Model Performance Ranking (by MSE - Lower is Better):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<25} {'MAPE (%)':<8} {'RMSE':<8} {'MSE':<12} {'Category':<15}")
        print("-" * 80)
        
        for i, result in enumerate(results, 1):
            print(f"{i:<4} {result['Model']:<25} {result['MAPE (%)']:<8} {result['RMSE']:<8} {result['MSE']:<12} {result['Category']:<15}")
        
        print()
        print("🏆 Top 5 Models by MSE:")
        print("-" * 40)
        for i in range(min(5, len(results))):
            model = results[i]
            print(f"{i+1}. {model['Model']}: MSE = {model['MSE']}, MAPE = {model['MAPE (%)']}%")
        
        print()
        print("📊 Summary Statistics:")
        mse_values = [float(r['MSE']) for r in results]
        print(f"   • Best MSE: {min(mse_values):.2f}")
        print(f"   • Worst MSE: {max(mse_values):.2f}")
        print(f"   • Mean MSE: {np.mean(mse_values):.2f}")
        print(f"   • Median MSE: {np.median(mse_values):.2f}")
        
        # Models meeting criteria (MAPE ≤ 10%)
        good_models = [r for r in results if float(r['MAPE (%)']) <= 10.0]
        print(f"   • Models with MAPE ≤ 10%: {len(good_models)}")
        
        if good_models:
            print()
            print("✅ Models Meeting Target Criteria (MAPE ≤ 10%):")
            for model in good_models:
                print(f"   • {model['Model']}: MSE = {model['MSE']}, MAPE = {model['MAPE (%)']}%")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find evaluation file at {eval_file}")
        return False
    except Exception as e:
        print(f"❌ Error loading evaluation data: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = load_and_calculate_mse()
    if success:
        print("\n✅ MSE calculation completed successfully!")
    else:
        print("\n❌ MSE calculation failed!")
