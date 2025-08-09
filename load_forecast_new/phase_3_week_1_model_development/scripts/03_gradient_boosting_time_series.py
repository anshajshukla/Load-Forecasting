"""
Delhi Load Forecasting - Phase 3 Week 1
Day 5-6: Gradient Boosting & Time Series Baselines Implementation

This script implements XGBoost and Facebook Prophet models with comprehensive
hyperparameter tuning and baseline ensemble combination testing.

Target MAPE: XGBoost (4-7%), Facebook Prophet (6-9%)
Timeline: Days 5-6 of Week 1 baseline establishment
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from prophet import Prophet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.ensemble import VotingRegressor
import joblib
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import randint, uniform

class GradientBoostingTimeSeriesBaselines:
    """
    Implementation of Gradient Boosting and Time Series baseline models for Delhi Load Forecasting
    
    Models:
    - XGBoost with hyperparameter tuning
    - Facebook Prophet for seasonal modeling
    - Baseline ensemble combinations
    """
    
    def __init__(self, data_dir, output_dir):
        """Initialize gradient boosting and time series models pipeline"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.create_output_directories()
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Previous models (from Day 3-4)
        self.previous_models = {}
        
        # New models
        self.models = {}
        self.model_results = {}
        self.feature_importance = {}
        self.ensemble_results = {}
        
        # Metadata
        self.advanced_metadata = {}
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            os.path.join(self.output_dir, 'models'),
            os.path.join(self.output_dir, 'results'),
            os.path.join(self.output_dir, 'ensemble'),
            os.path.join(self.output_dir, 'visualizations')
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print("✅ Output directories created successfully")
    
    def load_prepared_data_and_models(self):
        """Load prepared data and previous baseline models"""
        print("\n🔄 Loading prepared data and previous models...")
        
        try:
            # Load data splits
            self.X_train = pd.read_csv(
                os.path.join(self.data_dir, 'data', 'X_train_scaled.csv'),
                index_col=0, parse_dates=True
            )
            self.X_val = pd.read_csv(
                os.path.join(self.data_dir, 'data', 'X_val_scaled.csv'),
                index_col=0, parse_dates=True
            )
            self.X_test = pd.read_csv(
                os.path.join(self.data_dir, 'data', 'X_test_scaled.csv'),
                index_col=0, parse_dates=True
            )
            
            self.y_train = pd.read_csv(
                os.path.join(self.data_dir, 'data', 'y_train.csv'),
                index_col=0, parse_dates=True
            )
            self.y_val = pd.read_csv(
                os.path.join(self.data_dir, 'data', 'y_val.csv'),
                index_col=0, parse_dates=True
            )
            self.y_test = pd.read_csv(
                os.path.join(self.data_dir, 'data', 'y_test.csv'),
                index_col=0, parse_dates=True
            )
            
            print("✅ Prepared data loaded successfully")
            
            # Load previous models for ensemble
            models_dir = os.path.join(self.data_dir, 'models')
            previous_model_files = ['ridge_model.pkl', 'lasso_model.pkl', 'random_forest_model.pkl']
            
            for model_file in previous_model_files:
                model_path = os.path.join(models_dir, model_file)
                if os.path.exists(model_path):
                    model_name = model_file.replace('_model.pkl', '')
                    self.previous_models[model_name] = joblib.load(model_path)
                    print(f"   📁 Loaded: {model_name}")
            
            print(f"✅ Loaded {len(self.previous_models)} previous models for ensemble")
            
        except Exception as e:
            print(f"❌ Error loading data/models: {str(e)}")
            raise
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Ensure we have DataFrame inputs
        if isinstance(y_true, np.ndarray):
            y_true = pd.DataFrame(y_true, columns=self.y_train.columns)
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.DataFrame(y_pred, columns=self.y_train.columns)
        
        # Core metrics for each target
        for target in self.y_train.columns:
            if target in y_true.columns and target in y_pred.columns:
                mask = np.isfinite(y_true[target]) & np.isfinite(y_pred[target])
                y_true_clean = y_true[target][mask]
                y_pred_clean = y_pred[target][mask]
                
                if len(y_true_clean) > 0:
                    target_name = target.replace('_load', '')
                    metrics[f'{target_name}_mape'] = mean_absolute_percentage_error(y_true_clean, y_pred_clean) * 100
                    metrics[f'{target_name}_mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
                    metrics[f'{target_name}_rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # Overall performance
        if 'delhi_load' in y_true.columns and 'delhi_load' in y_pred.columns:
            mask = np.isfinite(y_true['delhi_load']) & np.isfinite(y_pred['delhi_load'])
            if mask.sum() > 0:
                metrics['overall_mape'] = mean_absolute_percentage_error(
                    y_true['delhi_load'][mask], y_pred['delhi_load'][mask]
                ) * 100
                metrics['overall_mae'] = mean_absolute_error(
                    y_true['delhi_load'][mask], y_pred['delhi_load'][mask]
                )
                metrics['overall_rmse'] = np.sqrt(mean_squared_error(
                    y_true['delhi_load'][mask], y_pred['delhi_load'][mask]
                ))
        
        return metrics
    
    def implement_xgboost(self):
        """Implement XGBoost with comprehensive hyperparameter tuning"""
        print("\n🔄 Implementing XGBoost with hyperparameter tuning...")
        
        # Define parameter distribution for XGBoost
        xgb_params = {
            'n_estimators': [100, 200, 300, 500, 800],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }
        
        # Use MultiOutputRegressor for multiple targets
        xgb_base = xgb.XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        xgb_multi = MultiOutputRegressor(xgb_base)
        
        # Randomized search for efficiency
        xgb_search = RandomizedSearchCV(
            xgb_multi,
            {f'estimator__{k}': v for k, v in xgb_params.items()},
            n_iter=50,
            cv=5,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the model
        print("   🔄 Training XGBoost (this may take a while)...")
        xgb_search.fit(self.X_train, self.y_train)
        
        # Store best model
        self.models['xgboost'] = xgb_search.best_estimator_
        
        # Make predictions
        xgb_train_pred = xgb_search.predict(self.X_train)
        xgb_val_pred = xgb_search.predict(self.X_val)
        
        # Calculate metrics
        train_metrics = self.calculate_comprehensive_metrics(self.y_train, xgb_train_pred, 'xgboost')
        val_metrics = self.calculate_comprehensive_metrics(self.y_val, xgb_val_pred, 'xgboost')
        
        # Extract feature importance
        feature_importance = {}
        for i, target in enumerate(self.y_train.columns):
            importances = xgb_search.best_estimator_.estimators_[i].feature_importances_
            feature_importance[target] = dict(zip(self.X_train.columns, importances))
        
        self.feature_importance['xgboost'] = feature_importance
        
        # Store results
        self.model_results['xgboost'] = {
            'best_params': xgb_search.best_params_,
            'best_score': xgb_search.best_score_,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_predictions': xgb_train_pred,
            'val_predictions': xgb_val_pred,
            'feature_importance': feature_importance
        }
        
        print("✅ XGBoost completed")
        print(f"   📊 Validation MAPE: {val_metrics.get('overall_mape', 'N/A'):.2f}%")
        print(f"   🎯 Target MAPE: 4-7%")
    
    def implement_facebook_prophet(self):
        """Implement Facebook Prophet for seasonal modeling"""
        print("\n🔄 Implementing Facebook Prophet for seasonal modeling...")
        
        # Prophet models for each target (Prophet doesn't support multi-output directly)
        prophet_models = {}
        prophet_predictions = {'train': {}, 'val': {}}
        prophet_metrics = {'train': {}, 'val': {}}
        
        for target in self.y_train.columns:
            print(f"   🔄 Training Prophet for {target}...")
            
            try:
                # Prepare data for Prophet (requires 'ds' and 'y' columns)
                train_prophet = pd.DataFrame({
                    'ds': self.y_train.index,
                    'y': self.y_train[target].values
                })
                
                # Initialize Prophet with Delhi-specific parameters
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    holidays_prior_scale=0.5,
                    seasonality_prior_scale=1.0,
                    changepoint_prior_scale=0.05,
                    interval_width=0.95
                )
                
                # Add custom seasonalities for Delhi patterns
                model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
                
                # Fit the model
                model.fit(train_prophet)
                prophet_models[target] = model
                
                # Make predictions for training set
                train_future = model.make_future_dataframe(periods=0, freq='H')
                train_forecast = model.predict(train_future)
                prophet_predictions['train'][target] = train_forecast['yhat'].values
                
                # Make predictions for validation set
                val_future = pd.DataFrame({
                    'ds': self.y_val.index
                })
                val_forecast = model.predict(val_future)
                prophet_predictions['val'][target] = val_forecast['yhat'].values
                
                # Calculate metrics
                train_mape = mean_absolute_percentage_error(
                    self.y_train[target], prophet_predictions['train'][target]
                ) * 100
                val_mape = mean_absolute_percentage_error(
                    self.y_val[target], prophet_predictions['val'][target]
                ) * 100
                
                prophet_metrics['train'][f'{target.replace("_load", "")}_mape'] = train_mape
                prophet_metrics['val'][f'{target.replace("_load", "")}_mape'] = val_mape
                
                print(f"      ✅ {target}: {val_mape:.2f}% MAPE")
                
            except Exception as e:
                print(f"      ❌ Error with {target}: {str(e)}")
                # Use simple mean as fallback
                mean_pred = np.full(len(self.y_train), self.y_train[target].mean())
                prophet_predictions['train'][target] = mean_pred
                prophet_predictions['val'][target] = np.full(len(self.y_val), self.y_train[target].mean())
        
        # Store Prophet models and results
        self.models['prophet'] = prophet_models
        
        # Combine predictions into arrays
        train_pred_array = np.column_stack([
            prophet_predictions['train'][target] for target in self.y_train.columns
        ])
        val_pred_array = np.column_stack([
            prophet_predictions['val'][target] for target in self.y_train.columns
        ])
        
        # Calculate comprehensive metrics
        train_metrics = self.calculate_comprehensive_metrics(self.y_train, train_pred_array, 'prophet')
        val_metrics = self.calculate_comprehensive_metrics(self.y_val, val_pred_array, 'prophet')
        
        self.model_results['prophet'] = {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'train_predictions': train_pred_array,
            'val_predictions': val_pred_array,
            'individual_metrics': prophet_metrics
        }
        
        print("✅ Facebook Prophet completed")
        print(f"   📊 Validation MAPE: {val_metrics.get('overall_mape', 'N/A'):.2f}%")
        print(f"   🎯 Target MAPE: 6-9%")
    
    def implement_baseline_ensemble(self):
        """Implement baseline ensemble combination testing"""
        print("\n🔄 Implementing baseline ensemble combinations...")
        
        # Collect all available models and their predictions
        ensemble_models = {}
        val_predictions = {}
        
        # Add previous models
        for model_name, model in self.previous_models.items():
            try:
                pred = model.predict(self.X_val)
                val_predictions[model_name] = pred
                ensemble_models[model_name] = model
                print(f"   ✅ Added {model_name} to ensemble")
            except Exception as e:
                print(f"   ⚠️ Failed to use {model_name}: {str(e)}")
        
        # Add new models
        for model_name in ['xgboost']:  # Prophet handled separately
            if model_name in self.model_results:
                val_predictions[model_name] = self.model_results[model_name]['val_predictions']
                ensemble_models[model_name] = self.models[model_name]
                print(f"   ✅ Added {model_name} to ensemble")
        
        # Add Prophet separately
        if 'prophet' in self.model_results:
            val_predictions['prophet'] = self.model_results['prophet']['val_predictions']
            print(f"   ✅ Added prophet to ensemble")
        
        if len(val_predictions) < 2:
            print("   ⚠️ Need at least 2 models for ensemble. Skipping ensemble.")
            return
        
        # Test different ensemble combinations
        ensemble_combinations = [
            ['ridge', 'random_forest'],
            ['ridge', 'xgboost'],
            ['random_forest', 'xgboost'],
            ['ridge', 'random_forest', 'xgboost'],
            ['prophet', 'xgboost'],
            ['random_forest', 'prophet'],
        ]
        
        # Add all models combination if we have enough
        if len(val_predictions) >= 3:
            all_models = list(val_predictions.keys())
            ensemble_combinations.append(all_models)
        
        ensemble_results = {}
        
        for combination in ensemble_combinations:
            # Check if all models in combination are available
            available_models = [m for m in combination if m in val_predictions]
            if len(available_models) < 2:
                continue
                
            combo_name = '_'.join(available_models)
            print(f"   🔄 Testing ensemble: {combo_name}")
            
            try:
                # Simple average ensemble
                ensemble_pred = np.mean([val_predictions[model] for model in available_models], axis=0)
                
                # Calculate metrics
                ensemble_metrics = self.calculate_comprehensive_metrics(
                    self.y_val, ensemble_pred, combo_name
                )
                
                ensemble_results[combo_name] = {
                    'models': available_models,
                    'metrics': ensemble_metrics,
                    'predictions': ensemble_pred
                }
                
                print(f"      📊 {combo_name} MAPE: {ensemble_metrics.get('overall_mape', 'N/A'):.2f}%")
                
            except Exception as e:
                print(f"      ❌ Failed {combo_name}: {str(e)}")
        
        self.ensemble_results = ensemble_results
        
        # Find best ensemble
        if ensemble_results:
            best_ensemble = min(
                ensemble_results.items(),
                key=lambda x: x[1]['metrics'].get('overall_mape', float('inf'))
            )
            print(f"\n🏆 Best ensemble: {best_ensemble[0]}")
            print(f"   📊 MAPE: {best_ensemble[1]['metrics'].get('overall_mape', 'N/A'):.2f}%")
        
        print("✅ Baseline ensemble testing completed")
    
    def perform_cross_validation_analysis(self):
        """Perform cross-validation analysis and comparison"""
        print("\n🔄 Performing cross-validation analysis...")
        
        cv_results = {}
        
        # Cross-validate available models
        for model_name in ['xgboost']:
            if model_name in self.models:
                print(f"   🔄 CV analysis for {model_name}...")
                
                try:
                    # Use the trained model for CV
                    model = self.models[model_name]
                    
                    # Perform 5-fold CV on training data
                    cv_scores = cross_val_score(
                        model, self.X_train, self.y_train,
                        cv=5, scoring='neg_mean_absolute_percentage_error',
                        n_jobs=-1
                    )
                    
                    cv_results[model_name] = {
                        'cv_scores': -cv_scores * 100,  # Convert to positive MAPE
                        'cv_mean': -cv_scores.mean() * 100,
                        'cv_std': cv_scores.std() * 100
                    }
                    
                    print(f"      📊 CV MAPE: {cv_results[model_name]['cv_mean']:.2f}% ± {cv_results[model_name]['cv_std']:.2f}%")
                    
                except Exception as e:
                    print(f"      ❌ CV failed for {model_name}: {str(e)}")
        
        self.advanced_metadata['cross_validation'] = cv_results
        print("✅ Cross-validation analysis completed")
    
    def create_advanced_visualizations(self):
        """Create comprehensive visualizations for advanced models"""
        print("\n🔄 Creating advanced model visualizations...")
        
        plt.figure(figsize=(20, 15))
        
        # 1. Model Performance Comparison (including new models)
        plt.subplot(3, 4, 1)
        all_models = {}
        
        # Load previous results
        try:
            with open(os.path.join(self.data_dir, 'results', 'linear_tree_baselines_results.json'), 'r') as f:
                previous_results = json.load(f)
            for model, results in previous_results.items():
                all_models[model] = results['val_metrics']['overall_mape']
        except:
            pass
        
        # Add new models
        for model, results in self.model_results.items():
            all_models[model] = results['val_metrics'].get('overall_mape', np.nan)
        
        model_names = list(all_models.keys())
        mapes = [all_models[model] for model in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        bars = plt.bar(model_names, mapes, color=colors)
        plt.title('All Models Performance Comparison')
        plt.ylabel('MAPE (%)')
        plt.xticks(rotation=45)
        
        # Add target lines
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Week 1 Target')
        plt.axhline(y=7, color='orange', linestyle='--', alpha=0.7, label='XGBoost Target')
        plt.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Excellent')
        plt.legend()
        
        # 2. XGBoost Feature Importance
        if 'xgboost' in self.feature_importance:
            plt.subplot(3, 4, 2)
            xgb_importance = self.feature_importance['xgboost']['delhi_load']
            top_features = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            
            feature_names = [f.split('_')[0][:10] for f, _ in top_features]
            importances = [imp for _, imp in top_features]
            
            plt.barh(feature_names, importances)
            plt.title('XGBoost Feature Importance (Top 15)')
            plt.xlabel('Importance')
        
        # 3. Prophet Components (if available)
        if 'prophet' in self.models and 'delhi_load' in self.models['prophet']:
            plt.subplot(3, 4, 3)
            try:
                # Create a sample forecast for visualization
                model = self.models['prophet']['delhi_load']
                future = model.make_future_dataframe(periods=168, freq='H')  # 1 week ahead
                forecast = model.predict(future)
                
                # Plot trend
                plt.plot(forecast['ds'][-168:], forecast['trend'][-168:], label='Trend')
                plt.plot(forecast['ds'][-168:], forecast['weekly'][-168:], label='Weekly')
                plt.title('Prophet Components (Delhi Load)')
                plt.xlabel('Date')
                plt.ylabel('Component')
                plt.legend()
                plt.xticks(rotation=45)
            except:
                plt.text(0.5, 0.5, 'Prophet\nVisualization\nUnavailable', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Prophet Components')
        
        # 4. Ensemble Performance
        plt.subplot(3, 4, 4)
        if self.ensemble_results:
            ensemble_names = list(self.ensemble_results.keys())
            ensemble_mapes = [result['metrics'].get('overall_mape', np.nan) 
                            for result in self.ensemble_results.values()]
            
            plt.bar(range(len(ensemble_names)), ensemble_mapes)
            plt.title('Ensemble Combinations Performance')
            plt.ylabel('MAPE (%)')
            plt.xticks(range(len(ensemble_names)), 
                      [name.replace('_', '\n') for name in ensemble_names], 
                      rotation=45, fontsize=8)
        
        # 5. Prediction vs Actual (XGBoost)
        if 'xgboost' in self.model_results:
            plt.subplot(3, 4, 5)
            actual = self.y_val['delhi_load']
            predicted = pd.DataFrame(
                self.model_results['xgboost']['val_predictions'],
                columns=self.y_train.columns,
                index=self.y_val.index
            )['delhi_load']
            
            plt.scatter(actual, predicted, alpha=0.6, s=20)
            min_val, max_val = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            plt.xlabel('Actual Delhi Load (MW)')
            plt.ylabel('Predicted Delhi Load (MW)')
            plt.title('XGBoost: Prediction vs Actual')
        
        # 6. Time Series Comparison
        plt.subplot(3, 4, 6)
        sample_period = slice(None, 200)  # First 200 validation samples
        actual_sample = self.y_val['delhi_load'].iloc[sample_period]
        
        plt.plot(actual_sample.index, actual_sample, label='Actual', color='black', linewidth=2)
        
        for model_name in ['xgboost', 'prophet']:
            if model_name in self.model_results:
                pred_df = pd.DataFrame(
                    self.model_results[model_name]['val_predictions'],
                    columns=self.y_train.columns,
                    index=self.y_val.index
                )
                pred_sample = pred_df['delhi_load'].iloc[sample_period]
                plt.plot(pred_sample.index, pred_sample, label=model_name.title(), alpha=0.8)
        
        plt.title('Time Series Predictions Comparison')
        plt.xlabel('Date')
        plt.ylabel('Delhi Load (MW)')
        plt.legend()
        plt.xticks(rotation=45)
        
        # 7. Residuals Analysis (XGBoost)
        if 'xgboost' in self.model_results:
            plt.subplot(3, 4, 7)
            actual = self.y_val['delhi_load']
            predicted = pd.DataFrame(
                self.model_results['xgboost']['val_predictions'],
                columns=self.y_train.columns,
                index=self.y_val.index
            )['delhi_load']
            
            residuals = actual - predicted
            plt.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            plt.title('XGBoost Residuals Distribution')
            plt.xlabel('Residual (MW)')
            plt.ylabel('Frequency')
            plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # 8. DISCOM Performance Comparison
        plt.subplot(3, 4, 8)
        if 'xgboost' in self.model_results:
            discom_mapes = []
            discom_names = []
            for target in ['brpl_load', 'bypl_load', 'ndpl_load', 'ndmc_load', 'mes_load']:
                target_name = target.replace('_load', '')
                mape_key = f'{target_name}_mape'
                if mape_key in self.model_results['xgboost']['val_metrics']:
                    discom_mapes.append(self.model_results['xgboost']['val_metrics'][mape_key])
                    discom_names.append(target_name.upper())
            
            if discom_mapes:
                plt.bar(discom_names, discom_mapes)
                plt.title('DISCOM-wise MAPE (XGBoost)')
                plt.ylabel('MAPE (%)')
                plt.xticks(rotation=45)
        
        # 9. Cross-validation Results
        plt.subplot(3, 4, 9)
        if hasattr(self, 'advanced_metadata') and 'cross_validation' in self.advanced_metadata:
            cv_data = self.advanced_metadata['cross_validation']
            if cv_data:
                models = list(cv_data.keys())
                means = [cv_data[model]['cv_mean'] for model in models]
                stds = [cv_data[model]['cv_std'] for model in models]
                
                plt.bar(models, means, yerr=stds, capsize=5)
                plt.title('Cross-Validation Results')
                plt.ylabel('CV MAPE (%)')
                plt.xticks(rotation=45)
        
        # 10. Training vs Validation Performance
        plt.subplot(3, 4, 10)
        train_mapes = []
        val_mapes = []
        model_names = []
        
        for model_name, results in self.model_results.items():
            train_mape = results['train_metrics'].get('overall_mape', np.nan)
            val_mape = results['val_metrics'].get('overall_mape', np.nan)
            if not np.isnan(train_mape) and not np.isnan(val_mape):
                train_mapes.append(train_mape)
                val_mapes.append(val_mape)
                model_names.append(model_name)
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.bar(x - width/2, train_mapes, width, label='Training', alpha=0.8)
        plt.bar(x + width/2, val_mapes, width, label='Validation', alpha=0.8)
        plt.xlabel('Model')
        plt.ylabel('MAPE (%)')
        plt.title('Training vs Validation Performance')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        
        # 11. Error Distribution by Hour
        if 'xgboost' in self.model_results:
            plt.subplot(3, 4, 11)
            actual = self.y_val['delhi_load']
            predicted = pd.DataFrame(
                self.model_results['xgboost']['val_predictions'],
                columns=self.y_train.columns,
                index=self.y_val.index
            )['delhi_load']
            
            residuals = actual - predicted
            hourly_errors = residuals.groupby(residuals.index.hour).std()
            
            plt.bar(hourly_errors.index, hourly_errors.values)
            plt.title('Hourly Error Distribution (XGBoost)')
            plt.xlabel('Hour of Day')
            plt.ylabel('Error Std Dev (MW)')
        
        # 12. Model Complexity vs Performance
        plt.subplot(3, 4, 12)
        complexity_data = {
            'ridge': {'complexity': 1, 'performance': all_models.get('ridge', np.nan)},
            'lasso': {'complexity': 1, 'performance': all_models.get('lasso', np.nan)},
            'random_forest': {'complexity': 3, 'performance': all_models.get('random_forest', np.nan)},
            'xgboost': {'complexity': 4, 'performance': all_models.get('xgboost', np.nan)},
            'prophet': {'complexity': 2, 'performance': all_models.get('prophet', np.nan)},
        }
        
        complexities = [data['complexity'] for data in complexity_data.values()]
        performances = [data['performance'] for data in complexity_data.values()]
        model_labels = list(complexity_data.keys())
        
        valid_indices = [i for i, p in enumerate(performances) if not np.isnan(p)]
        if valid_indices:
            plt.scatter([complexities[i] for i in valid_indices], 
                       [performances[i] for i in valid_indices], s=100)
            for i in valid_indices:
                plt.annotate(model_labels[i], 
                           (complexities[i], performances[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Complexity (Relative)')
        plt.ylabel('Validation MAPE (%)')
        plt.title('Complexity vs Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.output_dir, 'visualizations', 'advanced_models_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Saved: advanced_models_analysis.png")
        print("✅ Advanced visualizations created successfully")
    
    def save_models_and_results(self):
        """Save all trained models and results"""
        print("\n🔄 Saving advanced models and results...")
        
        models_dir = os.path.join(self.output_dir, 'models')
        results_dir = os.path.join(self.output_dir, 'results')
        ensemble_dir = os.path.join(self.output_dir, 'ensemble')
        
        # Save XGBoost model
        if 'xgboost' in self.models:
            model_path = os.path.join(models_dir, 'xgboost_model.pkl')
            joblib.dump(self.models['xgboost'], model_path)
            print(f"   💾 Saved: xgboost_model.pkl")
        
        # Save Prophet models
        if 'prophet' in self.models:
            prophet_dir = os.path.join(models_dir, 'prophet_models')
            os.makedirs(prophet_dir, exist_ok=True)
            for target, model in self.models['prophet'].items():
                model_path = os.path.join(prophet_dir, f'prophet_{target}.pkl')
                joblib.dump(model, model_path)
            print(f"   💾 Saved: prophet_models/")
        
        # Save detailed results
        results_path = os.path.join(results_dir, 'advanced_models_results.json')
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for model_name, results in self.model_results.items():
            serializable_results[model_name] = {
                'train_metrics': {k: float(v) if not np.isnan(v) else None for k, v in results['train_metrics'].items()},
                'val_metrics': {k: float(v) if not np.isnan(v) else None for k, v in results['val_metrics'].items()}
            }
            
            # Add model-specific parameters
            if 'best_params' in results:
                serializable_results[model_name]['best_params'] = results['best_params']
            if 'best_score' in results:
                serializable_results[model_name]['best_score'] = float(results['best_score']) if not np.isnan(results['best_score']) else None
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"   💾 Saved: advanced_models_results.json")
        
        # Save ensemble results
        if self.ensemble_results:
            ensemble_path = os.path.join(ensemble_dir, 'ensemble_results.json')
            
            serializable_ensemble = {}
            for combo_name, results in self.ensemble_results.items():
                serializable_ensemble[combo_name] = {
                    'models': results['models'],
                    'metrics': {k: float(v) if not np.isnan(v) else None for k, v in results['metrics'].items()}
                }
            
            with open(ensemble_path, 'w') as f:
                json.dump(serializable_ensemble, f, indent=2)
            print(f"   💾 Saved: ensemble_results.json")
        
        # Save comprehensive metadata
        self.advanced_metadata['completion_timestamp'] = datetime.now().isoformat()
        self.advanced_metadata['models_trained'] = list(self.models.keys())
        self.advanced_metadata['ensemble_combinations'] = len(self.ensemble_results) if self.ensemble_results else 0
        
        metadata_path = os.path.join(results_dir, 'advanced_models_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.advanced_metadata, f, indent=2, default=str)
        print(f"   💾 Saved: advanced_models_metadata.json")
        
        print("✅ All advanced models and results saved successfully")
    
    def run_advanced_baselines(self):
        """Run the complete gradient boosting and time series baselines pipeline"""
        print("🚀 Starting Gradient Boosting & Time Series Baselines (Day 5-6)")
        print("=" * 80)
        
        try:
            # Step 1: Load prepared data and previous models
            self.load_prepared_data_and_models()
            
            # Step 2: Implement XGBoost
            self.implement_xgboost()
            
            # Step 3: Implement Facebook Prophet
            self.implement_facebook_prophet()
            
            # Step 4: Implement baseline ensemble
            self.implement_baseline_ensemble()
            
            # Step 5: Perform cross-validation analysis
            self.perform_cross_validation_analysis()
            
            # Step 6: Create advanced visualizations
            self.create_advanced_visualizations()
            
            # Step 7: Save models and results
            self.save_models_and_results()
            
            print("\n🎉 Gradient Boosting & Time Series Baselines Completed Successfully!")
            print("=" * 80)
            
            # Summary of results
            print("📊 ADVANCED BASELINE RESULTS SUMMARY:")
            for model_name, results in self.model_results.items():
                mape = results['val_metrics'].get('overall_mape', np.nan)
                if model_name == 'xgboost':
                    target_range = "4-7%"
                elif model_name == 'prophet':
                    target_range = "6-9%"
                else:
                    target_range = "N/A"
                
                status = "✅" if (not np.isnan(mape) and mape < 10) else "⚠️"
                print(f"   {status} {model_name.title()}: {mape:.2f}% MAPE (Target: {target_range})")
            
            # Best ensemble
            if self.ensemble_results:
                best_ensemble = min(
                    self.ensemble_results.items(),
                    key=lambda x: x[1]['metrics'].get('overall_mape', float('inf'))
                )
                print(f"\n🏆 BEST ENSEMBLE: {best_ensemble[0]} ({best_ensemble[1]['metrics'].get('overall_mape', 'N/A'):.2f}% MAPE)")
            
            # Overall best model
            all_mapes = {name: results['val_metrics'].get('overall_mape', np.nan) 
                        for name, results in self.model_results.items()}
            best_model = min(all_mapes.items(), key=lambda x: x[1] if not np.isnan(x[1]) else float('inf'))
            print(f"🎯 BEST INDIVIDUAL MODEL: {best_model[0].title()} ({best_model[1]:.2f}% MAPE)")
            
            week1_success = best_model[1] < 10.0 if not np.isnan(best_model[1]) else False
            print(f"🎯 WEEK 1 SUCCESS CRITERIA: {'✅ MET' if week1_success else '❌ NOT MET'} (<10% MAPE)")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Advanced baseline pipeline failed: {str(e)}")
            raise

def main():
    """Main execution function"""
    # Configuration
    data_dir = r"c:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_1_model_development"
    output_dir = r"c:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_1_model_development"
    
    # Initialize and run pipeline
    advanced_pipeline = GradientBoostingTimeSeriesBaselines(data_dir, output_dir)
    success = advanced_pipeline.run_advanced_baselines()
    
    if success:
        print("\n🎯 Next Steps:")
        print("   1. Proceed to Day 7: Baseline Evaluation & Documentation")
        print("   2. Create performance comparison dashboard")
        print("   3. Establish model selection criteria")
        print("   4. Document insights and plan Week 2 advanced architectures")

if __name__ == "__main__":
    main()
