"""
STEP 2.5.3: SYSTEMATIC FEATURE SELECTION
========================================
Feature Selection Methods:
- Random Forest feature importance ranking (top contributors)
- LASSO regularization for automatic feature selection
- Recursive Feature Elimination with Cross-Validation
- Permutation importance validation
- Reduce from 267 → 100-120 optimal features

Duration: Day 2 Afternoon (4-5 hours)
Priority: CRITICAL
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import warnings
from typing import List, Dict, Tuple, Optional
import joblib

# ML libraries
from sklearn.feature_selection import (
    SelectFromModel, RFE, RFECV, SelectKBest, 
    f_regression, mutual_info_regression
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

class SystematicFeatureSelector:
    """
    Comprehensive systematic feature selection for Delhi Load Forecasting
    """
    
    def __init__(self, data_path: str, target_column: str = 'delhi_load_mw'):
        """
        Initialize systematic feature selector
        
        Args:
            data_path: Path to feature dataset
            target_column: Name of target variable
        """
        self.data_path = data_path
        self.target_column = target_column
        self.target_features = (100, 120)  # Target range
        self.results_dir = Path("phase_2_5_3_outputs")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Delhi-specific feature priorities
        self.delhi_priorities = {
            'dual_peak_features': ['dual_peak', 'morning_peak', 'evening_peak', 'peak_ratio'],
            'weather_interactions': ['temp_load', 'humidity_load', 'weather_pattern'],
            'temporal_patterns': ['hour', 'day_of_week', 'month', 'season', 'weekend'],
            'festival_cultural': ['festival', 'holiday', 'diwali', 'holi', 'cultural'],
            'thermal_comfort': ['thermal_index', 'heat_index', 'comfort_zone'],
            'load_components': ['BRPL', 'BYPL', 'NDPL', 'DTL', 'MES']
        }
        
        self.selection_results = {
            'random_forest_selection': {},
            'lasso_selection': {},
            'rfe_selection': {},
            'permutation_selection': {},
            'ensemble_selection': {},
            'final_features': [],
            'performance_comparison': {}
        }
        
        # Models for feature selection
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'lasso': LassoCV(
                alphas=np.logspace(-4, 1, 50), cv=5, random_state=42, max_iter=2000
            ),
            'elastic_net': ElasticNetCV(
                alphas=np.logspace(-4, 1, 20), cv=5, random_state=42, max_iter=2000
            )
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare dataset"""
        try:
            print("Loading dataset for feature selection...")
            
            if self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                try:
                    df = pd.read_parquet(self.data_path)
                except:
                    df = pd.read_csv(self.data_path)
            
            print(f"Dataset loaded: {df.shape}")
            
            # Handle datetime index
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            
            # Ensure target column exists
            if self.target_column not in df.columns:
                available_targets = [col for col in df.columns if 'load' in col.lower()]
                if available_targets:
                    self.target_column = available_targets[0]
                    print(f"Target column adjusted to: {self.target_column}")
                else:
                    raise ValueError("No suitable target column found")
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != self.target_column]
            
            # Clean data
            X = df[feature_columns].select_dtypes(include=[np.number])
            y = df[self.target_column]
            
            # Remove features with zero variance or all NaN
            X = X.loc[:, X.var() > 0]
            X = X.loc[:, X.notna().sum() > len(X) * 0.1]  # At least 10% non-null
            
            # Align X and y (common non-null indices)
            common_idx = X.index.intersection(y.dropna().index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            print(f"Cleaned dataset: Features={X.shape[1]}, Samples={len(X)}")
            
            return X, y
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def random_forest_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Random Forest based feature importance selection
        """
        print(f"\n{'='*60}")
        print("METHOD 1: RANDOM FOREST FEATURE IMPORTANCE")
        print(f"{'='*60}")
        
        # Standardize features for fair comparison
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Train Random Forest
        print("Training Random Forest model...")
        rf_model = self.models['random_forest']
        rf_model.fit(X_scaled, y)
        
        # Get feature importances
        feature_importances = pd.Series(
            rf_model.feature_importances_, 
            index=X.columns
        ).sort_values(ascending=False)
        
        # Calculate additional metrics
        rf_score = rf_model.score(X_scaled, y)
        
        # Select features using different thresholds
        selector_model = SelectFromModel(rf_model, prefit=True)
        
        # Try different thresholds to get optimal number of features
        thresholds = ['mean', 'median', '0.1*mean', '2*mean']
        threshold_results = {}
        
        for threshold in thresholds:
            selector = SelectFromModel(rf_model, threshold=threshold, prefit=True)
            try:
                X_selected = selector.transform(X_scaled)
                selected_features = X.columns[selector.get_support()].tolist()
                
                threshold_results[threshold] = {
                    'n_features': len(selected_features),
                    'features': selected_features,
                    'total_importance': feature_importances[selected_features].sum()
                }
            except:
                threshold_results[threshold] = {'n_features': 0, 'features': []}
        
        # Select threshold that gives us closest to target range
        target_min, target_max = self.target_features
        best_threshold = None
        best_score = float('inf')
        
        for threshold, results in threshold_results.items():
            n_feat = results['n_features']
            if target_min <= n_feat <= target_max:
                score = abs(n_feat - (target_min + target_max) / 2)
                if score < best_score:
                    best_score = score
                    best_threshold = threshold
        
        # If no threshold gives target range, select top features manually
        if best_threshold is None:
            target_count = int((target_min + target_max) / 2)  # Target middle of range
            selected_features = feature_importances.head(target_count).index.tolist()
        else:
            selected_features = threshold_results[best_threshold]['features']
        
        # Calculate Delhi-specific priorities
        delhi_scores = self._calculate_delhi_priorities(selected_features)
        
        rf_results = {
            'model': rf_model,
            'feature_importances': feature_importances.to_dict(),
            'model_score': rf_score,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'threshold_results': threshold_results,
            'best_threshold': best_threshold,
            'delhi_priority_scores': delhi_scores
        }
        
        # Create visualizations
        self._plot_rf_importance(feature_importances, selected_features)
        
        print(f"Random Forest Results:")
        print(f"  Model R²: {rf_score:.4f}")
        print(f"  Selected features: {len(selected_features)}")
        print(f"  Best threshold: {best_threshold}")
        
        self.selection_results['random_forest_selection'] = rf_results
        return rf_results
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        LASSO regularization for automatic feature selection
        """
        print(f"\n{'='*60}")
        print("METHOD 2: LASSO REGULARIZATION SELECTION")
        print(f"{'='*60}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Train LASSO with cross-validation
        print("Training LASSO with cross-validation...")
        lasso_model = self.models['lasso']
        lasso_model.fit(X_scaled, y)
        
        # Get selected features (non-zero coefficients)
        lasso_coef = pd.Series(lasso_model.coef_, index=X.columns)
        selected_features = lasso_coef[lasso_coef != 0].index.tolist()
        
        # If too many features, select top by absolute coefficient
        target_min, target_max = self.target_features
        if len(selected_features) > target_max:
            top_features = lasso_coef.abs().sort_values(ascending=False).head(target_max)
            selected_features = top_features.index.tolist()
        
        # Calculate feature scores
        feature_scores = lasso_coef.abs().sort_values(ascending=False)
        
        # Model performance
        lasso_score = lasso_model.score(X_scaled, y)
        
        # Try different alpha values for comparison
        alphas_test = [lasso_model.alpha_ * factor for factor in [0.1, 0.5, 2.0, 5.0]]
        alpha_results = {}
        
        for alpha in alphas_test:
            try:
                temp_lasso = LassoCV(alphas=[alpha], cv=3, random_state=42)
                temp_lasso.fit(X_scaled, y)
                temp_selected = X.columns[temp_lasso.coef_ != 0].tolist()
                
                alpha_results[alpha] = {
                    'n_features': len(temp_selected),
                    'features': temp_selected,
                    'score': temp_lasso.score(X_scaled, y)
                }
            except:
                alpha_results[alpha] = {'n_features': 0, 'features': [], 'score': 0}
        
        # Calculate Delhi priorities
        delhi_scores = self._calculate_delhi_priorities(selected_features)
        
        lasso_results = {
            'model': lasso_model,
            'feature_coefficients': lasso_coef.to_dict(),
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'model_score': lasso_score,
            'optimal_alpha': lasso_model.alpha_,
            'alpha_results': alpha_results,
            'feature_scores': feature_scores.to_dict(),
            'delhi_priority_scores': delhi_scores
        }
        
        # Create visualizations
        self._plot_lasso_path(lasso_model, X_scaled, y)
        self._plot_lasso_coefficients(lasso_coef, selected_features)
        
        print(f"LASSO Results:")
        print(f"  Model R²: {lasso_score:.4f}")
        print(f"  Optimal α: {lasso_model.alpha_:.6f}")
        print(f"  Selected features: {len(selected_features)}")
        
        self.selection_results['lasso_selection'] = lasso_results
        return lasso_results
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Recursive Feature Elimination with Cross-Validation
        """
        print(f"\n{'='*60}")
        print("METHOD 3: RECURSIVE FEATURE ELIMINATION (RFE-CV)")
        print(f"{'='*60}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        # Use time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Target number of features (middle of range)
        target_n_features = int((self.target_features[0] + self.target_features[1]) / 2)
        
        print(f"Running RFE with target features: {target_n_features}")
        
        # RFE with Cross Validation
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # RFECV to find optimal number of features
        rfecv = RFECV(
            estimator=estimator, 
            cv=tscv, 
            scoring='r2',
            min_features_to_select=50,
            n_jobs=-1
        )
        
        print("Training RFECV...")
        rfecv.fit(X_scaled, y)
        
        optimal_features = X.columns[rfecv.support_].tolist()
        
        # If not in target range, adjust
        if len(optimal_features) > self.target_features[1]:
            # Use standard RFE with target number
            rfe = RFE(estimator=estimator, n_features_to_select=target_n_features)
            rfe.fit(X_scaled, y)
            selected_features = X.columns[rfe.support_].tolist()
            feature_rankings = pd.Series(rfe.ranking_, index=X.columns)
        else:
            selected_features = optimal_features
            feature_rankings = pd.Series(rfecv.ranking_, index=X.columns)
        
        # Calculate feature importance from final estimator
        if hasattr(rfecv.estimator_, 'feature_importances_'):
            importance_scores = pd.Series(
                rfecv.estimator_.feature_importances_[rfecv.support_],
                index=selected_features
            ).sort_values(ascending=False)
        else:
            importance_scores = pd.Series(index=selected_features)
        
        # Calculate Delhi priorities
        delhi_scores = self._calculate_delhi_priorities(selected_features)
        
        rfe_results = {
            'model': rfecv,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'optimal_n_features': rfecv.n_features_,
            'feature_rankings': feature_rankings.to_dict(),
            'cv_scores': rfecv.cv_results_.get('mean_test_score', []).tolist() if hasattr(rfecv.cv_results_, 'get') else [],
            'importance_scores': importance_scores.to_dict(),
            'delhi_priority_scores': delhi_scores
        }
        
        # Create visualizations
        self._plot_rfe_results(rfecv, feature_rankings)
        
        print(f"RFE Results:")
        print(f"  Optimal features (RFECV): {rfecv.n_features_}")
        print(f"  Selected features: {len(selected_features)}")
        print(f"  Feature ranking range: {feature_rankings.min()}-{feature_rankings.max()}")
        
        self.selection_results['rfe_selection'] = rfe_results
        return rfe_results
    
    def permutation_importance_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Permutation importance based feature selection
        """
        print(f"\n{'='*60}")
        print("METHOD 4: PERMUTATION IMPORTANCE SELECTION")
        print(f"{'='*60}")
        
        # Use a subset of features to speed up computation
        if X.shape[1] > 150:
            # Pre-select top features using correlation with target
            target_corrs = X.corrwith(y).abs().sort_values(ascending=False)
            top_features = target_corrs.head(150).index.tolist()
            X_subset = X[top_features]
            print(f"Using top 150 features by target correlation for permutation importance")
        else:
            X_subset = X.copy()
            top_features = X.columns.tolist()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_subset), 
            columns=X_subset.columns, 
            index=X_subset.index
        )
        
        # Train base model
        base_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        base_model.fit(X_scaled, y)
        
        print("Calculating permutation importance...")
        # Calculate permutation importance
        perm_importance = permutation_importance(
            base_model, X_scaled, y, 
            n_repeats=5, random_state=42, n_jobs=-1,
            scoring='r2'
        )
        
        # Create feature importance series
        perm_scores = pd.Series(
            perm_importance.importances_mean, 
            index=X_subset.columns
        ).sort_values(ascending=False)
        
        perm_std = pd.Series(
            perm_importance.importances_std, 
            index=X_subset.columns
        )
        
        # Select top features
        target_n_features = int((self.target_features[0] + self.target_features[1]) / 2)
        selected_features = perm_scores.head(target_n_features).index.tolist()
        
        # Calculate Delhi priorities
        delhi_scores = self._calculate_delhi_priorities(selected_features)
        
        perm_results = {
            'model': base_model,
            'permutation_scores': perm_scores.to_dict(),
            'permutation_std': perm_std.to_dict(),
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'base_model_score': base_model.score(X_scaled, y),
            'delhi_priority_scores': delhi_scores
        }
        
        # Create visualizations
        self._plot_permutation_importance(perm_scores, perm_std, selected_features)
        
        print(f"Permutation Importance Results:")
        print(f"  Base model R²: {base_model.score(X_scaled, y):.4f}")
        print(f"  Selected features: {len(selected_features)}")
        print(f"  Mean importance: {perm_scores.mean():.6f}")
        
        self.selection_results['permutation_selection'] = perm_results
        return perm_results
    
    def ensemble_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Ensemble method combining all selection techniques
        """
        print(f"\n{'='*60}")
        print("METHOD 5: ENSEMBLE FEATURE SELECTION")
        print(f"{'='*60}")
        
        # Get results from all methods
        rf_features = set(self.selection_results['random_forest_selection']['selected_features'])
        lasso_features = set(self.selection_results['lasso_selection']['selected_features'])
        rfe_features = set(self.selection_results['rfe_selection']['selected_features'])
        perm_features = set(self.selection_results['permutation_selection']['selected_features'])
        
        all_methods = {
            'random_forest': rf_features,
            'lasso': lasso_features,
            'rfe': rfe_features,
            'permutation': perm_features
        }
        
        # Count votes for each feature
        feature_votes = {}
        all_features = rf_features | lasso_features | rfe_features | perm_features
        
        for feature in all_features:
            votes = sum(1 for method_features in all_methods.values() if feature in method_features)
            feature_votes[feature] = votes
        
        # Sort by votes
        feature_votes_sorted = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        
        # Select features with multiple votes first
        consensus_features = []
        majority_features = []
        minority_features = []
        
        for feature, votes in feature_votes_sorted:
            if votes >= 3:  # Selected by at least 3 methods
                consensus_features.append(feature)
            elif votes == 2:  # Selected by 2 methods
                majority_features.append(feature)
            else:  # Selected by 1 method
                minority_features.append(feature)
        
        print(f"Consensus features (3+ methods): {len(consensus_features)}")
        print(f"Majority features (2 methods): {len(majority_features)}")
        print(f"Minority features (1 method): {len(minority_features)}")
        
        # Build final feature set
        target_min, target_max = self.target_features
        final_features = consensus_features.copy()
        
        # Add majority features if needed
        remaining_slots = target_max - len(final_features)
        if remaining_slots > 0:
            # Prioritize by Delhi-specific importance
            majority_delhi_scores = [(f, self._calculate_single_delhi_priority(f)) 
                                   for f in majority_features]
            majority_delhi_sorted = sorted(majority_delhi_scores, key=lambda x: x[1], reverse=True)
            
            for feature, _ in majority_delhi_sorted[:remaining_slots]:
                final_features.append(feature)
        
        # If still need more features, add from minority
        remaining_slots = target_max - len(final_features)
        if remaining_slots > 0:
            minority_delhi_scores = [(f, self._calculate_single_delhi_priority(f)) 
                                   for f in minority_features]
            minority_delhi_sorted = sorted(minority_delhi_scores, key=lambda x: x[1], reverse=True)
            
            for feature, _ in minority_delhi_sorted[:remaining_slots]:
                final_features.append(feature)
        
        # If too many features, remove least voted first
        if len(final_features) > target_max:
            # Sort by votes and Delhi priority
            final_scores = [(f, feature_votes[f], self._calculate_single_delhi_priority(f)) 
                          for f in final_features]
            final_scores_sorted = sorted(final_scores, 
                                       key=lambda x: (x[1], x[2]), reverse=True)
            final_features = [f[0] for f in final_scores_sorted[:target_max]]
        
        # Validate final selection with model performance
        ensemble_performance = self._validate_feature_set(X, y, final_features)
        
        # Calculate comprehensive feature analysis
        final_analysis = self._analyze_final_features(final_features)
        
        ensemble_results = {
            'method_features': {k: list(v) for k, v in all_methods.items()},
            'feature_votes': feature_votes,
            'consensus_features': consensus_features,
            'majority_features': majority_features,
            'minority_features': minority_features,
            'final_features': final_features,
            'n_final': len(final_features),
            'ensemble_performance': ensemble_performance,
            'final_analysis': final_analysis
        }
        
        # Create visualizations
        self._plot_ensemble_analysis(feature_votes, final_features, all_methods)
        
        print(f"Ensemble Selection Results:")
        print(f"  Final features selected: {len(final_features)}")
        print(f"  Model performance: R² = {ensemble_performance['cv_score_mean']:.4f}")
        print(f"  Within target range: {target_min <= len(final_features) <= target_max}")
        
        self.selection_results['ensemble_selection'] = ensemble_results
        self.selection_results['final_features'] = final_features
        
        return ensemble_results
    
    def _calculate_delhi_priorities(self, features: List[str]) -> Dict:
        """Calculate Delhi-specific priority scores for feature list"""
        scores = {}
        for feature in features:
            scores[feature] = self._calculate_single_delhi_priority(feature)
        return scores
    
    def _calculate_single_delhi_priority(self, feature_name: str) -> float:
        """Calculate Delhi-specific priority score for single feature"""
        feature_lower = feature_name.lower()
        score = 0.0
        
        for category, keywords in self.delhi_priorities.items():
            for keyword in keywords:
                if keyword.lower() in feature_lower:
                    if category == 'dual_peak_features':
                        score += 0.5  # Highest priority for Delhi
                    elif category in ['weather_interactions', 'temporal_patterns']:
                        score += 0.3
                    elif category == 'festival_cultural':
                        score += 0.4  # Very important for Delhi
                    elif category == 'thermal_comfort':
                        score += 0.25
                    elif category == 'load_components':
                        score += 0.2
                    break
        
        return min(score, 1.0)
    
    def _validate_feature_set(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> Dict:
        """Validate feature set with cross-validation"""
        X_selected = X[features]
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Test with multiple models
        models_test = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LassoCV(random_state=42)
        }
        
        results = {}
        for model_name, model in models_test.items():
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
            results[f'{model_name}_scores'] = scores.tolist()
            results[f'{model_name}_mean'] = scores.mean()
            results[f'{model_name}_std'] = scores.std()
        
        # Overall performance
        results['cv_score_mean'] = np.mean([results['random_forest_mean'], results['linear_mean']])
        results['cv_score_std'] = np.mean([results['random_forest_std'], results['linear_std']])
        
        return results
    
    def _analyze_final_features(self, features: List[str]) -> Dict:
        """Analyze final feature selection"""
        analysis = {
            'n_features': len(features),
            'delhi_categories': {},
            'feature_types': {},
            'priority_distribution': []
        }
        
        # Categorize by Delhi priorities
        for category, keywords in self.delhi_priorities.items():
            category_features = []
            for feature in features:
                feature_lower = feature.lower()
                if any(keyword.lower() in feature_lower for keyword in keywords):
                    category_features.append(feature)
            analysis['delhi_categories'][category] = category_features
        
        # Calculate priority scores
        for feature in features:
            priority_score = self._calculate_single_delhi_priority(feature)
            analysis['priority_distribution'].append({
                'feature': feature,
                'priority_score': priority_score
            })
        
        return analysis
    
    def _plot_rf_importance(self, importance: pd.Series, selected: List[str]):
        """Plot Random Forest importance"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Top 20 importances
        top_20 = importance.head(20)
        axes[0].barh(range(len(top_20)), top_20.values)
        axes[0].set_yticks(range(len(top_20)))
        axes[0].set_yticklabels([f[:25] for f in top_20.index])
        axes[0].set_xlabel('Feature Importance')
        axes[0].set_title('Top 20 Random Forest Feature Importances')
        axes[0].grid(True, alpha=0.3)
        
        # Importance distribution
        axes[1].hist(importance.values, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(x=importance[selected].min(), color='red', linestyle='--', 
                       label=f'Selection Threshold')
        axes[1].set_xlabel('Feature Importance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Feature Importance Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'random_forest_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lasso_path(self, lasso_model, X, y):
        """Plot LASSO regularization path"""
        from sklearn.linear_model import lasso_path
        
        alphas, coefs, _ = lasso_path(X, y, alphas=lasso_model.alphas_)
        
        plt.figure(figsize=(12, 8))
        plt.plot(alphas, coefs.T, alpha=0.5)
        plt.axvline(x=lasso_model.alpha_, color='red', linestyle='--', 
                   label=f'Selected α = {lasso_model.alpha_:.6f}')
        plt.xlabel('Alpha (Regularization strength)')
        plt.ylabel('Coefficient value')
        plt.title('LASSO Regularization Path')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'lasso_path.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_lasso_coefficients(self, coefs: pd.Series, selected: List[str]):
        """Plot LASSO coefficients"""
        # Non-zero coefficients
        non_zero_coefs = coefs[coefs != 0].sort_values(key=abs, ascending=False)
        
        if len(non_zero_coefs) > 0:
            plt.figure(figsize=(12, 8))
            colors = ['red' if abs(c) > coefs.abs().quantile(0.8) else 'blue' 
                     for c in non_zero_coefs.values]
            
            plt.barh(range(len(non_zero_coefs)), non_zero_coefs.values, color=colors, alpha=0.7)
            plt.yticks(range(len(non_zero_coefs)), [f[:25] for f in non_zero_coefs.index])
            plt.xlabel('Coefficient Value')
            plt.title('LASSO Coefficients (Non-zero)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'lasso_coefficients.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_rfe_results(self, rfe_model, rankings: pd.Series):
        """Plot RFE results"""
        if hasattr(rfe_model, 'cv_results_'):
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # CV scores vs number of features
            n_features = range(1, len(rfe_model.cv_results_['mean_test_score']) + 1)
            axes[0].plot(n_features, rfe_model.cv_results_['mean_test_score'], 'b-', label='CV Score')
            axes[0].axvline(x=rfe_model.n_features_, color='red', linestyle='--', 
                           label=f'Optimal ({rfe_model.n_features_} features)')
            axes[0].set_xlabel('Number of Features')
            axes[0].set_ylabel('Cross-validation Score')
            axes[0].set_title('RFE Cross-validation Scores')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Feature rankings
            ranking_counts = rankings.value_counts().sort_index()
            axes[1].bar(ranking_counts.index, ranking_counts.values)
            axes[1].set_xlabel('Feature Ranking')
            axes[1].set_ylabel('Number of Features')
            axes[1].set_title('Feature Ranking Distribution')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.results_dir / 'rfe_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_permutation_importance(self, scores: pd.Series, std: pd.Series, selected: List[str]):
        """Plot permutation importance"""
        # Top 20 features
        top_20 = scores.head(20)
        top_20_std = std[top_20.index]
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(top_20))
        
        plt.barh(y_pos, top_20.values, xerr=top_20_std.values, 
                alpha=0.7, capsize=5, color='skyblue', edgecolor='black')
        plt.yticks(y_pos, [f[:25] for f in top_20.index])
        plt.xlabel('Permutation Importance')
        plt.title('Top 20 Permutation Importance Scores')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_ensemble_analysis(self, votes: Dict, final_features: List[str], methods: Dict):
        """Plot ensemble analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Vote distribution
        vote_counts = {}
        for feature, vote_count in votes.items():
            if vote_count not in vote_counts:
                vote_counts[vote_count] = 0
            vote_counts[vote_count] += 1
        
        axes[0, 0].bar(vote_counts.keys(), vote_counts.values())
        axes[0, 0].set_xlabel('Number of Methods Selecting Feature')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].set_title('Feature Selection Vote Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Method overlap
        method_sizes = [len(features) for features in methods.values()]
        axes[0, 1].bar(methods.keys(), method_sizes)
        axes[0, 1].set_ylabel('Number of Selected Features')
        axes[0, 1].set_title('Features Selected by Each Method')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Final feature votes
        final_votes = [votes[f] for f in final_features]
        axes[1, 0].hist(final_votes, bins=range(1, 6), alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Vote Count for Final Features')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Vote Distribution in Final Selection')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Delhi priority scores
        delhi_scores = [self._calculate_single_delhi_priority(f) for f in final_features]
        axes[1, 1].hist(delhi_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Delhi Priority Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Delhi Priority Distribution in Final Selection')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'ensemble_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_selection_report(self) -> Dict:
        """Generate comprehensive feature selection report"""
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'target_feature_range': f"{self.target_features[0]}-{self.target_features[1]}",
            'methods_used': [
                'Random Forest Feature Importance',
                'LASSO Regularization',
                'Recursive Feature Elimination',
                'Permutation Importance',
                'Ensemble Selection'
            ],
            'results_summary': {},
            'final_selection': {},
            'recommendations': []
        }
        
        # Summarize each method
        for method_name, results in self.selection_results.items():
            if method_name != 'final_features' and isinstance(results, dict):
                if 'selected_features' in results:
                    report['results_summary'][method_name] = {
                        'n_selected': results['n_selected'],
                        'features': results['selected_features']
                    }
        
        # Final selection summary
        final_features = self.selection_results.get('final_features', [])
        ensemble_results = self.selection_results.get('ensemble_selection', {})
        
        report['final_selection'] = {
            'n_features': len(final_features),
            'features': final_features,
            'within_target_range': self.target_features[0] <= len(final_features) <= self.target_features[1],
            'ensemble_performance': ensemble_results.get('ensemble_performance', {}),
            'delhi_analysis': ensemble_results.get('final_analysis', {})
        }
        
        # Generate recommendations
        if len(final_features) < self.target_features[0]:
            report['recommendations'].append(
                f"⚠️ Feature count ({len(final_features)}) below minimum target ({self.target_features[0]})"
            )
        elif len(final_features) > self.target_features[1]:
            report['recommendations'].append(
                f"⚠️ Feature count ({len(final_features)}) above maximum target ({self.target_features[1]})"
            )
        else:
            report['recommendations'].append(
                f"✅ Feature count ({len(final_features)}) within optimal range"
            )
        
        performance = ensemble_results.get('ensemble_performance', {})
        if performance.get('cv_score_mean', 0) > 0.8:
            report['recommendations'].append("✅ Strong model performance achieved")
        elif performance.get('cv_score_mean', 0) > 0.7:
            report['recommendations'].append("⚠️ Moderate model performance - consider feature engineering")
        else:
            report['recommendations'].append("❌ Low model performance - review feature selection")
        
        # Delhi-specific analysis
        delhi_analysis = ensemble_results.get('final_analysis', {})
        delhi_categories = delhi_analysis.get('delhi_categories', {})
        
        essential_categories = ['dual_peak_features', 'weather_interactions', 'festival_cultural']
        missing_categories = []
        for category in essential_categories:
            if not delhi_categories.get(category, []):
                missing_categories.append(category)
        
        if missing_categories:
            report['recommendations'].append(
                f"⚠️ Missing Delhi-specific features: {', '.join(missing_categories)}"
            )
        else:
            report['recommendations'].append("✅ All essential Delhi-specific categories represented")
        
        # Save report
        with open(self.results_dir / 'feature_selection_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def save_selected_features(self, X: pd.DataFrame, output_path: str = None):
        """Save dataset with selected features only"""
        final_features = self.selection_results.get('final_features', [])
        
        if not final_features:
            print("No features selected yet. Run feature selection first.")
            return
        
        if output_path is None:
            output_path = self.results_dir / 'selected_features_dataset.parquet'
        
        # Create dataset with selected features + target
        selected_columns = final_features + [self.target_column]
        available_columns = [col for col in selected_columns if col in X.columns]
        
        X_selected = X[available_columns]
        X_selected.to_parquet(output_path)
        
        print(f"Selected features dataset saved to: {output_path}")
        print(f"Features saved: {len(final_features)}")
        
        # Also save feature list
        feature_list_path = self.results_dir / 'selected_features_list.json'
        with open(feature_list_path, 'w') as f:
            json.dump({
                'selected_features': final_features,
                'n_features': len(final_features),
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
    
    def run_complete_selection(self, data_path: str = None) -> Dict:
        """
        Run complete systematic feature selection pipeline
        """
        print("="*80)
        print("SYSTEMATIC FEATURE SELECTION - DELHI LOAD FORECASTING")
        print("="*80)
        print(f"Analysis started at: {datetime.now()}")
        
        if data_path:
            self.data_path = data_path
        
        # Load data
        X, y = self.load_data()
        
        # Run all selection methods
        print(f"\nRunning feature selection on {X.shape[1]} features...")
        
        # Method 1: Random Forest
        rf_results = self.random_forest_selection(X, y)
        
        # Method 2: LASSO
        lasso_results = self.lasso_selection(X, y)
        
        # Method 3: RFE
        rfe_results = self.recursive_feature_elimination(X, y)
        
        # Method 4: Permutation Importance
        perm_results = self.permutation_importance_selection(X, y)
        
        # Method 5: Ensemble Selection
        ensemble_results = self.ensemble_feature_selection(X, y)
        
        # Generate comprehensive report
        final_report = self.generate_selection_report()
        
        # Save selected dataset
        self.save_selected_features(X)
        
        print(f"\n{'='*80}")
        print("SYSTEMATIC FEATURE SELECTION COMPLETED")
        print(f"{'='*80}")
        print(f"Analysis completed at: {datetime.now()}")
        print(f"Results saved to: {self.results_dir}")
        
        return {
            'random_forest': rf_results,
            'lasso': lasso_results,
            'rfe': rfe_results,
            'permutation': perm_results,
            'ensemble': ensemble_results,
            'final_report': final_report
        }

def main():
    """Main execution function"""
    
    # Configuration - use cleaned dataset from Step 1
    DATA_PATH = "phase_2_5_1_outputs/delhi_interaction_enhanced_cleaned.csv"
    TARGET_COLUMN = "delhi_load"  # Primary target
    
    print("Starting Systematic Feature Selection...")
    
    # Initialize selector
    selector = SystematicFeatureSelector(DATA_PATH, TARGET_COLUMN)
    
    # Run complete selection
    results = selector.run_complete_selection()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("SYSTEMATIC FEATURE SELECTION SUMMARY")
    print("="*80)
    
    final_features = selector.selection_results.get('final_features', [])
    target_min, target_max = selector.target_features
    
    print(f"Methods Applied:")
    for i, method in enumerate(['Random Forest', 'LASSO', 'RFE', 'Permutation', 'Ensemble'], 1):
        method_key = method.lower().replace(' ', '_') + '_selection'
        if method_key in selector.selection_results:
            n_selected = selector.selection_results[method_key].get('n_selected', 0)
            print(f"  {i}. {method}: {n_selected} features")
    
    print(f"\nFinal Selection:")
    print(f"  Target range: {target_min}-{target_max} features")
    print(f"  Final selection: {len(final_features)} features")
    print(f"  Within target: {'✅' if target_min <= len(final_features) <= target_max else '❌'}")
    
    # Performance summary
    ensemble_perf = selector.selection_results.get('ensemble_selection', {}).get('ensemble_performance', {})
    if ensemble_perf:
        cv_score = ensemble_perf.get('cv_score_mean', 0)
        print(f"  Cross-validation R²: {cv_score:.4f}")
        print(f"  Performance: {'✅ Strong' if cv_score > 0.8 else '⚠️ Moderate' if cv_score > 0.7 else '❌ Weak'}")
    
    # Delhi-specific analysis
    final_analysis = selector.selection_results.get('ensemble_selection', {}).get('final_analysis', {})
    if final_analysis:
        delhi_categories = final_analysis.get('delhi_categories', {})
        print(f"\nDelhi-Specific Features:")
        for category, features in delhi_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
    
    print(f"\nDetailed results saved to: {selector.results_dir}")

if __name__ == "__main__":
    main()
