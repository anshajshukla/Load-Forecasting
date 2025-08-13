"""
Phase 2.5.4: Feature Quality Validation
=====================================

Comprehensive feature quality assessment for Delhi load forecasting system.
Final validation step before Phase 3 modeling begins.

Validation Components:
1. Statistical Quality Assessment
2. Outlier Detection & Analysis  
3. Feature Distribution Validation
4. Business Logic Validation
5. Data Completeness Analysis
6. Feature Stability Assessment

Input: delhi_interaction_enhanced.csv (267 features)
Output: Comprehensive quality validation report

Author: SIH 2024 Team
Date: January 2025
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid GUI issues
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import json
import os

class FeatureQualityValidator:
    """
    Comprehensive feature quality validation for Delhi load forecasting.
    Ensures all features meet production-ready quality standards.
    """
    
    def __init__(self, data_path):
        """Initialize validator with dataset"""
        self.data_path = data_path
        self.data = None
        self.results = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_features': 0,
            'total_samples': 0,
            'statistical_quality': {},
            'outlier_analysis': {},
            'distribution_validation': {},
            'business_logic_validation': {},
            'completeness_analysis': {},
            'stability_assessment': {},
            'overall_quality_score': 0,
            'recommendations': []
        }
        
    def load_data(self):
        """Load the enhanced dataset with validation"""
        print("🔍 Loading enhanced dataset for quality validation...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            self.results['total_features'] = len(self.data.columns) - 1  # Exclude target
            self.results['total_samples'] = len(self.data)
            
            print(f"✅ Dataset loaded successfully:")
            print(f"   📊 Shape: {self.data.shape}")
            print(f"   🎯 Features: {self.results['total_features']}")
            print(f"   📈 Samples: {self.results['total_samples']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def validate_statistical_quality(self):
        """
        Comprehensive statistical quality assessment.
        Validates feature distributions, variance, and statistical properties.
        """
        print("\n🔬 Step 1: Statistical Quality Assessment")
        print("=" * 50)
        
        feature_cols = [col for col in self.data.columns if col != 'Load_MW']
        stats_results = {}
        
        # Key statistical metrics for each feature
        for col in feature_cols:
            try:
                series = self.data[col].dropna()
                
                # Basic statistics
                stats_dict = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'variance': float(series.var()),
                    'skewness': float(stats.skew(series)),
                    'kurtosis': float(stats.kurtosis(series)),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'range': float(series.max() - series.min()),
                    'iqr': float(series.quantile(0.75) - series.quantile(0.25)),
                    'cv': float(series.std() / series.mean()) if series.mean() != 0 else 0
                }
                
                # Quality flags
                quality_flags = []
                
                # Check for excessive skewness (>3 or <-3)
                if abs(stats_dict['skewness']) > 3:
                    quality_flags.append('high_skewness')
                
                # Check for excessive kurtosis (>10)
                if abs(stats_dict['kurtosis']) > 10:
                    quality_flags.append('high_kurtosis')
                
                # Check for zero variance
                if stats_dict['variance'] < 1e-10:
                    quality_flags.append('zero_variance')
                
                # Check for high coefficient of variation (>2)
                if stats_dict['cv'] > 2:
                    quality_flags.append('high_variability')
                
                # Normality test (Shapiro-Wilk for small samples)
                if len(series) <= 5000:
                    _, p_value = stats.shapiro(series.sample(min(5000, len(series))))
                    stats_dict['normality_p_value'] = float(p_value)
                    stats_dict['is_normal'] = p_value > 0.05
                else:
                    # Use Kolmogorov-Smirnov for larger samples
                    _, p_value = stats.kstest(series, 'norm')
                    stats_dict['normality_p_value'] = float(p_value)
                    stats_dict['is_normal'] = p_value > 0.05
                
                stats_dict['quality_flags'] = quality_flags
                stats_dict['quality_score'] = max(0, 1 - len(quality_flags) * 0.2)
                
                stats_results[col] = stats_dict
                
            except Exception as e:
                print(f"⚠️ Error processing {col}: {e}")
                stats_results[col] = {'error': str(e), 'quality_score': 0}
        
        # Summary statistics
        quality_scores = [v.get('quality_score', 0) for v in stats_results.values()]
        avg_quality = np.mean(quality_scores)
        
        # Identify problematic features
        problematic_features = [
            col for col, stats in stats_results.items() 
            if stats.get('quality_score', 0) < 0.7
        ]
        
        self.results['statistical_quality'] = {
            'feature_statistics': stats_results,
            'average_quality_score': float(avg_quality),
            'problematic_features': problematic_features,
            'total_problematic': len(problematic_features)
        }
        
        print(f"📈 Statistical Quality Results:")
        print(f"   🎯 Average Quality Score: {avg_quality:.3f}")
        print(f"   ⚠️ Problematic Features: {len(problematic_features)}")
        
        if problematic_features:
            print(f"   📋 Top Issues: {problematic_features[:5]}")
    
    def detect_outliers(self):
        """
        Advanced outlier detection using multiple methods.
        Identifies anomalous values that could impact model performance.
        """
        print("\n🎯 Step 2: Outlier Detection & Analysis")
        print("=" * 50)
        
        feature_cols = [col for col in self.data.columns if col != 'Load_MW']
        outlier_results = {}
        
        # Method 1: IQR-based outlier detection
        def iqr_outliers(series):
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        # Method 2: Z-score outliers (>3 std)
        def zscore_outliers(series):
            z_scores = np.abs(stats.zscore(series))
            return z_scores > 3
        
        # Method 3: Isolation Forest
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        
        for col in feature_cols:
            try:
                series = self.data[col].dropna()
                
                if len(series) == 0:
                    continue
                
                # IQR outliers
                iqr_mask = iqr_outliers(series)
                iqr_outliers_count = iqr_mask.sum()
                iqr_percentage = (iqr_outliers_count / len(series)) * 100
                
                # Z-score outliers
                zscore_mask = zscore_outliers(series)
                zscore_outliers_count = zscore_mask.sum()
                zscore_percentage = (zscore_outliers_count / len(series)) * 100
                
                # Isolation Forest outliers
                try:
                    if_predictions = isolation_forest.fit_predict(series.values.reshape(-1, 1))
                    if_outliers_count = (if_predictions == -1).sum()
                    if_percentage = (if_outliers_count / len(series)) * 100
                except:
                    if_outliers_count = 0
                    if_percentage = 0
                
                # Outlier severity assessment
                severity = 'low'
                if max(iqr_percentage, zscore_percentage, if_percentage) > 10:
                    severity = 'high'
                elif max(iqr_percentage, zscore_percentage, if_percentage) > 5:
                    severity = 'medium'
                
                outlier_results[col] = {
                    'iqr_outliers': int(iqr_outliers_count),
                    'iqr_percentage': float(iqr_percentage),
                    'zscore_outliers': int(zscore_outliers_count),
                    'zscore_percentage': float(zscore_percentage),
                    'isolation_forest_outliers': int(if_outliers_count),
                    'if_percentage': float(if_percentage),
                    'severity': severity,
                    'max_outlier_percentage': float(max(iqr_percentage, zscore_percentage, if_percentage))
                }
                
            except Exception as e:
                print(f"⚠️ Error in outlier detection for {col}: {e}")
                outlier_results[col] = {'error': str(e)}
        
        # Summary analysis
        high_outlier_features = [
            col for col, results in outlier_results.items()
            if results.get('severity') == 'high'
        ]
        
        medium_outlier_features = [
            col for col, results in outlier_results.items()
            if results.get('severity') == 'medium'
        ]
        
        self.results['outlier_analysis'] = {
            'feature_outliers': outlier_results,
            'high_outlier_features': high_outlier_features,
            'medium_outlier_features': medium_outlier_features,
            'total_high_outlier_features': len(high_outlier_features),
            'total_medium_outlier_features': len(medium_outlier_features)
        }
        
        print(f"🎯 Outlier Detection Results:")
        print(f"   🔴 High Outlier Features: {len(high_outlier_features)}")
        print(f"   🟡 Medium Outlier Features: {len(medium_outlier_features)}")
        
        if high_outlier_features:
            print(f"   📋 High Priority: {high_outlier_features[:3]}")
    
    def validate_feature_distributions(self):
        """
        Validate feature distributions for modeling suitability.
        Checks for appropriate ranges, patterns, and distribution shapes.
        """
        print("\n📊 Step 3: Feature Distribution Validation")
        print("=" * 50)
        
        feature_cols = [col for col in self.data.columns if col != 'Load_MW']
        distribution_results = {}
        
        for col in feature_cols:
            try:
                series = self.data[col].dropna()
                
                if len(series) == 0:
                    continue
                
                # Distribution characteristics
                dist_char = {
                    'unique_values': int(series.nunique()),
                    'unique_percentage': float((series.nunique() / len(series)) * 100),
                    'zero_count': int((series == 0).sum()),
                    'zero_percentage': float(((series == 0).sum() / len(series)) * 100),
                    'negative_count': int((series < 0).sum()),
                    'negative_percentage': float(((series < 0).sum() / len(series)) * 100),
                    'infinite_count': int(np.isinf(series).sum()),
                    'range_ratio': float(series.max() / series.min()) if series.min() != 0 else float('inf')
                }
                
                # Distribution quality flags
                quality_flags = []
                
                # Check for constant features
                if dist_char['unique_values'] == 1:
                    quality_flags.append('constant_feature')
                
                # Check for binary features
                elif dist_char['unique_values'] == 2:
                    quality_flags.append('binary_feature')
                
                # Check for high sparsity (>80% zeros)
                if dist_char['zero_percentage'] > 80:
                    quality_flags.append('high_sparsity')
                
                # Check for infinite values
                if dist_char['infinite_count'] > 0:
                    quality_flags.append('infinite_values')
                
                # Check for extreme range ratios
                if dist_char['range_ratio'] > 1e6:
                    quality_flags.append('extreme_range')
                
                # Distribution type detection
                distribution_type = 'continuous'
                if dist_char['unique_values'] <= 10:
                    distribution_type = 'categorical'
                elif dist_char['unique_values'] <= 50 and all(series == series.astype(int)):
                    distribution_type = 'discrete'
                
                dist_char['distribution_type'] = distribution_type
                dist_char['quality_flags'] = quality_flags
                dist_char['quality_score'] = max(0, 1 - len(quality_flags) * 0.25)
                
                distribution_results[col] = dist_char
                
            except Exception as e:
                print(f"⚠️ Error in distribution analysis for {col}: {e}")
                distribution_results[col] = {'error': str(e), 'quality_score': 0}
        
        # Summary analysis
        problematic_distributions = [
            col for col, results in distribution_results.items()
            if results.get('quality_score', 0) < 0.7
        ]
        
        constant_features = [
            col for col, results in distribution_results.items()
            if 'constant_feature' in results.get('quality_flags', [])
        ]
        
        sparse_features = [
            col for col, results in distribution_results.items()
            if 'high_sparsity' in results.get('quality_flags', [])
        ]
        
        self.results['distribution_validation'] = {
            'feature_distributions': distribution_results,
            'problematic_distributions': problematic_distributions,
            'constant_features': constant_features,
            'sparse_features': sparse_features,
            'total_problematic': len(problematic_distributions),
            'total_constant': len(constant_features),
            'total_sparse': len(sparse_features)
        }
        
        print(f"📊 Distribution Validation Results:")
        print(f"   ⚠️ Problematic Distributions: {len(problematic_distributions)}")
        print(f"   🔴 Constant Features: {len(constant_features)}")
        print(f"   🟡 Sparse Features: {len(sparse_features)}")
    
    def validate_business_logic(self):
        """
        Validate features against Delhi load forecasting business rules.
        Ensures features make sense in the energy domain context.
        """
        print("\n🏢 Step 4: Business Logic Validation")
        print("=" * 50)
        
        business_results = {}
        
        # Define expected ranges for different feature categories
        expected_ranges = {
            'temperature': (-5, 50),  # Delhi temperature range (°C)
            'humidity': (0, 100),     # Relative humidity (%)
            'pressure': (980, 1050),  # Atmospheric pressure (hPa)
            'wind_speed': (0, 25),    # Wind speed (m/s)
            'load': (1000, 8000),     # Load range (MW) for Delhi
            'thermal_comfort': (-10, 10),  # Thermal comfort indices
            'hour': (0, 23),          # Hour of day
            'month': (1, 12),         # Month of year
            'weekday': (0, 6),        # Day of week
        }
        
        # Business logic checks
        for col in self.data.columns:
            try:
                series = self.data[col].dropna()
                
                if len(series) == 0:
                    continue
                
                business_flags = []
                
                # Check temperature-related features
                if any(temp_word in col.lower() for temp_word in ['temp', 'temperature']):
                    min_temp, max_temp = expected_ranges['temperature']
                    if series.min() < min_temp or series.max() > max_temp:
                        business_flags.append('temperature_out_of_range')
                
                # Check humidity features
                if 'humidity' in col.lower():
                    min_hum, max_hum = expected_ranges['humidity']
                    if series.min() < min_hum or series.max() > max_hum:
                        business_flags.append('humidity_out_of_range')
                
                # Check pressure features
                if 'pressure' in col.lower():
                    min_press, max_press = expected_ranges['pressure']
                    if series.min() < min_press or series.max() > max_press:
                        business_flags.append('pressure_out_of_range')
                
                # Check wind speed features
                if 'wind' in col.lower() and 'speed' in col.lower():
                    min_wind, max_wind = expected_ranges['wind_speed']
                    if series.min() < 0 or series.max() > max_wind:
                        business_flags.append('wind_speed_out_of_range')
                
                # Check load features
                if col == 'Load_MW':
                    min_load, max_load = expected_ranges['load']
                    if series.min() < min_load or series.max() > max_load:
                        business_flags.append('load_out_of_range')
                
                # Check thermal comfort features
                if any(tc_word in col.lower() for tc_word in ['thermal', 'comfort', 'heat_index', 'feels_like']):
                    min_tc, max_tc = expected_ranges['thermal_comfort']
                    # More flexible range for derived thermal comfort indices
                    if series.min() < min_tc * 3 or series.max() > max_tc * 3:
                        business_flags.append('thermal_comfort_extreme')
                
                # Check temporal features
                if col.lower() == 'hour':
                    if series.min() < 0 or series.max() > 23:
                        business_flags.append('hour_out_of_range')
                
                if col.lower() == 'month':
                    if series.min() < 1 or series.max() > 12:
                        business_flags.append('month_out_of_range')
                
                # Check for reasonable correlation with load (for weather features)
                if col != 'Load_MW' and 'Load_MW' in self.data.columns:
                    correlation = self.data[col].corr(self.data['Load_MW'])
                    if abs(correlation) < 0.01:  # Very weak correlation
                        business_flags.append('weak_load_correlation')
                
                # Business logic score
                business_score = max(0, 1 - len(business_flags) * 0.2)
                
                business_results[col] = {
                    'business_flags': business_flags,
                    'business_score': float(business_score),
                    'min_value': float(series.min()),
                    'max_value': float(series.max()),
                    'load_correlation': float(correlation) if col != 'Load_MW' else 1.0
                }
                
            except Exception as e:
                print(f"⚠️ Error in business logic validation for {col}: {e}")
                business_results[col] = {'error': str(e), 'business_score': 0}
        
        # Summary analysis
        business_violations = [
            col for col, results in business_results.items()
            if results.get('business_score', 0) < 0.8
        ]
        
        weak_correlations = [
            col for col, results in business_results.items()
            if 'weak_load_correlation' in results.get('business_flags', [])
        ]
        
        self.results['business_logic_validation'] = {
            'feature_business_logic': business_results,
            'business_violations': business_violations,
            'weak_correlations': weak_correlations,
            'total_violations': len(business_violations),
            'total_weak_correlations': len(weak_correlations)
        }
        
        print(f"🏢 Business Logic Validation Results:")
        print(f"   ⚠️ Business Rule Violations: {len(business_violations)}")
        print(f"   🔗 Weak Load Correlations: {len(weak_correlations)}")
    
    def analyze_data_completeness(self):
        """
        Comprehensive data completeness analysis.
        Checks for missing values, data gaps, and temporal consistency.
        """
        print("\n📋 Step 5: Data Completeness Analysis")
        print("=" * 50)
        
        completeness_results = {}
        
        # Overall completeness metrics
        total_cells = self.data.shape[0] * self.data.shape[1]
        missing_cells = self.data.isnull().sum().sum()
        completeness_percentage = ((total_cells - missing_cells) / total_cells) * 100
        
        # Per-feature completeness
        for col in self.data.columns:
            series = self.data[col]
            
            missing_count = series.isnull().sum()
            missing_percentage = (missing_count / len(series)) * 100
            
            # Completeness quality score
            if missing_percentage == 0:
                completeness_score = 1.0
            elif missing_percentage < 5:
                completeness_score = 0.9
            elif missing_percentage < 10:
                completeness_score = 0.7
            elif missing_percentage < 20:
                completeness_score = 0.5
            else:
                completeness_score = 0.2
            
            completeness_results[col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_percentage),
                'completeness_score': float(completeness_score),
                'data_type': str(series.dtype)
            }
        
        # Identify problematic features
        high_missing_features = [
            col for col, results in completeness_results.items()
            if results['missing_percentage'] > 10
        ]
        
        medium_missing_features = [
            col for col, results in completeness_results.items()
            if 5 < results['missing_percentage'] <= 10
        ]
        
        self.results['completeness_analysis'] = {
            'overall_completeness_percentage': float(completeness_percentage),
            'total_missing_cells': int(missing_cells),
            'feature_completeness': completeness_results,
            'high_missing_features': high_missing_features,
            'medium_missing_features': medium_missing_features,
            'total_high_missing': len(high_missing_features),
            'total_medium_missing': len(medium_missing_features)
        }
        
        print(f"📋 Data Completeness Results:")
        print(f"   ✅ Overall Completeness: {completeness_percentage:.2f}%")
        print(f"   🔴 High Missing Features: {len(high_missing_features)}")
        print(f"   🟡 Medium Missing Features: {len(medium_missing_features)}")
    
    def assess_feature_stability(self):
        """
        Assess feature stability over time for temporal consistency.
        Critical for time series forecasting model reliability.
        """
        print("\n⏰ Step 6: Feature Stability Assessment")
        print("=" * 50)
        
        stability_results = {}
        
        # Split data into time periods for stability analysis
        # Assuming data is temporally ordered
        n_samples = len(self.data)
        period_size = n_samples // 4  # Quarterly analysis
        
        periods = [
            self.data.iloc[:period_size],
            self.data.iloc[period_size:2*period_size],
            self.data.iloc[2*period_size:3*period_size],
            self.data.iloc[3*period_size:]
        ]
        
        for col in self.data.columns:
            if col == 'Load_MW':
                continue
                
            try:
                # Calculate statistics for each period
                period_stats = []
                for i, period in enumerate(periods):
                    if col in period.columns and not period[col].dropna().empty:
                        stats_dict = {
                            'mean': period[col].mean(),
                            'std': period[col].std(),
                            'min': period[col].min(),
                            'max': period[col].max()
                        }
                        period_stats.append(stats_dict)
                
                if len(period_stats) < 2:
                    continue
                
                # Calculate stability metrics
                means = [s['mean'] for s in period_stats]
                stds = [s['std'] for s in period_stats]
                
                # Coefficient of variation across periods
                mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
                std_cv = np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else 0
                
                # Stability score (lower CV = higher stability)
                stability_score = max(0, 1 - max(mean_cv, std_cv))
                
                # Stability classification
                if stability_score > 0.9:
                    stability_class = 'highly_stable'
                elif stability_score > 0.7:
                    stability_class = 'stable'
                elif stability_score > 0.5:
                    stability_class = 'moderately_stable'
                else:
                    stability_class = 'unstable'
                
                stability_results[col] = {
                    'mean_cv': float(mean_cv),
                    'std_cv': float(std_cv),
                    'stability_score': float(stability_score),
                    'stability_class': stability_class,
                    'period_means': [float(m) for m in means],
                    'period_stds': [float(s) for s in stds]
                }
                
            except Exception as e:
                print(f"⚠️ Error in stability analysis for {col}: {e}")
                stability_results[col] = {'error': str(e), 'stability_score': 0}
        
        # Summary analysis
        unstable_features = [
            col for col, results in stability_results.items()
            if results.get('stability_class') == 'unstable'
        ]
        
        highly_stable_features = [
            col for col, results in stability_results.items()
            if results.get('stability_class') == 'highly_stable'
        ]
        
        avg_stability = np.mean([
            results.get('stability_score', 0) 
            for results in stability_results.values()
        ])
        
        self.results['stability_assessment'] = {
            'feature_stability': stability_results,
            'unstable_features': unstable_features,
            'highly_stable_features': highly_stable_features,
            'average_stability_score': float(avg_stability),
            'total_unstable': len(unstable_features),
            'total_highly_stable': len(highly_stable_features)
        }
        
        print(f"⏰ Feature Stability Results:")
        print(f"   📈 Average Stability Score: {avg_stability:.3f}")
        print(f"   🔴 Unstable Features: {len(unstable_features)}")
        print(f"   ✅ Highly Stable Features: {len(highly_stable_features)}")
    
    def calculate_overall_quality_score(self):
        """
        Calculate comprehensive quality score and generate recommendations.
        """
        print("\n🎯 Calculating Overall Quality Score")
        print("=" * 50)
        
        # Weight different validation components
        weights = {
            'statistical_quality': 0.25,
            'outlier_severity': 0.15,
            'distribution_quality': 0.20,
            'business_logic': 0.20,
            'completeness': 0.15,
            'stability': 0.05
        }
        
        # Extract component scores
        stat_score = self.results['statistical_quality'].get('average_quality_score', 0)
        
        # Outlier score (inverse of outlier severity)
        high_outliers = self.results['outlier_analysis'].get('total_high_outlier_features', 0)
        total_features = self.results['total_features']
        outlier_score = max(0, 1 - (high_outliers / total_features))
        
        # Distribution score
        problematic_dist = self.results['distribution_validation'].get('total_problematic', 0)
        dist_score = max(0, 1 - (problematic_dist / total_features))
        
        # Business logic score
        business_violations = self.results['business_logic_validation'].get('total_violations', 0)
        business_score = max(0, 1 - (business_violations / total_features))
        
        # Completeness score
        completeness_score = self.results['completeness_analysis'].get('overall_completeness_percentage', 0) / 100
        
        # Stability score
        stability_score = self.results['stability_assessment'].get('average_stability_score', 0)
        
        # Calculate weighted overall score
        overall_score = (
            weights['statistical_quality'] * stat_score +
            weights['outlier_severity'] * outlier_score +
            weights['distribution_quality'] * dist_score +
            weights['business_logic'] * business_score +
            weights['completeness'] * completeness_score +
            weights['stability'] * stability_score
        )
        
        self.results['overall_quality_score'] = float(overall_score)
        
        # Generate recommendations
        recommendations = []
        
        if stat_score < 0.7:
            recommendations.append("Review features with poor statistical properties - consider transformation or removal")
        
        if outlier_score < 0.8:
            recommendations.append("Investigate and handle outliers in high-severity features")
        
        if dist_score < 0.7:
            recommendations.append("Address distribution issues in problematic features")
        
        if business_score < 0.8:
            recommendations.append("Review business logic violations - ensure features make domain sense")
        
        if completeness_score < 0.95:
            recommendations.append("Handle missing values in incomplete features")
        
        if stability_score < 0.7:
            recommendations.append("Investigate temporal instability in features")
        
        if overall_score > 0.9:
            recommendations.append("✅ EXCELLENT: Dataset ready for modeling with minimal preprocessing")
        elif overall_score > 0.8:
            recommendations.append("✅ GOOD: Dataset ready for modeling with minor adjustments")
        elif overall_score > 0.7:
            recommendations.append("⚠️ MODERATE: Address key issues before modeling")
        else:
            recommendations.append("🔴 POOR: Significant data quality issues require attention")
        
        self.results['recommendations'] = recommendations
        
        print(f"🎯 Overall Quality Assessment:")
        print(f"   📊 Overall Quality Score: {overall_score:.3f}")
        print(f"   📈 Component Scores:")
        print(f"      Statistical: {stat_score:.3f}")
        print(f"      Outliers: {outlier_score:.3f}")
        print(f"      Distributions: {dist_score:.3f}")
        print(f"      Business Logic: {business_score:.3f}")
        print(f"      Completeness: {completeness_score:.3f}")
        print(f"      Stability: {stability_score:.3f}")
    
    def save_results(self):
        """Save comprehensive validation results"""
        # Ensure output directory exists
        output_dir = "phase_2_5_3_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"feature_quality_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {output_path}")
        return output_path
    
    def generate_summary_report(self):
        """Generate executive summary of validation results"""
        print("\n" + "="*70)
        print("🏆 PHASE 2.5.4: FEATURE QUALITY VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total Features: {self.results['total_features']}")
        print(f"   Total Samples: {self.results['total_samples']}")
        print(f"   Overall Quality Score: {self.results['overall_quality_score']:.3f}")
        
        print(f"\n🔍 Key Findings:")
        print(f"   Statistical Issues: {self.results['statistical_quality']['total_problematic']}")
        print(f"   High Outlier Features: {self.results['outlier_analysis']['total_high_outlier_features']}")
        print(f"   Distribution Problems: {self.results['distribution_validation']['total_problematic']}")
        print(f"   Business Logic Violations: {self.results['business_logic_validation']['total_violations']}")
        print(f"   Completeness: {self.results['completeness_analysis']['overall_completeness_percentage']:.1f}%")
        print(f"   Unstable Features: {self.results['stability_assessment']['total_unstable']}")
        
        print(f"\n📋 Recommendations:")
        for i, rec in enumerate(self.results['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n✅ Validation Complete - Ready for Phase 3 Modeling!")

def main():
    """
    Execute comprehensive feature quality validation workflow.
    """
    print("🚀 Starting Phase 2.5.4: Feature Quality Validation")
    print("=" * 60)
    
    # Initialize validator
    data_path = "../dataset/delhi_interaction_enhanced.csv"
    validator = FeatureQualityValidator(data_path)
    
    # Execute validation workflow
    try:
        # Step 1: Load data
        if not validator.load_data():
            return False
        
        # Step 2: Statistical quality assessment
        validator.validate_statistical_quality()
        
        # Step 3: Outlier detection
        validator.detect_outliers()
        
        # Step 4: Distribution validation
        validator.validate_feature_distributions()
        
        # Step 5: Business logic validation
        validator.validate_business_logic()
        
        # Step 6: Data completeness analysis
        validator.analyze_data_completeness()
        
        # Step 7: Feature stability assessment
        validator.assess_feature_stability()
        
        # Step 8: Calculate overall quality score
        validator.calculate_overall_quality_score()
        
        # Step 9: Save results
        report_path = validator.save_results()
        
        # Step 10: Generate summary
        validator.generate_summary_report()
        
        print(f"\n🎉 Phase 2.5.4 Complete!")
        print(f"📄 Detailed report: {report_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in validation workflow: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Feature quality validation completed successfully!")
        print("🚀 Ready to proceed to Phase 3: Baseline Model Implementation")
    else:
        print("\n❌ Feature quality validation failed!")
        print("🔧 Please check errors and retry")
