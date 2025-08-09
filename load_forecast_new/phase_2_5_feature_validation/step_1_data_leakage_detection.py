"""
STEP 2.5.1: DATA LEAKAGE DETECTION
==================================
Critical Tasks:
- Correlation analysis (target vs all 267 features)
- Flag features with correlation >0.95 (likely leakage)
- Investigate load component relationships (BRPL/BYPL/NDPL)
- Temporal leakage validation (lag features, rolling windows)
- Remove/fix leakage features immediately

Duration: Day 1 (4-6 hours)
Priority: CRITICAL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple

warnings.filterwarnings('ignore')

class DataLeakageDetector:
    """
    Comprehensive data leakage detection system for Delhi Load Forecasting
    """
    
    def __init__(self, data_path: str, target_column: str = 'delhi_load_mw'):
        """
        Initialize the data leakage detector
        
        Args:
            data_path: Path to the feature dataset
            target_column: Name of the target variable
        """
        self.data_path = data_path
        self.target_column = target_column
        self.correlation_threshold = 0.95
        self.results_dir = Path("results/leakage_detection")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Delhi-specific load components that need special attention
        self.load_components = ['BRPL', 'BYPL', 'NDPL', 'DTL', 'MES']
        
        self.leakage_report = {
            'high_correlation_features': [],
            'temporal_leakage_features': [],
            'load_component_issues': [],
            'suspicious_patterns': [],
            'recommendations': []
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset with proper datetime handling"""
        try:
            print("Loading dataset for leakage detection...")
            
            # Try different file formats
            if self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                # Try both formats
                try:
                    df = pd.read_parquet(self.data_path)
                except:
                    df = pd.read_csv(self.data_path)
            
            print(f"Dataset loaded: {df.shape}")
            print(f"Features available: {df.columns.tolist()[:10]}...")
            
            # Ensure datetime index if available
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime').sort_index()
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def detect_high_correlation_leakage(self, df: pd.DataFrame) -> Dict:
        """
        Detect features with suspiciously high correlation with target variable
        """
        print(f"\n{'='*60}")
        print("STEP 1: HIGH CORRELATION LEAKAGE DETECTION")
        print(f"{'='*60}")
        
        # Calculate correlations with target
        if self.target_column not in df.columns:
            available_targets = [col for col in df.columns if 'load' in col.lower()]
            if available_targets:
                self.target_column = available_targets[0]
                print(f"Target column not found, using: {self.target_column}")
            else:
                raise ValueError(f"Target column '{self.target_column}' not found")
        
        correlations = df.corr()[self.target_column].abs().sort_values(ascending=False)
        
        # Identify high correlation features (excluding target itself)
        high_corr_features = correlations[
            (correlations > self.correlation_threshold) & 
            (correlations.index != self.target_column)
        ]
        
        print(f"Found {len(high_corr_features)} features with correlation > {self.correlation_threshold}")
        
        if len(high_corr_features) > 0:
            print("\nSUSPICIOUS HIGH CORRELATION FEATURES:")
            for feature, corr in high_corr_features.items():
                print(f"  {feature}: {corr:.4f}")
                self.leakage_report['high_correlation_features'].append({
                    'feature': feature,
                    'correlation': corr,
                    'risk_level': 'CRITICAL' if corr > 0.98 else 'HIGH'
                })
        
        # Create correlation visualization
        self._plot_correlation_distribution(correlations)
        
        return {
            'high_correlation_features': high_corr_features.to_dict(),
            'correlation_stats': {
                'mean': correlations.mean(),
                'std': correlations.std(),
                'max': correlations.max(),
                'features_above_threshold': len(high_corr_features)
            }
        }
    
    def detect_load_component_leakage(self, df: pd.DataFrame) -> Dict:
        """
        Investigate load component relationships for potential leakage
        """
        print(f"\n{'='*60}")
        print("STEP 2: LOAD COMPONENT LEAKAGE ANALYSIS")
        print(f"{'='*60}")
        
        # Find load component features
        load_features = []
        for component in self.load_components:
            component_features = [col for col in df.columns if component.lower() in col.lower()]
            load_features.extend(component_features)
        
        if not load_features:
            print("No specific load component features found")
            return {}
        
        print(f"Found {len(load_features)} load component features")
        
        # Analyze cross-correlations
        load_df = df[[self.target_column] + load_features]
        load_corr_matrix = load_df.corr()
        
        # Check for suspicious relationships
        suspicious_pairs = []
        for i, feature1 in enumerate(load_features):
            for j, feature2 in enumerate(load_features[i+1:], i+1):
                corr = load_corr_matrix.loc[feature1, feature2]
                if abs(corr) > 0.95:
                    suspicious_pairs.append((feature1, feature2, corr))
        
        # Check relationships with target
        target_relationships = []
        for feature in load_features:
            corr = load_corr_matrix.loc[self.target_column, feature]
            if abs(corr) > 0.90:
                target_relationships.append((feature, corr))
                self.leakage_report['load_component_issues'].append({
                    'feature': feature,
                    'target_correlation': corr,
                    'component': self._identify_component(feature),
                    'risk_level': 'HIGH' if abs(corr) > 0.95 else 'MEDIUM'
                })
        
        print(f"Found {len(suspicious_pairs)} suspicious component pairs")
        print(f"Found {len(target_relationships)} high target correlations")
        
        # Visualize load component relationships
        self._plot_load_component_matrix(load_corr_matrix)
        
        return {
            'suspicious_pairs': suspicious_pairs,
            'target_relationships': target_relationships,
            'component_features': load_features
        }
    
    def detect_temporal_leakage(self, df: pd.DataFrame) -> Dict:
        """
        Detect temporal leakage in lag features and rolling windows
        """
        print(f"\n{'='*60}")
        print("STEP 3: TEMPORAL LEAKAGE DETECTION")
        print(f"{'='*60}")
        
        temporal_issues = []
        
        # Check lag features
        lag_features = [col for col in df.columns if 'lag' in col.lower()]
        rolling_features = [col for col in df.columns if any(term in col.lower() for term in ['roll', 'ma_', 'mean_', 'std_'])]
        future_features = [col for col in df.columns if any(term in col.lower() for term in ['next', 'future', 'ahead'])]
        
        print(f"Found {len(lag_features)} lag features")
        print(f"Found {len(rolling_features)} rolling window features")
        print(f"Found {len(future_features)} potential future-looking features")
        
        # Analyze lag features for proper temporal structure
        for feature in lag_features:
            if feature in df.columns:
                # Extract lag period if possible
                lag_period = self._extract_lag_period(feature)
                
                # Check correlation with target
                corr = df[feature].corr(df[self.target_column])
                
                if abs(corr) > 0.98:
                    temporal_issues.append({
                        'feature': feature,
                        'issue_type': 'LAG_HIGH_CORRELATION',
                        'correlation': corr,
                        'lag_period': lag_period,
                        'risk_level': 'HIGH'
                    })
        
        # Check future-looking features (major red flag)
        for feature in future_features:
            temporal_issues.append({
                'feature': feature,
                'issue_type': 'FUTURE_LOOKING',
                'risk_level': 'CRITICAL',
                'recommendation': 'REMOVE_IMMEDIATELY'
            })
        
        # Validate rolling window features
        for feature in rolling_features:
            # Check if rolling window might include future data
            if self._check_rolling_window_validity(df, feature):
                temporal_issues.append({
                    'feature': feature,
                    'issue_type': 'ROLLING_WINDOW_LOOKAHEAD',
                    'risk_level': 'MEDIUM',
                    'recommendation': 'VERIFY_IMPLEMENTATION'
                })
        
        self.leakage_report['temporal_leakage_features'].extend(temporal_issues)
        
        print(f"Found {len(temporal_issues)} temporal leakage issues")
        
        return {
            'lag_features': lag_features,
            'rolling_features': rolling_features,
            'future_features': future_features,
            'temporal_issues': temporal_issues
        }
    
    def detect_statistical_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detect statistical patterns that might indicate leakage
        """
        print(f"\n{'='*60}")
        print("STEP 4: STATISTICAL ANOMALY DETECTION")
        print(f"{'='*60}")
        
        anomalies = []
        
        # Check for perfect correlations (excluding self-correlation)
        corr_matrix = df.corr()
        perfect_correlations = []
        
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns[i+1:], i+1):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.999 and col1 != col2:
                    perfect_correlations.append((col1, col2, corr))
        
        # Check for features with zero variance
        zero_variance_features = df.columns[df.var() == 0].tolist()
        
        # Check for features with identical distributions
        identical_features = self._find_identical_features(df)
        
        # Check for suspiciously clean relationships
        clean_relationships = self._find_suspiciously_clean_relationships(df)
        
        anomalies = {
            'perfect_correlations': perfect_correlations,
            'zero_variance_features': zero_variance_features,
            'identical_features': identical_features,
            'clean_relationships': clean_relationships
        }
        
        print(f"Found {len(perfect_correlations)} perfect correlations")
        print(f"Found {len(zero_variance_features)} zero variance features")
        print(f"Found {len(identical_features)} identical feature pairs")
        
        return anomalies
    
    def _plot_correlation_distribution(self, correlations: pd.Series):
        """Plot correlation distribution"""
        plt.figure(figsize=(12, 8))
        
        # Main correlation distribution
        plt.subplot(2, 2, 1)
        plt.hist(correlations.values, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=self.correlation_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.correlation_threshold})')
        plt.xlabel('Absolute Correlation with Target')
        plt.ylabel('Frequency')
        plt.title('Feature-Target Correlation Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Top correlations bar plot
        plt.subplot(2, 2, 2)
        top_corrs = correlations.head(15)
        plt.barh(range(len(top_corrs)), top_corrs.values)
        plt.yticks(range(len(top_corrs)), [f"{name[:30]}..." if len(name) > 30 else name 
                                         for name in top_corrs.index])
        plt.xlabel('Correlation')
        plt.title('Top 15 Feature Correlations')
        plt.grid(True, alpha=0.3)
        
        # Correlation vs feature index
        plt.subplot(2, 2, 3)
        plt.plot(correlations.values, 'o-', alpha=0.6)
        plt.axhline(y=self.correlation_threshold, color='red', linestyle='--')
        plt.xlabel('Feature Index')
        plt.ylabel('Correlation')
        plt.title('Correlation by Feature Order')
        plt.grid(True, alpha=0.3)
        
        # High correlation features
        plt.subplot(2, 2, 4)
        high_corr = correlations[correlations > self.correlation_threshold]
        if len(high_corr) > 0:
            plt.bar(range(len(high_corr)), high_corr.values)
            plt.xticks(range(len(high_corr)), [f"{name[:20]}..." if len(name) > 20 else name 
                                              for name in high_corr.index], rotation=45)
            plt.ylabel('Correlation')
            plt.title(f'Features Above {self.correlation_threshold} Threshold')
        else:
            plt.text(0.5, 0.5, 'No features above threshold', 
                    ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_load_component_matrix(self, corr_matrix: pd.DataFrame):
        """Plot load component correlation matrix"""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title('Load Component Cross-Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'load_component_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _identify_component(self, feature_name: str) -> str:
        """Identify which load component a feature belongs to"""
        feature_lower = feature_name.lower()
        for component in self.load_components:
            if component.lower() in feature_lower:
                return component
        return 'UNKNOWN'
    
    def _extract_lag_period(self, feature_name: str) -> int:
        """Extract lag period from feature name"""
        import re
        match = re.search(r'lag_?(\d+)', feature_name.lower())
        return int(match.group(1)) if match else None
    
    def _check_rolling_window_validity(self, df: pd.DataFrame, feature: str) -> bool:
        """Check if rolling window feature might have look-ahead bias"""
        # This is a heuristic check - would need more sophisticated analysis
        # For now, flag features with suspiciously high correlation
        if feature in df.columns and self.target_column in df.columns:
            corr = df[feature].corr(df[self.target_column])
            return abs(corr) > 0.95
        return False
    
    def _find_identical_features(self, df: pd.DataFrame) -> List[Tuple]:
        """Find features that are identical"""
        identical_pairs = []
        
        # Compare features pairwise
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns[i+1:], i+1):
                if col1 != col2:
                    # Check if values are identical (allowing for small floating point differences)
                    if np.allclose(df[col1].fillna(0), df[col2].fillna(0), rtol=1e-10):
                        identical_pairs.append((col1, col2))
        
        return identical_pairs
    
    def _find_suspiciously_clean_relationships(self, df: pd.DataFrame) -> List[Dict]:
        """Find relationships that are suspiciously clean/perfect"""
        clean_relationships = []
        
        # Check for linear relationships that are too perfect
        from scipy.stats import linregress
        
        target_data = df[self.target_column].dropna()
        
        for col in df.columns:
            if col != self.target_column and df[col].dtype in ['int64', 'float64']:
                # Get common non-null indices
                common_idx = target_data.index.intersection(df[col].dropna().index)
                if len(common_idx) > 100:  # Need sufficient data
                    
                    x = df.loc[common_idx, col].values
                    y = target_data.loc[common_idx].values
                    
                    try:
                        slope, intercept, r_value, p_value, std_err = linregress(x, y)
                        
                        # Flag suspiciously perfect relationships
                        if r_value**2 > 0.99 and p_value < 1e-10:
                            clean_relationships.append({
                                'feature': col,
                                'r_squared': r_value**2,
                                'p_value': p_value,
                                'relationship_type': 'PERFECT_LINEAR'
                            })
                    except:
                        continue
        
        return clean_relationships
    
    def generate_leakage_report(self) -> Dict:
        """Generate comprehensive leakage detection report"""
        
        # Add recommendations based on findings
        if self.leakage_report['high_correlation_features']:
            self.leakage_report['recommendations'].append(
                "CRITICAL: Remove or investigate features with correlation >0.95 immediately"
            )
        
        if self.leakage_report['temporal_leakage_features']:
            future_features = [f for f in self.leakage_report['temporal_leakage_features'] 
                             if f.get('issue_type') == 'FUTURE_LOOKING']
            if future_features:
                self.leakage_report['recommendations'].append(
                    f"CRITICAL: Remove {len(future_features)} future-looking features immediately"
                )
        
        if self.leakage_report['load_component_issues']:
            self.leakage_report['recommendations'].append(
                "Investigate load component relationships - may indicate component leakage"
            )
        
        # Save report
        with open(self.results_dir / 'leakage_detection_report.json', 'w') as f:
            json.dump(self.leakage_report, f, indent=2, default=str)
        
        return self.leakage_report
    
    def run_complete_analysis(self, data_path: str = None) -> Dict:
        """
        Run complete data leakage detection analysis
        """
        print("="*80)
        print("DATA LEAKAGE DETECTION - DELHI LOAD FORECASTING")
        print("="*80)
        print(f"Analysis started at: {datetime.now()}")
        
        if data_path:
            self.data_path = data_path
            
        # Load data
        df = self.load_data()
        
        # Run all detection methods
        results = {}
        
        # 1. High correlation detection
        results['correlation_analysis'] = self.detect_high_correlation_leakage(df)
        
        # 2. Load component analysis
        results['load_component_analysis'] = self.detect_load_component_leakage(df)
        
        # 3. Temporal leakage detection
        results['temporal_analysis'] = self.detect_temporal_leakage(df)
        
        # 4. Statistical anomalies
        results['statistical_anomalies'] = self.detect_statistical_anomalies(df)
        
        # 5. Generate comprehensive report
        results['leakage_report'] = self.generate_leakage_report()
        
        print(f"\n{'='*80}")
        print("DATA LEAKAGE DETECTION COMPLETED")
        print(f"{'='*80}")
        print(f"Analysis completed at: {datetime.now()}")
        print(f"Results saved to: {self.results_dir}")
        
        return results

def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = "../data/delhi_features_267.parquet"  # Adjust path as needed
    TARGET_COLUMN = "delhi_load_mw"
    
    print("Starting Data Leakage Detection Analysis...")
    
    # Initialize detector
    detector = DataLeakageDetector(DATA_PATH, TARGET_COLUMN)
    
    # Run complete analysis
    results = detector.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("LEAKAGE DETECTION SUMMARY")
    print("="*80)
    
    high_corr_count = len(results['leakage_report']['high_correlation_features'])
    temporal_issues_count = len(results['leakage_report']['temporal_leakage_features'])
    load_issues_count = len(results['leakage_report']['load_component_issues'])
    
    print(f"High Correlation Features (>{detector.correlation_threshold}): {high_corr_count}")
    print(f"Temporal Leakage Issues: {temporal_issues_count}")
    print(f"Load Component Issues: {load_issues_count}")
    
    if high_corr_count == 0 and temporal_issues_count == 0:
        print("\n✅ NO CRITICAL DATA LEAKAGE DETECTED")
    else:
        print(f"\n⚠️  {high_corr_count + temporal_issues_count} POTENTIAL LEAKAGE ISSUES FOUND")
        print("Review the detailed report and take corrective action immediately!")
    
    print(f"\nDetailed results saved to: {detector.results_dir}")

if __name__ == "__main__":
    main()
