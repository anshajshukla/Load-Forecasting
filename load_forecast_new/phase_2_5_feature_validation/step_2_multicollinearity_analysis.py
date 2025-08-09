"""
STEP 2.5.2: MULTICOLLINEARITY ANALYSIS
=====================================
Analysis Tasks:
- Calculate VIF (Variance Inflation Factor) for all 267 features
- Generate comprehensive correlation heatmap visualization
- Identify highly correlated feature groups (>0.95 correlation)
- Prioritize features within correlated groups based on domain knowledge

Duration: Day 2 Morning (3-4 hours)
Priority: CRITICAL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import json
import warnings
from typing import List, Dict, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

class MulticollinearityAnalyzer:
    """
    Comprehensive multicollinearity analysis for Delhi Load Forecasting features
    """
    
    def __init__(self, data_path: str, target_column: str = 'delhi_load_mw'):
        """
        Initialize multicollinearity analyzer
        
        Args:
            data_path: Path to feature dataset
            target_column: Name of target variable
        """
        self.data_path = data_path
        self.target_column = target_column
        self.vif_threshold = 5.0
        self.correlation_threshold = 0.95
        self.results_dir = Path("results/multicollinearity_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Delhi-specific feature priorities for selection
        self.feature_priorities = {
            'dual_peak': ['dual_peak', 'peak_', 'morning', 'evening'],
            'weather': ['temp', 'humidity', 'wind', 'weather'],
            'temporal': ['hour', 'day', 'month', 'season', 'weekend'],
            'festivals': ['festival', 'holiday', 'diwali', 'holi'],
            'thermal': ['thermal', 'comfort', 'heat'],
            'load_components': ['BRPL', 'BYPL', 'NDPL', 'DTL', 'MES']
        }
        
        self.analysis_results = {
            'vif_analysis': {},
            'correlation_analysis': {},
            'feature_groups': {},
            'recommended_features': [],
            'removed_features': [],
            'summary_stats': {}
        }
    
    def load_data(self) -> pd.DataFrame:
        """Load dataset with proper handling"""
        try:
            print("Loading dataset for multicollinearity analysis...")
            
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
                    raise ValueError(f"No suitable target column found")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def calculate_vif_scores(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Variance Inflation Factor for all features
        """
        print(f"\n{'='*60}")
        print("STEP 1: VIF (VARIANCE INFLATION FACTOR) ANALYSIS")
        print(f"{'='*60}")
        
        # Prepare features for VIF calculation
        feature_columns = [col for col in df.columns if col != self.target_column]
        
        # Remove non-numeric columns
        numeric_features = []
        for col in feature_columns:
            if df[col].dtype in ['int64', 'float64']:
                if not df[col].isnull().all() and df[col].var() > 0:
                    numeric_features.append(col)
        
        print(f"Calculating VIF for {len(numeric_features)} numeric features...")
        
        # Prepare clean dataset for VIF
        vif_df = df[numeric_features].dropna()
        
        if vif_df.empty:
            print("Warning: No clean data available for VIF calculation")
            return {}
        
        # Standardize features to avoid scale issues
        scaler = StandardScaler()
        vif_data_scaled = scaler.fit_transform(vif_df)
        vif_df_scaled = pd.DataFrame(vif_data_scaled, columns=numeric_features)
        
        # Calculate VIF scores
        vif_scores = {}
        high_vif_features = []
        
        print("Computing VIF scores...")
        for i, feature in enumerate(numeric_features):
            try:
                vif_score = variance_inflation_factor(vif_df_scaled.values, i)
                vif_scores[feature] = vif_score
                
                if vif_score > self.vif_threshold:
                    high_vif_features.append((feature, vif_score))
                    
            except Exception as e:
                print(f"Error calculating VIF for {feature}: {e}")
                vif_scores[feature] = np.inf
        
        # Sort by VIF score
        vif_sorted = sorted(vif_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nVIF Analysis Results:")
        print(f"Features with VIF > {self.vif_threshold}: {len(high_vif_features)}")
        
        # Display top high VIF features
        if high_vif_features:
            print("\nTOP HIGH VIF FEATURES:")
            high_vif_sorted = sorted(high_vif_features, key=lambda x: x[1], reverse=True)
            for feature, vif in high_vif_sorted[:15]:
                print(f"  {feature}: {vif:.2f}")
        
        # Create VIF visualization
        self._plot_vif_analysis(vif_sorted)
        
        self.analysis_results['vif_analysis'] = {
            'vif_scores': vif_scores,
            'high_vif_features': high_vif_features,
            'vif_threshold': self.vif_threshold,
            'stats': {
                'mean_vif': np.mean(list(vif_scores.values())),
                'median_vif': np.median(list(vif_scores.values())),
                'max_vif': max(vif_scores.values()) if vif_scores else 0,
                'features_above_threshold': len(high_vif_features)
            }
        }
        
        return self.analysis_results['vif_analysis']
    
    def analyze_correlation_structure(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive correlation analysis and visualization
        """
        print(f"\n{'='*60}")
        print("STEP 2: CORRELATION STRUCTURE ANALYSIS")
        print(f"{'='*60}")
        
        # Get numeric features
        numeric_features = [col for col in df.columns 
                          if df[col].dtype in ['int64', 'float64'] and 
                          not df[col].isnull().all() and df[col].var() > 0]
        
        print(f"Analyzing correlation structure for {len(numeric_features)} features...")
        
        # Calculate correlation matrix
        corr_df = df[numeric_features]
        correlation_matrix = corr_df.corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        corr_upper = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        for col in corr_upper.columns:
            for idx in corr_upper.index:
                corr_val = corr_upper.loc[idx, col]
                if pd.notna(corr_val) and abs(corr_val) > self.correlation_threshold:
                    high_corr_pairs.append((idx, col, corr_val))
        
        print(f"Found {len(high_corr_pairs)} feature pairs with correlation > {self.correlation_threshold}")
        
        # Create feature groups based on correlation clustering
        feature_groups = self._create_correlation_clusters(correlation_matrix)
        
        # Generate comprehensive correlation visualizations
        self._create_correlation_heatmap(correlation_matrix)
        self._create_correlation_network(high_corr_pairs)
        
        # Analyze correlation with target
        target_correlations = correlation_matrix[self.target_column].abs().sort_values(ascending=False)
        
        correlation_analysis = {
            'correlation_matrix': correlation_matrix,
            'high_corr_pairs': high_corr_pairs,
            'feature_groups': feature_groups,
            'target_correlations': target_correlations.to_dict(),
            'correlation_stats': {
                'mean_abs_correlation': np.mean(np.abs(correlation_matrix.values)),
                'max_correlation': np.max(np.abs(corr_upper.values)),
                'pairs_above_threshold': len(high_corr_pairs)
            }
        }
        
        self.analysis_results['correlation_analysis'] = correlation_analysis
        
        # Display results
        if high_corr_pairs:
            print("\nHIGH CORRELATION PAIRS:")
            high_corr_sorted = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)
            for feat1, feat2, corr in high_corr_sorted[:10]:
                print(f"  {feat1} <-> {feat2}: {corr:.4f}")
        
        return correlation_analysis
    
    def _create_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> Dict:
        """Create feature clusters based on correlation"""
        
        # Convert correlation to distance
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        # Extract clusters
        cluster_labels = fcluster(linkage_matrix, t=0.3, criterion='distance')
        
        # Group features by cluster
        feature_groups = {}
        for i, feature in enumerate(correlation_matrix.columns):
            cluster_id = cluster_labels[i]
            if cluster_id not in feature_groups:
                feature_groups[cluster_id] = []
            feature_groups[cluster_id].append(feature)
        
        # Filter out single-feature clusters for multicollinearity focus
        multicollinear_groups = {k: v for k, v in feature_groups.items() if len(v) > 1}
        
        print(f"Identified {len(multicollinear_groups)} multicollinear feature groups")
        
        return multicollinear_groups
    
    def prioritize_features_in_groups(self, feature_groups: Dict, df: pd.DataFrame) -> Dict:
        """
        Prioritize features within correlated groups based on domain knowledge
        """
        print(f"\n{'='*60}")
        print("STEP 3: FEATURE PRIORITIZATION WITHIN GROUPS")
        print(f"{'='*60}")
        
        prioritization_results = {}
        
        for group_id, features in feature_groups.items():
            if len(features) <= 1:
                continue
                
            print(f"\nAnalyzing Group {group_id} ({len(features)} features):")
            
            # Calculate feature importance metrics
            feature_scores = {}
            
            for feature in features:
                score = 0
                
                # Domain knowledge priority
                domain_score = self._calculate_domain_priority(feature)
                
                # Statistical importance
                target_corr = abs(df[feature].corr(df[self.target_column]))
                variance = df[feature].var()
                non_null_ratio = (1 - df[feature].isnull().sum() / len(df))
                
                # Combined score
                score = (domain_score * 0.4 + 
                        target_corr * 0.3 + 
                        (variance / (variance + 1)) * 0.2 + 
                        non_null_ratio * 0.1)
                
                feature_scores[feature] = {
                    'total_score': score,
                    'domain_score': domain_score,
                    'target_correlation': target_corr,
                    'variance': variance,
                    'completeness': non_null_ratio
                }
            
            # Sort features by score
            sorted_features = sorted(feature_scores.items(), 
                                   key=lambda x: x[1]['total_score'], reverse=True)
            
            # Select top feature(s) from group
            selected_features = [sorted_features[0][0]]  # Select top feature
            removed_features = [f[0] for f in sorted_features[1:]]
            
            prioritization_results[group_id] = {
                'all_features': features,
                'feature_scores': feature_scores,
                'selected_features': selected_features,
                'removed_features': removed_features
            }
            
            print(f"  Selected: {selected_features[0]} (score: {sorted_features[0][1]['total_score']:.3f})")
            print(f"  Removed: {len(removed_features)} features")
        
        return prioritization_results
    
    def _calculate_domain_priority(self, feature_name: str) -> float:
        """Calculate domain knowledge priority score"""
        feature_lower = feature_name.lower()
        score = 0.5  # Base score
        
        # Delhi-specific priorities
        for category, keywords in self.feature_priorities.items():
            for keyword in keywords:
                if keyword.lower() in feature_lower:
                    if category == 'dual_peak':
                        score += 0.4  # Highest priority for Delhi
                    elif category in ['weather', 'temporal']:
                        score += 0.3
                    elif category == 'festivals':
                        score += 0.25  # Important for Delhi
                    elif category == 'thermal':
                        score += 0.2
                    else:
                        score += 0.1
                    break
        
        return min(score, 1.0)  # Cap at 1.0
    
    def select_optimal_features(self, df: pd.DataFrame) -> Dict:
        """
        Select optimal feature set using combined VIF and correlation analysis
        """
        print(f"\n{'='*60}")
        print("STEP 4: OPTIMAL FEATURE SELECTION")
        print(f"{'='*60}")
        
        # Get all features
        all_features = [col for col in df.columns if col != self.target_column]
        selected_features = all_features.copy()
        removed_features = []
        
        # Remove high VIF features iteratively
        vif_analysis = self.analysis_results.get('vif_analysis', {})
        high_vif_features = vif_analysis.get('high_vif_features', [])
        
        if high_vif_features:
            print(f"Removing features with VIF > {self.vif_threshold}...")
            
            # Sort by VIF and remove highest first
            high_vif_sorted = sorted(high_vif_features, key=lambda x: x[1], reverse=True)
            
            for feature, vif_score in high_vif_sorted:
                if feature in selected_features:
                    # Check if feature is high priority
                    domain_score = self._calculate_domain_priority(feature)
                    
                    if domain_score < 0.8:  # Remove if not critical
                        selected_features.remove(feature)
                        removed_features.append({
                            'feature': feature,
                            'reason': 'HIGH_VIF',
                            'vif_score': vif_score,
                            'domain_score': domain_score
                        })
        
        # Apply correlation-based removal
        correlation_analysis = self.analysis_results.get('correlation_analysis', {})
        high_corr_pairs = correlation_analysis.get('high_corr_pairs', [])
        
        if high_corr_pairs:
            print(f"Resolving {len(high_corr_pairs)} high correlation pairs...")
            
            for feat1, feat2, corr in high_corr_pairs:
                if feat1 in selected_features and feat2 in selected_features:
                    # Keep the feature with higher domain priority
                    score1 = self._calculate_domain_priority(feat1)
                    score2 = self._calculate_domain_priority(feat2)
                    
                    # Also consider target correlation
                    target_corr1 = abs(df[feat1].corr(df[self.target_column]))
                    target_corr2 = abs(df[feat2].corr(df[self.target_column]))
                    
                    combined_score1 = score1 * 0.6 + target_corr1 * 0.4
                    combined_score2 = score2 * 0.6 + target_corr2 * 0.4
                    
                    if combined_score1 >= combined_score2:
                        remove_feature = feat2
                        keep_feature = feat1
                    else:
                        remove_feature = feat1
                        keep_feature = feat2
                    
                    if remove_feature in selected_features:
                        selected_features.remove(remove_feature)
                        removed_features.append({
                            'feature': remove_feature,
                            'reason': 'HIGH_CORRELATION',
                            'correlation_with': keep_feature,
                            'correlation_value': corr
                        })
        
        # Final optimization to target range (100-120 features)
        target_feature_count = 110  # Target middle of range
        
        if len(selected_features) > 120:
            print(f"Further reducing from {len(selected_features)} to ~{target_feature_count} features...")
            
            # Score remaining features for final selection
            feature_final_scores = {}
            for feature in selected_features:
                domain_score = self._calculate_domain_priority(feature)
                target_corr = abs(df[feature].corr(df[self.target_column]))
                
                # Get VIF if available
                vif_scores = vif_analysis.get('vif_scores', {})
                vif_penalty = 1.0 / (1.0 + vif_scores.get(feature, 1.0))
                
                final_score = domain_score * 0.5 + target_corr * 0.3 + vif_penalty * 0.2
                feature_final_scores[feature] = final_score
            
            # Select top features
            sorted_final = sorted(feature_final_scores.items(), key=lambda x: x[1], reverse=True)
            final_selected = [f[0] for f in sorted_final[:target_feature_count]]
            
            # Add remaining to removed list
            for feature, score in sorted_final[target_feature_count:]:
                removed_features.append({
                    'feature': feature,
                    'reason': 'FINAL_OPTIMIZATION',
                    'final_score': score
                })
            
            selected_features = final_selected
        
        selection_results = {
            'selected_features': selected_features,
            'removed_features': removed_features,
            'selection_stats': {
                'original_count': len(all_features),
                'final_count': len(selected_features),
                'removed_count': len(removed_features),
                'reduction_percentage': (len(removed_features) / len(all_features)) * 100
            }
        }
        
        print(f"\nFEATURE SELECTION RESULTS:")
        print(f"Original features: {len(all_features)}")
        print(f"Selected features: {len(selected_features)}")
        print(f"Removed features: {len(removed_features)}")
        print(f"Reduction: {selection_results['selection_stats']['reduction_percentage']:.1f}%")
        
        self.analysis_results['feature_selection'] = selection_results
        
        return selection_results
    
    def _plot_vif_analysis(self, vif_sorted: List[Tuple]):
        """Create VIF analysis visualizations"""
        
        if not vif_sorted:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # VIF distribution
        vif_values = [score for _, score in vif_sorted if not np.isinf(score)]
        
        axes[0, 0].hist(vif_values, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=self.vif_threshold, color='red', linestyle='--', 
                          label=f'Threshold ({self.vif_threshold})')
        axes[0, 0].set_xlabel('VIF Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('VIF Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top VIF features
        top_vif = vif_sorted[:20]  # Top 20
        features = [f[:25] for f, _ in top_vif]  # Truncate names
        scores = [s for _, s in top_vif if not np.isinf(s)]
        features = features[:len(scores)]  # Match lengths
        
        axes[0, 1].barh(range(len(features)), scores)
        axes[0, 1].set_yticks(range(len(features)))
        axes[0, 1].set_yticklabels(features)
        axes[0, 1].set_xlabel('VIF Score')
        axes[0, 1].set_title('Top 20 VIF Scores')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=self.vif_threshold, color='red', linestyle='--')
        
        # VIF vs Index
        axes[1, 0].plot([s for _, s in vif_sorted if not np.isinf(s)], 'o-', alpha=0.6)
        axes[1, 0].axhline(y=self.vif_threshold, color='red', linestyle='--')
        axes[1, 0].set_xlabel('Feature Index (sorted by VIF)')
        axes[1, 0].set_ylabel('VIF Score')
        axes[1, 0].set_title('VIF Scores by Rank')
        axes[1, 0].grid(True, alpha=0.3)
        
        # High VIF count by threshold
        thresholds = [2, 5, 10, 15, 20, 30]
        high_counts = [sum(1 for _, s in vif_sorted if s > t and not np.isinf(s)) for t in thresholds]
        
        axes[1, 1].bar(range(len(thresholds)), high_counts)
        axes[1, 1].set_xticks(range(len(thresholds)))
        axes[1, 1].set_xticklabels([f'>{t}' for t in thresholds])
        axes[1, 1].set_xlabel('VIF Threshold')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Features Above VIF Thresholds')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'vif_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_heatmap(self, correlation_matrix: pd.DataFrame):
        """Create comprehensive correlation heatmap"""
        
        # Full correlation matrix (sample if too large)
        if correlation_matrix.shape[0] > 50:
            # Sample top features by target correlation
            target_corrs = correlation_matrix[self.target_column].abs().sort_values(ascending=False)
            top_features = target_corrs.head(50).index.tolist()
            corr_sample = correlation_matrix.loc[top_features, top_features]
        else:
            corr_sample = correlation_matrix
        
        plt.figure(figsize=(12, 10))
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_sample, dtype=bool))
        
        # Generate heatmap
        sns.heatmap(corr_sample, mask=mask, annot=False, cmap='RdBu_r', center=0,
                    square=True, linewidths=0.1, cbar_kws={"shrink": 0.8})
        
        plt.title(f'Feature Correlation Heatmap (Top {len(corr_sample)} Features)')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_network(self, high_corr_pairs: List[Tuple]):
        """Create network visualization of high correlations"""
        
        if not high_corr_pairs:
            return
            
        # For simplicity, create a basic visualization
        plt.figure(figsize=(12, 8))
        
        # Extract unique features
        all_features = set()
        for feat1, feat2, _ in high_corr_pairs:
            all_features.add(feat1)
            all_features.add(feat2)
        
        feature_list = list(all_features)
        
        # Create adjacency matrix
        n_features = len(feature_list)
        adj_matrix = np.zeros((n_features, n_features))
        
        for feat1, feat2, corr in high_corr_pairs:
            i = feature_list.index(feat1)
            j = feature_list.index(feat2)
            adj_matrix[i, j] = abs(corr)
            adj_matrix[j, i] = abs(corr)
        
        # Plot as heatmap
        sns.heatmap(adj_matrix, xticklabels=[f[:15] for f in feature_list], 
                    yticklabels=[f[:15] for f in feature_list],
                    annot=False, cmap='Reds', square=True)
        
        plt.title(f'High Correlation Network ({len(high_corr_pairs)} pairs)')
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_network.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_analysis_report(self) -> Dict:
        """Generate comprehensive multicollinearity analysis report"""
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'parameters': {
                'vif_threshold': self.vif_threshold,
                'correlation_threshold': self.correlation_threshold,
                'target_feature_range': '100-120'
            },
            'results': self.analysis_results,
            'recommendations': []
        }
        
        # Generate recommendations
        vif_analysis = self.analysis_results.get('vif_analysis', {})
        if vif_analysis.get('features_above_threshold', 0) > 0:
            report['recommendations'].append(
                f"Address {vif_analysis['features_above_threshold']} features with VIF > {self.vif_threshold}"
            )
        
        correlation_analysis = self.analysis_results.get('correlation_analysis', {})
        if correlation_analysis.get('correlation_stats', {}).get('pairs_above_threshold', 0) > 0:
            pairs_count = correlation_analysis['correlation_stats']['pairs_above_threshold']
            report['recommendations'].append(
                f"Resolve {pairs_count} highly correlated feature pairs (>{self.correlation_threshold})"
            )
        
        selection_results = self.analysis_results.get('feature_selection', {})
        if selection_results:
            final_count = selection_results['selection_stats']['final_count']
            if 100 <= final_count <= 120:
                report['recommendations'].append("✅ Feature count within optimal range (100-120)")
            else:
                report['recommendations'].append(f"⚠️ Feature count ({final_count}) outside optimal range")
        
        # Save report
        with open(self.results_dir / 'multicollinearity_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def run_complete_analysis(self, data_path: str = None) -> Dict:
        """
        Run complete multicollinearity analysis
        """
        print("="*80)
        print("MULTICOLLINEARITY ANALYSIS - DELHI LOAD FORECASTING")
        print("="*80)
        print(f"Analysis started at: {datetime.now()}")
        
        if data_path:
            self.data_path = data_path
        
        # Load data
        df = self.load_data()
        
        # Run analysis steps
        
        # Step 1: VIF Analysis
        vif_results = self.calculate_vif_scores(df)
        
        # Step 2: Correlation Analysis
        correlation_results = self.analyze_correlation_structure(df)
        
        # Step 3: Feature Group Prioritization
        feature_groups = correlation_results.get('feature_groups', {})
        if feature_groups:
            prioritization_results = self.prioritize_features_in_groups(feature_groups, df)
            self.analysis_results['feature_prioritization'] = prioritization_results
        
        # Step 4: Final Feature Selection
        selection_results = self.select_optimal_features(df)
        
        # Step 5: Generate Report
        final_report = self.generate_analysis_report()
        
        print(f"\n{'='*80}")
        print("MULTICOLLINEARITY ANALYSIS COMPLETED")
        print(f"{'='*80}")
        print(f"Analysis completed at: {datetime.now()}")
        print(f"Results saved to: {self.results_dir}")
        
        return {
            'vif_analysis': vif_results,
            'correlation_analysis': correlation_results,
            'feature_selection': selection_results,
            'final_report': final_report
        }

def main():
    """Main execution function"""
    
    # Configuration
    DATA_PATH = "../data/delhi_features_267.parquet"  # Adjust path as needed
    TARGET_COLUMN = "delhi_load_mw"
    
    print("Starting Multicollinearity Analysis...")
    
    # Initialize analyzer
    analyzer = MulticollinearityAnalyzer(DATA_PATH, TARGET_COLUMN)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Print summary
    print("\n" + "="*80)
    print("MULTICOLLINEARITY ANALYSIS SUMMARY")
    print("="*80)
    
    vif_stats = results['vif_analysis']['stats']
    corr_stats = results['correlation_analysis']['correlation_stats']
    selection_stats = results['feature_selection']['selection_stats']
    
    print(f"VIF Analysis:")
    print(f"  Mean VIF: {vif_stats['mean_vif']:.2f}")
    print(f"  Features with VIF > {analyzer.vif_threshold}: {vif_stats['features_above_threshold']}")
    
    print(f"\nCorrelation Analysis:")
    print(f"  High correlation pairs: {corr_stats['pairs_above_threshold']}")
    print(f"  Max correlation: {corr_stats['max_correlation']:.4f}")
    
    print(f"\nFeature Selection:")
    print(f"  Original features: {selection_stats['original_count']}")
    print(f"  Selected features: {selection_stats['final_count']}")
    print(f"  Reduction: {selection_stats['reduction_percentage']:.1f}%")
    
    final_count = selection_stats['final_count']
    if 100 <= final_count <= 120:
        print(f"\n✅ OPTIMAL FEATURE COUNT ACHIEVED ({final_count})")
    else:
        print(f"\n⚠️ FEATURE COUNT ({final_count}) OUTSIDE TARGET RANGE (100-120)")
    
    print(f"\nDetailed results saved to: {analyzer.results_dir}")

if __name__ == "__main__":
    main()
