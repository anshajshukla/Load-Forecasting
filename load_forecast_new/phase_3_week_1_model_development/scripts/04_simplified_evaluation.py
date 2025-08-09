"""
Week 1 Final Evaluation & Documentation - Simplified Version
==============================================================================
This script provides a comprehensive evaluation of all Week 1 baseline models
and generates documentation for model selection and Week 2 planning.
"""

import os
import pandas as pd
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Week1FinalEvaluation:
    def __init__(self):
        """Initialize the evaluation pipeline"""
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_dir = os.path.join(self.project_dir, 'results')
        self.output_dir = os.path.join(self.project_dir, 'evaluation')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'reports'), exist_ok=True)
        
        self.all_results = {}
        self.model_summary = {}
        
    def load_all_results(self):
        """Load all available results files"""
        print("🔄 Loading Week 1 results...")
        
        # Load CSV results
        csv_files = [
            'enhanced_baseline_results.csv',
            'advanced_models_results.csv', 
            'super_fast_baseline_results.csv'
        ]
        
        for csv_file in csv_files:
            filepath = os.path.join(self.results_dir, csv_file)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    self.all_results[csv_file.replace('.csv', '')] = df
                    print(f"   ✅ Loaded: {csv_file}")
                except Exception as e:
                    print(f"   ⚠️  Error loading {csv_file}: {e}")
            else:
                print(f"   ⚠️  Not found: {csv_file}")
                
        # Load JSON results
        json_files = [
            'ensemble_results.json',
            'baseline_results.json'
        ]
        
        for json_file in json_files:
            filepath = os.path.join(self.results_dir, json_file)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    self.all_results[json_file.replace('.json', '')] = data
                    print(f"   ✅ Loaded: {json_file}")
                except Exception as e:
                    print(f"   ⚠️  Error loading {json_file}: {e}")
        
        print(f"✅ Loaded {len(self.all_results)} result files")
    
    def create_comprehensive_summary(self):
        """Create comprehensive model performance summary"""
        print("\n🔄 Creating comprehensive model summary...")
        
        all_models = []
        
        # Process CSV results
        for key, df in self.all_results.items():
            if isinstance(df, pd.DataFrame):
                for _, row in df.iterrows():
                    model_info = {
                        'model_name': row.get('model', 'unknown'),
                        'validation_mape': row.get('val_mape', None),
                        'training_mape': row.get('train_mape', None),
                        'validation_rmse': row.get('val_rmse', None),
                        'source': key,
                        'category': self.categorize_model(row.get('model', ''), key)
                    }
                    if model_info['validation_mape'] is not None:
                        all_models.append(model_info)
        
        # Convert to DataFrame for easy analysis
        self.model_summary = pd.DataFrame(all_models)
        
        if not self.model_summary.empty:
            # Sort by validation MAPE
            self.model_summary = self.model_summary.sort_values('validation_mape')
            print(f"✅ Analyzed {len(self.model_summary)} models")
        else:
            print("⚠️  No valid model results found")
        
        return self.model_summary
    
    def categorize_model(self, model_name, source):
        """Categorize models based on name and source"""
        if 'enhanced' in source:
            return 'Enhanced Baseline'
        elif 'advanced' in source:
            return 'Advanced Models'
        elif 'super_fast' in source:
            return 'Initial Baseline'
        else:
            return 'Other'
    
    def analyze_performance(self):
        """Analyze model performance against targets"""
        print("\n🔄 Analyzing performance against targets...")
        
        if self.model_summary.empty:
            print("⚠️  No models to analyze")
            return
        
        # Performance categories
        excellent = self.model_summary[self.model_summary['validation_mape'] <= 5.0]
        good = self.model_summary[(self.model_summary['validation_mape'] > 5.0) & 
                                 (self.model_summary['validation_mape'] <= 7.5)]
        acceptable = self.model_summary[(self.model_summary['validation_mape'] > 7.5) & 
                                       (self.model_summary['validation_mape'] <= 10.0)]
        needs_improvement = self.model_summary[self.model_summary['validation_mape'] > 10.0]
        
        print("📊 Performance Analysis:")
        print(f"   🏆 Excellent (≤5%): {len(excellent)} models")
        print(f"   ✅ Good (5-7.5%): {len(good)} models")
        print(f"   📈 Acceptable (7.5-10%): {len(acceptable)} models")
        print(f"   ⚠️  Needs Improvement (>10%): {len(needs_improvement)} models")
        
        # Best models
        best_models = self.model_summary.head(5)
        print(f"\n🏆 Top 5 Models:")
        for _, model in best_models.iterrows():
            print(f"   {model['model_name']}: {model['validation_mape']:.2f}% MAPE ({model['category']})")
        
        return {
            'excellent': len(excellent),
            'good': len(good),
            'acceptable': len(acceptable),
            'needs_improvement': len(needs_improvement),
            'best_model': best_models.iloc[0] if not best_models.empty else None
        }
    
    def check_week1_criteria(self):
        """Check if Week 1 completion criteria are met"""
        print("\n🎯 Checking Week 1 Completion Criteria...")
        
        if self.model_summary.empty:
            print("❌ No models available for evaluation")
            return False
        
        best_mape = self.model_summary['validation_mape'].min()
        models_under_10 = len(self.model_summary[self.model_summary['validation_mape'] <= 10.0])
        
        print(f"   📊 Best MAPE achieved: {best_mape:.2f}%")
        print(f"   📈 Models under 10% MAPE: {models_under_10}")
        
        if best_mape <= 10.0:
            print("✅ Week 1 COMPLETION CRITERIA MET!")
            print("🎉 Successfully established baseline models with MAPE ≤ 10%")
            return True
        else:
            print("❌ Week 1 completion criteria not met")
            print("💡 Consider further optimization or ensemble methods")
            return False
    
    def generate_final_report(self):
        """Generate final evaluation report"""
        print("\n📝 Generating final evaluation report...")
        
        # Create report content
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'week_1_summary': {
                'total_models_evaluated': len(self.model_summary) if not self.model_summary.empty else 0,
                'best_mape': self.model_summary['validation_mape'].min() if not self.model_summary.empty else None,
                'models_meeting_criteria': len(self.model_summary[self.model_summary['validation_mape'] <= 10.0]) if not self.model_summary.empty else 0
            },
            'model_performance': self.model_summary.to_dict('records') if not self.model_summary.empty else [],
            'recommendations': self.generate_recommendations()
        }
        
        # Save JSON report
        report_path = os.path.join(self.output_dir, 'reports', 'week1_final_evaluation.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save CSV summary
        if not self.model_summary.empty:
            csv_path = os.path.join(self.output_dir, 'reports', 'week1_model_summary.csv')
            self.model_summary.to_csv(csv_path, index=False)
            print(f"✅ Reports saved to: {self.output_dir}/reports/")
        
        return report
    
    def generate_recommendations(self):
        """Generate recommendations for Week 2"""
        recommendations = [
            "Focus on ensemble methods combining best performing models",
            "Implement advanced feature engineering techniques",
            "Explore time series specific models (ARIMA, Prophet)",
            "Consider neural network approaches for complex patterns",
            "Implement cross-validation for robust model selection"
        ]
        
        if not self.model_summary.empty:
            best_category = self.model_summary.iloc[0]['category']
            recommendations.insert(0, f"Build upon {best_category} approach which showed best performance")
        
        return recommendations
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("🚀 Starting Week 1 Final Evaluation")
        print("="*80)
        
        try:
            # Load all results
            self.load_all_results()
            
            # Create summary
            self.create_comprehensive_summary()
            
            # Analyze performance
            performance_analysis = self.analyze_performance()
            
            # Check completion criteria
            criteria_met = self.check_week1_criteria()
            
            # Generate report
            report = self.generate_final_report()
            
            print("\n" + "="*80)
            print("🎉 Week 1 Evaluation Completed Successfully!")
            
            if criteria_met:
                print("✅ Ready to proceed to Week 2 advanced model development")
            else:
                print("📋 Review recommendations before proceeding to Week 2")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            return False

def main():
    """Main execution function"""
    evaluation = Week1FinalEvaluation()
    success = evaluation.run_complete_evaluation()
    
    if success:
        print("\n🎯 Week 1 baseline establishment pipeline completed!")
    else:
        print("\n⚠️  Evaluation encountered issues. Please review and retry.")

if __name__ == "__main__":
    main()
