"""
Delhi Load Forecasting - Phase 3 Week 1
COMPLETE WEEK 1 BASELINE ESTABLISHMENT PIPELINE

This master script executes the complete Week 1 baseline establishment pipeline
following the exact Day 1-7 timeline as specified in the project flow.

Timeline: Days 1-7 of Week 1 baseline establishment
Success Criteria: Baseline MAPE <10% established across all models
"""

import sys
import os
import subprocess
import time
from datetime import datetime
import json

class Week1BaselineExecutor:
    """
    Master executor for complete Week 1 baseline establishment pipeline
    
    Executes:
    - Day 1-2: Data Preparation Pipeline
    - Day 3-4: Linear & Tree-Based Baselines  
    - Day 5-6: Gradient Boosting & Time Series Baselines
    - Day 7: Baseline Evaluation & Documentation
    """
    
    def __init__(self, project_dir):
        """Initialize Week 1 pipeline executor"""
        self.project_dir = project_dir
        self.scripts_dir = os.path.join(project_dir, 'scripts')
        self.execution_log = []
        self.start_time = datetime.now()
        
        # Define pipeline scripts
        self.pipeline_scripts = [
            {
                'day': '1-2',
                'name': 'Data Preparation Pipeline',
                'script': '01_data_preparation_pipeline.py',
                'description': 'Time-based splits, feature scaling, cross-validation setup',
                'success_criteria': 'Data prepared for 111 features across train/val/test sets'
            },
            {
                'day': '3-4', 
                'name': 'Linear & Tree-Based Baselines',
                'script': '02_linear_tree_baselines.py',
                'description': 'Ridge/Lasso regression, Random Forest with hyperparameter tuning',
                'success_criteria': 'Ridge/Lasso <12% MAPE, Random Forest <8% MAPE'
            },
            {
                'day': '5-6',
                'name': 'Gradient Boosting & Time Series Baselines', 
                'script': '03_gradient_boosting_time_series.py',
                'description': 'XGBoost optimization, Facebook Prophet, ensemble combinations',
                'success_criteria': 'XGBoost <7% MAPE, Prophet <9% MAPE, functional ensembles'
            },
            {
                'day': '7',
                'name': 'Baseline Evaluation & Documentation',
                'script': '04_baseline_evaluation_documentation.py',
                'description': 'Performance comparison, model selection, Week 2 planning',
                'success_criteria': 'Complete evaluation report, Week 2 architecture plan'
            }
        ]
        
    def log_execution(self, message, status='INFO'):
        """Log execution progress"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {status}: {message}"
        print(log_entry)
        self.execution_log.append(log_entry)
    
    def check_prerequisites(self):
        """Check if all prerequisites are met"""
        self.log_execution("🔍 Checking Week 1 pipeline prerequisites...")
        
        # Check if project directory exists
        if not os.path.exists(self.project_dir):
            raise FileNotFoundError(f"Project directory not found: {self.project_dir}")
        
        # Check if scripts directory exists
        if not os.path.exists(self.scripts_dir):
            raise FileNotFoundError(f"Scripts directory not found: {self.scripts_dir}")
        
        # Check if all required scripts exist
        missing_scripts = []
        for script_info in self.pipeline_scripts:
            script_path = os.path.join(self.scripts_dir, script_info['script'])
            if not os.path.exists(script_path):
                missing_scripts.append(script_info['script'])
        
        if missing_scripts:
            raise FileNotFoundError(f"Missing scripts: {missing_scripts}")
        
        # Check if data file exists
        data_file = r"C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_2_5_3_outputs\delhi_selected_features.csv"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Required dataset not found: {data_file}")
        
        self.log_execution("✅ All prerequisites checked successfully")
    
    def execute_script(self, script_info):
        """Execute a single pipeline script"""
        script_name = script_info['script']
        script_path = os.path.join(self.scripts_dir, script_name)
        
        self.log_execution(f"🚀 Starting Day {script_info['day']}: {script_info['name']}")
        self.log_execution(f"📋 Description: {script_info['description']}")
        self.log_execution(f"🎯 Success Criteria: {script_info['success_criteria']}")
        
        start_time = time.time()
        
        try:
            # Execute the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                cwd=self.project_dir
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                self.log_execution(f"✅ Day {script_info['day']} completed successfully in {execution_time:.1f}s", "SUCCESS")
                self.log_execution(f"📊 Output preview: {result.stdout[-200:] if result.stdout else 'No output'}")
                return True
            else:
                self.log_execution(f"❌ Day {script_info['day']} failed after {execution_time:.1f}s", "ERROR")
                self.log_execution(f"Error details: {result.stderr}", "ERROR")
                return False
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.log_execution(f"❌ Day {script_info['day']} failed with exception after {execution_time:.1f}s: {str(e)}", "ERROR")
            return False
    
    def create_week1_summary_report(self):
        """Create comprehensive Week 1 summary report"""
        self.log_execution("📄 Creating Week 1 execution summary report...")
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        summary_report = {
            'week_1_execution_summary': {
                'project': 'Delhi Load Forecasting - Phase 3 Week 1',
                'execution_date': self.start_time.isoformat(),
                'total_execution_time_seconds': total_time,
                'total_execution_time_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
                'pipeline_scripts_executed': len(self.pipeline_scripts),
                'overall_status': 'COMPLETED' if all(script.get('success', False) for script in self.pipeline_scripts) else 'PARTIAL',
            },
            'script_execution_details': []
        }
        
        for i, script_info in enumerate(self.pipeline_scripts):
            script_detail = {
                'day': script_info['day'],
                'script_name': script_info['script'],
                'description': script_info['description'],
                'success_criteria': script_info['success_criteria'],
                'status': 'COMPLETED' if script_info.get('success', False) else 'FAILED',
                'execution_order': i + 1
            }
            summary_report['script_execution_details'].append(script_detail)
        
        # Save summary report
        summary_path = os.path.join(self.project_dir, 'WEEK_1_EXECUTION_SUMMARY.json')
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        # Save execution log
        log_path = os.path.join(self.project_dir, 'WEEK_1_EXECUTION_LOG.txt')
        with open(log_path, 'w') as f:
            f.write('\n'.join(self.execution_log))
        
        self.log_execution(f"💾 Summary report saved: {summary_path}")
        self.log_execution(f"💾 Execution log saved: {log_path}")
    
    def run_complete_week1_pipeline(self):
        """Execute the complete Week 1 baseline establishment pipeline"""
        self.log_execution("🚀 STARTING DELHI LOAD FORECASTING - WEEK 1 BASELINE ESTABLISHMENT")
        self.log_execution("=" * 80)
        self.log_execution("📋 Pipeline Overview:")
        self.log_execution("   Day 1-2: Data Preparation Pipeline") 
        self.log_execution("   Day 3-4: Linear & Tree-Based Baselines")
        self.log_execution("   Day 5-6: Gradient Boosting & Time Series Baselines")
        self.log_execution("   Day 7: Baseline Evaluation & Documentation")
        self.log_execution("🎯 Success Criteria: Baseline MAPE <10% established across all models")
        self.log_execution("=" * 80)
        
        try:
            # Check prerequisites
            self.check_prerequisites()
            
            # Execute each script in the pipeline
            all_successful = True
            for script_info in self.pipeline_scripts:
                success = self.execute_script(script_info)
                script_info['success'] = success
                
                if not success:
                    all_successful = False
                    self.log_execution(f"⚠️ Pipeline interrupted at Day {script_info['day']}", "WARNING")
                    self.log_execution("🔄 Continuing with next stage (non-interactive mode)", "INFO")
                    # Continue automatically instead of asking user
                
                self.log_execution("-" * 60)
            
            # Create summary report
            self.create_week1_summary_report()
            
            # Final summary
            self.log_execution("\n🎉 WEEK 1 BASELINE ESTABLISHMENT PIPELINE COMPLETED!")
            self.log_execution("=" * 80)
            
            successful_scripts = sum(1 for script in self.pipeline_scripts if script.get('success', False))
            total_scripts = len(self.pipeline_scripts)
            
            self.log_execution(f"📊 EXECUTION SUMMARY:")
            self.log_execution(f"   ✅ Scripts completed successfully: {successful_scripts}/{total_scripts}")
            self.log_execution(f"   ⏱️ Total execution time: {(datetime.now() - self.start_time).total_seconds():.0f} seconds")
            self.log_execution(f"   📁 Results location: {self.project_dir}")
            
            if all_successful:
                self.log_execution(f"   🏆 WEEK 1 STATUS: ✅ FULLY COMPLETED")
                self.log_execution(f"   🚀 READY FOR: Week 2 Advanced Model Development")
            else:
                self.log_execution(f"   ⚠️ WEEK 1 STATUS: 🔶 PARTIALLY COMPLETED")
                self.log_execution(f"   🔧 ACTION REQUIRED: Review failed components before Week 2")
            
            self.log_execution("\n🎯 NEXT STEPS:")
            self.log_execution("   1. Review Week 1 evaluation dashboard and report")
            self.log_execution("   2. Validate baseline model performance (target: <10% MAPE)")
            self.log_execution("   3. Begin Week 2 LSTM/GRU architecture development")
            self.log_execution("   4. Setup neural network training environment")
            
            return all_successful
            
        except Exception as e:
            self.log_execution(f"❌ CRITICAL ERROR: Week 1 pipeline failed: {str(e)}", "CRITICAL")
            raise
        
        finally:
            self.log_execution(f"📝 Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main execution function"""
    print("🚀 Delhi Load Forecasting - Week 1 Baseline Establishment Pipeline")
    print("=" * 80)
    
    # Configuration
    project_dir = r"C:\Users\ansha\Desktop\SIH_new\load_forecast\phase_3_week_1_model_development"
    
    # Initialize and run complete Week 1 pipeline
    try:
        executor = Week1BaselineExecutor(project_dir)
        success = executor.run_complete_week1_pipeline()
        
        if success:
            print("\n✅ WEEK 1 BASELINE ESTABLISHMENT COMPLETED SUCCESSFULLY!")
            print("🚀 Ready to proceed to Week 2: Advanced Model Development")
        else:
            print("\n⚠️ WEEK 1 BASELINE ESTABLISHMENT PARTIALLY COMPLETED")
            print("🔧 Please review errors and re-run failed components")
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("🛠️ Please check prerequisites and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
