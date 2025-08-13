"""
Delhi Load Forecasting - Phase 3 Week 3
Advanced Architectures Development Pipeline

This script orchestrates the complete advanced architecture development pipeline:
- Day 1-2: Hybrid models combining multiple approaches
- Day 3-4: Attention mechanisms and advanced feature fusion
- Day 5-6: Multi-scale and multi-horizon forecasting
- Day 7: Performance optimization and architecture selection

Target: Achieve MAPE <5% with advanced architectures
Timeline: Complete Week 3 advanced architecture development
"""

import os
import sys
import time
import subprocess
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class Week3AdvancedArchitecturesPipeline:
    """
    Master pipeline for Week 3 advanced architecture development
    
    Features:
    - Hybrid model architectures
    - Attention mechanisms
    - Multi-scale forecasting
    - Advanced ensemble methods
    - Performance optimization
    """
    
    def __init__(self, project_dir):
        """Initialize the advanced architectures pipeline"""
        self.project_dir = project_dir
        self.scripts_dir = os.path.join(project_dir, 'scripts')
        self.week1_dir = os.path.join(os.path.dirname(project_dir), 'phase_3_week_1_model_development')
        self.week2_dir = os.path.join(os.path.dirname(project_dir), 'phase_3_week_2_neural_networks')
        
        # Pipeline configuration
        self.pipeline_config = {
            'timeout_minutes': 180,  # 3 hours max per script
            'memory_limit_gb': 8,    # Memory limit for advanced models
            'max_retries': 2
        }
        
        # Pipeline stages
        self.stages = [
            {
                'name': 'Hybrid Model Development',
                'script': '01_hybrid_models.py',
                'description': 'Combine traditional ML with neural networks',
                'estimated_time': '30-45 minutes',
                'day': '1-2'
            },
            {
                'name': 'Attention Mechanisms',
                'script': '02_attention_mechanisms.py',
                'description': 'Implement attention-based forecasting models',
                'estimated_time': '45-60 minutes',
                'day': '3-4'
            },
            {
                'name': 'Multi-Scale Forecasting',
                'script': '03_multiscale_forecasting.py',
                'description': 'Multi-horizon and multi-resolution forecasting',
                'estimated_time': '30-45 minutes',
                'day': '5-6'
            },
            {
                'name': 'Advanced Ensemble Methods',
                'script': '04_advanced_ensemble.py',
                'description': 'Sophisticated ensemble and meta-learning',
                'estimated_time': '25-35 minutes',
                'day': '7'
            },
            {
                'name': 'Architecture Optimization',
                'script': '05_architecture_optimization.py',
                'description': 'Model compression and optimization',
                'estimated_time': '20-30 minutes',
                'day': '7'
            }
        ]
        
        # Execution log
        self.execution_log = []
        self.start_time = None
        self.total_stages = len(self.stages)
        self.completed_stages = 0
    
    def check_prerequisites(self):
        """Check system prerequisites for advanced architecture development"""
        print("[CHECKING] System prerequisites for advanced architecture development...")
        
        # Check Python environment
        python_version = sys.version_info
        print(f"   [INFO] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version.major < 3 or python_version.minor < 8:
            raise RuntimeError("Python 3.8+ required for advanced architectures")
        
        # Check required packages
        required_packages = ['tensorflow', 'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn', 'joblib']
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   [OK] {package} available")
            except ImportError:
                raise RuntimeError(f"Required package not installed: {package}")
        
        # Check previous week results
        for week_name, week_dir in [('Week 1', self.week1_dir), ('Week 2', self.week2_dir)]:
            if not os.path.exists(week_dir):
                print(f"   [WARN] {week_name} directory not found: {week_dir}")
                print(f"   [INFO] {week_name} baseline comparison will be skipped")
            else:
                print(f"   [OK] {week_name} results found for integration")
        
        # Check available resources
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.project_dir)
            free_gb = free // (1024**3)
            print(f"   [INFO] Available disk space: {free_gb} GB")
            
            if free_gb < 3:
                print(f"   [WARN] Low disk space. Advanced models require ~3GB")
        except:
            print(f"   [INFO] Could not check disk space")
        
        print("[OK] Prerequisites check completed")
        return True
    
    def log_execution(self, stage_name, status, duration=None, error=None):
        """Log stage execution details"""
        log_entry = {
            'stage': stage_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'error': error
        }
        self.execution_log.append(log_entry)
    
    def run_script(self, script_path, stage_name, estimated_time):
        """Run a pipeline script with timeout and error handling"""
        print(f"\\n[STARTING] {stage_name}")
        print(f"[INFO] Estimated time: {estimated_time}")
        print(f"[RUNNING] {os.path.basename(script_path)}")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, script_path],
                cwd=self.project_dir,
                capture_output=False,  # Show output in real-time
                timeout=self.pipeline_config['timeout_minutes'] * 60
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print("-" * 80)
                print(f"[SUCCESS] {stage_name} completed in {duration:.1f} seconds")
                self.log_execution(stage_name, 'SUCCESS', duration)
                return True
            else:
                print("-" * 80)
                print(f"[ERROR] {stage_name} failed with return code {result.returncode}")
                self.log_execution(stage_name, 'FAILED', duration, f"Return code: {result.returncode}")
                return False
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print("-" * 80)
            print(f"[TIMEOUT] {stage_name} timed out after {duration:.1f} seconds")
            self.log_execution(stage_name, 'TIMEOUT', duration, "Script timeout")
            return False
        
        except Exception as e:
            duration = time.time() - start_time
            print("-" * 80)
            print(f"[ERROR] {stage_name} failed: {str(e)}")
            self.log_execution(stage_name, 'ERROR', duration, str(e))
            return False
    
    def print_progress(self, current_stage, total_stages):
        """Print pipeline progress"""
        progress_pct = (current_stage / total_stages) * 100
        progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
        
        print(f"\\n[PROGRESS] Week 3 Advanced Architectures Pipeline")
        print(f"[{progress_bar}] {progress_pct:.0f}% ({current_stage}/{total_stages} stages)")
        
        if current_stage > 0:
            elapsed = time.time() - self.start_time
            avg_time = elapsed / current_stage
            remaining_stages = total_stages - current_stage
            estimated_remaining = avg_time * remaining_stages
            
            print(f"[INFO] Elapsed: {elapsed/60:.1f} min | Est. remaining: {estimated_remaining/60:.1f} min")
    
    def generate_pipeline_report(self):
        """Generate execution report"""
        print(f"\\n{'='*80}")
        print("WEEK 3 ADVANCED ARCHITECTURES PIPELINE EXECUTION REPORT")
        print(f"{'='*80}")
        
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        print(f"Pipeline Start: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Pipeline End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {total_duration/60:.1f} minutes")
        print(f"Stages Completed: {self.completed_stages}/{self.total_stages}")
        
        print(f"\\nSTAGE EXECUTION SUMMARY:")
        print("-" * 80)
        
        for i, log_entry in enumerate(self.execution_log, 1):
            status_icon = {
                'SUCCESS': '✓',
                'FAILED': '✗',
                'TIMEOUT': '⏱',
                'ERROR': '✗'
            }.get(log_entry['status'], '?')
            
            duration_str = f"{log_entry['duration']:.1f}s" if log_entry['duration'] else "N/A"
            
            print(f"{i:2d}. {status_icon} {log_entry['stage']:<40} [{log_entry['status']:7s}] {duration_str:>8s}")
            
            if log_entry['error']:
                print(f"    Error: {log_entry['error']}")
        
        # Success rate
        successful_stages = sum(1 for log in self.execution_log if log['status'] == 'SUCCESS')
        success_rate = (successful_stages / len(self.execution_log)) * 100 if self.execution_log else 0
        
        print(f"\\nPIPELINE STATISTICS:")
        print(f"• Success Rate: {success_rate:.1f}% ({successful_stages}/{len(self.execution_log)})")
        print(f"• Average Stage Duration: {total_duration/len(self.execution_log)/60:.1f} minutes")
        print(f"• Project Directory: {self.project_dir}")
        
        # Next steps
        if self.completed_stages == self.total_stages:
            print(f"\\n[NEXT STEPS]")
            print("✓ All advanced architectures have been developed and evaluated")
            print("✓ Review results in the 'results/' directory")
            print("✓ Check architecture optimization recommendations")
            print("✓ Proceed to Week 4: Model Optimization & Deployment")
        else:
            print(f"\\n[PIPELINE INCOMPLETE]")
            print(f"• {self.total_stages - self.completed_stages} stages remaining")
            print("• Review error messages above for troubleshooting")
            print("• Re-run pipeline after addressing issues")
        
        print(f"{'='*80}")
    
    def run_complete_pipeline(self):
        """Execute the complete advanced architectures development pipeline"""
        self.start_time = time.time()
        
        print("[STARTING] Delhi Load Forecasting - Phase 3 Week 3")
        print("Advanced Architectures Development Pipeline")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Project Directory: {self.project_dir}")
        print(f"Total Stages: {self.total_stages}")
        print("\\nFOCUS: Advanced architectures for <5% MAPE target")
        
        try:
            # Check prerequisites
            self.check_prerequisites()
            
            # Execute each stage
            for i, stage in enumerate(self.stages, 1):
                self.print_progress(self.completed_stages, self.total_stages)
                
                print(f"\\n[STAGE {i}/{self.total_stages}] Day {stage['day']}: {stage['name']}")
                print(f"[DESCRIPTION] {stage['description']}")
                
                script_path = os.path.join(self.scripts_dir, stage['script'])
                
                if not os.path.exists(script_path):
                    print(f"[ERROR] Script not found: {script_path}")
                    self.log_execution(stage['name'], 'ERROR', 0, f"Script not found: {stage['script']}")
                    continue
                
                # Run the stage
                success = self.run_script(script_path, stage['name'], stage['estimated_time'])
                
                if success:
                    self.completed_stages += 1
                else:
                    print(f"[FAILED] Stage {i} failed. Check logs above.")
                    print("[AUTO-CONTINUE] Continuing with remaining stages (non-interactive mode)")
            
            # Final progress
            self.print_progress(self.completed_stages, self.total_stages)
            
            # Generate report
            self.generate_pipeline_report()
            
            # Return success status
            return self.completed_stages == self.total_stages
            
        except KeyboardInterrupt:
            print(f"\\n[INTERRUPTED] Pipeline execution interrupted by user")
            self.generate_pipeline_report()
            return False
        
        except Exception as e:
            print(f"\\n[CRITICAL ERROR] Pipeline execution failed: {str(e)}")
            self.generate_pipeline_report()
            return False

def main():
    """Main execution function"""
    # Configuration
    project_dir = r"C:\\Users\\ansha\\Desktop\\SIH_new\\load_forecast\\phase_3_week_3_advanced_architectures"
    
    # Verify project directory exists
    if not os.path.exists(project_dir):
        print(f"[ERROR] Project directory not found: {project_dir}")
        print("Please ensure Week 3 directory structure exists")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = Week3AdvancedArchitecturesPipeline(project_dir)
    success = pipeline.run_complete_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
