#!/usr/bin/env python3
"""
Comprehensive Cleanup Script for Delhi Load Forecasting Project
Removes irrelevant, duplicate, and temporary files
"""

import os
import shutil
import glob
from datetime import datetime

def cleanup_project():
    """Clean up irrelevant files and directories"""
    base_dir = r"C:\Users\ansha\Desktop\SIH_new"
    os.chdir(base_dir)
    
    print("🧹 Starting comprehensive cleanup...")
    print(f"📍 Working directory: {base_dir}")
    
    # Files and directories to delete
    items_to_delete = [
        # Duplicate phase directories (we have them in load_forecast/)
        "phase_3_week_1_model_development",
        "phase_3_week_2_neural_networks",
        
        # Old Load-Forecasting directory (replaced by load_forecast/)
        "Load-Forecasting",
        
        # Temporary and deployment files
        "temp_deployment",
        "deployment.log",
        
        # Old deployment scripts (keep only deploy_final.py)
        "deploy_lightweight.py",
        "deploy_to_github.py", 
        "deploy_quick.ps1",
        
        # Output directories (can be regenerated)
        "outputs",
        "results",
        
        # Old dataset files
        "final_dataset_solar_treated.csv",
        
        # Source directory at root (we have it in load_forecast/src/)
        "src",
        
        # Virtual environment duplicates
        ".venv",  # Keep the one in load_forecast if it exists
    ]
    
    deleted_count = 0
    skipped_count = 0
    
    for item in items_to_delete:
        item_path = os.path.join(base_dir, item)
        
        if os.path.exists(item_path):
            try:
                if os.path.isdir(item_path):
                    print(f"🗂️  Deleting directory: {item}")
                    shutil.rmtree(item_path)
                else:
                    print(f"📄 Deleting file: {item}")
                    os.remove(item_path)
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {item}: {e}")
                skipped_count += 1
        else:
            print(f"⏭️  Already deleted or doesn't exist: {item}")
            skipped_count += 1
    
    # Clean up cache files
    print("\n🔄 Cleaning cache files...")
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo",
        "**/.pytest_cache",
        "**/.ipynb_checkpoints",
    ]
    
    for pattern in cache_patterns:
        for item in glob.glob(pattern, recursive=True):
            try:
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                print(f"🧹 Cleaned cache: {item}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to clean {item}: {e}")
    
    # Summary
    print(f"\n✅ Cleanup completed!")
    print(f"📊 Summary:")
    print(f"   - Items deleted: {deleted_count}")
    print(f"   - Items skipped: {skipped_count}")
    
    # Show remaining structure
    print(f"\n📁 Remaining project structure:")
    remaining_items = [item for item in os.listdir(base_dir) 
                      if not item.startswith('.') and item != '__pycache__']
    
    for item in sorted(remaining_items):
        if os.path.isdir(item):
            print(f"   📂 {item}/")
        else:
            print(f"   📄 {item}")
    
    print(f"\n🎯 Project is now clean and optimized!")
    print(f"🚀 Main components remaining:")
    print(f"   - load_forecast/ (main project)")
    print(f"   - deploy_final.py (deployment script)")
    print(f"   - cleanup_irrelevant_files.py (existing cleanup)")
    print(f"   - 27-Jun-2025_Rev Prov summary Apr 25.pdf (documentation)")

if __name__ == "__main__":
    cleanup_project()
