#!/usr/bin/env python3
"""
Week 1 Model Development Directory Cleanup Script
================================================

This script removes unnecessary, duplicate, and redundant files from the 
phase_3_week_1_model_development directory to clean up the workspace.

Target files for removal:
- Duplicate scripts with "_fixed", "_fast", "_super_fast" suffixes
- Empty directories
- Redundant execution logs
- Nested duplicate directories
- Cache directories

Author: Cleanup Script
Date: August 8, 2025
"""

import os
import shutil
import sys
from pathlib import Path

def remove_file_safe(file_path):
    """Safely remove a file with error handling."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✓ Removed file: {file_path}")
            return True
        else:
            print(f"⚠ File not found: {file_path}")
            return False
    except Exception as e:
        print(f"✗ Error removing file {file_path}: {e}")
        return False

def remove_directory_safe(dir_path):
    """Safely remove a directory with error handling."""
    try:
        if os.path.exists(dir_path):
            if os.path.isdir(dir_path):
                # Check if directory is empty or only contains empty subdirectories
                is_empty = len(os.listdir(dir_path)) == 0
                if is_empty:
                    shutil.rmtree(dir_path)
                    print(f"✓ Removed empty directory: {dir_path}")
                    return True
                else:
                    # Force remove if specified in the list
                    shutil.rmtree(dir_path)
                    print(f"✓ Removed directory: {dir_path}")
                    return True
            else:
                print(f"⚠ Not a directory: {dir_path}")
                return False
        else:
            print(f"⚠ Directory not found: {dir_path}")
            return False
    except Exception as e:
        print(f"✗ Error removing directory {dir_path}: {e}")
        return False

def main():
    """Main cleanup function."""
    print("🧹 Starting Week 1 Model Development Directory Cleanup")
    print("=" * 60)
    
    # Get the base directory
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "scripts"
    
    print(f"Base directory: {base_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print()
    
    # Files to remove from scripts directory
    files_to_remove = [
        # Duplicate/fixed versions of scripts
        "scripts/02_fast_baselines.py",
        "scripts/02_linear_tree_baselines_fast.py", 
        "scripts/02_linear_tree_baselines_fixed.py",
        "scripts/02_super_fast_baselines.py",
        "scripts/03_gradient_boosting_time_series_fast.py",
        "scripts/03_gradient_boosting_time_series_fixed.py",
        "scripts/04_baseline_evaluation_documentation_fixed.py",
        # Any remaining _fixed or _fast files
    ]
    
    # Directories to remove
    directories_to_remove = [
        # Empty or redundant directories
        "notebooks",  # Empty
        "scripts/data",  # Empty
        "scripts/logs",  # Empty  
        "phase_3_week_1_model_development",  # Nested duplicate
        "scripts/__pycache__",  # Python cache
    ]
    
    # Execution log files to remove (keep only the main one)
    log_files_to_remove = [
        "WEEK_1_FAST_EXECUTION_LOG.txt",
        "WEEK_1_FAST_EXECUTION_SUMMARY.json",
    ]
    
    # Remove duplicate script files
    print("📁 Removing duplicate script files...")
    removed_files = 0
    for file_rel_path in files_to_remove:
        file_path = base_dir / file_rel_path
        if remove_file_safe(file_path):
            removed_files += 1
    
    print(f"Removed {removed_files} duplicate script files.\n")
    
    # Remove execution log files
    print("📄 Removing redundant execution logs...")
    removed_logs = 0
    for log_file in log_files_to_remove:
        log_path = base_dir / log_file
        if remove_file_safe(log_path):
            removed_logs += 1
    
    print(f"Removed {removed_logs} redundant log files.\n")
    
    # Remove empty/redundant directories
    print("📂 Removing empty/redundant directories...")
    removed_dirs = 0
    for dir_rel_path in directories_to_remove:
        dir_path = base_dir / dir_rel_path
        if remove_directory_safe(dir_path):
            removed_dirs += 1
    
    print(f"Removed {removed_dirs} redundant directories.\n")
    
    # Additional cleanup: Look for any remaining _fixed, _fast, _super_fast files
    print("🔍 Scanning for additional duplicate files...")
    additional_files = []
    
    for root, dirs, files in os.walk(scripts_dir):
        for file in files:
            if any(suffix in file for suffix in ['_fixed.py', '_fast.py', '_super_fast.py']):
                file_path = os.path.join(root, file)
                additional_files.append(file_path)
    
    if additional_files:
        print("Found additional duplicate files:")
        for file_path in additional_files:
            if remove_file_safe(file_path):
                removed_files += 1
    else:
        print("No additional duplicate files found.")
    
    print()
    print("=" * 60)
    print("🎉 Cleanup Summary:")
    print(f"   • Removed {removed_files} duplicate script files")
    print(f"   • Removed {removed_logs} redundant log files") 
    print(f"   • Removed {removed_dirs} redundant directories")
    print(f"   • Total items cleaned: {removed_files + removed_logs + removed_dirs}")
    print()
    print("✅ Week 1 directory cleanup completed successfully!")
    
    # Show remaining structure
    print("\n📋 Remaining important files in scripts directory:")
    if scripts_dir.exists():
        for item in sorted(scripts_dir.iterdir()):
            if item.is_file() and item.suffix == '.py':
                print(f"   📄 {item.name}")
            elif item.is_dir() and not item.name.startswith('.'):
                print(f"   📁 {item.name}/")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during cleanup: {e}")
        sys.exit(1)
