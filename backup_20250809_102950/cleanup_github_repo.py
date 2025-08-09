#!/usr/bin/env python3
"""
Simple Delhi Load Forecasting Repository Cleanup Script
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def scan_current_directory():
    """Show all files and folders in current directory."""
    current_path = Path(".")
    print(f"Scanning directory: {current_path.absolute()}")
    print("=" * 60)
    
    all_items = []
    for item in current_path.iterdir():
        if not item.name.startswith('.'):
            item_type = "FOLDER" if item.is_dir() else "FILE"
            size = ""
            if item.is_file():
                try:
                    size = f"({item.stat().st_size} bytes)"
                except:
                    size = ""
            all_items.append((item.name, item_type, size))
    
    # Sort and display
    all_items.sort()
    print(f"Found {len(all_items)} items:")
    print()
    
    for name, item_type, size in all_items:
        print(f"  {item_type}: {name} {size}")
    
    return [item[0] for item in all_items]

def categorize_files(file_list):
    """Categorize files based on Delhi forecasting project structure."""
    
    # Define categories based on what we expect in a Delhi forecasting project
    keep_files = set()
    archive_files = set()
    delete_files = set()
    
    for filename in file_list:
        filename_lower = filename.lower()
        
        # KEEP - Critical files
        if any(pattern in filename_lower for pattern in [
            'dashboard', 'src', 'requirements.txt', 'readme.md', 'dataset', 
            'notebook', 'phase_4', 'model_evaluation', 'final'
        ]):
            keep_files.add(filename)
        
        # ARCHIVE - Important but not critical
        elif any(pattern in filename_lower for pattern in [
            'config', 'docs', 'eda', 'feature_selection', 'validation'
        ]):
            archive_files.add(filename)
        
        # DELETE - Clearly temporary/intermediate
        elif any(pattern in filename_lower for pattern in [
            'analyze_', 'debug_', 'cleanup_', 'scrape_', 'verify_', 'test_',
            'phase_2_5', 'phase_3_week_1', 'phase_3_week_2', 'phase_3_week_3',
            'temp', 'old', 'backup'
        ]):
            delete_files.add(filename)
    
    # Anything not categorized is "unknown"
    all_files = set(file_list)
    unknown_files = all_files - keep_files - archive_files - delete_files
    
    return {
        'keep': sorted(keep_files),
        'archive': sorted(archive_files), 
        'delete': sorted(delete_files),
        'unknown': sorted(unknown_files)
    }

def show_categorization(categories):
    """Show the categorization results."""
    print("\n" + "=" * 60)
    print("FILE CATEGORIZATION RESULTS")
    print("=" * 60)
    
    print(f"\nKEEP (Critical - {len(categories['keep'])} files):")
    for f in categories['keep']:
        print(f"  KEEP: {f}")
    
    print(f"\nARCHIVE (Important but not critical - {len(categories['archive'])} files):")
    for f in categories['archive']:
        print(f"  ARCHIVE: {f}")
    
    print(f"\nDELETE (Temporary/Intermediate - {len(categories['delete'])} files):")
    for f in categories['delete']:
        print(f"  DELETE: {f}")
    
    print(f"\nUNKNOWN (Need manual review - {len(categories['unknown'])} files):")
    for f in categories['unknown']:
        print(f"  UNKNOWN: {f}")
    
    print(f"\nSUMMARY:")
    print(f"  Total files: {sum(len(v) for v in categories.values())}")
    print(f"  Will keep: {len(categories['keep'])}")
    print(f"  Will archive: {len(categories['archive'])}")
    print(f"  Will delete: {len(categories['delete'])}")
    print(f"  Need review: {len(categories['unknown'])}")

def manual_review_unknown(unknown_files):
    """Manually review unknown files."""
    if not unknown_files:
        return {'keep': [], 'archive': [], 'delete': []}
    
    print(f"\n" + "=" * 60)
    print("MANUAL REVIEW OF UNKNOWN FILES")
    print("=" * 60)
    print("For each file, choose what to do:")
    print("  k = Keep (important)")
    print("  a = Archive (reference)")
    print("  d = Delete (not needed)")
    print("  ? = Show file info")
    print()
    
    manual_categories = {'keep': [], 'archive': [], 'delete': []}
    
    for i, filename in enumerate(unknown_files):
        while True:
            print(f"\nFile {i+1}/{len(unknown_files)}: {filename}")
            choice = input("What to do with this file? (k/a/d/?): ").lower().strip()
            
            if choice == 'k':
                manual_categories['keep'].append(filename)
                print(f"  -> Will KEEP {filename}")
                break
            elif choice == 'a':
                manual_categories['archive'].append(filename)
                print(f"  -> Will ARCHIVE {filename}")
                break
            elif choice == 'd':
                manual_categories['delete'].append(filename)
                print(f"  -> Will DELETE {filename}")
                break
            elif choice == '?':
                # Show file info
                file_path = Path(filename)
                if file_path.exists():
                    if file_path.is_dir():
                        try:
                            contents = list(file_path.iterdir())[:5]  # First 5 items
                            print(f"    FOLDER contents (first 5): {[f.name for f in contents]}")
                        except:
                            print(f"    FOLDER (cannot read contents)")
                    else:
                        try:
                            size = file_path.stat().st_size
                            print(f"    FILE size: {size} bytes")
                        except:
                            print(f"    FILE (cannot read size)")
            else:
                print("Please enter k, a, d, or ?")
    
    return manual_categories

def execute_cleanup(categories):
    """Execute the cleanup based on categories."""
    print(f"\n" + "=" * 60)
    print("EXECUTING CLEANUP")
    print("=" * 60)
    
    # Create backup first
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = Path(backup_dir)
    
    items_to_backup = categories['archive'] + categories['delete']
    if items_to_backup:
        print(f"Creating backup in {backup_dir}/")
        backup_path.mkdir(exist_ok=True)
        
        for item_name in items_to_backup:
            source = Path(item_name)
            if source.exists():
                dest = backup_path / item_name
                try:
                    if source.is_dir():
                        shutil.copytree(source, dest)
                    else:
                        shutil.copy2(source, dest)
                    print(f"  Backed up: {item_name}")
                except Exception as e:
                    print(f"  ERROR backing up {item_name}: {e}")
    
    # Create archive directory
    archive_path = Path("archive")
    if categories['archive']:
        archive_path.mkdir(exist_ok=True)
        print(f"\nMoving files to archive/")
        
        for item_name in categories['archive']:
            source = Path(item_name)
            dest = archive_path / item_name
            if source.exists():
                try:
                    shutil.move(str(source), str(dest))
                    print(f"  Archived: {item_name}")
                except Exception as e:
                    print(f"  ERROR archiving {item_name}: {e}")
    
    # Delete files
    if categories['delete']:
        print(f"\nDeleting files:")
        for item_name in categories['delete']:
            item_path = Path(item_name)
            if item_path.exists():
                try:
                    if item_path.is_dir():
                        shutil.rmtree(item_path)
                    else:
                        item_path.unlink()
                    print(f"  Deleted: {item_name}")
                except Exception as e:
                    print(f"  ERROR deleting {item_name}: {e}")
    
    print(f"\nCleanup completed!")
    print(f"Backup created in: {backup_dir}/")
    if categories['archive']:
        print(f"Archived files moved to: archive/")

def main():
    """Main function."""
    print("Delhi Load Forecasting Repository Cleanup Tool")
    print("SIH 2024 Winner Project")
    print("=" * 60)
    
    # Step 1: Scan directory
    print("Step 1: Scanning current directory...")
    all_files = scan_current_directory()
    
    if not all_files:
        print("No files found! Make sure you're in the right directory.")
        return
    
    # Step 2: Auto-categorize
    print("\nStep 2: Auto-categorizing files...")
    categories = categorize_files(all_files)
    show_categorization(categories)
    
    # Step 3: Handle unknown files
    if categories['unknown']:
        print(f"\nStep 3: {len(categories['unknown'])} files need manual review...")
        review_choice = input("Do you want to review unknown files now? (y/n): ").lower()
        
        if review_choice == 'y':
            manual_cats = manual_review_unknown(categories['unknown'])
            # Merge manual categorization
            categories['keep'].extend(manual_cats['keep'])
            categories['archive'].extend(manual_cats['archive'])
            categories['delete'].extend(manual_cats['delete'])
            categories['unknown'] = []  # All reviewed
            
            print("\nUpdated categorization after manual review:")
            show_categorization(categories)
    
    # Step 4: Confirm and execute
    total_changes = len(categories['archive']) + len(categories['delete'])
    if total_changes == 0:
        print("\nNo changes to make. All files will remain in place.")
        return
    
    print(f"\nReady to make {total_changes} changes.")
    print("This will:")
    print(f"  - Move {len(categories['archive'])} files to archive/")
    print(f"  - Delete {len(categories['delete'])} files")
    print(f"  - Keep {len(categories['keep'])} files in place")
    print(f"  - Create backup before making changes")
    
    confirm = input("\nProceed with cleanup? (y/n): ").lower()
    if confirm == 'y':
        execute_cleanup(categories)
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()