"""
SIH 2024 Load Forecasting Project
Main Application Configuration and Setup
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Directory structure
DIRECTORIES = {
    'src': PROJECT_ROOT / 'src',
    'core': PROJECT_ROOT / 'src' / 'core',
    'models': PROJECT_ROOT / 'src' / 'models',
    'api': PROJECT_ROOT / 'src' / 'api',
    'utils': PROJECT_ROOT / 'src' / 'utils',
    'data': PROJECT_ROOT / 'data',
    'config': PROJECT_ROOT / 'config',
    'tests': PROJECT_ROOT / 'tests',
    'docs': PROJECT_ROOT / 'docs',
    'deployment': PROJECT_ROOT / 'deployment',
    'frontend': PROJECT_ROOT / 'frontend',
    'static': PROJECT_ROOT / 'static',
    'templates': PROJECT_ROOT / 'templates',
    'scripts': PROJECT_ROOT / 'scripts',
    'sih_2024': PROJECT_ROOT / 'sih_2024'
}

# Application metadata
APP_INFO = {
    'name': 'SIH 2024 Load Forecasting System',
    'version': '1.0.0',
    'description': 'Advanced Load Forecasting System for Smart Grid Management',
    'authors': ['SIH Team 2024'],
    'license': 'MIT',
    'python_version': '>=3.8',
    'competition': 'Smart India Hackathon 2024'
}

# Phase implementation tracking
PHASES = {
    'phase_1': {
        'name': 'Production Database Setup',
        'duration': 'Weeks 1-3',
        'status': 'planned',
        'deliverables': [
            'Cloud database connection',
            'Data migration scripts',
            'Real-time data pipeline'
        ],
        'files': [
            'sih_2024/database/',
            'sih_2024/scripts/migrate_to_supabase.py'
        ]
    },
    'phase_2': {
        'name': 'Enhanced Forecasting Engine',
        'duration': 'Weeks 4-6',
        'status': 'planned',
        'deliverables': [
            'Weather integration',
            'Holiday impact analysis',
            'Spike detection system'
        ],
        'files': [
            'sih_2024/enhancements/core_enhancements.py',
            'src/core/'
        ]
    },
    'phase_3': {
        'name': 'Production API & Dashboard',
        'duration': 'Weeks 7-9',
        'status': 'planned',
        'deliverables': [
            'RESTful API',
            'Real-time dashboard',
            'Mobile-responsive UI'
        ],
        'files': [
            'src/api/',
            'frontend/',
            'deployment/'
        ]
    },
    'phase_4': {
        'name': 'Advanced Analytics & ML',
        'duration': 'Weeks 10-12',
        'status': 'planned',
        'deliverables': [
            'Ensemble models',
            'Automated model retraining',
            'Performance analytics'
        ],
        'files': [
            'src/models/',
            'tests/',
            'docs/'
        ]
    }
}

def get_phase_status():
    """Get current implementation phase status"""
    return PHASES

def get_project_info():
    """Get project metadata"""
    return APP_INFO

def validate_structure():
    """Validate project directory structure"""
    missing_dirs = []
    for name, path in DIRECTORIES.items():
        if not path.exists():
            missing_dirs.append(path)
    
    if missing_dirs:
        print("⚠️  Missing directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        return False
    
    print("✅ Project structure validated successfully!")
    return True

if __name__ == "__main__":
    print(f"🚀 {APP_INFO['name']} v{APP_INFO['version']}")
    print(f"📁 Project root: {PROJECT_ROOT}")
    validate_structure()
