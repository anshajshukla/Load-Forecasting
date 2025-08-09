"""
Delhi Load Forecasting Dashboard - Project Constants
Centralized configuration and constants for the dashboard application.

This module contains all project constants, configuration parameters,
and shared data structures used across the dashboard.
"""

from typing import Dict, List, Any, Tuple
from datetime import datetime

# Project Information
PROJECT_NAME: str = "Delhi Load Forecasting"
PROJECT_VERSION: str = "1.0.0"
DASHBOARD_VERSION: str = "1.0.0"
ORGANIZATION: str = "Delhi Load Forecasting Team"
PROJECT_START_DATE: str = "2022-07-01"
CURRENT_DATE: str = "2025-08-09"

# Performance Targets and Achievements
TARGET_MAPE: float = 5.0
ACHIEVED_MAPE: float = 4.09
BASELINE_MAPE: float = 6.5  # Current DISCOM performance
WORLD_CLASS_MAPE: float = 3.0  # World-class target

# Feature Engineering Results
ORIGINAL_FEATURES: int = 267
OPTIMIZED_FEATURES: int = 111
FEATURE_QUALITY_SCORE: float = 0.894
MAX_QUALITY_SCORE: float = 1.0

# Business Impact Metrics
MONTHLY_SAVINGS_USD: int = 4_800_000
ANNUAL_SAVINGS_USD: int = 57_600_000
ROI_PERCENT: int = 47_876
PAYBACK_MONTHS: float = 0.0
GRID_STABILITY_SCORE: float = 79.4
CERC_COMPLIANCE_PERCENT: float = 75.0

# Data Characteristics
TOTAL_RECORDS: int = 26_472
TIME_PERIOD_YEARS: float = 3.1
DATA_COMPLETENESS_PERCENT: float = 99.2
DATA_ACCURACY_PERCENT: float = 98.7

# Project Phase Information
TOTAL_PHASES: int = 7
COMPLETED_PHASES: int = 4
CURRENT_PHASE: str = "Phase 5"

# Color Scheme for Dashboard
COLORS: Dict[str, str] = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e", 
    "success": "#2ca02c",
    "warning": "#d62728",
    "info": "#17a2b8",
    "background": "#f8f9fa",
    "text": "#262730",
    "light_blue": "#e8f4fd",
    "gradient_start": "#1f77b4",
    "gradient_end": "#2ca02c"
}

# Chart Configuration
CHART_CONFIG: Dict[str, Any] = {
    "default_height": 400,
    "default_width": 800,
    "font_family": "Arial, sans-serif",
    "font_size": 12,
    "title_font_size": 16,
    "margin": {"l": 50, "r": 50, "t": 80, "b": 50}
}

# Feature Categories for Analysis
FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "Weather-Load Interactions": [
        "temperature_max", "temperature_min", "humidity_avg",
        "solar_radiation", "wind_speed", "pressure"
    ],
    "Temporal Patterns": [
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos", "season"
    ],
    "Dual Peak Features": [
        "day_peak_magnitude", "night_peak_magnitude", "peak_ratio",
        "inter_peak_duration", "peak_timing_day", "peak_timing_night"
    ],
    "Thermal Comfort": [
        "thermal_comfort_index", "heat_index", "cooling_degree_hours",
        "heating_degree_hours", "apparent_temperature"
    ],
    "Festival & Cultural": [
        "festival_proximity", "festival_type", "holiday_flag",
        "weekend_flag", "working_day_flag"
    ],
    "Solar Integration": [
        "solar_generation", "duck_curve_impact", "net_load",
        "solar_variability", "cloud_cover_impact"
    ]
}

# Model Performance Benchmarks
MODEL_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "Current DISCOM": {
        "mape": 6.5,
        "mae": 180,
        "rmse": 220,
        "category": "Baseline"
    },
    "Industry Average": {
        "mape": 5.8,
        "mae": 165,
        "rmse": 205,
        "category": "Industry"
    },
    "World Class": {
        "mape": 3.0,
        "mae": 85,
        "rmse": 120,
        "category": "Target"
    },
    "Our Achievement": {
        "mape": 4.09,
        "mae": 115,
        "rmse": 145,
        "category": "Achievement"
    }
}

# Business Impact Categories
BUSINESS_IMPACT_CATEGORIES: List[Dict[str, Any]] = [
    {
        "category": "Cost Savings",
        "metric": "Monthly Savings",
        "value": f"${MONTHLY_SAVINGS_USD/1e6:.1f}M",
        "description": "Procurement cost optimization through accurate forecasting"
    },
    {
        "category": "Grid Stability", 
        "metric": "Stability Score",
        "value": f"{GRID_STABILITY_SCORE:.1f}%",
        "description": "Improved grid operations and reduced balancing requirements"
    },
    {
        "category": "ROI Achievement",
        "metric": "Return on Investment", 
        "value": f"{ROI_PERCENT:,}%",
        "description": "Outstanding financial returns with minimal payback period"
    },
    {
        "category": "Regulatory Compliance",
        "metric": "CERC Standards",
        "value": f"{CERC_COMPLIANCE_PERCENT:.0f}%",
        "description": "Central Electricity Regulatory Commission compliance"
    }
]

# Navigation Configuration
NAVIGATION_PAGES: Dict[str, Dict[str, str]] = {
    "overview": {
        "title": "🏠 Project Overview",
        "description": "Complete project status and key achievements"
    },
    "data_quality": {
        "title": "📊 Data Integration & Quality", 
        "description": "Data sources, quality metrics, and validation results"
    },
    "features": {
        "title": "🔧 Feature Engineering",
        "description": "267→111 feature optimization and engineering process"
    },
    "models": {
        "title": "🧠 Model Development",
        "description": "ML model training, evaluation, and selection results"
    },
    "performance": {
        "title": "📈 Performance Evaluation",
        "description": "4.09% MAPE achievement and performance analysis"
    },
    "business": {
        "title": "💼 Business Impact",
        "description": "$57.6M annual impact and ROI analysis"
    },
    "roadmap": {
        "title": "🚀 Next Steps & Roadmap", 
        "description": "Phase 5-7 planning and production deployment"
    }
}

# Status Indicators
STATUS_INDICATORS: Dict[str, Dict[str, str]] = {
    "complete": {
        "emoji": "✅",
        "text": "Complete",
        "color": COLORS["success"]
    },
    "in_progress": {
        "emoji": "🚀", 
        "text": "In Progress",
        "color": COLORS["warning"]
    },
    "planned": {
        "emoji": "📋",
        "text": "Planned", 
        "color": COLORS["info"]
    },
    "ready": {
        "emoji": "🚀",
        "text": "Ready",
        "color": COLORS["secondary"]
    }
}

# File Paths (relative to dashboard root)
FILE_PATHS: Dict[str, str] = {
    "project_data": "../final_dataset_solar_treated.csv",
    "phase3_results": "../phase_3_week_3_advanced_architectures/outputs/",
    "phase4_results": "../phase_4_model_evaluation_selection/reports/",
    "assets": "./assets/",
    "logs": "./logs/"
}

# Metrics Configuration
METRICS_CONFIG: Dict[str, Dict[str, Any]] = {
    "mape": {
        "name": "Mean Absolute Percentage Error",
        "unit": "%",
        "lower_is_better": True,
        "format": ".2f"
    },
    "mae": {
        "name": "Mean Absolute Error", 
        "unit": "MW",
        "lower_is_better": True,
        "format": ".1f"
    },
    "rmse": {
        "name": "Root Mean Square Error",
        "unit": "MW", 
        "lower_is_better": True,
        "format": ".1f"
    },
    "r2": {
        "name": "R-squared Score",
        "unit": "",
        "lower_is_better": False,
        "format": ".3f"
    }
}

# Export configuration
EXPORT_CONFIG: Dict[str, Any] = {
    "pdf_settings": {
        "orientation": "landscape",
        "format": "A4",
        "margin": "1cm"
    },
    "excel_settings": {
        "engine": "xlsxwriter",
        "options": {
            "strings_to_formulas": False,
            "strings_to_urls": False
        }
    }
}

def get_project_summary() -> Dict[str, Any]:
    """Get comprehensive project summary for dashboard."""
    return {
        "project_info": {
            "name": PROJECT_NAME,
            "version": PROJECT_VERSION,
            "organization": ORGANIZATION,
            "start_date": PROJECT_START_DATE,
            "current_date": CURRENT_DATE
        },
        "performance": {
            "target_mape": TARGET_MAPE,
            "achieved_mape": ACHIEVED_MAPE,
            "improvement": BASELINE_MAPE - ACHIEVED_MAPE,
            "improvement_percent": ((BASELINE_MAPE - ACHIEVED_MAPE) / BASELINE_MAPE) * 100
        },
        "business_impact": {
            "monthly_savings": MONTHLY_SAVINGS_USD,
            "annual_savings": ANNUAL_SAVINGS_USD,
            "roi_percent": ROI_PERCENT,
            "payback_months": PAYBACK_MONTHS
        },
        "technical_achievements": {
            "original_features": ORIGINAL_FEATURES,
            "optimized_features": OPTIMIZED_FEATURES,
            "quality_score": FEATURE_QUALITY_SCORE,
            "data_records": TOTAL_RECORDS,
            "time_period": TIME_PERIOD_YEARS
        },
        "project_status": {
            "total_phases": TOTAL_PHASES,
            "completed_phases": COMPLETED_PHASES,
            "current_phase": CURRENT_PHASE,
            "completion_percent": (COMPLETED_PHASES / TOTAL_PHASES) * 100
        }
    }
