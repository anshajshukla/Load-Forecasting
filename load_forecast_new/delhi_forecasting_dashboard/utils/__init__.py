"""
Delhi Load Forecasting Dashboard - Utilities Package
Professional utility modules for the Streamlit dashboard application.

This package contains optimized utility functions for data loading,
visualization, constants management, and metrics calculation.
"""

from .constants import (
    PROJECT_NAME, PROJECT_VERSION, COLORS, CHART_CONFIG,
    MODEL_BENCHMARKS, FEATURE_CATEGORIES, get_project_summary
)

from .data_loader import (
    load_project_data, load_phase_results, load_model_performance_data,
    load_business_impact_data, load_feature_analysis_data,
    load_data_quality_metrics, validate_data_integrity, get_data_summary
)

from .visualizations import (
    create_performance_comparison_chart, create_timeline_chart,
    create_feature_importance_chart, create_business_impact_dashboard,
    create_correlation_heatmap, create_time_series_plot,
    create_distribution_plot, create_model_comparison_table,
    style_metric_card, apply_theme_to_plotly_figure
)

__version__ = "1.0.0"
__author__ = "Delhi Load Forecasting Team"

__all__ = [
    # Constants
    "PROJECT_NAME", "PROJECT_VERSION", "COLORS", "CHART_CONFIG",
    "MODEL_BENCHMARKS", "FEATURE_CATEGORIES", "get_project_summary",
    
    # Data Loading
    "load_project_data", "load_phase_results", "load_model_performance_data",
    "load_business_impact_data", "load_feature_analysis_data",
    "load_data_quality_metrics", "validate_data_integrity", "get_data_summary",
    
    # Visualizations
    "create_performance_comparison_chart", "create_timeline_chart",
    "create_feature_importance_chart", "create_business_impact_dashboard",
    "create_correlation_heatmap", "create_time_series_plot",
    "create_distribution_plot", "create_model_comparison_table",
    "style_metric_card", "apply_theme_to_plotly_figure"
]
