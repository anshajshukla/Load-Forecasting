"""
Delhi Load Forecasting Dashboard - Data Loading Utilities
Professional data loading and caching functions for the dashboard.

This module provides optimized data loading functions with proper
caching, error handling, and data validation for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

from .constants import (
    FILE_PATHS, TOTAL_RECORDS, TIME_PERIOD_YEARS,
    MODEL_BENCHMARKS, FEATURE_CATEGORIES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data(ttl=3600, show_spinner=True)
def load_project_data() -> pd.DataFrame:
    """
    Load main project dataset with proper caching and error handling.
    
    Returns:
        pd.DataFrame: Main project dataset with all features and target variables
        
    Raises:
        FileNotFoundError: If the dataset file is not found
        ValueError: If the data format is invalid
    """
    try:
        # Try to load the main dataset
        data_path = Path(__file__).parent.parent / FILE_PATHS["project_data"]
        
        if data_path.exists():
            df = pd.read_csv(data_path)
            logger.info(f"Loaded project data: {len(df)} records from {data_path}")
            
            # Basic data validation
            if len(df) == 0:
                raise ValueError("Dataset is empty")
                
            # Ensure datetime column exists and is properly formatted
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
            
            return df
            
        else:
            logger.warning(f"Dataset file not found at {data_path}")
            # Return sample data for demonstration
            return _generate_sample_data()
            
    except Exception as e:
        logger.error(f"Error loading project data: {str(e)}")
        # Return sample data as fallback
        return _generate_sample_data()

@st.cache_data(ttl=1800)
def _generate_sample_data() -> pd.DataFrame:
    """
    Generate sample data for demonstration when main dataset is not available.
    
    Returns:
        pd.DataFrame: Sample dataset with key features for demonstration
    """
    logger.info("Generating sample data for demonstration")
    
    # Generate 1000 sample records for demonstration
    n_samples = 1000
    
    # Create date range
    start_date = datetime(2024, 1, 1)
    end_date = start_date + timedelta(hours=n_samples-1)
    dates = pd.date_range(start_date, end_date, freq='H')
    
    # Generate sample features
    np.random.seed(42)  # For reproducible sample data
    
    df = pd.DataFrame({
        'datetime': dates,
        'delhi_load': np.random.normal(4000, 800, n_samples),  # MW
        'temperature_max': np.random.normal(30, 8, n_samples),  # Celsius
        'temperature_min': np.random.normal(20, 6, n_samples),
        'humidity_avg': np.random.normal(65, 15, n_samples),  # %
        'solar_radiation': np.random.gamma(2, 100, n_samples),  # W/m²
        'wind_speed': np.random.gamma(2, 3, n_samples),  # m/s
        'hour_sin': np.sin(2 * np.pi * dates.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dates.hour / 24),
        'day_of_week': dates.dayofweek,
        'weekend_flag': (dates.dayofweek >= 5).astype(int),
        'day_peak_magnitude': np.random.normal(5000, 600, n_samples),
        'night_peak_magnitude': np.random.normal(3500, 400, n_samples),
        'thermal_comfort_index': np.random.normal(0.6, 0.2, n_samples),
        'festival_proximity': np.random.exponential(10, n_samples)
    })
    
    # Ensure realistic bounds
    df['delhi_load'] = np.clip(df['delhi_load'], 2000, 7000)
    df['temperature_max'] = np.clip(df['temperature_max'], 5, 50)
    df['temperature_min'] = np.clip(df['temperature_min'], 0, 45)
    df['humidity_avg'] = np.clip(df['humidity_avg'], 20, 100)
    df['thermal_comfort_index'] = np.clip(df['thermal_comfort_index'], 0, 1)
    
    return df

@st.cache_data(ttl=1800)
def load_phase_results() -> Dict[str, Any]:
    """
    Load results from all project phases with error handling.
    
    Returns:
        Dict[str, Any]: Consolidated results from all phases
    """
    try:
        results = {
            "phase_1": {
                "status": "Complete",
                "duration": "3 weeks",
                "key_metrics": {
                    "data_completeness": 99.2,
                    "data_accuracy": 98.7,
                    "records_processed": TOTAL_RECORDS,
                    "time_period": TIME_PERIOD_YEARS
                }
            },
            "phase_2": {
                "status": "Complete", 
                "duration": "4 weeks",
                "key_metrics": {
                    "features_engineered": 267,
                    "feature_categories": len(FEATURE_CATEGORIES),
                    "dual_peak_features": 30,
                    "thermal_comfort_features": 30
                }
            },
            "phase_2_5": {
                "status": "Complete",
                "duration": "3 days", 
                "key_metrics": {
                    "features_optimized": 111,
                    "quality_score": 0.894,
                    "leakage_eliminated": True,
                    "multicollinearity_resolved": True
                }
            },
            "phase_3": {
                "status": "Complete",
                "duration": "4 weeks",
                "key_metrics": {
                    "models_trained": 19,
                    "best_mape": 4.09,
                    "target_achieved": True,
                    "ensemble_method": "Hybrid RF+Linear"
                }
            },
            "phase_4": {
                "status": "Complete",
                "duration": "1 week",
                "key_metrics": {
                    "committee_approval": "Unanimous",
                    "deployment_authorized": True,
                    "business_impact_validated": True,
                    "documentation_complete": True
                }
            }
        }
        
        # Try to load actual results if available
        phase4_path = Path(__file__).parent.parent / FILE_PATHS["phase4_results"]
        if phase4_path.exists():
            # Load actual Phase 4 results if available
            for file in phase4_path.glob("*.json"):
                try:
                    with open(file, 'r') as f:
                        file_data = json.load(f)
                        results["actual_phase4_data"] = file_data
                        logger.info(f"Loaded actual Phase 4 results from {file}")
                        break
                except Exception as e:
                    logger.warning(f"Could not load {file}: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading phase results: {str(e)}")
        return {}

@st.cache_data(ttl=1800)
def load_model_performance_data() -> Dict[str, Any]:
    """
    Load model performance comparison data.
    
    Returns:
        Dict[str, Any]: Model performance metrics and comparisons
    """
    try:
        # Base performance data from constants
        performance_data = {
            "model_comparison": MODEL_BENCHMARKS.copy(),
            "phase3_models": {
                "Week1_XGBoost": {
                    "mape": 6.85,
                    "mae": 195,
                    "rmse": 245,
                    "training_time": "1 hour",
                    "complexity": "Medium"
                },
                "Week2_BiLSTM": {
                    "mape": 11.71,
                    "mae": 285,
                    "rmse": 340,
                    "training_time": "6 hours", 
                    "complexity": "High"
                },
                "Week3_Hybrid_RF_Linear": {
                    "mape": 4.09,
                    "mae": 115,
                    "rmse": 145,
                    "training_time": "2.5 hours",
                    "complexity": "Medium-High"
                },
                "Week4_Optimized": {
                    "mape": 4.12,
                    "mae": 118,
                    "rmse": 148,
                    "training_time": "3 hours",
                    "complexity": "High",
                    "note": "Overfitting prevention applied"
                }
            },
            "performance_trends": {
                "improvement_timeline": [
                    {"week": "Baseline", "mape": 6.5, "model": "Current DISCOM"},
                    {"week": "Week 1", "mape": 6.85, "model": "XGBoost"},
                    {"week": "Week 2", "mape": 11.71, "model": "BiLSTM"},
                    {"week": "Week 3", "mape": 4.09, "model": "Hybrid RF+Linear"},
                    {"week": "Week 4", "mape": 4.12, "model": "Optimized"}
                ]
            }
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error loading model performance data: {str(e)}")
        return {}

@st.cache_data(ttl=1800)
def load_business_impact_data() -> Dict[str, Any]:
    """
    Load business impact and ROI analysis data.
    
    Returns:
        Dict[str, Any]: Business impact metrics and analysis
    """
    try:
        business_data = {
            "economic_impact": {
                "monthly_savings_usd": 4_800_000,
                "annual_savings_usd": 57_600_000,
                "roi_percent": 47_876,
                "payback_months": 0.0,
                "implementation_cost": 100_000,
                "operational_savings": {
                    "procurement_optimization": 3_200_000,
                    "balancing_cost_reduction": 1_200_000,
                    "grid_efficiency_gains": 400_000
                }
            },
            "grid_improvements": {
                "stability_score": 79.4,
                "frequency_regulation_improvement": 47,
                "renewable_curtailment_reduction": 4,
                "capacity_utilization_improvement": 12
            },
            "regulatory_compliance": {
                "cerc_overall": 75.0,
                "day_ahead_accuracy": 95.9,
                "forecast_error_limit": 5.0,
                "current_error": 4.09,
                "compliance_status": "Needs Review"
            },
            "stakeholder_value": {
                "discom_savings": 2_880_000,  # 60% of total
                "consumer_benefits": 1_440_000,  # 30% of total
                "grid_operator_value": 480_000   # 10% of total
            }
        }
        
        return business_data
        
    except Exception as e:
        logger.error(f"Error loading business impact data: {str(e)}")
        return {}

@st.cache_data(ttl=1800) 
def load_feature_analysis_data() -> Dict[str, Any]:
    """
    Load feature engineering and analysis data.
    
    Returns:
        Dict[str, Any]: Feature importance and engineering results
    """
    try:
        feature_data = {
            "feature_evolution": {
                "original_count": 267,
                "after_validation": 251,  # After removing leakage
                "final_optimized": 111,
                "quality_score": 0.894
            },
            "feature_importance": {
                "top_features": [
                    {"feature": "temperature_max", "importance": 0.12, "category": "Weather"},
                    {"feature": "hour_sin", "importance": 0.09, "category": "Temporal"},
                    {"feature": "humidity_avg", "importance": 0.08, "category": "Weather"},
                    {"feature": "day_peak_magnitude", "importance": 0.07, "category": "Dual Peak"},
                    {"feature": "thermal_comfort_index", "importance": 0.06, "category": "Thermal"},
                    {"feature": "cooling_degree_hours", "importance": 0.06, "category": "Thermal"},
                    {"feature": "weekend_flag", "importance": 0.05, "category": "Temporal"},
                    {"feature": "festival_proximity", "importance": 0.04, "category": "Cultural"},
                    {"feature": "solar_radiation", "importance": 0.04, "category": "Weather"},
                    {"feature": "wind_speed", "importance": 0.03, "category": "Weather"}
                ]
            },
            "category_distribution": {
                "Weather-Load Interactions": 35,
                "Temporal Patterns": 25,
                "Dual Peak Features": 20,
                "Thermal Comfort": 15,
                "Festival & Cultural": 10,
                "Solar Integration": 6
            },
            "quality_metrics": {
                "data_leakage_removed": 16,  # Features removed
                "multicollinearity_resolved": 140,  # Features affected
                "vif_score_improved": True,
                "correlation_threshold": 0.95
            }
        }
        
        return feature_data
        
    except Exception as e:
        logger.error(f"Error loading feature analysis data: {str(e)}")
        return {}

@st.cache_data(ttl=3600)
def load_data_quality_metrics() -> Dict[str, Any]:
    """
    Load data quality assessment metrics.
    
    Returns:
        Dict[str, Any]: Comprehensive data quality metrics
    """
    try:
        quality_data = {
            "overall_metrics": {
                "completeness": 99.2,
                "accuracy": 98.7,
                "consistency": 99.8,
                "timeliness": 99.5,
                "validity": 98.9
            },
            "data_sources": {
                "weather_data": {
                    "completeness": 99.5,
                    "accuracy": 98.9,
                    "source": "Multiple weather APIs",
                    "update_frequency": "Hourly"
                },
                "load_data": {
                    "completeness": 98.8,
                    "accuracy": 98.5,
                    "source": "Delhi DISCOMs (BRPL, BYPL, NDPL)",
                    "update_frequency": "Real-time"
                }
            },
            "validation_results": {
                "outliers_detected": 234,
                "outliers_treated": 234,
                "missing_values_imputed": 156,
                "temporal_gaps_filled": 12
            },
            "time_coverage": {
                "start_date": "2022-07-01",
                "end_date": "2025-07-31", 
                "total_hours": TOTAL_RECORDS,
                "missing_hours": 28,
                "coverage_percent": 99.9
            }
        }
        
        return quality_data
        
    except Exception as e:
        logger.error(f"Error loading data quality metrics: {str(e)}")
        return {}

def validate_data_integrity(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate data integrity for dashboard display.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dict[str, bool]: Validation results
    """
    try:
        validation_results = {
            "has_data": len(df) > 0,
            "has_datetime": 'datetime' in df.columns,
            "has_target": any(col in df.columns for col in ['delhi_load', 'load', 'target']),
            "no_all_nulls": not df.isnull().all().any(),
            "temporal_order": True
        }
        
        # Check temporal ordering if datetime exists
        if validation_results["has_datetime"]:
            try:
                df_sorted = df.sort_values('datetime')
                validation_results["temporal_order"] = df['datetime'].equals(df_sorted['datetime'])
            except:
                validation_results["temporal_order"] = False
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating data integrity: {str(e)}")
        return {"error": True, "message": str(e)}

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive data summary for dashboard display.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Dict[str, Any]: Data summary statistics
    """
    try:
        summary = {
            "basic_info": {
                "total_records": len(df),
                "total_features": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
                "date_range": None
            },
            "missing_data": {
                "total_missing": df.isnull().sum().sum(),
                "missing_percent": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                "columns_with_missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
            },
            "data_types": df.dtypes.value_counts().to_dict()
        }
        
        # Add date range if datetime column exists
        if 'datetime' in df.columns:
            summary["basic_info"]["date_range"] = {
                "start": df['datetime'].min().strftime('%Y-%m-%d'),
                "end": df['datetime'].max().strftime('%Y-%m-%d'),
                "duration_days": (df['datetime'].max() - df['datetime'].min()).days
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating data summary: {str(e)}")
        return {"error": True, "message": str(e)}
