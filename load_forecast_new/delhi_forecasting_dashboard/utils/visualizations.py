"""
Delhi Load Forecasting Dashboard - Visualization Utilities
Professional visualization functions with Plotly and Seaborn integration.

This module provides optimized visualization functions with consistent
styling, interactive features, and responsive design for the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from .constants import COLORS, CHART_CONFIG, MODEL_BENCHMARKS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default color palette
DEFAULT_COLORS = [
    COLORS["primary"], COLORS["secondary"], COLORS["success"], 
    COLORS["warning"], COLORS["info"]
]

def apply_theme_to_plotly_figure(fig: go.Figure, title: str = "") -> go.Figure:
    """
    Apply consistent theme and styling to Plotly figures.
    
    Args:
        fig: Plotly figure object
        title: Figure title
        
    Returns:
        go.Figure: Styled figure with consistent theme
    """
    try:
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': CHART_CONFIG["title_font_size"], 'family': CHART_CONFIG["font_family"]}
            },
            font={'family': CHART_CONFIG["font_family"], 'size': CHART_CONFIG["font_size"]},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=CHART_CONFIG["margin"],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right", 
                x=1
            ),
            hovermode='closest'
        )
        
        # Update axes styling
        fig.update_xaxes(
            gridcolor='lightgray',
            gridwidth=0.5,
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        )
        
        fig.update_yaxes(
            gridcolor='lightgray', 
            gridwidth=0.5,
            showgrid=True,
            zeroline=True,
            zerolinecolor='lightgray'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error applying theme to figure: {str(e)}")
        return fig

def create_performance_comparison_chart(
    performance_data: Dict[str, Any],
    metric: str = "mape",
    title: str = "Model Performance Comparison"
) -> go.Figure:
    """
    Create interactive performance comparison chart.
    
    Args:
        performance_data: Dictionary containing model performance metrics
        metric: Performance metric to display (mape, mae, rmse)
        title: Chart title
        
    Returns:
        go.Figure: Interactive performance comparison chart
    """
    try:
        models = list(performance_data.keys())
        values = [performance_data[model][metric] for model in models]
        
        # Color coding based on performance
        colors = []
        for value in values:
            if metric == "mape":
                if value <= 3.0:
                    colors.append(COLORS["success"])  # Excellent
                elif value <= 5.0:
                    colors.append(COLORS["primary"])  # Good
                elif value <= 7.0:
                    colors.append(COLORS["secondary"])  # Fair
                else:
                    colors.append(COLORS["warning"])  # Needs improvement
            else:
                colors.append(COLORS["primary"])
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=values,
                marker_color=colors,
                text=[f"{value:.2f}" for value in values],
                textposition='auto',
                hovertemplate=f"<b>%{{x}}</b><br>{metric.upper()}: %{{y:.2f}}<extra></extra>"
            )
        ])
        
        # Add target line for MAPE
        if metric == "mape":
            fig.add_hline(
                y=5.0,
                line_dash="dash",
                line_color=COLORS["warning"],
                annotation_text="Target (5.0%)",
                annotation_position="top right"
            )
        
        fig = apply_theme_to_plotly_figure(fig, title)
        fig.update_layout(
            xaxis_title="Models",
            yaxis_title=f"{metric.upper()} ({'%' if metric == 'mape' else 'MW'})",
            height=CHART_CONFIG["default_height"]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating performance comparison chart: {str(e)}")
        return go.Figure()

def create_timeline_chart(
    timeline_data: List[Dict[str, Any]],
    title: str = "Project Timeline"
) -> go.Figure:
    """
    Create interactive project timeline chart.
    
    Args:
        timeline_data: List of timeline events with dates and descriptions
        title: Chart title
        
    Returns:
        go.Figure: Interactive timeline chart
    """
    try:
        fig = go.Figure()
        
        # Create timeline scatter plot
        for i, event in enumerate(timeline_data):
            status = event.get('status', 'complete')
            
            if status == 'complete':
                color = COLORS["success"]
                symbol = 'circle'
            elif status == 'in_progress':
                color = COLORS["secondary"]
                symbol = 'diamond'
            else:
                color = COLORS["info"]
                symbol = 'square'
            
            fig.add_trace(go.Scatter(
                x=[i],
                y=[1],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color=color,
                    symbol=symbol,
                    line=dict(width=2, color='white')
                ),
                text=event.get('phase', f"Event {i+1}"),
                textposition='top center',
                name=event.get('name', f"Event {i+1}"),
                hovertemplate=f"<b>%{{text}}</b><br>" +
                             f"Status: {status}<br>" +
                             f"Duration: {event.get('duration', 'N/A')}<br>" +
                             "<extra></extra>"
            ))
        
        # Connect timeline events
        fig.add_trace(go.Scatter(
            x=list(range(len(timeline_data))),
            y=[1] * len(timeline_data),
            mode='lines',
            line=dict(color=COLORS["primary"], width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig = apply_theme_to_plotly_figure(fig, title)
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(len(timeline_data))),
                ticktext=[event.get('phase', f"Event {i+1}") for i, event in enumerate(timeline_data)]
            ),
            yaxis=dict(visible=False),
            height=300,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating timeline chart: {str(e)}")
        return go.Figure()

def create_feature_importance_chart(
    feature_data: List[Dict[str, Any]],
    top_n: int = 15,
    title: str = "Top Feature Importance Rankings"
) -> go.Figure:
    """
    Create horizontal bar chart for feature importance.
    
    Args:
        feature_data: List of feature importance data
        top_n: Number of top features to display
        title: Chart title
        
    Returns:
        go.Figure: Feature importance chart
    """
    try:
        # Sort by importance and take top N
        sorted_features = sorted(feature_data, key=lambda x: x['importance'], reverse=True)[:top_n]
        
        features = [f['feature'] for f in sorted_features]
        importance = [f['importance'] for f in sorted_features]
        categories = [f.get('category', 'Other') for f in sorted_features]
        
        # Color mapping for categories
        category_colors = {
            'Weather': COLORS["primary"],
            'Temporal': COLORS["secondary"],
            'Dual Peak': COLORS["success"],
            'Thermal': COLORS["warning"],
            'Cultural': COLORS["info"],
            'Other': '#9467bd'
        }
        
        colors = [category_colors.get(cat, category_colors['Other']) for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker_color=colors,
                text=[f"{imp:.3f}" for imp in importance],
                textposition='auto',
                hovertemplate="<b>%{y}</b><br>" +
                            "Importance: %{x:.3f}<br>" +
                            "Category: %{customdata}<br>" +
                            "<extra></extra>",
                customdata=categories
            )
        ])
        
        fig = apply_theme_to_plotly_figure(fig, title)
        fig.update_layout(
            xaxis_title="Feature Importance",
            yaxis_title="Features",
            height=max(400, top_n * 25),  # Dynamic height based on number of features
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating feature importance chart: {str(e)}")
        return go.Figure()

def create_business_impact_dashboard(
    business_data: Dict[str, Any],
    title: str = "Business Impact Dashboard"
) -> go.Figure:
    """
    Create comprehensive business impact dashboard with subplots.
    
    Args:
        business_data: Dictionary containing business impact metrics
        title: Dashboard title
        
    Returns:
        go.Figure: Multi-panel business impact dashboard
    """
    try:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Monthly Cost Savings', 'ROI Analysis',
                'Grid Improvements', 'Regulatory Compliance'
            ],
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 1. Monthly Savings Indicator
        monthly_savings = business_data.get('economic_impact', {}).get('monthly_savings_usd', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=monthly_savings / 1e6,  # Convert to millions
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Monthly Savings ($M)"},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': COLORS["success"]},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 5], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 3
                    }
                }
            ),
            row=1, col=1
        )
        
        # 2. ROI Analysis Bar Chart
        roi_data = business_data.get('economic_impact', {})
        roi_categories = ['Implementation Cost', 'Annual Savings', 'Net Benefit']
        roi_values = [
            roi_data.get('implementation_cost', 100000) / 1e6,
            roi_data.get('annual_savings_usd', 57600000) / 1e6,
            (roi_data.get('annual_savings_usd', 57600000) - roi_data.get('implementation_cost', 100000)) / 1e6
        ]
        
        fig.add_trace(
            go.Bar(
                x=roi_categories,
                y=roi_values,
                marker_color=[COLORS["warning"], COLORS["success"], COLORS["primary"]],
                text=[f"${val:.1f}M" for val in roi_values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Grid Improvements
        grid_data = business_data.get('grid_improvements', {})
        grid_metrics = ['Stability Score', 'Frequency Regulation', 'Curtailment Reduction']
        grid_values = [
            grid_data.get('stability_score', 79.4),
            grid_data.get('frequency_regulation_improvement', 47),
            grid_data.get('renewable_curtailment_reduction', 4)
        ]
        
        fig.add_trace(
            go.Bar(
                x=grid_metrics,
                y=grid_values,
                marker_color=COLORS["success"],
                text=[f"{val:.1f}%" for val in grid_values],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Regulatory Compliance Indicator
        compliance_data = business_data.get('regulatory_compliance', {})
        compliance_score = compliance_data.get('cerc_overall', 75.0)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=compliance_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CERC Compliance (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': COLORS["primary"]},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 95], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "green", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=title,
            height=600,
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating business impact dashboard: {str(e)}")
        return go.Figure()

def create_correlation_heatmap(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    title: str = "Feature Correlation Matrix"
) -> go.Figure:
    """
    Create interactive correlation heatmap using Plotly.
    
    Args:
        df: DataFrame containing features
        features: List of features to include (if None, use all numeric features)
        title: Chart title
        
    Returns:
        go.Figure: Interactive correlation heatmap
    """
    try:
        # Select features
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[features].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>"
        ))
        
        fig = apply_theme_to_plotly_figure(fig, title)
        fig.update_layout(
            width=800,
            height=800,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure()

def create_time_series_plot(
    df: pd.DataFrame,
    date_col: str,
    value_cols: Union[str, List[str]],
    title: str = "Time Series Analysis"
) -> go.Figure:
    """
    Create interactive time series plot.
    
    Args:
        df: DataFrame containing time series data
        date_col: Name of date/datetime column
        value_cols: Column name(s) for values to plot
        title: Chart title
        
    Returns:
        go.Figure: Interactive time series plot
    """
    try:
        fig = go.Figure()
        
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        
        for i, col in enumerate(value_cols):
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                mode='lines',
                name=col,
                line=dict(color=DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                hovertemplate=f"<b>{col}</b><br>" +
                            "Date: %{x}<br>" +
                            "Value: %{y:.2f}<br>" +
                            "<extra></extra>"
            ))
        
        fig = apply_theme_to_plotly_figure(fig, title)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value",
            height=CHART_CONFIG["default_height"]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating time series plot: {str(e)}")
        return go.Figure()

def create_distribution_plot(
    data: Union[List, np.ndarray, pd.Series],
    title: str = "Data Distribution",
    bins: int = 50
) -> go.Figure:
    """
    Create histogram with distribution overlay.
    
    Args:
        data: Data to plot
        title: Chart title
        bins: Number of histogram bins
        
    Returns:
        go.Figure: Distribution plot
    """
    try:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name="Histogram",
            marker_color=COLORS["primary"],
            opacity=0.7,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"
        ))
        
        # Add distribution curve if enough data points
        if len(data) > 10:
            from scipy import stats
            # Calculate KDE
            x_range = np.linspace(min(data), max(data), 100)
            kde = stats.gaussian_kde(data)
            y_kde = kde(x_range)
            
            # Scale KDE to match histogram
            y_kde_scaled = y_kde * len(data) * (max(data) - min(data)) / bins
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde_scaled,
                mode='lines',
                name="Distribution",
                line=dict(color=COLORS["secondary"], width=3),
                hovertemplate="Value: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>"
            ))
        
        fig = apply_theme_to_plotly_figure(fig, title)
        fig.update_layout(
            xaxis_title="Value",
            yaxis_title="Frequency",
            height=CHART_CONFIG["default_height"]
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating distribution plot: {str(e)}")
        return go.Figure()

@st.cache_data(ttl=1800)
def create_model_comparison_table(performance_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Create formatted model comparison table for display.
    
    Args:
        performance_data: Dictionary containing model performance data
        
    Returns:
        pd.DataFrame: Formatted comparison table
    """
    try:
        table_data = []
        
        for model_name, metrics in performance_data.items():
            row = {
                'Model': model_name.replace('_', ' '),
                'MAPE (%)': f"{metrics.get('mape', 0):.2f}",
                'MAE (MW)': f"{metrics.get('mae', 0):.1f}",
                'RMSE (MW)': f"{metrics.get('rmse', 0):.1f}",
                'Category': metrics.get('category', 'Unknown'),
                'Target Met': '✅' if metrics.get('mape', 100) <= 5.0 else '❌'
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
        
    except Exception as e:
        logger.error(f"Error creating model comparison table: {str(e)}")
        return pd.DataFrame()

def style_metric_card(
    value: Union[str, float, int],
    label: str,
    delta: Optional[str] = None,
    color: str = "primary"
) -> str:
    """
    Create styled metric card HTML.
    
    Args:
        value: Metric value to display
        label: Metric label
        delta: Optional delta/change indicator
        color: Color theme (primary, success, warning, info)
        
    Returns:
        str: HTML string for styled metric card
    """
    try:
        color_value = COLORS.get(color, COLORS["primary"])
        
        delta_html = ""
        if delta:
            delta_html = f'<p style="margin: 0.3rem 0; font-size: 0.8rem; color: {color_value};">{delta}</p>'
        
        return f"""
        <div style="
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid {color_value};
            margin-bottom: 1rem;
            text-align: center;
        ">
            <h2 style="
                font-size: 2rem;
                font-weight: 700;
                color: {color_value};
                margin: 0 0 0.5rem 0;
            ">{value}</h2>
            <p style="
                font-size: 0.9rem;
                color: #666;
                margin: 0;
                font-weight: 500;
            ">{label}</p>
            {delta_html}
        </div>
        """
        
    except Exception as e:
        logger.error(f"Error creating metric card: {str(e)}")
        return f"<div>Error: {str(e)}</div>"
