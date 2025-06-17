import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Add the src directory to the path for importing custom modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import your actual data processor and model factory (uncomment and use when ready)
# from data_processor import DataProcessor
# from model_factory import ModelFactory

# Set page config
st.set_page_config(
    page_title="Load Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_metrics' not in st.session_state:
    st.session_state.current_metrics = None
if 'current_y_true' not in st.session_state:
    st.session_state.current_y_true = None
if 'current_y_pred' not in st.session_state:
    st.session_state.current_y_pred = None
if 'current_history' not in st.session_state:
    st.session_state.current_history = None

# --- Dummy Backend Functions (Replace with your actual implementations) ---
def get_available_models():
    """Dummy function to return available model names."""
    return ['GRU', 'LSTM', 'Bi-LSTM', 'CNN-LSTM', 'Improved']

def train_model(model_name, hyperparams_dict):
    """Dummy function to simulate model training."""
    st.info(f"Simulating training for {model_name} with {hyperparams_dict}")
    
    # Simulate history data
    epochs = hyperparams_dict.get('epochs', 100)
    history = {
        'loss': np.linspace(0.5, 0.1, epochs).tolist(),
        'val_loss': np.linspace(0.6, 0.15, epochs).tolist(),
        'mae': np.linspace(20, 5, epochs).tolist(),
        'val_mae': np.linspace(25, 7, epochs).tolist()
    }
    
    # Simulate metrics
    metrics = {
        'rmse': np.random.uniform(10, 15),
        'mae': np.random.uniform(7, 10),
        'mape': np.random.uniform(2, 5),
        'r2': np.random.uniform(0.85, 0.95)
    }
    
    # Simulate y_true and y_pred for plotting
    num_points = 100
    y_true = np.random.rand(num_points, 5) * 100 # 5 regions
    y_pred = y_true * (1 + np.random.uniform(-0.1, 0.1, size=y_true.shape))
    
    return history, metrics, y_true, y_pred

def plot_forecast(y_true, y_pred):
    """Dummy function to plot forecast."""
    fig = go.Figure()
    for i in range(y_true.shape[1]):
        fig.add_trace(go.Scatter(y=y_true[:, i], mode='lines', name=f'Actual Region {i+1}'))
        fig.add_trace(go.Scatter(y=y_pred[:, i], mode='lines', name=f'Predicted Region {i+1}', line=dict(dash='dash')))
    fig.update_layout(title='Dummy Forecast: Actual vs Predicted Load', xaxis_title='Time Step', yaxis_title='Load')
    st.plotly_chart(fig, use_container_width=True)

def plot_training_loss(history):
    """Dummy function to plot training loss."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title='Dummy Training Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig, use_container_width=True)
# --- End Dummy Backend Functions ---

def create_hyperparameter_inputs():
    """Create input widgets for model hyperparameters."""
    st.sidebar.header("Model Hyperparameters")
    
    hyperparams = {
        'epochs': st.sidebar.slider('Number of Epochs', 10, 200, 100),
        'learning_rate': st.sidebar.slider('Learning Rate', 0.0001, 0.01, 0.001, format="%.4f"),
        'gru_units': st.sidebar.slider('Number of GRU Units', 32, 256, 128, step=32),
        'batch_size': st.sidebar.selectbox('Batch Size', [16, 32, 64, 128, 256]),
        'optimizer': st.sidebar.selectbox('Optimizer', ['adam', 'rmsprop', 'sgd'])
    }
    
    return hyperparams

def display_metrics(metrics):
    """Display model metrics in a clean format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
    with col2:
        st.metric("MAE", f"{metrics['mae']:.4f}")
    with col3:
        st.metric("MAPE", f"{metrics['mape']:.2f}%")
    with col4:
        st.metric("R²", f"{metrics['r2']:.4f}")

def plot_metrics_comparison(training_history_list):
    """Create a bar chart comparing metrics across models."""
    if not training_history_list:
        return
    
    # Extract relevant data for plotting
    plot_data = []
    for entry in training_history_list:
        model_name = entry['model_name']
        metrics = entry['metrics']
        plot_data.append({
            'model_name': model_name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'mape': metrics['mape'],
            'r2': metrics['r2']
        })

    df = pd.DataFrame(plot_data)
    
    fig = go.Figure()
    metrics_to_plot = ['rmse', 'mae', 'mape', 'r2']
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric.upper(),
            x=df['model_name'],
            y=df[metric],
            text=[f'{v:.4f}' for v in df[metric]],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        xaxis_title='Model',
        yaxis_title='Metric Value',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def save_training_result(model_name, hyperparams, metrics, history, y_true, y_pred):
    """Save training results to session state and update current model."""
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'hyperparams': hyperparams,
        'metrics': metrics,
        'history': history,
        'y_true': y_true,
        'y_pred': y_pred
    }
    
    st.session_state.training_history.append(result)
    
    # Update current model details for immediate display
    st.session_state.current_model = model_name
    st.session_state.current_metrics = metrics
    st.session_state.current_y_true = y_true
    st.session_state.current_y_pred = y_pred
    st.session_state.current_history = history

def main():
    st.title("📊 Load Forecasting Dashboard")
    
    # Sidebar
    st.sidebar.title("Model Configuration")
    
    # Model selection
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox(
        "Select Model",
        available_models,
        index=0
    )
    
    # Hyperparameter inputs
    hyperparams = create_hyperparameter_inputs()
    
    # Training button
    if st.sidebar.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Train model (dummy call)
            history, metrics, y_true, y_pred = train_model(selected_model, hyperparams)
            
            # Save results (this also updates current_model, metrics, y_true, y_pred)
            save_training_result(selected_model, hyperparams, metrics, history, y_true, y_pred)
            
            st.success("Model training completed!")
    
    # Main content area - display results if a model has been trained
    if st.session_state.current_model:
        st.header(f"Results for {st.session_state.current_model}")
        
        # Display metrics
        display_metrics(st.session_state.current_metrics)
        
        # Create two columns for plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Forecast Plot")
            # Retrieve from session state
            if st.session_state.current_y_true is not None and st.session_state.current_y_pred is not None:
                plot_forecast(st.session_state.current_y_true, st.session_state.current_y_pred)
            else:
                st.info("No forecast data available. Train a model first.")
        
        with col2:
            st.subheader("Training Loss")
            # Retrieve from session state
            if st.session_state.current_history is not None:
                plot_training_loss(st.session_state.current_history)
            else:
                st.info("No training history available. Train a model first.")
    
    # Model comparison section
    if st.session_state.training_history:
        st.header("Model Comparison")
        
        # Convert history to DataFrame for display
        history_df = pd.DataFrame([
            {
                'Timestamp': h['timestamp'],
                'Model': h['model_name'],
                'RMSE': h['metrics']['rmse'],
                'MAE': h['metrics']['mae'],
                'MAPE': h['metrics']['mape'],
                'R²': h['metrics']['r2']
            }
            for h in st.session_state.training_history
        ])
        
        # Display comparison table
        st.dataframe(history_df, use_container_width=True)
        
        # Plot metrics comparison
        plot_metrics_comparison(st.session_state.training_history)

if __name__ == "__main__":
    main() 