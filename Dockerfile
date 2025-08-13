FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY load_forecast_new/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY load_forecast_new/ ./load_forecast_new/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit dashboard
CMD ["streamlit", "run", "load_forecast_new/delhi_forecasting_dashboard/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
