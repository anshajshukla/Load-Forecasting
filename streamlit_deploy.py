"""
Quick Streamlit Deployment Script
This script helps deploy the dashboard using different methods
"""

import os
import subprocess
import sys

def run_local_dashboard():
    """Run the dashboard locally"""
    print("🚀 Starting Local Dashboard...")
    os.chdir("load_forecast_new/delhi_forecasting_dashboard")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "main.py", "--server.port", "8501"])

def show_deployment_options():
    """Show all deployment options"""
    print("""
🚀 DELHI LOAD FORECASTING DASHBOARD DEPLOYMENT OPTIONS

📁 Repository: anshajshukla/Load-Forecasting
📄 Main File: load_forecast_new/delhi_forecasting_dashboard/main.py

🌟 OPTION 1: Streamlit Cloud (Free)
   - URL: https://streamlit.io/cloud
   - Repository: anshajshukla/Load-Forecasting
   - Branch: main
   - Main file: load_forecast_new/delhi_forecasting_dashboard/main.py

🐳 OPTION 2: Local Docker
   - Command: docker build -t delhi-dashboard .
   - Run: docker run -p 8501:8501 delhi-dashboard

☁️ OPTION 3: Heroku
   - Command: heroku create delhi-load-forecasting
   - Deploy: git push heroku main

🖥️ OPTION 4: Local Development
   - Run this script to start locally
    """)

if __name__ == "__main__":
    choice = input("""
Select deployment option:
1. Run Local Dashboard
2. Show All Deployment Options
Enter choice (1 or 2): """)
    
    if choice == "1":
        run_local_dashboard()
    else:
        show_deployment_options()
