FROM python:3.11-slim

WORKDIR /app
COPY requirements-pipeline.txt requirements-app.txt ./
RUN pip install --no-cache-dir -r requirements-app.txt

COPY pipeline/ ./pipeline/
COPY app/ ./app/
COPY data/posoco/ ./data/posoco/
COPY reports/ ./reports/

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
