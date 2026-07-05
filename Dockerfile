FROM python:3.11-slim

WORKDIR /app

# Python deps only — no system DB drivers needed (parquet-backed, no ODBC/Azure SQL).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + precomputed parquet data (app/data, ~51 MB) ship in the image.
COPY . .

# Cloud hosts inject $PORT; app.py binds to it (defaults to 8000 locally).
EXPOSE 8000
CMD ["python3", "app.py"]
