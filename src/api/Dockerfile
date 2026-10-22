FROM python:3.11-slim

WORKDIR /app

# Dépendances d'abord (meilleur cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'API
COPY main.py .

EXPOSE 8080

# Lancement de l'API (le modèle s'entraîne au démarrage via l'événement startup)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
