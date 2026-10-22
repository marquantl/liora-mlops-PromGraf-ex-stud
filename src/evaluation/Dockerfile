FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_evaluation.py .

# Récupère les données de février et interroge l'endpoint /evaluate de l'API
CMD ["python", "run_evaluation.py"]
