#!/usr/bin/env bash
# Génère du trafic simple sur l'endpoint /predict pour remplir les métriques d'API.
set -e
N="${1:-50}"
echo "Envoi de $N requêtes vers http://localhost:8080/predict ..."
for i in $(seq 1 "$N"); do
  curl -s -o /dev/null -X POST http://localhost:8080/predict \
    -H 'Content-Type: application/json' \
    -d '{"temp":0.24,"atemp":0.2879,"hum":0.81,"windspeed":0.0,"mnth":1,"hr":0,"weekday":6,"season":1,"holiday":0,"workingday":0,"weathersit":1,"dteday":"2011-01-01"}'
done
echo "Terminé : $N requêtes envoyées."
