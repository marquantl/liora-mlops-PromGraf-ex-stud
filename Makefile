# Docker Compose v2 ("docker compose"). Si ta machine n'a que l'ancien binaire,
# remplace "docker compose" par "docker-compose".
COMPOSE = docker compose

all:
	$(COMPOSE) up --build -d bike-api prometheus grafana node-exporter

stop:
	$(COMPOSE) down --remove-orphans

# Arrêt COMPLET + suppression des volumes (repart de zéro : réentraînement, dashboards réimportés)
reset:
	$(COMPOSE) down -v --remove-orphans

# Voir l'état des conteneurs du projet
ps:
	$(COMPOSE) ps

# Le modèle s'entraîne au démarrage de l'API ; cette cible (re)construit juste l'API.
train:
	$(COMPOSE) up --build -d bike-api

# Lance le script d'évaluation (va chercher février et appelle /evaluate -> maj des métriques)
evaluation:
	$(COMPOSE) up --build evaluation

# Génère du trafic sur /predict
traffic:
	bash scripts/generate_traffic.sh 50

# Déclenche volontairement l'alerte "ModelRmseHigh" (force model_rmse_score=9999 via /trigger-drift)
fire-alert:
	curl -X POST http://localhost:8080/trigger-drift
