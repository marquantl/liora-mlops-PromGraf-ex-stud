# Notes de rendu — Monitoring Bike Sharing (Prometheus & Grafana)

## Métriques exposées (`/metrics`)
- `api_requests_total` (Counter) — labels `endpoint`, `method`, `status_code`
- `api_request_duration_seconds` (Histogram) — mêmes labels (→ latence P95)
- `model_rmse_score`, `model_mae_score`, `model_r2_score` (Gauge) — mis à jour par `/evaluate`
- **Métriques au choix :**
  - `model_mape_score` (Gauge) : l'erreur moyenne **en pourcentage**. Choisie car, contrairement au RMSE, elle est indépendante de l'échelle de `cnt` → plus facile à interpréter et à comparer dans le temps.
  - `evidently_data_drift_detected_status` (Gauge, 0/1) : vaut 1 quand Evidently détecte une dérive des données (part de colonnes dérivées ≥ 50 %). C'est un **signal direct de data drift**, le cœur du monitoring ML.

## Endpoint `/evaluate`
Reçoit des données "courantes" (envoyées par `run_evaluation.py`), fait prédire le modèle, lance un rapport **Evidently** (`RMSE`, `MAE`, `R2Score` + `DataDriftPreset`) entre la **référence = janvier 2011** et ces données, en extrait les scores et met à jour les Gauges Prometheus.

## Alerting
- **Prometheus** (`deployment/prometheus/rules/alert_rules.yml`) :
  - `BikeApiDown` : l'API ne répond plus (`up == 0`).
  - `ModelRmseHigh` : `model_rmse_score > 300` (performance dégradée / dérive).
- **Grafana** : à créer dans l'UI (ex. sur `model_rmse_score` ou `model_mape_score`) — voir étape de test.

## `make fire-alert` — quelle alerte est testée ?
`make fire-alert` appelle `POST /trigger-drift`, qui force `model_rmse_score = 9999` et `drift = 1`.
→ cela déclenche l'alerte **`ModelRmseHigh`** (RMSE > 300). C'est l'alerte "métrique ML" demandée.

## Lancer le projet
```bash
make all          # démarre bike-api(8080), prometheus(9090), grafana(3000), node-exporter(9100)
make traffic      # génère du trafic sur /predict (remplit les métriques d'API)
make evaluation   # calcule RMSE/MAE/R2/MAPE/drift (met à jour les Gauges)
make fire-alert   # déclenche volontairement l'alerte RMSE
make stop         # arrête tout
```
> Au tout premier démarrage, `bike-api` télécharge le dataset UCI et entraîne le modèle : compter ~1 min avant que l'API réponde.

## Métrique "au choix" — rappel
Le choix (`model_mape_score` + statut de drift) est justifié dans les commentaires de `src/api/main.py`
et ci-dessus.
