# --- Librairies standard ---
import io
import time
import zipfile
import datetime
import logging
import warnings
from typing import Any, Optional

# --- Librairies data / ML ---
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

# --- Evidently (servira plus tard, pour l'endpoint /evaluate) ---
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset

# --- API & monitoring ---
from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge

warnings.filterwarnings('ignore')

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Prometheus Metrics Definitions ---
# Son propre registre = on n'expose QUE nos métriques (pas celles par défaut de Python).
registry = CollectorRegistry()

# Métriques d'APPLICATION (API)
api_requests_total = Counter(
    'api_requests_total', 'Total number of API requests',
    ['endpoint', 'method', 'status_code'], registry=registry)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds', 'API request duration in seconds',
    ['endpoint', 'method', 'status_code'], registry=registry)

# Métriques de MODÈLE (régression) — mises à jour par /evaluate (ce sont des Gauge car elles varient)
model_rmse_score = Gauge('model_rmse_score', 'Model RMSE (via /evaluate)', registry=registry)
model_mae_score = Gauge('model_mae_score', 'Model MAE (via /evaluate)', registry=registry)
model_r2_score = Gauge('model_r2_score', 'Model R2 score (via /evaluate)', registry=registry)

# Métriques "au choix" :
# - MAPE : erreur en POURCENTAGE, plus parlante que le RMSE (indépendante de l'échelle de cnt)
# - drift status : 1 si Evidently détecte une dérive des données, 0 sinon -> signal direct de drift
model_mape_score = Gauge('model_mape_score', 'Model MAPE in % (via /evaluate)', registry=registry)
data_drift_detected = Gauge('evidently_data_drift_detected_status',
                            '1 if Evidently detected dataset drift, else 0', registry=registry)



# --- Global Variables for Model and Data ---
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']

# Rempli au démarrage de l'API (voir l'événement 'startup' plus bas)
MODEL = None
REFERENCE_DATA = None


# --- Data Ingestion and Preparation Functions ---
def _fetch_data() -> pd.DataFrame:
    """Télécharge le dataset Bike Sharing (UCI) et renvoie le DataFrame horaire ('hour.csv')."""
    logger.info("Téléchargement du dataset Bike Sharing...")
    content = requests.get(DATASET_URL, verify=False, timeout=60).content
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        df = pd.read_csv(z.open("hour.csv"), header=0, sep=',', parse_dates=['dteday'])
    return df


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Pose un index temporel (date + heure) pour pouvoir sélectionner par période (ex: janvier)."""
    raw_data['hr'] = raw_data['hr'].astype(int)
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(row['dteday'].date(), datetime.time(row.hr)),
        axis=1
    )
    return raw_data.sort_index()


def _train_and_predict_reference_model(data: pd.DataFrame):
    """
    Entraîne le RandomForest sur les données de JANVIER 2011 (période de référence).
    Renvoie (le modèle, les données de référence AVEC une colonne 'prediction').
    Cette colonne 'prediction' servira de point de comparaison pour Evidently (étape /evaluate).
    """
    reference = data.loc['2011-01-01 00:00:00':'2011-01-31 23:00:00'].copy()
    model = RandomForestRegressor(random_state=42)
    model.fit(reference[NUM_FEATS + CAT_FEATS], reference[TARGET])
    reference[PREDICTION] = model.predict(reference[NUM_FEATS + CAT_FEATS])
    logger.info(f"Modèle entraîné sur {reference.shape[0]} lignes de janvier 2011.")
    return model, reference


# --- Entraînement au démarrage de l'API (une seule fois) ---
@app.on_event("startup")
def load_model_and_reference():
    """Au lancement du conteneur : on récupère les données, on prépare, on entraîne le modèle UNE fois."""
    global MODEL, REFERENCE_DATA
    raw = _process_data(_fetch_data())
    MODEL, REFERENCE_DATA = _train_and_predict_reference_model(raw)
    logger.info("Modèle et données de référence prêts.")


# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}


@app.post("/predict", response_model=PredictionOutput)
async def predict(features: BikeSharingInput):
    """Prédit le nombre de vélos (cnt) pour un enregistrement."""
    start_time = time.time()
    status_code = "200"
    try:
        # On transforme l'entrée en DataFrame d'1 ligne et on garde les colonnes du modèle
        # (dteday est ignorée : elle ne sert pas à la prédiction).
        X = pd.DataFrame([features.model_dump()])[NUM_FEATS + CAT_FEATS]
        prediction = float(MODEL.predict(X)[0])
        return PredictionOutput(predicted_count=prediction)
    except Exception as e:
        status_code = "500"
        logger.error(f"Erreur pendant /predict : {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        # On enregistre les métriques dans TOUS les cas (succès ou erreur).
        duration = time.time() - start_time
        api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()


def _run_evidently_report(current_df: pd.DataFrame):
    """Construit le rapport Evidently (régression + drift) entre la référence (janvier) et les données courantes."""
    data_definition = DataDefinition(
        numerical_columns=NUM_FEATS,
        categorical_columns=CAT_FEATS,
        regression=[Regression(target=TARGET, prediction=PREDICTION)]
    )
    # On restreint référence ET données courantes aux MÊMES colonnes (sinon Evidently compare
    # des tableaux de formes différentes : la référence 'hour.csv' a des colonnes en plus).
    cols = NUM_FEATS + CAT_FEATS + [TARGET, PREDICTION]
    ref_ds = Dataset.from_pandas(REFERENCE_DATA[cols], data_definition=data_definition)
    cur_ds = Dataset.from_pandas(current_df[cols], data_definition=data_definition)
    report = Report(metrics=[RMSE(), MAE(), R2Score(), DataDriftPreset()])
    return report.run(current_data=cur_ds, reference_data=ref_ds)


def _extract_scores(snapshot) -> dict:
    """Lit RMSE, MAE, R2 et la part de colonnes en dérive dans le rapport Evidently."""
    scores = {'rmse': None, 'mae': None, 'r2': None, 'drift_share': 0.0}
    for m in snapshot.dict()['metrics']:
        name, value = m['metric_name'], m['value']
        if name.startswith('RMSE'):
            scores['rmse'] = float(value)
        elif name.startswith('MAE'):                       # MAE renvoie {'mean':..,'std':..}
            scores['mae'] = float(value['mean']) if isinstance(value, dict) else float(value)
        elif name.startswith('R2Score'):
            scores['r2'] = float(value)
        elif name.startswith('DriftedColumnsCount'):       # {'count':.., 'share':..}
            scores['drift_share'] = float(value.get('share', 0.0))
    return scores


@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(payload: EvaluationData):
    """Évalue le modèle sur des données 'courantes', calcule les scores Evidently et met à jour les métriques."""
    start_time = time.time()
    status_code = "200"
    try:
        current = pd.DataFrame(payload.data)
        if current.empty:
            status_code = "400"
            raise HTTPException(status_code=400, detail="No data provided for evaluation.")

        # 1) prédictions du modèle sur les données courantes
        current[PREDICTION] = MODEL.predict(current[NUM_FEATS + CAT_FEATS])

        # 2) rapport Evidently (régression + dérive des données)
        snapshot = _run_evidently_report(current)
        scores = _extract_scores(snapshot)

        # 3) MAPE (métrique au choix) : erreur moyenne en %
        mape = float(mean_absolute_percentage_error(current[TARGET], current[PREDICTION]) * 100)
        # dérive détectée si au moins la moitié des colonnes ont dérivé (seuil Evidently par défaut)
        drift = 1 if scores['drift_share'] >= 0.5 else 0

        # 4) mise à jour des métriques Prometheus (Gauge -> .set())
        model_rmse_score.set(scores['rmse'])
        model_mae_score.set(scores['mae'])
        model_r2_score.set(scores['r2'])
        model_mape_score.set(mape)
        data_drift_detected.set(drift)

        logger.info(f"[{payload.evaluation_period_name}] RMSE={scores['rmse']:.2f} "
                    f"MAE={scores['mae']:.2f} R2={scores['r2']:.3f} MAPE={mape:.1f}% drift={drift}")
        return EvaluationReportOutput(
            message=f"Evaluation completed for {payload.evaluation_period_name}",
            rmse=scores['rmse'], mape=mape, mae=scores['mae'], r2score=scores['r2'],
            drift_detected=drift, evaluated_items=len(current)
        )
    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        status_code = "500"
        logger.error(f"Erreur pendant /evaluate : {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    finally:
        duration = time.time() - start_time
        api_request_duration_seconds.labels(endpoint="/evaluate", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/evaluate", method="POST", status_code=status_code).inc()


@app.post("/trigger-drift")
async def trigger_drift():
    """
    Force volontairement des métriques 'dégradées' pour déclencher l'alerte (cible Makefile 'fire-alert').
    On simule un modèle qui déraille : RMSE énorme + flag de dérive à 1.
    """
    model_rmse_score.set(9999.0)
    data_drift_detected.set(1)
    api_requests_total.labels(endpoint="/trigger-drift", method="POST", status_code="200").inc()
    logger.warning("DRIFT SIMULÉ : model_rmse_score=9999 et drift=1 (test d'alerte).")
    return {"message": "Drift simulated: model_rmse_score=9999 and drift flag=1."}


@app.get("/metrics")
async def metrics():
    """Expose les métriques au format texte que Prometheus vient scraper."""
    return Response(content=generate_latest(registry), media_type="text/plain")

