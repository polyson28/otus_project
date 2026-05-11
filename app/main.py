from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from urllib.parse import urlparse
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, Histogram, generate_latest

from app.health import health_payload
from app.model_loader import ModelBundle, load_model_bundle
from app.prediction_logger import log_prediction
from app.predictor import predict_from_features, predict_from_records
from app.schemas import FeaturesRequest, HealthResponse, PredictionResponse, RecordsRequest
from src.io.s3 import S3Error


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

REQUEST_COUNT = Gauge("request_count", "Total HTTP requests.", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("request_latency_seconds", "HTTP request latency in seconds.", ["method", "endpoint"])
PREDICTION_COUNT = Gauge("prediction_count", "Total successful predictions.", ["endpoint"])
PREDICTION_ERRORS_COUNT = Gauge("prediction_errors_count", "Total prediction errors.", ["endpoint"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Starting FastAPI service.")
    app.state.model_bundle = None
    try:
        app.state.model_bundle = load_model_bundle()
    except (RuntimeError, FileNotFoundError, S3Error) as error:
        LOGGER.error("Model load failed during startup. Service will stay up without a loaded model: %s", error)
        app.state.model_load_error = str(error)
    except Exception as error:
        LOGGER.exception("Model load failed during startup. Service will stay up without a loaded model.")
        app.state.model_load_error = str(error)
    else:
        app.state.model_load_error = None
    yield
    LOGGER.info("Stopping FastAPI service.")


app = FastAPI(title="TS Project Forecast API", version="1.0.0", lifespan=lifespan)


def _bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        detail = "Model is not loaded."
        load_error = getattr(request.app.state, "model_load_error", None)
        if load_error:
            detail = f"{detail} Startup load error: {load_error}"
        raise HTTPException(status_code=503, detail=detail)
    return bundle


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return str(path or request.url.path)


def _response_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("_")}


def _log_prediction_result(endpoint: str, bundle: ModelBundle, result: dict[str, Any]) -> None:
    features = result.get("_features")
    if not isinstance(features, dict):
        features = {}
    try:
        log_prediction(bundle=bundle, features=features, prediction=result["prediction"])
    except Exception as error:
        LOGGER.warning("Prediction logging failed: endpoint=%s error=%s", endpoint, error)


def _model_name(bundle: ModelBundle | None) -> str | None:
    if bundle is None:
        return None
    for key in ("model_name", "registered_model_name", "registered_model"):
        value = bundle.metadata.get(key)
        if value not in (None, ""):
            return str(value)
    if bundle.model_uri.startswith("models:/"):
        remainder = bundle.model_uri.removeprefix("models:/").lstrip("/")
        if "@" in remainder:
            return remainder.rsplit("@", 1)[0] or None
        if "/" in remainder:
            return remainder.rsplit("/", 1)[0] or None
        return remainder or None
    if bundle.model_uri.startswith("runs:/"):
        return None
    parsed = urlparse(bundle.model_uri)
    return os.path.splitext(os.path.basename(parsed.path or bundle.model_uri))[0] or None


@app.get("/")
def root(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "model_bundle", None)
    return {
        "service": "TS Project Forecast API",
        "status": "ok",
        "model_loaded": bundle is not None,
        "endpoints": [
            "/health",
            "/metadata",
            "/model_status",
            "/reload_model",
            "/predict_from_features",
            "/predict",
            "/metrics",
        ],
    }


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    endpoint = _endpoint_label(request)
    status_code = "500"
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        REQUEST_COUNT.labels(request.method, endpoint, status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(time.perf_counter() - start_time)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": "validation_error", "detail": exc.errors()},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled API error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> dict[str, Any]:
    return health_payload(getattr(request.app.state, "model_bundle", None))


@app.get("/metadata")
def metadata(request: Request) -> dict[str, Any]:
    return _bundle(request).metadata


@app.get("/model_status")
def model_status(request: Request) -> dict[str, Any]:
    bundle = getattr(request.app.state, "model_bundle", None)
    return {
        "model_name": _model_name(bundle),
        "model_version": bundle.model_version if bundle is not None else None,
        "loaded_at": bundle.loaded_at if bundle is not None else None,
        "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI"),
    }


@app.post("/reload_model")
def reload_model(request: Request) -> dict[str, Any]:
    old_bundle = getattr(request.app.state, "model_bundle", None)
    old_model_version = old_bundle.model_version if old_bundle is not None else None
    try:
        new_bundle = load_model_bundle()
    except Exception as error:
        LOGGER.exception("Model reload failed: %s", error)
        request.app.state.model_load_error = str(error)
        raise HTTPException(status_code=503, detail=f"Model reload failed: {error}") from error

    request.app.state.model_bundle = new_bundle
    request.app.state.model_load_error = None
    LOGGER.info(
        "Model reloaded: old_model_version=%s new_model_version=%s model_uri=%s",
        old_model_version,
        new_bundle.model_version,
        new_bundle.model_uri,
    )
    return {
        "status": "reloaded",
        "old_model_version": old_model_version,
        "new_model_version": new_bundle.model_version,
    }


@app.post("/predict_from_features", response_model=PredictionResponse)
def predict_features(payload: FeaturesRequest, request: Request) -> dict[str, Any]:
    endpoint = "/predict_from_features"
    LOGGER.info("Prediction request: endpoint=%s", endpoint)
    bundle = _bundle(request)
    try:
        result = predict_from_features(payload.features, bundle)
    except HTTPException as error:
        PREDICTION_ERRORS_COUNT.labels(endpoint).inc()
        LOGGER.error("Prediction error: endpoint=%s detail=%s", endpoint, error.detail)
        raise
    except ValueError as error:
        PREDICTION_ERRORS_COUNT.labels(endpoint).inc()
        LOGGER.error("Prediction validation error: endpoint=%s error=%s", endpoint, error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        PREDICTION_ERRORS_COUNT.labels(endpoint).inc()
        LOGGER.exception("Prediction error: endpoint=%s error=%s", endpoint, error)
        raise HTTPException(status_code=400, detail=str(error)) from error

    PREDICTION_COUNT.labels(endpoint).inc()
    _log_prediction_result(endpoint, bundle, result)
    return _response_payload(result)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: RecordsRequest, request: Request) -> dict[str, Any]:
    endpoint = "/predict"
    LOGGER.info("Prediction request: endpoint=%s rows=%s", endpoint, len(payload.records))
    bundle = _bundle(request)
    try:
        result = predict_from_records(payload.records, bundle)
    except HTTPException as error:
        PREDICTION_ERRORS_COUNT.labels(endpoint).inc()
        LOGGER.error("Prediction error: endpoint=%s detail=%s", endpoint, error.detail)
        raise
    except ValueError as error:
        PREDICTION_ERRORS_COUNT.labels(endpoint).inc()
        LOGGER.error("Prediction validation error: endpoint=%s error=%s", endpoint, error)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        PREDICTION_ERRORS_COUNT.labels(endpoint).inc()
        LOGGER.exception("Prediction error: endpoint=%s error=%s", endpoint, error)
        raise HTTPException(status_code=400, detail=str(error)) from error

    PREDICTION_COUNT.labels(endpoint).inc()
    _log_prediction_result(endpoint, bundle, result)
    return _response_payload(result)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
