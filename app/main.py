"""CardioRisk prediction API.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment:
    MODEL_PATH       path to the joblib pipeline (default models/model.joblib)
    MODEL_CARD_PATH  path to model_card.json   (default: next to MODEL_PATH)
    LOG_LEVEL        INFO (default) or DEBUG; DEBUG also logs probe requests
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import metrics
from app.schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse
from cardiorisk import __version__, config
from cardiorisk.predict import load_model, load_model_card, predict_records

log = logging.getLogger("cardiorisk.api")
QUIET_ROUTES = {"/health", "/ready", "/metrics"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, so logs are easy to grep, ship and query."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    log.handlers[:] = [handler]
    log.setLevel(level.upper())
    log.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    model_path = Path(os.getenv("MODEL_PATH", str(config.MODEL_PATH)))
    card_path = Path(os.getenv("MODEL_CARD_PATH", str(model_path.with_name("model_card.json"))))
    app.state.model, app.state.card = None, {}
    try:
        app.state.model = load_model(model_path)
        app.state.card = load_model_card(card_path)
    except (OSError, EOFError, ValueError) as exc:
        # Stay up so /health answers, but report not-ready until a model is present.
        log.error(
            "model_load_failed", extra={"fields": {"path": str(model_path), "error": str(exc)}}
        )
    version = app.state.card.get("model_version", "unknown")
    metrics.MODEL_LOADED.set(1 if app.state.model is not None else 0)
    if app.state.model is not None:
        metrics.MODEL_INFO.labels(version, app.state.card.get("model_family", "unknown")).set(1)
        log.info("model_loaded", extra={"fields": {"path": str(model_path), "version": version}})
    yield
    metrics.MODEL_LOADED.set(0)


app = FastAPI(
    title="CardioRisk API",
    version=__version__,
    description="Predicts heart-disease risk from 13 clinical features (UCI Heart Disease).",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    metrics.IN_PROGRESS.inc()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        metrics.IN_PROGRESS.dec()
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        # Use the route template (not the raw path) to keep metric cardinality bounded.
        route_path = getattr(route, "path", "unmatched")
        metrics.REQUESTS.labels(request.method, route_path, str(status)).inc()
        metrics.LATENCY.labels(request.method, route_path).observe(elapsed)
        level = logging.DEBUG if route_path in QUIET_ROUTES and status < 400 else logging.INFO
        log.log(
            level,
            "http_request",
            extra={
                "fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "latency_ms": round(elapsed * 1000, 2),
                    "client": request.client.host if request.client else None,
                }
            },
        )


def _require_model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _predict(request: Request, patients: list[PatientFeatures]) -> list[dict]:
    model = _require_model(request)
    threshold = request.app.state.card.get("decision_threshold", 0.5)
    start = time.perf_counter()
    results = predict_records(model, [p.model_dump() for p in patients], threshold)
    metrics.INFERENCE.observe(time.perf_counter() - start)
    for result in results:
        metrics.PREDICTIONS.labels(result["label"], result["risk_band"]).inc()
        metrics.PROBABILITY.observe(result["probability_disease"])
        # Log outcomes only: raw clinical inputs are health data and stay out of logs.
        log.info(
            "prediction",
            extra={
                "fields": {
                    "request_id": request.state.request_id,
                    "prediction": result["prediction"],
                    "probability_disease": result["probability_disease"],
                    "risk_band": result["risk_band"],
                }
            },
        )
    return results


@app.get("/", tags=["service"])
def root(request: Request) -> dict:
    return {
        "service": "cardiorisk-api",
        "version": __version__,
        "model_version": request.app.state.card.get("model_version", "unknown"),
        "docs": "/docs",
    }


@app.get("/health", tags=["service"])
def health() -> dict:
    """Liveness: the process is up (does not require a model)."""
    return {"status": "ok"}


@app.get("/ready", tags=["service"])
def ready(request: Request) -> dict:
    """Readiness: a model is loaded and requests can be served."""
    _require_model(request)
    return {"status": "ready", "model_version": request.app.state.card.get("model_version")}


@app.get("/model-info", tags=["model"])
def model_info(request: Request) -> dict:
    _require_model(request)
    card = request.app.state.card
    keys = ("model_name", "model_version", "model_family", "trained_at", "best_params",
            "decision_threshold", "features", "metrics", "sklearn_version")  # fmt: skip
    return {k: card[k] for k in keys if k in card}


@app.post("/predict", response_model=PredictionResponse, tags=["model"])
def predict(patient: PatientFeatures, request: Request) -> dict:
    """Predict heart-disease risk for one patient."""
    result = _predict(request, [patient])[0]
    return {
        **result,
        "model_version": request.app.state.card.get("model_version", "unknown"),
        "request_id": request.state.request_id,
    }


@app.post("/predict/batch", response_model=BatchResponse, tags=["model"])
def predict_batch(batch: BatchRequest, request: Request) -> dict:
    """Predict for up to 100 patients in one call."""
    return {
        "predictions": _predict(request, batch.instances),
        "model_version": request.app.state.card.get("model_version", "unknown"),
        "request_id": request.state.request_id,
    }


@app.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
