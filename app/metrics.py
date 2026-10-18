"""Prometheus metrics exposed on /metrics (one process per pod, default registry)."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "cardiorisk_http_requests_total", "HTTP requests handled", ["method", "route", "status"]
)
LATENCY = Histogram(
    "cardiorisk_http_request_duration_seconds",
    "End-to-end request latency",
    ["method", "route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
IN_PROGRESS = Gauge("cardiorisk_http_requests_in_progress", "Requests currently being served")
PREDICTIONS = Counter(
    "cardiorisk_predictions_total", "Predictions returned", ["label", "risk_band"]
)
PROBABILITY = Histogram(
    "cardiorisk_prediction_probability",
    "Predicted probability of heart disease (drift indicator)",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
INFERENCE = Histogram(
    "cardiorisk_inference_duration_seconds",
    "Model predict_proba time per request",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
MODEL_LOADED = Gauge("cardiorisk_model_loaded", "1 if a model is loaded and ready")
MODEL_INFO = Gauge(
    "cardiorisk_model_info", "Loaded model metadata (value is always 1)", ["version", "family"]
)
