"""HTTP contract of the FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_health_and_readiness(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}
    ready = api_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["model_version"] == "test-v0"


def test_predict_returns_prediction_and_confidence(api_client, patient):
    response = api_client.post("/predict", json=patient)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability_disease"] <= 1
    assert 0.5 <= body["confidence"] <= 1
    assert body["risk_band"] in {"low", "moderate", "high"}
    assert body["model_version"] == "test-v0"
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_high_risk_example_is_flagged(api_client, patient):
    assert api_client.post("/predict", json=patient).json()["label"] == "heart_disease"


def test_optional_fields_can_be_omitted(api_client, patient):
    del patient["ca"], patient["thal"]
    assert api_client.post("/predict", json=patient).status_code == 200


def test_client_request_id_is_echoed(api_client, patient):
    response = api_client.post("/predict", json=patient, headers={"X-Request-ID": "abc-123"})
    assert response.json()["request_id"] == "abc-123"


@pytest.mark.parametrize(
    "change",
    [
        {"age": 5},  # below plausible range
        {"cp": 9},  # not a valid chest-pain code
        {"chol": "high"},  # wrong type
        {"unexpected": 1},  # unknown fields are rejected
    ],
)
def test_invalid_input_is_rejected(api_client, patient, change):
    assert api_client.post("/predict", json={**patient, **change}).status_code == 422


def test_missing_required_field_is_rejected(api_client, patient):
    del patient["age"]
    assert api_client.post("/predict", json=patient).status_code == 422


def test_batch_prediction(api_client, patient):
    response = api_client.post("/predict/batch", json={"instances": [patient] * 3})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 3


def test_empty_batch_is_rejected(api_client):
    assert api_client.post("/predict/batch", json={"instances": []}).status_code == 422


def test_metrics_endpoint_counts_predictions(api_client, patient):
    api_client.post("/predict", json=patient)
    text = api_client.get("/metrics").text
    assert "cardiorisk_predictions_total" in text
    assert 'route="/predict"' in text
    assert "cardiorisk_model_loaded 1.0" in text


def test_model_info_exposes_card(api_client):
    assert api_client.get("/model-info").json()["model_family"] == "logistic_regression"


def test_service_reports_not_ready_without_model(monkeypatch, tmp_path, patient):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("MODEL_CARD_PATH", str(tmp_path / "missing.json"))
    from app.main import app

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=patient).status_code == 503
