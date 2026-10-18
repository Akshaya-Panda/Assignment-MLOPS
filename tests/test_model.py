"""Model building, evaluation, inference helpers and the training entry point."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest
from sklearn.model_selection import ParameterGrid

from cardiorisk import config, train
from cardiorisk.evaluation import compute_metrics
from cardiorisk.features import build_pipeline, split_features_target
from cardiorisk.predict import predict_records, risk_band


def test_compute_metrics_on_known_predictions():
    y = [0, 0, 1, 1]
    proba = [0.1, 0.6, 0.4, 0.9]  # one FP, one FN at threshold 0.5
    m = compute_metrics(y, proba)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["specificity"] == 0.5
    assert m["false_negatives"] == 1
    assert m["roc_auc"] == 0.75


def test_compute_metrics_perfect_classifier():
    m = compute_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert m["accuracy"] == m["roc_auc"] == m["f1"] == 1.0


@pytest.mark.parametrize("quick", [True, False])
def test_every_grid_parameter_is_valid_for_its_pipeline(quick):
    for name, (estimator, grid) in train.model_specs(quick).items():
        for params in ParameterGrid(grid):
            build_pipeline(estimator).set_params(**params)  # raises on a typo
        assert name in {"logistic_regression", "random_forest", "gradient_boosting"}


def test_pipeline_learns_signal(fitted_pipeline, clean_df):
    X, y = split_features_target(clean_df)
    proba = fitted_pipeline.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert compute_metrics(y, proba)["roc_auc"] > 0.85


def test_pipeline_handles_missing_optional_fields(fitted_pipeline, patient):
    patient.pop("ca")
    patient.pop("thal")
    result = predict_records(fitted_pipeline, [patient])[0]
    assert 0 <= result["probability_disease"] <= 1


def test_serialised_model_gives_identical_predictions(tmp_path, fitted_pipeline, clean_df):
    X, _ = split_features_target(clean_df)
    joblib.dump(fitted_pipeline, tmp_path / "m.joblib")
    reloaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_array_equal(reloaded.predict_proba(X), fitted_pipeline.predict_proba(X))


def test_predict_records_contract(fitted_pipeline, patient):
    result = predict_records(fitted_pipeline, [patient])[0]
    assert set(result) == {"prediction", "label", "probability_disease", "confidence", "risk_band"}
    assert result["confidence"] >= 0.5
    assert result["label"] == ("heart_disease" if result["prediction"] else "no_heart_disease")


@pytest.mark.parametrize(
    ("p", "band"),
    [(0.0, "low"), (0.29, "low"), (0.3, "moderate"), (0.59, "moderate"), (0.6, "high")],
)
def test_risk_band_boundaries(p, band):
    assert risk_band(p) == band


@pytest.mark.slow
def test_training_run_end_to_end(tmp_path, monkeypatch):
    """Quick-mode training: MLflow logging, registry, model export and report files."""
    monkeypatch.setattr(
        config, "MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    )
    monkeypatch.setattr(config, "MLFLOW_ARTIFACT_ROOT", (tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "models" / "model.joblib")
    monkeypatch.setattr(config, "MODEL_CARD_PATH", tmp_path / "models" / "model_card.json")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "reports" / "figures")

    card = train.run(quick=True)

    assert card["model_family"] in {"logistic_regression", "random_forest"}
    assert card["metrics"]["test_roc_auc"] > 0.8
    assert json.loads((tmp_path / "models" / "model_card.json").read_text()) == card
    assert joblib.load(tmp_path / "models" / "model.joblib").predict_proba is not None
    assert (tmp_path / "reports" / "model_comparison.csv").exists()
    assert (tmp_path / "reports" / "figures" / "champion_roc_curve.png").exists()
