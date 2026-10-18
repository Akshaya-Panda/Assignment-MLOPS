"""Shared fixtures: real cleaned data, a small trained pipeline and an API client."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.schemas import EXAMPLE_PATIENT
from cardiorisk import config
from cardiorisk.data import load_clean
from cardiorisk.features import build_pipeline, split_features_target


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    return load_clean(config.CLEAN_FILE)


@pytest.fixture(scope="session")
def fitted_pipeline(clean_df):
    X, y = split_features_target(clean_df)
    return build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y)


@pytest.fixture(scope="session")
def model_files(tmp_path_factory, fitted_pipeline):
    """A model.joblib + model_card.json pair in a temp dir (independent of models/)."""
    root = tmp_path_factory.mktemp("model")
    joblib.dump(fitted_pipeline, root / "model.joblib")
    card = {
        "model_version": "test-v0",
        "model_family": "logistic_regression",
        "decision_threshold": 0.5,
    }
    (root / "model_card.json").write_text(json.dumps(card))
    return root / "model.joblib", root / "model_card.json"


@pytest.fixture
def api_client(monkeypatch, model_files):
    model_path, card_path = model_files
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_CARD_PATH", str(card_path))
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def patient() -> dict:
    return dict(EXAMPLE_PATIENT)


def make_raw_frame(rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS)
