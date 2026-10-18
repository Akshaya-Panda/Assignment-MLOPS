"""Feature engineering and preprocessing, packaged as scikit-learn transformers.

Everything that turns raw clinical inputs into model features lives in one
Pipeline, so training, evaluation and the API apply identical transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cardiorisk import config

ENGINEERED_FEATURES = ["pct_max_hr", "bp_stage"]

# ACC/AHA 2017 systolic thresholds: normal <120, elevated 120-129,
# stage 1 hypertension 130-139, stage 2 hypertension >=140.
BP_STAGE_EDGES = [120, 130, 140]


class ClinicalFeatureAdder(BaseEstimator, TransformerMixin):
    """Adds age-normalised heart-rate response and blood-pressure stage.

    pct_max_hr: thalach / (220 - age), the share of age-predicted maximum
        heart rate reached during the exercise test.
    bp_stage:   0-3 resting blood-pressure category (ACC/AHA 2017).
    Missing inputs propagate as NaN and are imputed downstream.
    """

    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in config.FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing columns: {missing}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["pct_max_hr"] = X["thalach"] / (220 - X["age"]).clip(lower=1)
        stage = np.digitize(X["trestbps"], BP_STAGE_EDGES).astype(float)
        X["bp_stage"] = np.where(X["trestbps"].isna(), np.nan, stage)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else config.FEATURE_COLUMNS
        return np.array(base + ENGINEERED_FEATURES, dtype=object)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, config.NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("binary", binary, config.BINARY_FEATURES),
            ("categorical", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Full inference pipeline: raw clinical columns in, class probabilities out."""
    return Pipeline(
        [
            ("clinical", ClinicalFeatureAdder()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURE_COLUMNS].astype(float), df[config.TARGET].astype(int)
