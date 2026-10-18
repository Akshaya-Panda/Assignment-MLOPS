"""Feature engineering and preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk import config
from cardiorisk.features import (
    ENGINEERED_FEATURES,
    ClinicalFeatureAdder,
    build_preprocessor,
    split_features_target,
)


def _frame(**overrides) -> pd.DataFrame:
    row = dict.fromkeys(config.FEATURE_COLUMNS, 1.0)
    row.update(age=60.0, thalach=128.0, trestbps=135.0, cp=4.0, thal=7.0, slope=2.0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_pct_max_hr_is_share_of_age_predicted_maximum():
    out = ClinicalFeatureAdder().fit_transform(_frame(age=60.0, thalach=128.0))
    assert out.loc[0, "pct_max_hr"] == pytest.approx(128 / 160)


@pytest.mark.parametrize(
    ("trestbps", "stage"), [(110, 0), (119.9, 0), (120, 1), (129, 1), (130, 2), (140, 3), (180, 3)]
)
def test_bp_stage_follows_acc_aha_thresholds(trestbps, stage):
    out = ClinicalFeatureAdder().fit_transform(_frame(trestbps=float(trestbps)))
    assert out.loc[0, "bp_stage"] == stage


def test_missing_inputs_propagate_as_nan():
    out = ClinicalFeatureAdder().fit_transform(_frame(trestbps=np.nan, thalach=np.nan))
    assert out[ENGINEERED_FEATURES].isna().all(axis=None)


def test_adder_does_not_mutate_input():
    frame = _frame()
    ClinicalFeatureAdder().fit_transform(frame)
    assert list(frame.columns) == config.FEATURE_COLUMNS


def test_adder_requires_all_feature_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ClinicalFeatureAdder().fit(_frame().drop(columns="age"))


def test_preprocessor_output_is_complete_and_numeric(clean_df):
    X, _ = split_features_target(clean_df)
    Xt = build_preprocessor().fit_transform(ClinicalFeatureAdder().fit_transform(X))
    # 8 scaled numeric (6 raw + 2 engineered) + 3 binary + 13 one-hot (cp 4, restecg 3,
    # slope 3, thal 3)
    assert Xt.shape == (len(X), 24)
    assert not np.isnan(Xt).any()


def test_preprocessor_ignores_unseen_categories(clean_df):
    X, _ = split_features_target(clean_df)
    pre = build_preprocessor().fit(ClinicalFeatureAdder().fit_transform(X))
    unseen = ClinicalFeatureAdder().transform(_frame(restecg=9.0))
    assert pre.transform(unseen).shape == (1, 24)


def test_scaled_numeric_features_are_standardised(clean_df):
    X, _ = split_features_target(clean_df)
    Xt = build_preprocessor().fit_transform(ClinicalFeatureAdder().fit_transform(X))
    numeric = Xt[:, :8]
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)
