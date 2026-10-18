"""Data acquisition, cleaning and validation."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile

import numpy as np
import pytest

from cardiorisk import config, data
from tests.conftest import make_raw_frame

VALID_ROW = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0]


def test_load_raw_treats_question_mark_as_missing(tmp_path):
    raw = tmp_path / "raw.data"
    raw.write_text("63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,?,?,0\n")
    df = data.load_raw(raw)
    assert list(df.columns) == config.RAW_COLUMNS
    assert df[["ca", "thal"]].isna().all(axis=None)


@pytest.mark.parametrize(("num", "expected"), [(0, 0), (1, 1), (2, 1), (3, 1), (4, 1)])
def test_clean_binarises_target(num, expected):
    df = data.clean(make_raw_frame([VALID_ROW[:-1] + [num]]))
    assert "num" not in df.columns
    assert df[config.TARGET].tolist() == [expected]


def test_clean_nulls_out_invalid_codes_and_implausible_values():
    bad = list(VALID_ROW)
    bad[2] = 9  # cp has no code 9
    bad[4] = 0  # cholesterol of 0 is a recording error
    df = data.clean(make_raw_frame([bad]))
    assert np.isnan(df.loc[0, "cp"])
    assert np.isnan(df.loc[0, "chol"])
    assert df.loc[0, "age"] == 63


def test_clean_drops_duplicates_and_rows_without_label():
    no_label = VALID_ROW[:-1] + [None]
    df = data.clean(make_raw_frame([VALID_ROW, VALID_ROW, no_label]))
    assert len(df) == 1


def test_clean_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        data.clean(make_raw_frame([VALID_ROW]).drop(columns="thal"))


def test_validate_rejects_non_binary_target(clean_df):
    broken = clean_df.copy()
    broken.loc[0, config.TARGET] = 2
    with pytest.raises(ValueError, match="binary"):
        data.validate(broken)


def test_validate_rejects_invalid_codes(clean_df):
    broken = clean_df.copy()
    broken.loc[0, "thal"] = 5
    with pytest.raises(ValueError, match="thal"):
        data.validate(broken)


def test_committed_dataset_matches_expected_profile(clean_df):
    assert clean_df.shape == (303, 14)
    assert clean_df[config.TARGET].sum() == 139
    assert clean_df.isna().sum().sum() == 6  # 4 x ca + 2 x thal
    data.validate(clean_df)


def test_save_and_load_round_trip(tmp_path, clean_df):
    path = data.save_clean(clean_df, tmp_path / "clean.csv")
    reloaded = data.load_clean(path)
    assert reloaded.shape == clean_df.shape
    assert np.allclose(reloaded.fillna(-1), clean_df.fillna(-1))


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_fetch", lambda url, timeout=30: b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.download(tmp_path / "raw.data", force=True)


def test_download_falls_back_to_zip_archive(tmp_path, monkeypatch):
    payload = b"63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(config.UCI_ZIP_MEMBER, payload)

    def fake_fetch(url, timeout=30):
        if url == config.UCI_URL:
            raise urllib.error.URLError("primary mirror down")
        return archive.getvalue()

    monkeypatch.setattr(data, "_fetch", fake_fetch)
    monkeypatch.setattr(config, "RAW_SHA256", hashlib.sha256(payload).hexdigest())
    dest = data.download(tmp_path / "raw.data", force=True)
    assert dest.read_bytes() == payload
