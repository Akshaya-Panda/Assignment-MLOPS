"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset.

Usage:
    python -m cardiorisk.data            # download (if missing), verify, clean, save
    python -m cardiorisk.data --force    # force a fresh download
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from cardiorisk import config

log = logging.getLogger(__name__)

# Valid codes for the discrete clinical fields (UCI dataset documentation).
ALLOWED_CODES: dict[str, set[int]] = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}

# Physiologically plausible ranges. Values outside are data-entry errors and
# are set to missing so the model pipeline imputes them consistently.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "age": (18, 100),
    "trestbps": (60, 250),
    "chol": (80, 700),
    "thalach": (50, 230),
    "oldpeak": (-3, 10),
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "cardiorisk-mlops/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download(dest: Path = config.RAW_FILE, force: bool = False, verify: bool = True) -> Path:
    """Download the raw Cleveland file, falling back to the UCI zip archive."""
    dest = Path(dest)
    if dest.exists() and not force:
        log.info("Raw file already present: %s", dest)
    else:
        try:
            payload = _fetch(config.UCI_URL)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("Direct download failed (%s); trying the zip archive", exc)
            with zipfile.ZipFile(io.BytesIO(_fetch(config.UCI_ZIP_URL))) as archive:
                payload = archive.read(config.UCI_ZIP_MEMBER)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        log.info("Downloaded %d bytes to %s", len(payload), dest)

    if verify:
        actual = sha256sum(dest)
        if actual != config.RAW_SHA256:
            raise ValueError(
                f"Checksum mismatch for {dest}: expected {config.RAW_SHA256}, got {actual}. "
                "The upstream file changed; re-run with --no-verify after reviewing it."
            )
    return dest


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    """Read the headerless raw file; '?' marks missing values."""
    return pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values="?")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, binarise the target, null out invalid values, drop duplicates."""
    missing_cols = [c for c in config.RAW_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Raw data is missing columns: {missing_cols}")

    df = raw[config.RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["num"])
    df[config.TARGET] = (df["num"] > 0).astype(int)
    df = df.drop(columns="num")

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        df[col] = df[col].where(df[col].between(low, high))
    for col, codes in ALLOWED_CODES.items():
        df[col] = df[col].where(df[col].isin(codes))

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < before:
        log.info("Dropped %d duplicate rows", before - len(df))
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise ValueError if the cleaned frame violates the expected schema."""
    expected = config.FEATURE_COLUMNS + [config.TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if df.empty:
        raise ValueError("Dataset is empty")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be binary (0/1)")
    for col, codes in ALLOWED_CODES.items():
        bad = df[col].dropna()[~df[col].dropna().isin(codes)]
        if not bad.empty:
            raise ValueError(f"Column '{col}' has invalid codes: {sorted(bad.unique())}")
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        values = df[col].dropna()
        if ((values < low) | (values > high)).any():
            raise ValueError(f"Column '{col}' has values outside [{low}, {high}]")


def save_clean(df: pd.DataFrame, path: Path = config.CLEAN_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%g")
    return path


def load_clean(path: Path = config.CLEAN_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def summarise(df: pd.DataFrame) -> dict:
    counts = df[config.TARGET].value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "features": len(config.FEATURE_COLUMNS),
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n},
        "class_counts": {int(k): int(v) for k, v in counts.items()},
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download the raw file")
    parser.add_argument("--no-verify", action="store_true", help="skip the SHA-256 check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_path = download(force=args.force, verify=not args.no_verify)
    df = clean(load_raw(raw_path))
    validate(df)
    out = save_clean(df)
    log.info("Saved cleaned dataset to %s", out)
    print(json.dumps(summarise(df), indent=2))


if __name__ == "__main__":
    main()
