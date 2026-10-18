"""Request/response contracts. Validation ranges reuse the data-cleaning rules."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cardiorisk.data import PLAUSIBLE_RANGES

EXAMPLE_PATIENT = {
    "age": 67,
    "sex": 1,
    "cp": 4,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 3,
    "thal": 3,
}


def _range(name: str, description: str):
    low, high = PLAUSIBLE_RANGES[name]
    return Field(ge=low, le=high, description=description)


class PatientFeatures(BaseModel):
    """One patient's clinical measurements (UCI Cleveland encoding)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    age: int = _range("age", "Age in years")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest pain: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = _range("trestbps", "Resting systolic blood pressure (mm Hg)")
    chol: float = _range("chol", "Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG: 0 normal, 1 ST-T, 2 LVH")
    thalach: float = _range("thalach", "Maximum heart rate during exercise test (bpm)")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = _range("oldpeak", "ST depression induced by exercise (mm)")
    slope: Literal[1, 2, 3] = Field(description="Peak exercise ST slope: 1 up, 2 flat, 3 down")
    ca: Literal[0, 1, 2, 3] | None = Field(
        default=None, description="Major vessels coloured by fluoroscopy (optional, imputed)"
    )
    thal: Literal[3, 6, 7] | None = Field(
        default=None, description="Thallium test: 3 normal, 6 fixed, 7 reversible (optional)"
    )


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instances: list[PatientFeatures] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    prediction: Literal[0, 1] = Field(description="1 = heart disease predicted")
    label: Literal["no_heart_disease", "heart_disease"]
    probability_disease: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0.5, le=1, description="Probability of the predicted class")
    risk_band: Literal["low", "moderate", "high"]


class PredictionResponse(Prediction):
    model_version: str
    request_id: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_version: str
    request_id: str
