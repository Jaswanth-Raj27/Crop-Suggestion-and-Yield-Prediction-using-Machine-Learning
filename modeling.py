"""Shared preprocessing, training, validation, and prediction for the demo."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_config import categorical_features_for

NUMERIC_FEATURES = ("N", "P", "K", "temperature", "humidity", "ph", "rainfall")
BASE_CATEGORICAL_FEATURES = categorical_features_for("crop")
CROP_FEATURES = categorical_features_for("yield_tons_per_acre")
DATASET_COLUMNS = (*BASE_CATEGORICAL_FEATURES, *NUMERIC_FEATURES, "crop", "yield_tons_per_acre")
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "crop_models.joblib"


class InputValidationError(ValueError):
    """Raised when a submitted form value is outside the accepted contract."""

# Broad input sanity limits; they reject impossible or hazardous input values,
# not certify agronomic validity. The bundled synthetic data is not a source of
# scientifically valid ranges.
NUMERIC_LIMITS = {
    "N": (0, 2000),
    "P": (0, 2000),
    "K": (0, 2000),
    "temperature": (-50, 70),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 10000),
}


def _make_pipeline(categorical_features: tuple[str, ...], estimator: Any) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical_features),
            ),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def train_models(dataset_path: Path | str) -> tuple[dict[str, Any], dict[str, float]]:
    """Train both pipelines on the dataset and return bundle plus holdout metrics."""
    source = Path(dataset_path).resolve()
    data = pd.read_csv(source)
    missing = sorted(set(DATASET_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if data[list(DATASET_COLUMNS)].isna().any().any():
        raise ValueError("Dataset contains missing values in required model columns")
    if len(data) < 10:
        raise ValueError("Dataset is too small for a train/holdout split")

    classifier = _make_pipeline(
        BASE_CATEGORICAL_FEATURES,
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    )
    cls_features = [*NUMERIC_FEATURES, *BASE_CATEGORICAL_FEATURES]
    x_cls_train, x_cls_test, y_cls_train, y_cls_test = train_test_split(
        data[cls_features],
        data["crop"],
        test_size=0.2,
        random_state=42,
        stratify=data["crop"],
    )
    classifier.fit(x_cls_train, y_cls_train)

    regressor = _make_pipeline(
        CROP_FEATURES,
        RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    )
    reg_features = [*NUMERIC_FEATURES, *CROP_FEATURES]
    x_reg_train, x_reg_test, y_reg_train, y_reg_test = train_test_split(
        data[reg_features],
        data["yield_tons_per_acre"],
        test_size=0.2,
        random_state=42,
    )
    regressor.fit(x_reg_train, y_reg_train)

    metrics = {
        "classifier_accuracy": float(accuracy_score(y_cls_test, classifier.predict(x_cls_test))),
        "yield_mae": float(mean_absolute_error(y_reg_test, regressor.predict(x_reg_test))),
        "yield_r2": float(r2_score(y_reg_test, regressor.predict(x_reg_test))),
    }
    category_values = {
        column: sorted(str(value) for value in data[column].unique())
        for column in BASE_CATEGORICAL_FEATURES
    }
    bundle = {
        "classifier": classifier,
        "regressor": regressor,
        "category_values": category_values,
        "crops": sorted(str(value) for value in data["crop"].unique()),
        "metrics": metrics,
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
    }
    return bundle, metrics


def save_model_bundle(bundle: dict[str, Any], destination: Path | str = MODEL_PATH) -> Path:
    """Atomically write the trusted local model artifact."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as temp:
            temporary_path = temp.name
        joblib.dump(bundle, temporary_path)
        os.replace(temporary_path, output)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return output


def load_model_bundle(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    """Load a locally generated trusted artifact; never load user-supplied files."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError("Model artifacts are missing. Run `python train_models.py` first.")
    bundle = joblib.load(artifact)
    if not isinstance(bundle, dict) or not {"classifier", "regressor", "category_values", "crops"} <= bundle.keys():
        raise ValueError("Model artifact has an unsupported format; retrain the models.")
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise ValueError("Model artifact uses a different scikit-learn version; retrain the models.")
    return bundle


def validate_input(raw: dict[str, Any], category_values: dict[str, list[str]]) -> dict[str, Any]:
    """Validate and normalize user-supplied features before inference."""
    normalized: dict[str, Any] = {}
    errors: list[str] = []
    for feature, (minimum, maximum) in NUMERIC_LIMITS.items():
        value = raw.get(feature)
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{feature} must be a number")
            continue
        if not math.isfinite(number):
            errors.append(f"{feature} must be finite")
        elif not minimum <= number <= maximum:
            errors.append(f"{feature} must be between {minimum} and {maximum}")
        else:
            normalized[feature] = number

    for feature in BASE_CATEGORICAL_FEATURES:
        value = raw.get(feature)
        allowed = category_values.get(feature, [])
        if not isinstance(value, str) or value not in allowed:
            errors.append(f"Select a valid {feature.replace('_', ' ')} from the available options")
        else:
            normalized[feature] = value

    if errors:
        raise InputValidationError("; ".join(errors))
    return normalized


def predict(bundle: dict[str, Any], raw: dict[str, Any]) -> dict[str, str]:
    values = validate_input(raw, bundle["category_values"])
    classifier_input = pd.DataFrame([values])
    crop = str(bundle["classifier"].predict(classifier_input)[0])
    regressor_input = pd.DataFrame([{**values, "crop": crop}])
    estimate = float(bundle["regressor"].predict(regressor_input)[0])
    if not math.isfinite(estimate):
        raise ValueError("The model returned an invalid estimate")
    return {"crop": crop, "yield": f"{estimate:.2f} tons per acre"}
