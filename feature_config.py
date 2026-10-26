"""Feature configuration shared by training and prediction code."""

BASE_CATEGORICAL_FEATURES = (
    "state",
    "district",
    "soil_type",
    "pest_problem",
)


def categorical_features_for(target_label: str) -> tuple[str, ...]:
    """Return categorical inputs for a model target.

    Crop is a prediction target for the classifier and an input for the yield
    regressor, so it must not be included as a classifier feature.
    """
    if target_label == "crop":
        return BASE_CATEGORICAL_FEATURES
    if target_label == "yield_tons_per_acre":
        return (*BASE_CATEGORICAL_FEATURES, "crop")
    raise ValueError(f"Unsupported model target: {target_label}")
