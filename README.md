# Crop Suggestion and Yield Prediction

A Flask prototype that suggests a crop category from soil, weather, and location inputs, then shows the latest available India-wide historical yield record for that crop.

## Important data limitation

The bundled `indian_agriculture_data.csv` contains 742 examples across five crop classes. The project README identifies the dataset as AI-generated. It is synthetic demonstration data, not verified agricultural measurements; a local data scan also found humidity values above 100%. Model scores and predictions must not be presented as evidence of real-world farming performance or used as agronomic advice.

For a decision-support application, replace the synthetic data with a documented, licensed dataset of measured agricultural observations. Record its source, collection period, geography, units, cleaning steps, and license, then evaluate on an independent holdout that reflects the locations and conditions where the app will be used.

## How the prototype works

1. A Random Forest classifier suggests one of five crop labels from N, P, K, temperature, humidity, pH, rainfall, state, district, soil type, and pest problem.
2. The suggestion is experimental because its training data is synthetic.
3. The Flask page pairs the suggestion with the latest India-wide FAOSTAT yield record for that crop. This is historical context, not a field-specific forecast.

The repository still contains an internal yield-regression training path for experimentation, but its synthetic yield estimate is no longer presented as the result. Crop labels are excluded from classifier inputs because crop is that model's target.

The home page also links to a historical data view based on FAOSTAT's India national annual crop-yield records. This real dataset is separate from the synthetic model data; it lacks district, season, soil, and weather inputs and is not used to claim personalized crop recommendations.

## Repository contents

- `app.py` — Flask web application and validated prediction route.
- `modeling.py` — shared scikit-learn preprocessing pipelines, training, validation, and prediction.
- `train_models.py` — offline model training and local artifact generation.
- `Yield.py` — original exploratory training and plotting helpers.
- `feature_config.py` — shared target-specific categorical feature configuration.
- `indian_agriculture_data.csv` — synthetic demonstration dataset.
- `data/processed/india_faostat_crop_yield.csv` — attributed FAOSTAT India national annual yield extract; details and limitations are in `data/README.md`.
- `templates/index.html` — input form and prediction result page.
- `tests/` — unit tests for feature configuration, training, validation, and prediction.

## Setup and run

Python 3.11 or newer and pip are required. Install the tested dependency versions in a virtual environment, train the model artifact, then start the app:

```sh
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_models.py
python app.py
```

Training is a separate step. It creates `artifacts/crop_models.joblib`, which is ignored by Git and can be regenerated from the bundled dataset. The artifact contains scikit-learn objects and must only be loaded from a trusted local build; never accept a user-supplied model file. The web process does not retrain models on import. If the artifact is missing, the page explains that training is required.

Run tests after installing the dependencies:

```sh
python -m unittest discover -s tests
```

## Evaluation

The training command reports classifier accuracy and yield MAE/R² on deterministic 80/20 holdouts; the classifier split is stratified by crop. This remains a basic check, not a robust estimate of performance. Compare against simple baselines, report per-class metrics, and use location-grouped validation to measure generalization to new districts or states. Since the source data is synthetic, these metrics are only useful for checking the code path.

## Before deployment

- Replace or clearly retain the synthetic dataset limitation; do not describe synthetic results as validated agricultural predictions.
- Obtain measured soil, weather, location, crop, and yield observations before treating the crop suggestion as agricultural decision support.
- Obtain and document a measured dataset's source, license, collection period, geography, units, and cleaning process before decision-support use.
- Validate data provenance, crop and yield units, input ranges, and geographic coverage with an agricultural domain expert.
- Before public deployment, add production server configuration, CSRF protection, rate limiting, monitoring, and a reviewed model-artifact release process. Flask's development server is for local use only.
