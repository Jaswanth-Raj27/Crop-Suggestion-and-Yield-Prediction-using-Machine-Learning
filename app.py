"""Flask UI for the crop/yield demonstration model."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from flask import Flask, render_template, request
import pandas as pd

from modeling import InputValidationError, load_model_bundle, predict

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
HISTORY_PATH = Path(__file__).resolve().parent / "data" / "processed" / "india_faostat_crop_yield.csv"
PREDICTION_CROP_ITEMS = {
    "coffee": "Coffee, green",
    "maize": "Maize (corn)",
    "rice": "Rice",
    "sugarcane": "Sugar cane",
    "wheat": "Wheat",
}
YIELD_FLAGS = {
    "A": "Official figure",
    "E": "Estimated value",
    "I": "Imputed by a receiving agency",
    "M": "Missing value",
    "X": "Figure from an external organization",
}

CROP_VISUALS_PATH = Path(__file__).resolve().parent / "data" / "crop_visuals.json"


@lru_cache(maxsize=1)
def get_crop_visuals():
    """Return the exact-item visual allow-list and attribution metadata."""
    return json.loads(CROP_VISUALS_PATH.read_text(encoding="utf-8-sig"))


def crop_visual_for(item):
    """Return the dedicated visual configured for one FAOSTAT item."""
    try:
        return get_crop_visuals()[item]
    except KeyError as exc:
        raise ValueError(f"No unique crop photo is configured for {item!r}.") from exc


@lru_cache(maxsize=1)
def get_model_bundle():
    return load_model_bundle()


@lru_cache(maxsize=1)
def get_india_yield_data():
    return pd.read_csv(HISTORY_PATH)


def latest_india_yield_reference(crop):
    """Return the latest national historical record for a demo crop label."""
    item = PREDICTION_CROP_ITEMS.get(str(crop).casefold())
    if not item:
        return None
    data = get_india_yield_data()
    rows = data.loc[data["Item"] == item].sort_values("Year", ascending=False)
    if rows.empty:
        return None
    row = rows.iloc[0]
    return {
        "value": f"{float(row['Value']):,.1f} {row['Unit']}",
        "year": int(row["Year"]),
        "status": YIELD_FLAGS.get(row["Flag"], "See FAOSTAT flag metadata"),
    }


def render_page(*, prediction=None, error=None, values=None, status_code=200):
    values = values or {}
    setup_error = None
    try:
        bundle = get_model_bundle()
        categories = bundle["category_values"]
    except (FileNotFoundError, ValueError):
        categories = {}
        setup_error = "Prediction models are not ready. Run the documented training command, then restart the app."
    response = render_template(
        "index.html",
        prediction=prediction,
        error=error,
        setup_error=setup_error,
        values=values,
        categories=categories,
        active_page="predictor",
    )
    return response, status_code


@app.route("/")
def home():
    return render_page()


@app.route("/historical-yields")
def historical_yields():
    try:
        data = get_india_yield_data()
    except (FileNotFoundError, pd.errors.ParserError):
        return render_template(
            "history.html",
            error="The historical data file is not available.",
            items=[],
            records=[],
            selected_visual=crop_visual_for("Wheat"),
        ), 503

    items = sorted(data["Item"].dropna().unique().tolist())
    selected_item = request.args.get("item", "Wheat")
    if selected_item not in items:
        selected_item = "Wheat" if "Wheat" in items else (items[0] if items else "")
    selected = data.loc[data["Item"] == selected_item].sort_values("Year", ascending=False).head(20)
    records = [
        {
            "year": int(row.Year),
            "yield": f"{float(row.Value):,.1f}",
            "unit": row.Unit,
            "flag": row.Flag,
            "flag_description": YIELD_FLAGS.get(row.Flag, "See FAOSTAT flag metadata"),
        }
        for row in selected.itertuples(index=False)
    ]
    return render_template(
        "history.html",
        error=None,
        items=items,
        selected_item=selected_item,
        records=records,
        item_count=len(items),
        year_start=int(data["Year"].min()),
        year_end=int(data["Year"].max()),
        year_count=data["Year"].nunique(),
        selected_visual=crop_visual_for(selected_item),
        crop_visuals={item: crop_visual_for(item) for item in items},
        active_page="history",
    )


@app.route("/predict", methods=["POST"])
def predict_route():
    raw = {name: request.form.get(name, "") for name in (
        "N", "P", "K", "temperature", "humidity", "ph", "rainfall",
        "state", "district", "soil_type", "pest_problem",
    )}
    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError):
        return render_page(
            error="Prediction models are not available yet. Please try again after setup is complete.",
            values=raw,
            status_code=503,
        )
    try:
        result = predict(bundle, raw)
        # The regressor is trained on synthetic demonstration data. Show the
        # real India-wide FAOSTAT figure as context instead of presenting its
        # synthetic estimate as an agricultural forecast.
        result["yield_reference"] = latest_india_yield_reference(result["crop"])
        return render_page(prediction=result)
    except InputValidationError as exc:
        return render_page(error=str(exc), values=raw, status_code=400)
    except Exception:
        logger.exception("Prediction request failed")
        return render_page(
            error="The prediction could not be completed. Please review your inputs and try again.",
            values=raw,
            status_code=500,
        )


if __name__ == "__main__":
    app.run(debug=False)
