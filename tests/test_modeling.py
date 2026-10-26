import unittest
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from modeling import (
    BASE_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_model_bundle,
    predict,
    save_model_bundle,
    train_models,
    validate_input,
)


ROOT = Path(__file__).resolve().parents[1]


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle, cls.metrics = train_models(ROOT / "indian_agriculture_data.csv")
        cls.example = {feature: 10 for feature in NUMERIC_FEATURES}
        cls.example.update({
            "N": 95,
            "P": 45,
            "K": 48,
            "temperature": 27.5,
            "humidity": 86.2,
            "ph": 6.6,
            "rainfall": 225.5,
        })
        cls.example.update({
            name: cls.bundle["category_values"][name][0]
            for name in BASE_CATEGORICAL_FEATURES
        })

    def test_training_returns_finite_holdout_metrics(self):
        self.assertTrue(all(math.isfinite(value) for value in self.metrics.values()))
        self.assertEqual(self.bundle["sklearn_version"], __import__("sklearn").__version__)

    def test_prediction_runs_both_pipeline_models(self):
        result = predict(self.bundle, self.example)
        self.assertIn(result["crop"], self.bundle["crops"])
        self.assertTrue(result["yield"].endswith("tons per acre"))

    def test_crop_is_a_yield_feature_but_not_a_classifier_feature(self):
        classifier_columns = self.bundle["classifier"].named_steps["preprocess"].transformers_[1][2]
        regressor_columns = self.bundle["regressor"].named_steps["preprocess"].transformers_[1][2]
        self.assertNotIn("crop", classifier_columns)
        self.assertIn("crop", regressor_columns)

    def test_artifact_round_trip(self):
        with TemporaryDirectory() as directory:
            artifact = save_model_bundle(self.bundle, Path(directory) / "models.joblib")
            loaded = load_model_bundle(artifact)
        self.assertEqual(loaded["dataset_sha256"], self.bundle["dataset_sha256"])

    def test_unknown_category_is_rejected_explicitly(self):
        example = dict(self.example, state="Unknown State")
        with self.assertRaisesRegex(ValueError, "valid state"):
            predict(self.bundle, example)

    def test_non_finite_and_out_of_range_values_are_rejected(self):
        example = dict(self.example, N=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_input(example, self.bundle["category_values"])
        example = dict(self.example, humidity=101)
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            validate_input(example, self.bundle["category_values"])


if __name__ == "__main__":
    unittest.main()
