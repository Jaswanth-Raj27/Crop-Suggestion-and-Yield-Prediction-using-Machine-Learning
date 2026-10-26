import json
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

import app as flask_app


class PredictionRouteTests(unittest.TestCase):
    def setUp(self):
        flask_app.app.config.update(TESTING=True)
        self.client = flask_app.app.test_client()
        self.bundle = {
            "category_values": {
                "state": ["State"],
                "district": ["District"],
                "soil_type": ["Soil"],
                "pest_problem": ["None"],
            }
        }
        flask_app.get_model_bundle.cache_clear()
        flask_app.get_india_yield_data.cache_clear()

    def tearDown(self):
        flask_app.get_model_bundle.cache_clear()
        flask_app.get_india_yield_data.cache_clear()

    def test_home_page_renders(self):
        with patch.object(flask_app, "load_model_bundle", return_value=self.bundle):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"fieldnote", response.data)
        self.assertIn(b"A closer look at what grows.", response.data)

    def test_missing_artifact_fails_gracefully(self):
        with patch.object(flask_app, "load_model_bundle", side_effect=FileNotFoundError):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Prediction models are not ready", response.data)

    def test_historical_yield_page_uses_real_flagged_india_records(self):
        response = self.client.get("/historical-yields?item=Wheat")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"India yield", response.data)
        self.assertIn(b"kg/ha", response.data)
        self.assertIn(b"Official figure", response.data)
        self.assertIn(b"2024", response.data)

    def test_all_history_items_have_unique_crop_photos(self):
        visuals = flask_app.get_crop_visuals()
        dataset_items = set(flask_app.get_india_yield_data()["Item"].dropna().unique())

        self.assertEqual(set(visuals), dataset_items)
        urls = [visual["url"] for visual in visuals.values()]
        sources = [visual["source_url"] for visual in visuals.values()]
        self.assertEqual(len(urls), len(set(urls)))
        self.assertEqual(len(sources), len(set(sources)))

        credit_path = Path(flask_app.__file__).resolve().parent / "static" / "images" / "crops" / "credits.json"
        credits = json.loads(credit_path.read_text(encoding="utf-8-sig"))
        self.assertEqual({row["item"] for row in credits}, dataset_items)
        self.assertTrue(all(row["license"] and row["page"] and row["credit"] for row in credits))

        project_root = Path(flask_app.__file__).resolve().parent
        image_hashes = {}
        for item, visual in visuals.items():
            self.assertTrue(visual["title"], item)
            self.assertTrue(visual["credit"], item)
            self.assertTrue(visual["source_url"].startswith("https://commons.wikimedia.org/"), item)
            if visual["url"].startswith("/static/"):
                image_path = project_root / visual["url"].lstrip("/")
                self.assertTrue(image_path.is_file(), item)
                digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
                self.assertNotIn(digest, image_hashes, f"{item} reuses the image for {image_hashes.get(digest)}")
                image_hashes[digest] = item
            else:
                self.assertTrue(visual["url"].startswith("https://"), item)

    def test_crop_photo_changes_to_the_selected_exact_item(self):
        yams = self.client.get("/historical-yields?item=Yams")
        strawberries = self.client.get("/historical-yields?item=Strawberries")
        self.assertIn(b"Closed up photo of a Water yam plant.jpg", yams.data)
        self.assertIn(b"strawberry-field-panchgani.jpg", strawberries.data)
        self.assertNotEqual(
            flask_app.crop_visual_for("Yams")["url"],
            flask_app.crop_visual_for("Strawberries")["url"],
        )
        self.assertIn(b"crop-visual-map", yams.data)

    def test_invalid_request_returns_clear_validation_error(self):
        with patch.object(flask_app, "load_model_bundle", return_value=self.bundle):
            response = self.client.post("/predict", data={"N": "not-a-number"})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"must be a number", response.data)

    def test_prediction_uses_real_historical_yield_reference_not_demo_estimate(self):
        with patch.object(flask_app, "load_model_bundle", return_value=self.bundle), patch.object(
            flask_app,
            "predict",
            return_value={"crop": "wheat", "yield": "999.99 tons per acre"},
        ):
            response = self.client.post("/predict", data={"N": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Latest India yield record", response.data)
        self.assertIn(b"3,559.0 kg/ha", response.data)
        self.assertIn(b"2024", response.data)
        self.assertIn(b"not a field-specific forecast", response.data)
        self.assertNotIn(b"999.99 tons per acre", response.data)

    def test_unexpected_failure_does_not_leak_exception_details(self):
        with self.assertLogs(flask_app.logger, level="ERROR"):
            with patch.object(flask_app, "load_model_bundle", return_value=self.bundle), patch.object(
                flask_app, "predict", side_effect=RuntimeError("internal detail")
            ):
                response = self.client.post("/predict", data={})
        self.assertEqual(response.status_code, 500)
        self.assertIn(b"could not be completed", response.data)
        self.assertNotIn(b"internal detail", response.data)


if __name__ == "__main__":
    unittest.main()
