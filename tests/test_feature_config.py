import unittest

from feature_config import categorical_features_for


class CategoricalFeatureConfigTests(unittest.TestCase):
    def test_classifier_does_not_include_its_target_as_a_feature(self):
        self.assertNotIn("crop", categorical_features_for("crop"))

    def test_yield_regressor_includes_the_selected_crop(self):
        self.assertIn("crop", categorical_features_for("yield_tons_per_acre"))

    def test_unknown_target_is_rejected(self):
        with self.assertRaises(ValueError):
            categorical_features_for("unknown_target")


if __name__ == "__main__":
    unittest.main()
