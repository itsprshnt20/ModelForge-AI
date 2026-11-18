import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.compose import ColumnTransformer

from workbench import MODELS, ValidationError, evaluate, fit_models, model_card, parse_csv, split_labels
from samples.generate_churn import generate

DEMO = Path(__file__).resolve().parents[1] / "samples" / "synthetic_churn.csv"
EXCLUDED = ("customer_id", "cancellation_ticket")


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = parse_csv(DEMO.read_bytes())
        cls.run_result = fit_models(cls.frame, "churn", "Yes", EXCLUDED, "Synthetic customer churn")

    def test_target_and_exclusions_never_reach_pipeline(self):
        fitted = self.capture_transforms(self.frame)
        expected = set(self.frame.columns) - {"churn", *EXCLUDED}
        for transformer in fitted:
            self.assertEqual(set(transformer.feature_names_in_), expected)
        self.assertEqual(set(self.run_result.features), expected)

    def capture_transforms(self, frame):
        fitted = []

        def record(*args, **kwargs):
            transformer = ColumnTransformer(*args, **kwargs)
            fitted.append(transformer)
            return transformer

        with patch("workbench.ColumnTransformer", side_effect=record):
            fit_models(frame, "churn", "Yes", EXCLUDED, "Synthetic customer churn")
        return fitted

    def test_preprocessing_fits_train_only_and_handles_missing_unseen(self):
        frame = self.frame.copy()
        train, validation = split_labels(frame["churn"].eq("Yes").astype(int))
        numeric = frame.iloc[train]["monthly_charge"].dropna().astype(float)
        frame.loc[validation, "monthly_charge"] = "99999"
        frame.loc[validation, "region"] = "Validation-only region"
        frame.loc[validation[0], "monthly_charge"] = None
        frame.loc[validation[1], "region"] = None
        for transformer in self.capture_transforms(frame):
            prep = transformer.named_transformers_
            median = prep["numeric"].named_steps["impute"].statistics_[1]
            self.assertAlmostEqual(median, numeric.median())
            region_categories = prep["categorical"].named_steps["encode"].categories_[1]
            self.assertNotIn("Validation-only region", region_categories)

    def test_seed_reproduces_training_and_validation_rows(self):
        labels = self.frame["churn"].eq("Yes").astype(int)
        left = split_labels(labels)
        right = split_labels(labels)
        for first, second in zip(left, right):
            np.testing.assert_array_equal(first, second)
        self.assertEqual(tuple(left[1]), self.run_result.validation_rows)
        self.assertEqual(len(set(left[0]) & set(left[1])), 0)
        self.assertEqual(set(self.run_result.probabilities), set(MODELS))

    def test_malformed_and_missing_inputs(self):
        for raw, message in (
            (b'x,y\n1,"truncated', "Malformed CSV"),
            (b"x,y\n1,2,3", "Expected 2"),
            (b"x,x\n1,2", "unique"),
            (b"x,y\n", "headers only"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValidationError, message):
                parse_csv(raw)
        with self.assertRaisesRegex(ValidationError, "missing values"):
            fit_models(parse_csv(b"x,y\n1,\n2,Yes"), "y", "Yes", [], "Test")
        with self.assertRaisesRegex(ValidationError, "exactly two classes"):
            fit_models(parse_csv(b"x,y\n1,No\n2,Yes\n3,Maybe"), "y", "Yes", [], "Test")
        with self.assertRaisesRegex(ValidationError, "stratified split"):
            fit_models(parse_csv(b"x,y\n1,No\n2,No\n3,No\n4,Yes"), "y", "Yes", [], "Test")

    def test_unsupported_feature_and_explicit_positive(self):
        with self.assertRaisesRegex(ValidationError, "unsupported"):
            fit_models(self.frame, "churn", "Yes", ["cancellation_ticket"], "Test")
        with self.assertRaisesRegex(ValidationError, "Explicitly choose"):
            fit_models(self.frame, "churn", None, EXCLUDED, "Test")
        with self.assertRaisesRegex(ValidationError, "at least one feature"):
            fit_models(self.frame, "churn", "Yes", list(self.frame.columns), "Test")

    def test_threshold_uses_saved_scores_and_includes_exact_boundary(self):
        actual = (0, 0, 1, 1)
        scores = (0.1, 0.5, 0.5, 0.9)
        with patch("workbench.Pipeline.fit", side_effect=AssertionError("unexpected fit")):
            low = evaluate(actual, scores, 0.50)
            high = evaluate(actual, scores, 0.75)
        self.assertEqual([low[k] for k in ("TN", "FP", "FN", "TP")], [1, 1, 0, 2])
        self.assertEqual([high[k] for k in ("TN", "FP", "FN", "TP")], [2, 0, 1, 1])
        self.assertEqual(low["Average precision"], high["Average precision"])

    def test_card_uses_selected_model_threshold_and_omits_raw_rows(self):
        name, threshold = "Random forest", 0.72
        text = model_card(self.run_result, name, threshold)
        metric = evaluate(self.run_result.observed, self.run_result.probabilities[name], threshold)
        self.assertIn("- Model: Random forest", text)
        self.assertIn("- Threshold: 0.72", text)
        self.assertIn(f"FP {metric['FP']}", text)
        self.assertIn(f"FN {metric['FN']}", text)
        self.assertIn(f"Average precision: {metric['Average precision']:.3f}", text)
        self.assertIn("Excluded fields: customer_id, cancellation_ticket", text)
        self.assertNotIn("SYN-0001", text)

    def test_demo_generator_reproduces_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "demo.csv"
            generate(path)
            self.assertEqual(path.read_bytes().splitlines(), DEMO.read_bytes().splitlines())
