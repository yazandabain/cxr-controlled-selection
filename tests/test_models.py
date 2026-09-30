"""Synthetic-only tests: python -m unittest discover -s tests."""

from __future__ import annotations

import inspect
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from cxr_selection import models


class PatientSelectionTests(unittest.TestCase):
    def test_selection_precedes_eligibility_and_is_order_invariant(self):
        frame = pd.DataFrame({
            "subject_id": [2, 1, 1, 2, 1],
            "study_id": [11, 10, 2, 11, 2],
            "dicom_id": ["b", "c", "b", "a", "a"],
            "report_state": [1, 1, 1, 0, np.nan],
        }, index=[9, 8, 7, 6, 5])
        selected = models.select_one_study_per_patient(frame)
        self.assertEqual(selected.index.tolist(), [5, 6])
        self.assertTrue(pd.isna(selected.iloc[0]["report_state"]))
        pd.testing.assert_frame_equal(selected, models.select_one_study_per_patient(frame.iloc[::-1]))
        self.assertEqual(len(frame), 5)

    def test_numeric_study_strings_and_dicom_lexical_tie_break(self):
        frame = pd.DataFrame({"subject_id": ["1", "1", "1"],
                              "study_id": ["10", "2", "2"],
                              "dicom_id": ["a", "2", "10"]})
        self.assertEqual(models.select_one_study_per_patient(frame).index.tolist(), [2])

    def test_invalid_or_duplicate_keys_fail(self):
        row = {"subject_id": 1, "study_id": 2, "dicom_id": "a"}
        with self.assertRaises(ValueError):
            models.select_one_study_per_patient(pd.DataFrame([row, row]))
        for column in row:
            bad = row.copy()
            bad[column] = None
            with self.subTest(column=column), self.assertRaises(ValueError):
                models.select_one_study_per_patient(pd.DataFrame([bad]))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.arange(100, 164)
        self.states = np.resize(np.array([0.0, 1.0, -1.0, np.nan]), len(self.ids))
        self.y = ((np.arange(len(self.ids)) // 2) % 2).astype(int)
        self.budgets = (4, 12, 24)

    def test_exact_budgets_counts_nested_sets_and_reproducibility(self):
        first = models.sample_training_panels(self.ids, self.states, self.y, self.budgets)
        second = models.sample_training_panels(self.ids, self.states, self.y, self.budgets)
        self.assertEqual(len(first), 20)
        for seed, panels in first.items():
            previous = {arm: set() for arm in ("explicit", "random", "prevalence_matched")}
            for budget, arms in panels.items():
                for arm, index in arms.items():
                    with self.subTest(seed=seed, budget=budget, arm=arm):
                        self.assertEqual(index.dtype.kind, "i")
                        self.assertEqual(len(index), budget)
                        self.assertEqual(len(set(index)), budget)
                        self.assertTrue(np.all((index >= 0) & (index < len(self.ids))))
                        self.assertTrue(previous[arm].issubset(set(index)))
                        previous[arm] = set(index)
                        np.testing.assert_array_equal(index, second[seed][budget][arm])
                self.assertTrue(np.isin(self.states[arms["explicit"]], [0, 1]).all())
                self.assertEqual(self.y[arms["explicit"]].sum(), self.y[arms["prevalence_matched"]].sum())

    def test_input_order_and_budget_list_do_not_change_selected_patients(self):
        order = np.random.default_rng(2).permutation(len(self.ids))
        first = models.sample_training_panels(self.ids, self.states, self.y, self.budgets, [7])
        second = models.sample_training_panels(self.ids[order], self.states[order], self.y[order], [24], [7])
        for arm in first[7][24]:
            np.testing.assert_array_equal(self.ids[first[7][24][arm]], self.ids[order][second[7][24][arm]])

    def test_sampling_validation_and_no_input_mutation(self):
        ids, states, labels = self.ids.copy(), self.states.copy(), self.y.copy()
        models.sample_training_panels(ids, states, labels, [4], [0])
        np.testing.assert_array_equal(ids, self.ids)
        np.testing.assert_array_equal(states, self.states)
        np.testing.assert_array_equal(labels, self.y)
        bad_arguments = [
            (np.repeat(1, len(ids)), states, labels, [4], [0]),
            (ids, states, labels, [100], [0]),
            (ids, states, labels, [4, 4], [0]),
            (ids, states, labels, [4], [0, 0]),
            (ids, states, labels, [4], [-1]),
            (ids, np.repeat(2, len(ids)), labels, [4], [0]),
            (ids, states, np.repeat(np.nan, len(ids)), [4], [0]),
        ]
        for arguments in bad_arguments:
            with self.subTest(case=len(arguments)), self.assertRaises(ValueError):
                models.sample_training_panels(*arguments)


class WeightedMetricTests(unittest.TestCase):
    def test_ties_zero_weights_and_integer_multiplicities_match_sklearn(self):
        rng = np.random.default_rng(93)
        for size in [10, 31, 100]:
            for trial in range(20):
                y = np.r_[0, 1, rng.integers(0, 2, size - 2)]
                scores = rng.integers(0, 5, size) / 4
                weights = np.r_[1, 2, rng.integers(0, 5, size - 2)]
                actual = models.weighted_auc_ap(y, scores, weights)
                expanded = np.repeat(np.arange(size), weights)
                with self.subTest(size=size, trial=trial):
                    self.assertAlmostEqual(actual["auroc"], roc_auc_score(y, scores, sample_weight=weights), places=14)
                    self.assertAlmostEqual(actual["average_precision"], average_precision_score(y, scores, sample_weight=weights), places=14)
                    self.assertAlmostEqual(actual["auroc"], roc_auc_score(y[expanded], scores[expanded]), places=14)
                    self.assertAlmostEqual(actual["average_precision"], average_precision_score(y[expanded], scores[expanded]), places=14)

    def test_constant_scores_fractional_weights_and_known_rankings(self):
        y = np.array([0, 1, 0, 1])
        weights = np.array([0.5, 2.0, 0.0, 0.75])
        result = models.weighted_auc_ap(y, np.ones(4), weights)
        self.assertEqual(result["auroc"], 0.5)
        self.assertAlmostEqual(result["average_precision"], weights[y == 1].sum() / weights.sum())
        perfect = models.weighted_auc_ap(y, y)
        self.assertEqual(perfect, {"auroc": 1.0, "average_precision": 1.0})
        self.assertEqual(models.weighted_auc_ap(y, 1 - y)["auroc"], 0.0)

    def test_degenerate_or_invalid_metric_inputs_fail(self):
        invalid = [([1, 1], [0.1, 0.2], None), ([0, 1], [0.1, 0.2], [1, 0]),
                   ([0, 1], [np.nan, 0.2], None), ([0, 1], [0.1, 0.2], [-1, 2]),
                   ([0, 1], [0.1, 0.2], [0, 0]), ([0, 2], [0.1, 0.2], None),
                   ([0, 1], [0.1], None), ([0, 1], [0.1, 0.2], [1, np.inf])]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                models.weighted_auc_ap(*arguments)


class ModelFittingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.X = rng.normal(size=(120, 4))
        self.y = (self.X[:, 0] + 0.4 * self.X[:, 1] + rng.normal(size=120) > 0).astype(int)
        self.X_val = rng.normal(size=(50, 4))
        self.y_val = (self.X_val[:, 0] + rng.normal(size=50) > 0).astype(int)
        self.X_test = rng.normal(size=(13, 4))

    def test_logistic_outputs_and_validation_selection(self):
        result = models.fit_logistic(self.X, self.y, self.X_val, self.y_val, self.X_test)
        self.assertEqual(len(result["validation_metrics"]), 4)
        best = max(row["auroc"] for row in result["validation_metrics"])
        expected = min(row["C"] for row in result["validation_metrics"] if best - row["auroc"] <= 1e-8)
        self.assertEqual(result["selected_C"], expected)
        self.assertEqual(result["coefficient_count"], 4)
        self.assertLess(result["iterations"], 2000)
        self.assertEqual(result["test_probabilities"].shape, (13,))
        self.assertEqual(result["validation_probabilities"].shape, (50,))
        selected = next(row for row in result["validation_metrics"] if row["C"] == expected)
        self.assertAlmostEqual(selected["auroc"], roc_auc_score(self.y_val, result["validation_probabilities"]))

    def test_lr_bootstrap_weights_match_explicit_repetition(self):
        rng = np.random.default_rng(55)
        weights = rng.multinomial(len(self.X), np.ones(len(self.X)) / len(self.X))
        val_weights = rng.multinomial(len(self.X_val), np.ones(len(self.X_val)) / len(self.X_val))
        weighted = models.fit_logistic(self.X, self.y, self.X_val, self.y_val, self.X_test,
                                       sample_weight=weights, validation_sample_weight=val_weights)
        train_index = np.repeat(np.arange(len(weights)), weights)
        val_index = np.repeat(np.arange(len(val_weights)), val_weights)
        repeated = models.fit_logistic(self.X[train_index], self.y[train_index], self.X_val[val_index],
                                       self.y_val[val_index], self.X_test)
        self.assertEqual(weighted["selected_C"], repeated["selected_C"])
        np.testing.assert_allclose(weighted["test_probabilities"], repeated["test_probabilities"], atol=1e-10)

    def test_all_ties_select_smallest_C_and_nonconvergence_fails(self):
        result = models.fit_logistic(np.zeros_like(self.X), self.y, np.zeros_like(self.X_val),
                                     self.y_val, np.zeros_like(self.X_test))
        self.assertEqual(result["selected_C"], 0.01)
        with patch.object(models.LogisticRegression, "fit", side_effect=ConvergenceWarning("synthetic")):
            with self.assertRaises(RuntimeError):
                models.fit_logistic(self.X, self.y, self.X_val, self.y_val, self.X_test)

    def test_test_features_do_not_affect_choice_and_no_test_label_argument(self):
        first = models.fit_logistic(self.X, self.y, self.X_val, self.y_val, self.X_test)
        second = models.fit_logistic(self.X, self.y, self.X_val, self.y_val, self.X_test * 50)
        self.assertEqual(first["selected_C"], second["selected_C"])
        self.assertEqual(first["validation_metrics"], second["validation_metrics"])
        for function in [models.fit_logistic, models.fit_xgboost]:
            self.assertNotIn("y_test", inspect.signature(function).parameters)
            with self.assertRaises(TypeError):
                function(self.X, self.y, self.X_val, self.y_val, self.X_test, y_test=np.zeros(13))

    def test_invalid_model_inputs_fail_before_fitting(self):
        for function in [models.fit_logistic, models.fit_xgboost]:
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function(self.X, np.zeros_like(self.y), self.X_val, self.y_val, self.X_test)
                with self.assertRaises(ValueError):
                    function(self.X, self.y, self.X_val, np.ones_like(self.y_val), self.X_test)
                with self.assertRaises(ValueError):
                    function(self.X, self.y, self.X_val, self.y_val, self.X_test[:, :3])
                with self.assertRaises(ValueError):
                    function(self.X, self.y, self.X_val, self.y_val, self.X_test,
                             sample_weight=(self.y == 1).astype(int))
                with self.assertRaises(ValueError):
                    function(self.X, self.y, self.X_val, self.y_val, self.X_test,
                             validation_sample_weight=(self.y_val == 0).astype(int))

    def test_xgboost_fixed_grid_runs_on_synthetic_data(self):
        result = models.fit_xgboost(self.X, self.y, self.X_val, self.y_val, self.X_test)
        self.assertEqual([row["max_depth"] for row in result["validation_metrics"]], [2, 4, 6])
        self.assertEqual(result["tree_count"], 300)
        self.assertEqual(result["test_probabilities"].shape, (13,))
        best = max(row["auroc"] for row in result["validation_metrics"])
        expected = min(row["max_depth"] for row in result["validation_metrics"] if best - row["auroc"] <= 1e-8)
        self.assertEqual(result["selected_max_depth"], expected)


if __name__ == "__main__":
    unittest.main()


