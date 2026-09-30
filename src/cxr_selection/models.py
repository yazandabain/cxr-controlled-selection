"""Pure supervised-analysis helpers for the additive CXR study.

No function reads patient files, opens test labels, fits a preprocessor, or writes
artifacts. Inputs to the model routines must already have been transformed by a
preprocessor fitted only on the caller's declared training reference. The main
protocol freezes a common unlabeled preprocessing reference; any resampling
interpretation must explicitly retain that conditioning. Sampling indices are positional indices
into the supplied training panel; no identifier is included in model outputs.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


DEFAULT_BUDGETS = (512, 1536, 3072)
DEFAULT_SEEDS = tuple(range(20))
LOGISTIC_C_VALUES = (0.01, 0.1, 1.0, 10.0)
XGBOOST_DEPTHS = (2, 4, 6)
TIE_TOLERANCE = 1e-8


def _identifier_key(value: Any) -> tuple[int, Any]:
    """Sort integer-like study/patient identifiers numerically, others lexically."""
    if pd.isna(value):
        raise ValueError("Identifiers must be nonmissing scalars.")
    if isinstance(value, numbers.Integral):
        return 0, int(value)
    if isinstance(value, numbers.Real):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError("Numeric identifiers must be finite integers.")
        return 0, int(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Identifiers must be nonempty strings or integers.")
    value = value.strip()
    if value.isdecimal():
        return 0, int(value)
    return 1, value


def select_one_study_per_patient(
    frame: pd.DataFrame,
    patient_col: str = "subject_id",
    study_col: str = "study_id",
    dicom_col: str = "dicom_id",
) -> pd.DataFrame:
    """Select lowest study ID, then lexical DICOM ID, before report eligibility.

    The result is sorted by patient ID and preserves input columns and original
    row indices. This function does not inspect report states or outcome labels.
    Ambiguous duplicate patient/study/DICOM keys fail rather than depend on input
    row order. Apply this to the pre-eligibility pool, not an explicit-only pool.
    """
    required = (patient_col, study_col, dicom_col)
    if any(column not in frame.columns for column in required):
        raise ValueError("The patient, study, and DICOM columns are required.")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are unsupported.")
    if frame.empty:
        return frame.copy()
    patient_keys = [_identifier_key(value) for value in frame[patient_col]]
    study_keys = [_identifier_key(value) for value in frame[study_col]]
    dicom_keys = []
    for value in frame[dicom_col]:
        if pd.isna(value) or not isinstance(value, str) or not value.strip():
            raise ValueError("DICOM identifiers must be nonempty strings.")
        dicom_keys.append(value.strip())
    row_keys = list(zip(patient_keys, study_keys, dicom_keys))
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("Duplicate patient/study/DICOM keys are ambiguous.")
    order = sorted(range(len(frame)), key=row_keys.__getitem__)
    selected = []
    seen = set()
    for position in order:
        if patient_keys[position] not in seen:
            selected.append(position)
            seen.add(patient_keys[position])
    return frame.iloc[selected].copy()


def _positive_integers(values: Iterable[int], name: str, *, allow_zero: bool = False) -> tuple[int, ...]:
    values = tuple(values)
    minimum = 0 if allow_zero else 1
    if not values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Integral)
        or value < minimum
        for value in values
    ):
        raise ValueError(f"{name} must be a nonempty sequence of valid integers.")
    result = tuple(int(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _binary_labels(labels: Any, *, expected_length: int | None = None) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or (expected_length is not None and len(labels) != expected_length):
        raise ValueError("Labels must be a one-dimensional vector matching the observations.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Labels must be observed binary values, zero or one.")
    return labels.astype(np.int64, copy=False)


def sample_training_panels(
    patient_ids: Any,
    report_states: Any,
    labels: Any,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> dict[int, dict[int, dict[str, np.ndarray]]]:
    """Return seed -> budget -> arm -> training-panel positional indices.

    Each patient must occur once. Canonical patient sorting makes selected
    patients invariant to input row order. Every arm uses a fixed seeded
    permutation, so smaller budgets are subsets of larger budgets within a seed.
    ``prevalence_matched`` takes case/control permutation prefixes with exactly
    the positive/negative counts of that seed's explicit sample at that budget.
    Its arrays need not be literal prefixes, but their patient sets are nested.

    Arms are ``explicit`` (report state zero/one), ``random`` (all states), and
    ``prevalence_matched`` (all states). Sampling is without replacement within
    each panel. Overlap between arms is permitted. No outcome-based retries are
    made; a one-class panel, if sampled, must fail at model fitting.
    """
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1 or not len(patient_ids):
        raise ValueError("A nonempty one-dimensional patient vector is required.")
    patient_keys = [_identifier_key(value) for value in patient_ids]
    if len(set(patient_keys)) != len(patient_keys):
        raise ValueError("The training panel must contain exactly one row per patient.")
    labels = _binary_labels(labels, expected_length=len(patient_ids))
    try:
        states = pd.to_numeric(pd.Series(report_states), errors="raise").to_numpy(
            dtype=float, na_value=np.nan
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Report states must be zero, one, minus one, or missing.") from exc
    if states.ndim != 1 or len(states) != len(patient_ids):
        raise ValueError("Report states must match the patient vector.")
    if not np.all(np.isnan(states) | np.isin(states, [-1, 0, 1])):
        raise ValueError("Report states must be zero, one, minus one, or missing.")
    budgets = tuple(sorted(_positive_integers(budgets, "Budgets")))
    seeds = _positive_integers(seeds, "Seeds", allow_zero=True)
    order = np.asarray(sorted(range(len(patient_ids)), key=patient_keys.__getitem__), dtype=np.int64)
    explicit_pool = order[np.isin(states[order], [0, 1])]
    if budgets[-1] > len(explicit_pool):
        raise ValueError("The largest budget exceeds the number of explicit-report patients.")
    positive_pool = order[labels[order] == 1]
    negative_pool = order[labels[order] == 0]
    result = {}
    for seed in seeds:
        # Separate fixed streams prevent an added budget from changing samples.
        streams = [np.random.default_rng(np.random.SeedSequence([seed, arm])) for arm in range(4)]
        explicit_order = streams[0].permutation(explicit_pool)
        random_order = streams[1].permutation(order)
        positive_order = streams[2].permutation(positive_pool)
        negative_order = streams[3].permutation(negative_pool)
        panels = {}
        for budget in budgets:
            explicit = explicit_order[:budget].copy()
            positive_count = int(labels[explicit].sum())
            negative_count = budget - positive_count
            matched = np.concatenate((positive_order[:positive_count], negative_order[:negative_count]))
            # Availability follows from explicit_pool being a subset of all rows.
            if len(matched) != budget:
                raise ValueError("Insufficient case/control support for the matched budget.")
            panels[budget] = {
                "explicit": explicit,
                "random": random_order[:budget].copy(),
                "prevalence_matched": matched,
            }
        result[seed] = panels
    return result


def _weights(sample_weight: Any, length: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(length, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1 or len(weights) != length:
        raise ValueError("Sample weights must match the observations.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Sample weights must be finite, nonnegative, and not all zero.")
    return weights


def _require_two_classes(labels: np.ndarray, weights: np.ndarray) -> None:
    if not np.any((labels == 0) & (weights > 0)) or not np.any((labels == 1) & (weights > 0)):
        raise ValueError("Both classes must have positive total sample weight.")


def weighted_auc_ap(labels: Any, scores: Any, sample_weight: Any = None) -> dict[str, float]:
    """Exact empirical AUROC and non-interpolated AP with tied-score groups.

    Integer weights are patient bootstrap multiplicities in the one-row-per-
    patient panel; nonnegative fractional weights are also accepted. The AUROC
    is the weighted positive-negative ranking probability, with half credit for
    ties. AP sums precision times recall increments at distinct score thresholds,
    matching sklearn.metrics.average_precision_score (not trapezoidal PR area).
    This direct implementation is independently checked against sklearn and
    explicit repeated-row calculations in test_models.py.
    """
    labels = _binary_labels(labels)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(labels) or not np.all(np.isfinite(scores)):
        raise ValueError("Scores must be a finite vector matching the labels.")
    weights = _weights(sample_weight, len(labels))
    _require_two_classes(labels, weights)
    active = weights > 0
    labels, scores, weights = labels[active], scores[active], weights[active]
    order = np.argsort(scores, kind="stable")
    labels, scores, weights = labels[order], scores[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
    positives = np.add.reduceat(weights * labels, starts)
    negatives = np.add.reduceat(weights * (1 - labels), starts)
    positive_total = float(positives.sum())
    negative_total = float(negatives.sum())
    negatives_before = np.cumsum(negatives) - negatives
    auc = float(np.sum(positives * (negatives_before + 0.5 * negatives)) / positive_total / negative_total)
    positive_desc = positives[::-1]
    negative_desc = negatives[::-1]
    cumulative_positive = np.cumsum(positive_desc)
    precision = cumulative_positive / np.cumsum(positive_desc + negative_desc)
    ap = float(np.sum((positive_desc / positive_total) * precision))
    return {"auroc": auc, "average_precision": ap}


def _model_inputs(X: Any, y: Any, X_val: Any, y_val: Any, X_test: Any,
                  sample_weight: Any, validation_sample_weight: Any) -> tuple[np.ndarray, ...]:
    matrices = [np.asarray(matrix, dtype=float) for matrix in (X, X_val, X_test)]
    if any(matrix.ndim != 2 or not np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("Training, validation, and test inputs must be finite two-dimensional matrices.")
    X, X_val, X_test = matrices
    if not X.shape[0] or not X_val.shape[0] or not X.shape[1]:
        raise ValueError("Training and validation sets and the feature dimension must be nonempty.")
    if X_val.shape[1] != X.shape[1] or X_test.shape[1] != X.shape[1]:
        raise ValueError("Training, validation, and test feature dimensions must agree.")
    y = _binary_labels(y, expected_length=len(X))
    y_val = _binary_labels(y_val, expected_length=len(X_val))
    train_weights = _weights(sample_weight, len(y))
    validation_weights = _weights(validation_sample_weight, len(y_val))
    _require_two_classes(y, train_weights)
    _require_two_classes(y_val, validation_weights)
    return X, y, X_val, y_val, X_test, train_weights, validation_weights


def _fit_candidates(
    X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, train_weights: np.ndarray, validation_weights: np.ndarray,
    candidates: tuple[Any, ...], parameter_name: str, factory: Any,
) -> tuple[Any, dict[str, Any]]:
    fitted = []
    validation_metrics = []
    for candidate in candidates:
        model = factory(candidate)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X, y, sample_weight=train_weights)
        except ConvergenceWarning as exc:
            raise RuntimeError("A candidate model did not converge; selection was aborted.") from exc
        probabilities = np.asarray(model.predict_proba(X_val)[:, 1], dtype=float)
        metrics = weighted_auc_ap(y_val, probabilities, validation_weights)
        metrics[parameter_name] = candidate
        if hasattr(model, "n_iter_"):
            metrics["iterations"] = int(np.max(model.n_iter_))
        validation_metrics.append(metrics)
        fitted.append((model, probabilities))
    # Compare every candidate to the global maximum to avoid tolerance chaining.
    best_auc = max(metrics["auroc"] for metrics in validation_metrics)
    chosen = next(index for index, metrics in enumerate(validation_metrics)
                  if best_auc - metrics["auroc"] <= TIE_TOLERANCE)
    model, probabilities = fitted[chosen]
    test_probabilities = (np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
                          if len(X_test) else np.empty(0, dtype=float))
    if not np.all(np.isfinite(test_probabilities)):
        raise RuntimeError("The selected model produced nonfinite test probabilities.")
    result = {
        "selected_" + parameter_name: candidates[chosen],
        "validation_metrics": validation_metrics,
        "validation_probabilities": probabilities,
        "test_probabilities": test_probabilities,
        "feature_count": int(X.shape[1]),
        "training_rows": int(len(y)),
        "training_weight_sum": float(train_weights.sum()),
        "validation_rows": int(len(y_val)),
        "validation_weight_sum": float(validation_weights.sum()),
    }
    return model, result


def fit_logistic(
    X: Any, y: Any, X_val: Any, y_val: Any, X_test: Any, *,
    sample_weight: Any = None, validation_sample_weight: Any = None,
) -> dict[str, Any]:
    """Tune fixed-grid L2 LR on validation AUROC; never access test labels.

    All four C values are fitted to the supplied training rows. Among values
    within 1e-8 of maximum validation AUROC, choose the smallest C. The selected
    training fit supplies predictions without refitting on validation patients.
    Weights are used both in training and in validation selection when provided.
    Any candidate's convergence warning fails the entire routine.
    """
    inputs = _model_inputs(X, y, X_val, y_val, X_test, sample_weight, validation_sample_weight)
    model, result = _fit_candidates(
        *inputs, candidates=LOGISTIC_C_VALUES, parameter_name="C",
        factory=lambda C: LogisticRegression(
            C=C, solver="lbfgs", tol=1e-6, max_iter=2000,
            fit_intercept=True, class_weight=None,
        ),
    )
    result.update({
        "coefficient_count": int(model.coef_.size),
        "nonzero_coefficient_count": int(np.count_nonzero(model.coef_)),
        "intercept_count": int(model.intercept_.size),
        "iterations": int(np.max(model.n_iter_)),
        "model_state": {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
                        "classes": model.classes_.tolist()},
    })
    return result


def _make_xgboost_classifier(**parameters: Any) -> Any:
    # Lazy import keeps score-only and LR replay independent of XGBoost loading.
    from xgboost import XGBClassifier
    return XGBClassifier(**parameters)


def fit_xgboost(
    X: Any, y: Any, X_val: Any, y_val: Any, X_test: Any, *,
    sample_weight: Any = None, validation_sample_weight: Any = None,
) -> dict[str, Any]:
    """Fit the locked small CPU-hist tree grid; no early stopping or test labels."""
    inputs = _model_inputs(X, y, X_val, y_val, X_test, sample_weight, validation_sample_weight)
    model, result = _fit_candidates(
        *inputs, candidates=XGBOOST_DEPTHS, parameter_name="max_depth",
        factory=lambda depth: _make_xgboost_classifier(
            max_depth=depth, n_estimators=300, learning_rate=0.03,
            min_child_weight=10, subsample=1.0, colsample_bytree=1.0,
            reg_lambda=10.0, n_jobs=4, tree_method="hist", device="cpu",
            objective="binary:logistic", eval_metric="logloss", random_state=0,
        ),
    )
    result.update({"tree_count": int(model.get_booster().num_boosted_rounds()),
                   "coefficient_count": None, "iterations": 300,
                   "model_bytes": bytes(model.get_booster().save_raw(raw_format="ubj"))})
    return result


