"""Metrics and the paired bootstrap used to compare two models on the same test set."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 from a bincount confusion matrix; ~50x faster than sklearn inside bootstraps.

    Classes absent from both truth and prediction are skipped, and a class with no true
    and no predicted samples in a resample contributes nothing rather than a zero.
    """
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes**2).reshape(n_classes, -1)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = support + predicted
    present = denom > 0
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1[present].mean())


def summarize(y_true: Sequence[int], y_pred: Sequence[int], labels: list[str]) -> dict:
    """Accuracy, macro-F1, per-class F1 and the confusion matrix for one prediction set."""
    ids = list(range(len(labels)))
    per_class = f1_score(y_true, y_pred, labels=ids, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=ids, average="macro", zero_division=0)),
        "per_class_f1": {name: float(v) for name, v in zip(labels, per_class, strict=True)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def per_group(
    y_true: Sequence[int], y_pred: Sequence[int], groups: Sequence[str], labels: list[str]
) -> dict[str, dict]:
    """Accuracy and macro-F1 for each group (script, length bucket). Small groups are flagged."""
    y_true_arr, y_pred_arr, group_arr = map(np.asarray, (y_true, y_pred, groups))
    out: dict[str, dict] = {}
    for group in sorted(set(group_arr.tolist())):
        mask = group_arr == group
        entry = summarize(y_true_arr[mask], y_pred_arr[mask], labels)
        entry["small_sample"] = bool(mask.sum() < 30)
        out[group] = entry
    return out


def _ci95(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def bootstrap_macro_f1(
    y_true: Sequence[int],
    pred_a: Sequence[int],
    pred_b: Sequence[int] | None,
    n_classes: int,
    n_boot: int,
    seed: int,
) -> dict:
    """Percentile bootstrap 95% CI on macro-F1 of A, and on B - A when B is given.

    Test rows are resampled once per replicate and both models are scored on the same
    resample, so the difference interval reflects the paired comparison, not two
    independent ones. A CI that contains 0 means the test set cannot separate the models.
    """
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b) if pred_b is not None else None
    rng = np.random.default_rng(seed)
    n = len(y)
    scores_a = np.empty(n_boot)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores_a[i] = macro_f1_fast(y[idx], a[idx], n_classes)
        if b is not None:
            diffs[i] = macro_f1_fast(y[idx], b[idx], n_classes) - scores_a[i]
    result = {
        "macro_f1_ci95": [
            float(np.percentile(scores_a, 2.5)),
            float(np.percentile(scores_a, 97.5)),
        ],
        "n_boot": n_boot,
    }
    if b is not None:
        point = macro_f1_fast(y, b, n_classes) - macro_f1_fast(y, a, n_classes)
        result["diff_macro_f1"] = float(point)
        result["diff_ci95"] = _ci95(diffs)
    return result
