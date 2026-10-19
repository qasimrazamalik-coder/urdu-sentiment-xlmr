"""Classical baseline: word and character TF-IDF into a linear classifier.

Character n-grams matter here. Roman Urdu has no standard spelling ("bohat", "bohot",
"boht"), and char n-grams absorb much of that variation without any tokenizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def build_pipeline(cfg: dict[str, Any], c_value: float, seed: int) -> Pipeline:
    """Assemble the TF-IDF union and the chosen linear classifier."""
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=tuple(cfg["word_ngram"]),
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w+\b",
                    max_features=int(cfg["max_features"]),
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(cfg["char_ngram"]),
                    sublinear_tf=True,
                    max_features=int(cfg["max_features"]),
                ),
            ),
        ]
    )
    kind = cfg["classifier"]
    if kind == "logreg":
        clf = LogisticRegression(C=c_value, max_iter=2000, random_state=seed)
    elif kind == "linsvm":
        clf = LinearSVC(C=c_value, random_state=seed)
    else:
        raise ValueError(f"baseline.classifier must be 'logreg' or 'linsvm', got {kind!r}")
    return Pipeline([("features", features), ("clf", clf)])


def fit_baseline(
    cfg: dict[str, Any],
    train_texts: list[str],
    train_y: list[int],
    val_texts: list[str],
    val_y: list[int],
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit on train; pick C by validation macro-F1 so the test set is never used for tuning."""
    best: tuple[float, float, Pipeline] | None = None
    tried = {}
    for c_value in cfg["c_grid"]:
        pipe = build_pipeline(cfg, float(c_value), seed).fit(train_texts, train_y)
        score = f1_score(val_y, pipe.predict(val_texts), average="macro", zero_division=0)
        tried[str(c_value)] = float(score)
        if best is None or score > best[0]:
            best = (float(score), float(c_value), pipe)
    assert best is not None
    return best[2], {"best_c": best[1], "val_macro_f1_by_c": tried, "classifier": cfg["classifier"]}


def predict_scores(pipe: Pipeline, texts: list[str]) -> np.ndarray:
    """Class probabilities (logreg) or softmax of margins (linear SVM, not calibrated)."""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(texts)
    margins = np.asarray(pipe.decision_function(texts))
    if margins.ndim == 1:
        margins = np.stack([-margins, margins], axis=1)
    margins = margins - margins.max(axis=1, keepdims=True)
    exp = np.exp(margins)
    return exp / exp.sum(axis=1, keepdims=True)


def save_baseline(pipe: Pipeline, labels: list[str], path: str | Path) -> None:
    """Persist the fitted pipeline with its label order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": labels}, path)


def load_baseline(path: str | Path) -> tuple[Pipeline, list[str]]:
    """Load a baseline written by `save_baseline`. joblib uses pickle: only load files you made."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No baseline at {path}. Run `make train` first.")
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle["labels"]
