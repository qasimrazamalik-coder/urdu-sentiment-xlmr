import numpy as np
import pytest

from urdu_sentiment.baseline import fit_baseline, load_baseline, predict_scores, save_baseline

CFG = {
    "classifier": "logreg",
    "c_grid": [1.0, 10.0],
    "word_ngram": [1, 2],
    "char_ngram": [2, 4],
    "max_features": 5000,
}


def _xy(splits):
    return (
        (list(splits.train["text"]), splits.encode(splits.train)),
        (list(splits.val["text"]), splits.encode(splits.val)),
        (list(splits.test["text"]), splits.encode(splits.test)),
    )


def test_baseline_fit_predict_and_roundtrip(splits, tmp_path):
    (tx, ty), (vx, vy), (sx, sy) = _xy(splits)
    pipe, info = fit_baseline(CFG, tx, ty, vx, vy, seed=0)
    assert info["best_c"] in CFG["c_grid"]
    scores = predict_scores(pipe, sx)
    assert scores.shape == (len(sx), 3)
    assert np.allclose(scores.sum(axis=1), 1.0)
    # Synthetic sentences are trivially separable: this checks wiring, not quality.
    assert (scores.argmax(axis=1) == np.array(sy)).mean() > 0.9

    save_baseline(pipe, splits.labels, tmp_path / "b.joblib")
    loaded, labels = load_baseline(tmp_path / "b.joblib")
    assert labels == splits.labels
    assert np.allclose(predict_scores(loaded, sx), scores)


def test_linear_svm_scores_are_a_distribution(splits):
    (tx, ty), (vx, vy), (sx, _) = _xy(splits)
    pipe, _ = fit_baseline({**CFG, "classifier": "linsvm"}, tx, ty, vx, vy, seed=0)
    scores = predict_scores(pipe, sx)
    assert np.allclose(scores.sum(axis=1), 1.0)


def test_unknown_classifier_fails_loudly(splits):
    (tx, ty), (vx, vy), _ = _xy(splits)
    with pytest.raises(ValueError, match="classifier"):
        fit_baseline({**CFG, "classifier": "svm-rbf"}, tx, ty, vx, vy, seed=0)


def test_load_missing_baseline(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        load_baseline(tmp_path / "nope.joblib")
