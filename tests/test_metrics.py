import numpy as np
from sklearn.metrics import f1_score

from urdu_sentiment.metrics import bootstrap_macro_f1, macro_f1_fast, per_group, summarize

LABELS = ["neg", "neu", "pos"]


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, 200))
    assert np.isclose(macro_f1_fast(y, p, 3), f1_score(y, p, average="macro"))


def test_summarize_perfect_and_confusion():
    y = [0, 1, 2, 2]
    out = summarize(y, y, LABELS)
    assert out["accuracy"] == 1.0 and out["macro_f1"] == 1.0
    wrong = summarize([0, 1, 2, 2], [0, 1, 2, 0], LABELS)
    assert wrong["confusion_matrix"][2] == [1, 0, 1]
    assert wrong["per_class_f1"]["neu"] == 1.0


def test_per_group_splits_and_flags_small():
    y = [0, 1, 2, 0, 1, 2]
    p = [0, 1, 2, 1, 1, 0]
    out = per_group(y, p, ["urdu"] * 3 + ["roman"] * 3, LABELS)
    assert out["urdu"]["accuracy"] == 1.0
    assert out["roman"]["accuracy"] < 1.0
    assert out["urdu"]["small_sample"] is True


def test_bootstrap_ci_brackets_point_and_is_seeded():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    good = np.where(rng.random(300) < 0.9, y, rng.integers(0, 3, 300))
    bad = np.where(rng.random(300) < 0.5, y, rng.integers(0, 3, 300))
    first = bootstrap_macro_f1(y, bad, good, 3, 300, seed=4)
    again = bootstrap_macro_f1(y, bad, good, 3, 300, seed=4)
    assert first == again
    lo, hi = first["macro_f1_ci95"]
    assert lo <= macro_f1_fast(y, bad, 3) <= hi
    dlo, dhi = first["diff_ci95"]
    assert dlo > 0, "a clearly better model should have a CI above zero"
    assert dlo <= first["diff_macro_f1"] <= dhi


def test_bootstrap_identical_models_ci_contains_zero():
    y = np.array([0, 1, 2] * 50)
    p = np.where(np.arange(150) % 5 == 0, 0, y)
    out = bootstrap_macro_f1(y, p, p, 3, 200, seed=0)
    assert out["diff_macro_f1"] == 0.0 and out["diff_ci95"] == [0.0, 0.0]
