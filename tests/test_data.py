import pandas as pd
import pytest

from urdu_sentiment.data import (
    clean_table,
    detect_script,
    load_table,
    normalize_text,
    split_dataset,
)
from urdu_sentiment.synthetic import make_synthetic


def test_detect_script_cases():
    assert detect_script("یہ فلم بہت اچھی ہے") == "urdu"
    assert detect_script("yeh film bohat achi hai") == "roman"
    assert detect_script("یہ movie بہت اچھی ہے") == "mixed"
    assert detect_script("1234 !!!") == "unknown"
    assert detect_script("") == "unknown"


def test_synthetic_generator_is_deterministic_unique_and_labelled():
    a, b = make_synthetic(120, seed=5), make_synthetic(120, seed=5)
    pd.testing.assert_frame_equal(a, b)
    assert a["text"].is_unique
    assert set(a["label"]) == {"negative", "neutral", "positive"}
    assert set(a["script"]) == {"urdu", "roman"}
    assert (a["text"].map(detect_script) == a["script"]).all()


def test_normalize_text_unifies_variants():
    assert normalize_text("  Yeh   FILM ​ ") == "yeh film"
    assert normalize_text("كتاب") == normalize_text("کتاب")
    assert normalize_text("اچھّا") == normalize_text("اچھا")


def test_split_is_deterministic_disjoint_and_stratified(synthetic_frame):
    kwargs = {"val_size": 0.15, "test_size": 0.25, "seed": 11}
    a, b = split_dataset(synthetic_frame, **kwargs), split_dataset(synthetic_frame, **kwargs)
    for part in ("train", "val", "test"):
        pd.testing.assert_frame_equal(getattr(a, part), getattr(b, part))
    other = split_dataset(synthetic_frame, **{**kwargs, "seed": 12})
    assert not a.test["text"].equals(other.test["text"])
    sets = [set(f["text"]) for f in (a.train, a.val, a.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(f) for f in (a.train, a.val, a.test)) == len(synthetic_frame)
    overall = synthetic_frame["label"].value_counts(normalize=True)
    test_share = a.test["label"].value_counts(normalize=True)
    assert (overall - test_share).abs().max() < 0.05


def test_split_rejects_tiny_classes():
    frame = pd.DataFrame(
        {"text": list("abcdefgh"), "label": ["a"] * 6 + ["b"] * 2, "script": ["roman"] * 8}
    )
    with pytest.raises(ValueError, match="at least 3"):
        split_dataset(frame, val_size=0.1, test_size=0.2, seed=0)


def test_clean_table_fills_script_and_reports_drops():
    raw = pd.DataFrame(
        {
            "text": ["yeh acha hai", "yeh acha hai", "یہ برا ہے", None, "x"],
            "label": ["pos", "pos", "neg", "neg", "neg"],
        }
    )
    clean, report = clean_table(raw)
    assert list(clean["script"]) == ["roman", "urdu"]
    assert report["dropped_missing"] == 1
    assert report["dropped_duplicates"] == 1
    assert report["dropped_too_short"] == 1


def test_clean_table_keeps_valid_script_and_fixes_invalid():
    raw = pd.DataFrame(
        {"text": ["yeh acha hai", "یہ برا ہے"], "label": ["p", "n"], "script": ["URDU", "garbage"]}
    )
    clean, _ = clean_table(raw)
    assert list(clean["script"]) == ["urdu", "urdu"]


def test_load_table_csv_jsonl_and_errors(tmp_path):
    frame = pd.DataFrame({"text": ["a b", "c d"], "label": ["x", "y"]})
    frame.to_csv(tmp_path / "d.csv", index=False)
    frame.to_json(tmp_path / "d.jsonl", orient="records", lines=True)
    assert len(load_table(tmp_path / "d.csv")) == 2
    assert len(load_table(tmp_path / "d.jsonl")) == 2
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_table(tmp_path / "missing.csv")
    frame.rename(columns={"text": "sentence"}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="text"):
        load_table(tmp_path / "bad.csv")
    (tmp_path / "d.txt").write_text("x")
    with pytest.raises(ValueError, match="Unsupported"):
        load_table(tmp_path / "d.txt")
