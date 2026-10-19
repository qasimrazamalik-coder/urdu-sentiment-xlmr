import json

import pandas as pd

from urdu_sentiment import error_analysis, evaluate
from urdu_sentiment.predict import MAX_CHARS, Predictor
from urdu_sentiment.train import run
from urdu_sentiment.utils import load_config


def _tiny_cfg(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg["data"]["synthetic_rows"] = 200
    cfg["model"].update(epochs=1, max_length=24)
    cfg["evaluation"]["bootstrap_samples"] = 50
    cfg["output"] = {
        "dir": str(tmp_path / "out"),
        "model_dir": str(tmp_path / "model"),
        "baseline_path": str(tmp_path / "b.joblib"),
    }
    return cfg


def test_end_to_end_synthetic_run(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    out = run(cfg)
    info = json.loads((out / "run_info.json").read_text(encoding="utf-8"))
    assert info["synthetic"] is True and info["seed"] == cfg["seed"]
    assert "torch" in info["versions"]

    predictions = pd.read_csv(out / "predictions_test.csv", dtype=str, keep_default_na=False)
    labels = info["data"]["labels"]
    metrics = evaluate.evaluate_predictions(predictions, labels, cfg["evaluation"], cfg["seed"])
    assert set(metrics["models"]) == {"baseline", "transformer"}
    assert len(metrics["comparison"]["diff_ci95"]) == 2
    assert set(metrics["models"]["baseline"]["per_script"]) == {"urdu", "roman"}
    table = evaluate.render_markdown({**metrics, "synthetic": True, "seed": 1})
    assert "SYNTHETIC" in table

    predictions["n_words"] = predictions["n_words"].astype(int)
    report, errors = error_analysis.analyse(predictions, labels, 5, 10, "transformer")
    assert set(report["transformer"]["by_length"]) <= {"short", "long"}
    assert {"text", "label", "predicted"} <= set(errors.columns)
    assert errors["text"].str.len().is_monotonic_decreasing

    predictor = Predictor(cfg)
    result = predictor.predict("یہ فلم بہت اچھی ہے")
    assert result["script"] == "urdu"
    assert set(result["models"]) == {"transformer", "baseline"}
    assert abs(sum(result["models"]["baseline"]) - 1.0) < 1e-6


def test_baseline_only_run_skips_transformer(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    out = run(cfg, baseline_only=True)
    predictions = pd.read_csv(out / "predictions_test.csv")
    assert "transformer_pred" not in predictions.columns


def test_app_builds_without_weights(tmp_path):
    from app import build_app, validate

    cfg = _tiny_cfg(tmp_path)
    predictor = Predictor(cfg)
    assert not predictor.ready
    assert build_app(predictor) is not None
    assert build_app(None) is not None
    assert validate("") and validate("   ") and validate(None)
    assert validate("x" * (MAX_CHARS + 1))
    assert validate("yeh film acha hai") is None
