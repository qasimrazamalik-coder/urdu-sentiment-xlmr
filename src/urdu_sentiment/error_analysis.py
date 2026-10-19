"""Error analysis on saved test predictions: confusions, long errors, short vs long, scripts.

Usage: python -m urdu_sentiment.error_analysis --config configs/default.yaml [--model transformer]
Writes error_analysis.json and errors.csv (misclassified rows, longest first) to the output dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import MODEL_COLUMNS
from .metrics import per_group
from .utils import PREDICTIONS_FILE, RUN_INFO_FILE, load_config, read_json, write_json


def top_confusions(frame: pd.DataFrame, pred_col: str, k: int = 5) -> list[dict[str, Any]]:
    """Most frequent (true, predicted) pairs among wrong predictions."""
    wrong = frame[frame["label"] != frame[pred_col]]
    counts = wrong.groupby(["label", pred_col]).size().sort_values(ascending=False).head(k)
    return [
        {"true": true, "predicted": pred, "count": int(n)} for (true, pred), n in counts.items()
    ]


def length_bucket(n_words: pd.Series, short_max: int) -> pd.Series:
    return n_words.map(lambda n: "short" if n <= short_max else "long")


def analyse(
    predictions: pd.DataFrame, labels: list[str], short_max_words: int, top_k: int, model: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Return the analysis dictionary and the error table for `model`."""
    index = {name: i for i, name in enumerate(labels)}
    frame = predictions.copy()
    frame["length_bucket"] = length_bucket(frame["n_words"], short_max_words)
    present = {n: c for n, c in MODEL_COLUMNS.items() if c in frame.columns}
    if model not in present:
        raise ValueError(f"Model {model!r} has no predictions; available: {sorted(present)}")

    report: dict[str, Any] = {"focus_model": model, "short_max_words": short_max_words}
    y_true = frame["label"].map(index)
    for name, col in present.items():
        y_pred = frame[col].map(index)
        report[name] = {
            "top_confusions": top_confusions(frame, col),
            "by_length": per_group(y_true, y_pred, frame["length_bucket"], labels),
            "by_script": per_group(y_true, y_pred, frame["script"], labels),
            "top_confusions_by_script": {
                script: top_confusions(part, col, 3) for script, part in frame.groupby("script")
            },
        }
    if len(present) == 2:
        base_ok = frame["baseline_pred"] == frame["label"]
        tr_ok = frame["transformer_pred"] == frame["label"]
        report["disagreement"] = {
            script: {
                "only_baseline_correct": int((base_ok & ~tr_ok)[part.index].sum()),
                "only_transformer_correct": int((~base_ok & tr_ok)[part.index].sum()),
                "both_wrong": int((~base_ok & ~tr_ok)[part.index].sum()),
            }
            for script, part in frame.groupby("script")
        }

    pred_col = present[model]
    errors = frame[frame["label"] != frame[pred_col]].copy()
    errors["n_chars"] = errors["text"].str.len()
    errors = errors.sort_values("n_chars", ascending=False).head(top_k)
    keep = ["text", "label", pred_col, "script", "length_bucket", "n_words", "n_chars"]
    return report, errors[keep].rename(columns={pred_col: "predicted"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default=None, help="baseline or transformer")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output"]["dir"])
    info = read_json(out_dir / RUN_INFO_FILE)
    path = out_dir / PREDICTIONS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `make train` first.")
    predictions = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={c: str for c in ("text", "label", "script", *MODEL_COLUMNS.values())},
    )
    model = args.model or ("transformer" if "transformer_pred" in predictions else "baseline")
    eval_cfg = cfg["evaluation"]
    report, errors = analyse(
        predictions,
        info["data"]["labels"],
        int(eval_cfg["short_max_words"]),
        int(eval_cfg["errors_top_k"]),
        model,
    )
    report["synthetic"] = info["synthetic"]
    write_json(report, out_dir / "error_analysis.json")
    errors.to_csv(out_dir / "errors.csv", index=False)
    print(f"Wrote {len(errors)} error rows to {out_dir / 'errors.csv'} and error_analysis.json")
    for pair in report[model]["top_confusions"]:
        print(f"  {pair['true']} -> {pair['predicted']}: {pair['count']}")


if __name__ == "__main__":
    main()
