"""Compute the reported metrics from saved test predictions.

Usage: python -m urdu_sentiment.evaluate --config configs/default.yaml
Writes metrics.json (everything, with seed and versions) and results.md (the README table).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import bootstrap_macro_f1, per_group, summarize
from .utils import (
    PREDICTIONS_FILE,
    RUN_INFO_FILE,
    load_config,
    package_versions,
    read_json,
    write_json,
)

MODEL_COLUMNS = {"baseline": "baseline_pred", "transformer": "transformer_pred"}


def evaluate_predictions(
    predictions: pd.DataFrame, labels: list[str], eval_cfg: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Score every model present in `predictions`; add a paired CI when both are present."""
    index = {name: i for i, name in enumerate(labels)}
    y_true = predictions["label"].map(index).to_numpy()
    scripts = predictions["script"].tolist()
    present = {n: c for n, c in MODEL_COLUMNS.items() if c in predictions.columns}
    n_boot = int(eval_cfg["bootstrap_samples"])
    results: dict[str, Any] = {}
    preds = {n: predictions[c].map(index).to_numpy() for n, c in present.items()}
    for name, y_pred in preds.items():
        entry = summarize(y_true, y_pred, labels)
        entry["per_script"] = per_group(y_true, y_pred, scripts, labels)
        entry.update(bootstrap_macro_f1(y_true, y_pred, None, len(labels), n_boot, seed))
        results[name] = entry
    comparison = None
    if "baseline" in preds and "transformer" in preds:
        comparison = bootstrap_macro_f1(
            y_true, preds["baseline"], preds["transformer"], len(labels), n_boot, seed
        )
        comparison["definition"] = "macro-F1(transformer) - macro-F1(baseline), paired bootstrap"
    return {"models": results, "comparison": comparison, "n_test": len(predictions)}


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(metrics: dict[str, Any]) -> str:
    """Render the results table that goes into the README."""
    lines = []
    if metrics["synthetic"]:
        lines += ["> SYNTHETIC template data: pipeline check only, not a result.", ""]
    lines += [
        f"Test rows: {metrics['n_test']}. Seed: {metrics['seed']}. "
        "Intervals are 95% bootstrap CIs on test macro-F1.",
        "",
        "| Model | Accuracy | Macro-F1 | Macro-F1 95% CI | F1 Urdu script | F1 Roman Urdu |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, entry in metrics["models"].items():
        lo, hi = entry["macro_f1_ci95"]
        scripts = entry["per_script"]
        cells = [
            _fmt(scripts[s]["macro_f1"]) + (" (n small)" if scripts[s]["small_sample"] else "")
            if s in scripts
            else "n/a"
            for s in ("urdu", "roman")
        ]
        lines.append(
            f"| {name} | {_fmt(entry['accuracy'])} | {_fmt(entry['macro_f1'])} "
            f"| {_fmt(lo)} to {_fmt(hi)} | {cells[0]} | {cells[1]} |"
        )
    comp = metrics["comparison"]
    if comp:
        lo, hi = comp["diff_ci95"]
        verdict = "interval excludes 0" if lo > 0 or hi < 0 else "interval includes 0"
        lines += [
            "",
            f"Transformer minus baseline, macro-F1: {comp['diff_macro_f1']:+.3f} "
            f"(95% CI {lo:+.3f} to {hi:+.3f}; {verdict}).",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output"]["dir"])
    info = read_json(out_dir / RUN_INFO_FILE)
    predictions_path = out_dir / PREDICTIONS_FILE
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing {predictions_path}. Run `make train` first.")
    predictions = pd.read_csv(
        predictions_path,
        keep_default_na=False,
        dtype={c: str for c in ("text", "label", "script", *MODEL_COLUMNS.values())},
    )
    labels = info["data"]["labels"]
    metrics = evaluate_predictions(predictions, labels, cfg["evaluation"], int(cfg["seed"]))
    metrics.update(
        synthetic=info["synthetic"],
        notice=info["notice"],
        seed=info["seed"],
        labels=labels,
        versions=package_versions(),
        training={"baseline": info["baseline"], "transformer": info["transformer"]},
    )
    write_json(metrics, out_dir / "metrics.json")
    table = render_markdown(metrics)
    (out_dir / "results.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
