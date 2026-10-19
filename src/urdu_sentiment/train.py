"""Train the baseline and the transformer on one fixed split and save test predictions.

Usage: python -m urdu_sentiment.train --config configs/default.yaml [--baseline-only]
Metrics are computed afterwards by `urdu_sentiment.evaluate` from the saved predictions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from .baseline import fit_baseline, predict_scores, save_baseline
from .data import load_splits
from .synthetic import SYNTHETIC_NOTICE
from .utils import (
    PREDICTIONS_FILE,
    RUN_INFO_FILE,
    load_config,
    package_versions,
    seed_everything,
    write_json,
)


def run(cfg: dict[str, Any], baseline_only: bool = False) -> Path:
    """Execute a full training run and return the output directory."""
    seed = int(cfg["seed"])
    seed_everything(seed)
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits, data_report = load_splits(cfg)
    labels = splits.labels
    train_x, val_x, test_x = (list(f["text"]) for f in (splits.train, splits.val, splits.test))
    train_y, val_y = splits.encode(splits.train), splits.encode(splits.val)

    started = time.time()
    pipe, base_info = fit_baseline(cfg["baseline"], train_x, train_y, val_x, val_y, seed)
    base_info["fit_seconds"] = round(time.time() - started, 1)
    save_baseline(pipe, labels, cfg["output"]["baseline_path"])

    predictions = splits.test.copy()
    predictions["n_words"] = predictions["text"].str.split().str.len()
    predictions["baseline_pred"] = [labels[i] for i in predict_scores(pipe, test_x).argmax(axis=1)]

    transformer_info: dict[str, Any] | None = None
    if cfg["model"]["enabled"] and not baseline_only:
        # torch is imported here so a baseline-only run works without it.
        from .backbone import make_backbone, save_backbone
        from .transformer import fine_tune, predict_proba

        seed_everything(seed)
        backbone = make_backbone(cfg["model"], labels, train_x, seed)
        transformer_info = fine_tune(
            backbone, (train_x, train_y), (val_x, val_y), cfg["model"], seed
        )
        transformer_info["model_name"] = cfg["model"]["name"]
        probs = predict_proba(backbone, test_x, int(cfg["model"]["eval_batch_size"]))
        predictions["transformer_pred"] = [labels[i] for i in probs.argmax(axis=1)]
        save_backbone(
            backbone,
            cfg["output"]["model_dir"],
            {"model_name": cfg["model"]["name"], "synthetic": data_report["synthetic"]},
        )

    predictions.to_csv(out_dir / PREDICTIONS_FILE, index=False)
    write_json(
        {
            "synthetic": data_report["synthetic"],
            "notice": SYNTHETIC_NOTICE if data_report["synthetic"] else None,
            "seed": seed,
            "versions": package_versions(),
            "config": cfg,
            "data": data_report,
            "baseline": base_info,
            "transformer": transformer_info,
        },
        out_dir / RUN_INFO_FILE,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-only", action="store_true", help="skip the transformer")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = run(cfg, baseline_only=args.baseline_only)
    print(f"Predictions and run info written to {out_dir}. Next: python -m urdu_sentiment.evaluate")


if __name__ == "__main__":
    main()
