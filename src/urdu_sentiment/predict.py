"""Inference wrapper used by the app: loads whatever trained models exist and scores text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .backbone import META_FILE, Backbone, load_backbone
from .baseline import load_baseline, predict_scores
from .data import detect_script, normalize_text
from .transformer import predict_proba

MAX_CHARS = 1000


class Predictor:
    """Holds the transformer and/or baseline; either may be missing."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.normalize = bool(cfg["data"]["normalize"])
        self.backbone: Backbone | None = None
        self.baseline = None
        self.labels: list[str] = []
        model_dir = Path(cfg["output"]["model_dir"])
        if (model_dir / META_FILE).is_file():
            self.backbone = load_backbone(model_dir)
            self.labels = self.backbone.labels
        baseline_path = Path(cfg["output"]["baseline_path"])
        if baseline_path.is_file():
            self.baseline, labels = load_baseline(baseline_path)
            self.labels = self.labels or labels

    @property
    def ready(self) -> bool:
        return self.backbone is not None or self.baseline is not None

    def predict(self, text: str) -> dict[str, Any]:
        """Return the detected script and per-model class probabilities for one text."""
        if not self.ready:
            raise RuntimeError("No trained model found. Run `make train` first.")
        prepared = normalize_text(text) if self.normalize else text.strip()
        models: dict[str, list[float]] = {}
        if self.backbone is not None:
            models["transformer"] = predict_proba(self.backbone, [prepared])[0].tolist()
        if self.baseline is not None:
            models["baseline"] = predict_scores(self.baseline, [prepared])[0].tolist()
        return {"script": detect_script(prepared), "labels": self.labels, "models": models}
