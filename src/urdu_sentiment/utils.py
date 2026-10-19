"""Seeding, config loading and run metadata shared by every entry point."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PREDICTIONS_FILE = "predictions_test.csv"
RUN_INFO_FILE = "run_info.json"


def seed_everything(seed: int) -> None:
    """Seed python, numpy and (if installed) torch in one place so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config. Fails loudly so a typo in the path is not silently ignored."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return cfg


def package_versions() -> dict[str, str]:
    """Versions that affect numbers, recorded in every results file."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:
            versions[name] = "not installed"
    return versions


def write_json(obj: Any, path: str | Path) -> Path:
    """Write JSON with stable key order, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def read_json(path: str | Path) -> Any:
    """Read JSON, with a message that says which step produces the missing file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the previous pipeline step first.")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
