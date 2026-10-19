"""Loading, cleaning, script detection and the fixed train/val/test split."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .synthetic import make_synthetic

SCRIPTS = ("urdu", "roman", "mixed", "unknown")

# Arabic, Arabic Supplement, Presentation Forms A and B: where Urdu letters live.
_ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
_ZERO_WIDTH = re.compile("[​‌‍‎‏‪-‮﻿]")
_DIACRITICS = re.compile("[ً-ٰٟۖ-ۭـ]")
_SPACES = re.compile(r"\s+")
_ARABIC_TO_URDU = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ۃ": "ہ", "ة": "ہ"})


def _is_arabic_script(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def arabic_ratio(text: str) -> float | None:
    """Share of letters written in Arabic script; None when the text has no letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    return sum(_is_arabic_script(c) for c in letters) / len(letters)


def detect_script(text: str, urdu_min: float = 0.7, roman_max: float = 0.3) -> str:
    """Label text as urdu, roman, mixed or unknown from its Arabic-script letter ratio.

    Roman Urdu is simply Urdu typed in Latin letters, so "roman" here means Latin-dominant
    text. The detector cannot tell Roman Urdu from English; the data source must guarantee
    the language, and the README says so.
    """
    ratio = arabic_ratio(text)
    if ratio is None:
        return "unknown"
    if ratio >= urdu_min:
        return "urdu"
    if ratio <= roman_max:
        return "roman"
    return "mixed"


def normalize_text(text: str) -> str:
    """Light normalisation applied identically to both models.

    Unifies Arabic and Urdu variants of yeh, kaf and heh, drops diacritics, tatweel and
    invisible direction marks, lowercases Latin text and collapses whitespace. Nothing is
    stemmed or removed beyond that, so the transformer sees nearly the raw text.
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = _ZERO_WIDTH.sub("", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_ARABIC_TO_URDU).lower()
    return _SPACES.sub(" ", text).strip()


def load_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or JSONL file that has `text` and `label` columns (and optionally `script`)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Fetch one with scripts/download_data.py "
            "(see README, 'Choosing a dataset') or point data.path at your own CSV/JSONL."
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".jsonl", ".json"}:
        frame = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file type {suffix!r}; use .csv or .jsonl")
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    return frame


def clean_table(
    frame: pd.DataFrame, *, normalize: bool = True, dedupe: bool = True, min_chars: int = 2
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate and clean a raw table; return it with counts of what was dropped.

    The `script` column is kept if it exists and is valid, otherwise it is detected from the
    text. Rows are dropped, never repaired, so the report below accounts for every change.
    """
    report = {"rows_in": len(frame)}
    out = frame.copy()
    out = out.dropna(subset=["text", "label"])
    report["dropped_missing"] = report["rows_in"] - len(out)
    out["text"] = out["text"].astype(str)
    if normalize:
        out["text"] = out["text"].map(normalize_text)
    else:
        out["text"] = out["text"].str.strip()
    before = len(out)
    out = out[out["text"].str.len() >= min_chars]
    report["dropped_too_short"] = before - len(out)
    out["label"] = out["label"].astype(str).str.strip()
    if "script" in out.columns:
        out["script"] = out["script"].astype(str).str.strip().str.lower()
        invalid = ~out["script"].isin(SCRIPTS)
        out.loc[invalid, "script"] = out.loc[invalid, "text"].map(detect_script)
    else:
        out["script"] = out["text"].map(detect_script)
    before = len(out)
    if dedupe:
        out = out.drop_duplicates(subset=["text"], keep="first")
    report["dropped_duplicates"] = before - len(out)
    out = out.reset_index(drop=True)
    report["rows_out"] = len(out)
    return out[["text", "label", "script"]], report


@dataclass(frozen=True)
class Splits:
    """The three frozen splits plus the label order used to turn names into integers."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    labels: list[str]

    def encode(self, frame: pd.DataFrame) -> list[int]:
        index = {name: i for i, name in enumerate(self.labels)}
        return [index[name] for name in frame["label"]]


def _strata(frame: pd.DataFrame) -> pd.Series:
    """Stratify on label+script when every group is big enough, else on label alone."""
    combined = frame["label"] + "|" + frame["script"]
    if combined.value_counts().min() >= 3:
        return combined
    return frame["label"]


def split_dataset(frame: pd.DataFrame, *, val_size: float, test_size: float, seed: int) -> Splits:
    """Fixed stratified split. The same seed and input always give the same three frames."""
    labels = sorted(frame["label"].unique())
    if len(labels) < 2:
        raise ValueError(f"Need at least two classes, found {labels}")
    if frame["label"].value_counts().min() < 3:
        raise ValueError("Every class needs at least 3 rows to be split into train/val/test")
    train_val, test = train_test_split(
        frame, test_size=test_size, random_state=seed, stratify=_strata(frame)
    )
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, random_state=seed, stratify=_strata(train_val)
    )
    reset = [part.reset_index(drop=True) for part in (train, val, test)]
    return Splits(train=reset[0], val=reset[1], test=reset[2], labels=labels)


def load_splits(cfg: dict[str, Any]) -> tuple[Splits, dict[str, Any]]:
    """Build splits from the config; returns them with a data report for the results file."""
    data_cfg = cfg["data"]
    synthetic = data_cfg["source"] == "synthetic"
    if synthetic:
        raw = make_synthetic(int(data_cfg["synthetic_rows"]), seed=int(cfg["seed"]))
    elif data_cfg["source"] == "file":
        raw = load_table(data_cfg["path"])
    else:
        raise ValueError(f"data.source must be 'file' or 'synthetic', got {data_cfg['source']!r}")
    frame, report = clean_table(
        raw,
        normalize=bool(data_cfg["normalize"]),
        dedupe=bool(data_cfg["dedupe"]),
        min_chars=int(data_cfg["min_chars"]),
    )
    splits = split_dataset(
        frame,
        val_size=float(data_cfg["val_size"]),
        test_size=float(data_cfg["test_size"]),
        seed=int(cfg["seed"]),
    )
    report.update(
        synthetic=synthetic,
        source=data_cfg["path"] if not synthetic else "synthetic template generator",
        labels=splits.labels,
        sizes={"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        script_counts={k: int(v) for k, v in frame["script"].value_counts().items()},
        label_counts={k: int(v) for k, v in frame["label"].value_counts().items()},
    )
    return splits, report
