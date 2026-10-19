"""Tokenizer + model loading behind one small interface.

Training and prediction only ever see a `Backbone`. Real runs build it from pretrained
Hugging Face weights; tests and `make smoke` build a tiny randomly initialised BERT with a
character tokenizer, so the exact same training and inference code runs without downloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
)

META_FILE = "urdu_sentiment_meta.json"
_VOCAB_FILE = "char_vocab.json"
_SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]


class CharTokenizer:
    """Character-level tokenizer with the call signature of a Hugging Face tokenizer.

    Only used for tests and the smoke run. It returns the same dict of padded tensors, so
    the training loop cannot tell it apart from a real tokenizer.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        self.vocab = vocab

    @classmethod
    def build(cls, texts: list[str]) -> CharTokenizer:
        chars = sorted({c for text in texts for c in text})
        vocab = {tok: i for i, tok in enumerate(_SPECIALS + chars)}
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __call__(self, texts: list[str], max_length: int = 64, **_: Any) -> dict[str, torch.Tensor]:
        pad, unk, cls_id, sep = (self.vocab[t] for t in _SPECIALS)
        rows = []
        for text in texts:
            ids = [self.vocab.get(c, unk) for c in text][: max_length - 2]
            rows.append([cls_id, *ids, sep])
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / _VOCAB_FILE).write_text(json.dumps(self.vocab), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> CharTokenizer:
        path = Path(directory) / _VOCAB_FILE
        return cls(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class Backbone:
    """A tokenizer and a sequence-classification model that belong together."""

    tokenizer: Any
    model: Any
    kind: str  # "hf" or "char"
    max_length: int
    labels: list[str]


def build_tiny_backbone(
    labels: list[str], texts: list[str], max_length: int, seed: int
) -> Backbone:
    """Tiny random BERT plus character tokenizer. For tests and smoke runs only."""
    tokenizer = CharTokenizer.build(texts)
    torch.manual_seed(seed)
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=max_length + 2,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
        pad_token_id=0,
    )
    return Backbone(tokenizer, BertForSequenceClassification(config), "char", max_length, labels)


def load_pretrained_backbone(name: str, labels: list[str], max_length: int) -> Backbone:
    """Load real pretrained weights by name or local path (needs the Hugging Face cache)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(
            name,
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
        )
    except OSError as err:
        raise RuntimeError(
            f"Could not load pretrained model {name!r}. Run on a machine with access to "
            "huggingface.co (Colab, Kaggle) or set model.name to a local folder."
        ) from err
    return Backbone(tokenizer, model, "hf", max_length, labels)


def make_backbone(
    model_cfg: dict[str, Any], labels: list[str], train_texts: list[str], seed: int
) -> Backbone:
    """Build the backbone the config asks for."""
    max_length = int(model_cfg["max_length"])
    source = model_cfg["source"]
    if source == "tiny_random":
        return build_tiny_backbone(labels, train_texts, max_length, seed)
    if source == "pretrained":
        return load_pretrained_backbone(model_cfg["name"], labels, max_length)
    raise ValueError(f"model.source must be 'pretrained' or 'tiny_random', got {source!r}")


def save_backbone(backbone: Backbone, directory: str | Path, extra: dict[str, Any]) -> None:
    """Write weights, tokenizer and a metadata file the app uses to reload them."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    backbone.model.save_pretrained(directory)
    backbone.tokenizer.save_pretrained(directory)
    meta = {
        "tokenizer_kind": backbone.kind,
        "max_length": backbone.max_length,
        "labels": backbone.labels,
        **extra,
    }
    (directory / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")


def load_backbone(directory: str | Path) -> Backbone:
    """Reload a backbone saved by `save_backbone`; raises if the folder is not one."""
    directory = Path(directory)
    meta_path = directory / META_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(f"No trained model in {directory}. Run `make train` first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["tokenizer_kind"] == "char":
        tokenizer: Any = CharTokenizer.from_pretrained(directory)
    else:
        tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory)
    model.eval()
    return Backbone(tokenizer, model, meta["tokenizer_kind"], meta["max_length"], meta["labels"])
