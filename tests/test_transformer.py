import numpy as np
import torch

from urdu_sentiment.backbone import (
    CharTokenizer,
    build_tiny_backbone,
    load_backbone,
    make_backbone,
    save_backbone,
)
from urdu_sentiment.transformer import fine_tune, predict_proba

MODEL_CFG = {
    "epochs": 2,
    "batch_size": 32,
    "eval_batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "patience": 2,
    "fp16": False,
    "class_weighted_loss": True,
    "num_threads": 2,
    "source": "tiny_random",
    "name": "tiny",
    "max_length": 32,
}


def test_char_tokenizer_pads_truncates_and_handles_unknown():
    tok = CharTokenizer.build(["abc", "ab"])
    enc = tok(["abc", "a"], max_length=4)
    assert enc["input_ids"].shape == (2, 4)
    assert enc["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 0]]
    assert tok(["zzz"], max_length=8)["input_ids"][0, 1].item() == tok.vocab["[UNK]"]


def test_tiny_backbone_forward_pass_shape(splits):
    texts = list(splits.train["text"])
    backbone = build_tiny_backbone(splits.labels, texts, max_length=32, seed=0)
    enc = backbone.tokenizer(
        texts[:4], truncation=True, max_length=32, padding=True, return_tensors="pt"
    )
    logits = backbone.model(**enc).logits
    assert logits.shape == (4, 3)
    probs = predict_proba(backbone, texts[:5], batch_size=2)
    assert probs.shape == (5, 3) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_fine_tune_runs_same_code_path_and_restores_best(splits, tmp_path):
    torch.set_num_threads(2)
    train = (list(splits.train["text"]), splits.encode(splits.train))
    val = (list(splits.val["text"]), splits.encode(splits.val))
    backbone = make_backbone(MODEL_CFG, splits.labels, train[0], seed=0)
    info = fine_tune(backbone, train, val, MODEL_CFG, seed=0)
    assert 1 <= info["epochs_run"] <= 2
    assert info["device"] in {"cpu", "cuda"}
    assert info["best_val_macro_f1"] >= max(h["val_macro_f1"] for h in info["history"]) - 1e-9

    before = predict_proba(backbone, val[0])
    save_backbone(backbone, tmp_path / "m", {"synthetic": True})
    reloaded = load_backbone(tmp_path / "m")
    assert reloaded.labels == splits.labels
    assert np.allclose(predict_proba(reloaded, val[0]), before, atol=1e-5)


def test_load_backbone_missing_folder(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError, match="make train"):
        load_backbone(tmp_path / "none")
