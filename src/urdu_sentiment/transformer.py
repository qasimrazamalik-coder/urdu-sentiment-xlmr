"""Fine-tuning loop and batched prediction for a `Backbone`.

A plain PyTorch loop instead of the Trainer: it is short, every step is visible, and early
stopping on validation macro-F1 (the metric we report) is a few lines.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .backbone import Backbone


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batches(texts: list[str], labels: list[int] | None, size: int, shuffle: bool, seed: int):
    items = [(text, labels[i] if labels is not None else -1) for i, text in enumerate(texts)]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=size, shuffle=shuffle, generator=generator, collate_fn=list)


def _encode(backbone: Backbone, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    enc = backbone.tokenizer(
        texts, truncation=True, max_length=backbone.max_length, padding=True, return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}


@torch.no_grad()
def predict_proba(
    backbone: Backbone, texts: list[str], batch_size: int = 64, device: torch.device | None = None
) -> np.ndarray:
    """Softmax probabilities, shape (n_texts, n_classes). Same path for eval and the app."""
    device = device or next(backbone.model.parameters()).device
    backbone.model.eval()
    out = []
    for batch in _batches(texts, None, batch_size, False, 0):
        enc = _encode(backbone, [t for t, _ in batch], device)
        out.append(torch.softmax(backbone.model(**enc).logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(backbone.labels)))


def fine_tune(
    backbone: Backbone,
    train: tuple[list[str], list[int]],
    val: tuple[list[str], list[int]],
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Train with AdamW, linear warmup/decay and early stopping on validation macro-F1.

    Mixed precision is used only when CUDA is available and `fp16` is true. The weights of
    the best epoch are restored at the end, so the saved model is the one validation chose.
    """
    from transformers import get_linear_schedule_with_warmup

    if cfg.get("num_threads"):
        torch.set_num_threads(int(cfg["num_threads"]))
    device = pick_device()
    use_amp = bool(cfg["fp16"]) and device.type == "cuda"
    model = backbone.model.to(device)
    train_x, train_y = train
    val_x, val_y = val
    epochs, batch_size = int(cfg["epochs"]), int(cfg["batch_size"])

    steps = epochs * ((len(train_x) + batch_size - 1) // batch_size)
    decay = [p for n, p in model.named_parameters() if not n.endswith(("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if n.endswith(("bias", "LayerNorm.weight"))]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(cfg["learning_rate"]),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(steps * float(cfg["warmup_ratio"])), steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    weights = None
    if cfg["class_weighted_loss"]:
        counts = np.bincount(train_y, minlength=len(backbone.labels)).clip(min=1)
        weights = torch.tensor(
            counts.sum() / (len(counts) * counts), dtype=torch.float, device=device
        )
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    history, best_f1, best_state, stale = [], -1.0, None, 0
    started = time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in _batches(train_x, train_y, batch_size, True, seed + epoch):
            enc = _encode(backbone, [t for t, _ in batch], device)
            target = torch.tensor([y for _, y in batch], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits
            loss = loss_fn(logits.float(), target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(batch)
        probs = predict_proba(backbone, val_x, int(cfg["eval_batch_size"]), device)
        val_f1 = float(f1_score(val_y, probs.argmax(axis=1), average="macro", zero_division=0))
        history.append(
            {"epoch": epoch + 1, "train_loss": total / len(train_x), "val_macro_f1": val_f1}
        )
        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
            if stale >= int(cfg["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {
        "history": history,
        "best_val_macro_f1": best_f1,
        "epochs_run": len(history),
        "device": device.type,
        "mixed_precision": use_amp,
        "train_seconds": round(time.time() - started, 1),
    }
