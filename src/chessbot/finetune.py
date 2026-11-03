"""Fine-tune a Maia 3 model on Justin's moves. Training never touches the base weights on disk."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F

from chessbot.examples import Example, collate


def masked_loss(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    base_logits: torch.Tensor | None = None,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """Weighted mean cross-entropy over legal moves, plus optional KL(base || current)."""
    neg_inf = float("-inf")
    log_probs = F.log_softmax(logits.float().masked_fill(~legal_mask, neg_inf), dim=-1)
    per_example = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    if base_logits is not None and kl_weight > 0:
        base_log = F.log_softmax(base_logits.float().masked_fill(~legal_mask, neg_inf), dim=-1)
        diff = (base_log.exp() * (base_log - log_probs)).masked_fill(~legal_mask, 0.0)
        per_example = per_example + kl_weight * diff.sum(dim=-1)
    return (per_example * weight).sum() / weight.sum()


@dataclass
class TrainResult:
    best_state: dict[str, torch.Tensor]
    best_epoch: int  # 0 means the starting model was never beaten on validation
    history: list[dict]


def _evaluate(model, examples: Sequence[Example], vocab_size: int, batch_size: int, device: str):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            tokens, se, oe, target, mask, _ = (t.to(device) for t in collate(examples[start:start + batch_size], vocab_size))
            logits = model(tokens, se, oe)[0].float()
            ones = torch.ones(len(target), device=device)
            total_loss += masked_loss(logits, mask, target, ones).item() * len(target)
            correct += (logits.masked_fill(~mask, float("-inf")).argmax(-1) == target).sum().item()
    return total_loss / len(examples), correct / len(examples)


def train(
    model,
    train_examples: Sequence[Example],
    val_examples: Sequence[Example],
    *,
    vocab_size: int,
    epochs: int = 3,
    lr: float = 1e-5,
    batch_size: int = 64,
    kl_weight: float = 0.0,
    freeze_layers: int = 0,
    patience: int = 1,
    device: str = "cpu",
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Fine-tune `model` in place; return the weights of the epoch with the best validation loss.

    Epoch 0 is the starting model, so a fine-tune that never beats it reports best_epoch 0.
    """
    model.to(device)
    for layer in list(model.transformer.layers)[:freeze_layers]:
        for p in layer.parameters():
            p.requires_grad_(False)
    base = copy.deepcopy(model).eval() if kl_weight > 0 else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    generator = torch.Generator().manual_seed(seed)

    def snapshot() -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # The overfitting gap compares like with like: an unweighted score of a fixed training
    # subsample (as large as validation), taken with the same function after every epoch.
    train_probe = list(train_examples[:len(val_examples)])

    def row(epoch: int, train_loss: float) -> dict:
        val_loss, val_top1 = _evaluate(model, val_examples, vocab_size, batch_size, device)
        probe_loss, probe_top1 = _evaluate(model, train_probe, vocab_size, batch_size, device)
        log(f"epoch {epoch}  train_loss {train_loss:.4f}  train_eval {probe_loss:.4f}/{probe_top1:.3f}"
            f"  val_loss {val_loss:.4f}  val_top1 {val_top1:.3f}")
        return {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_top1": val_top1,
                "train_eval_loss": probe_loss, "train_top1": probe_top1}

    history = [row(0, float("nan"))]
    val_loss = history[0]["val_loss"]
    best_state, best_loss, best_epoch = snapshot(), val_loss, 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_examples), generator=generator).tolist()
        losses = []
        for start in range(0, len(order), batch_size):
            batch = [train_examples[i] for i in order[start:start + batch_size]]
            tokens, se, oe, target, mask, weight = (t.to(device) for t in collate(batch, vocab_size))
            logits = model(tokens, se, oe)[0]
            base_logits = None
            if base is not None:
                with torch.no_grad():
                    base_logits = base(tokens, se, oe)[0]
            loss = masked_loss(logits, mask, target, weight, base_logits, kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(row(epoch, sum(losses) / len(losses)))
        val_loss = history[-1]["val_loss"]
        if val_loss < best_loss:
            best_state, best_loss, best_epoch = snapshot(), val_loss, epoch
        elif epoch - best_epoch >= patience:
            break
    model.eval()
    return TrainResult(best_state, best_epoch, history)


def save_checkpoint(state: dict[str, torch.Tensor], path: Path) -> None:
    """Write a maia3-loadable checkpoint; refuses to overwrite so base weights are never replaced."""
    if path.exists():
        raise FileExistsError(f"{path} already exists; choose a new file name")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state}, path)
