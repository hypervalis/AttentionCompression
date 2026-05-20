"""Training losses for activation distillation (relative MSE + directional cosine)."""
from __future__ import annotations

import torch

LOSS_KINDS = ("relative", "cosine", "both", "relative_plus_cosine", "mse_cosine")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Plain mean squared error over all elements."""
    return torch.mean((pred.float() - target.float()) ** 2)


def relative_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Batch mean of per-tensor relative MSE (matches script-32 ``rel_loss``)."""
    return torch.mean((pred.float() - target.float()) ** 2) / torch.var(target.float()).clamp_min(1e-6)


def directional_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean ``1 - cos(pred, target)`` over token positions (last dim is hidden)."""
    pred_f = pred.reshape(-1, pred.shape[-1]).float()
    target_f = target.reshape(-1, target.shape[-1]).float()
    cos = torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1)
    return torch.mean(1.0 - cos)


def compression_train_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_kind: str,
    relative_weight: float = 0.25,
    cosine_weight: float = 1.0,
    mse_weight: float = 0.25,
) -> torch.Tensor:
    """Combine relative MSE and/or cosine losses.

    ``both``: ``relative_weight * relative + (1 - relative_weight) * cosine`` (convex mix).

    ``relative_plus_cosine``: ``relative + cosine_weight * cosine`` (additive; FINDINGS-style).

    ``mse_cosine``: ``mse_weight * MSE + (1 - mse_weight) * cosine`` (plain MSE, convex mix).
    """
    kind = loss_kind.strip().lower()
    if kind == "relative":
        return relative_mse_loss(pred, target)
    if kind == "cosine":
        return directional_cosine_loss(pred, target)
    if kind == "both":
        w = float(relative_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError("relative_weight must be in [0, 1] when loss_kind is both")
        return w * relative_mse_loss(pred, target) + (1.0 - w) * directional_cosine_loss(pred, target)
    if kind == "relative_plus_cosine":
        return relative_mse_loss(pred, target) + float(cosine_weight) * directional_cosine_loss(pred, target)
    if kind == "mse_cosine":
        w = float(mse_weight)
        if not 0.0 <= w <= 1.0:
            raise ValueError("mse_weight must be in [0, 1] when loss_kind is mse_cosine")
        return w * mse_loss(pred, target) + (1.0 - w) * directional_cosine_loss(pred, target)
    raise ValueError(f"Unknown loss_kind {loss_kind!r}; expected one of {LOSS_KINDS}")
