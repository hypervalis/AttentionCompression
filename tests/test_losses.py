import torch

from attention_compression.losses import (
    compression_train_loss,
    directional_cosine_loss,
    mse_loss,
    relative_mse_loss,
)


def test_relative_plus_cosine_additive() -> None:
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    rel = relative_mse_loss(pred, target)
    cos = directional_cosine_loss(pred, target)
    mixed = compression_train_loss(
        pred, target, loss_kind="relative_plus_cosine", cosine_weight=2.0
    )
    assert torch.allclose(mixed, rel + 2.0 * cos)


def test_mse_cosine_convex_mix() -> None:
    pred = torch.randn(4, 8)
    target = torch.randn(4, 8)
    w = 0.3
    expected = w * mse_loss(pred, target) + (1.0 - w) * directional_cosine_loss(pred, target)
    got = compression_train_loss(pred, target, loss_kind="mse_cosine", mse_weight=w)
    assert torch.allclose(got, expected)


def test_both_convex_mix() -> None:
    pred = torch.randn(4, 8)
    target = torch.randn(4, 8)
    w = 0.25
    expected = w * relative_mse_loss(pred, target) + (1.0 - w) * directional_cosine_loss(pred, target)
    got = compression_train_loss(pred, target, loss_kind="both", relative_weight=w)
    assert torch.allclose(got, expected)
