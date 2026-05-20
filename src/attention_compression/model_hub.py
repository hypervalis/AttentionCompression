"""Load causal LMs and access transformer blocks in a layout-agnostic way."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from attention_compression.activations import find_transformer_layers


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    return getattr(torch, dtype)


def load_causal_lm(
    model_path: str | Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    """Load a Hugging Face causal LM from a hub id or local directory."""
    from transformers import AutoModelForCausalLM

    dev = resolve_device(device)
    dt = resolve_dtype(dev, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dt,
        trust_remote_code=trust_remote_code,
    )
    model.to(dev)
    return model, dev, dt


def get_transformer_layers(model: torch.nn.Module) -> Any:
    _path, layers = find_transformer_layers(model)
    return layers


def num_layers(model: torch.nn.Module) -> int:
    return len(get_transformer_layers(model))


def get_self_attn(model: torch.nn.Module, layer_index: int) -> Any:
    layers = get_transformer_layers(model)
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(f"layer_index {layer_index} out of range for {len(layers)} layers")
    return layers[layer_index].self_attn


def num_heads(model: torch.nn.Module, layer_index: int) -> int:
    attn = get_self_attn(model, layer_index)
    head_dim = int(attn.head_dim)
    return int(getattr(attn, "num_heads", attn.o_proj.in_features // head_dim))


def save_causal_lm(model: torch.nn.Module, output_path: str | Path) -> None:
    """Write model weights (and config if present) to a directory."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out)
        return
    torch.save(model.state_dict(), out / "pytorch_model.bin")
