"""Q/K low-rank + dense-V surgery for Hugging Face–style causal LMs.

Consumable API for applying trained low-rank Q/K checkpoints to an existing LM:

- Patch ``q_proj`` / ``k_proj`` with :class:`MultiHeadQKLowRankProjection` (dense V unchanged).
- Optional dense materialization helpers for fused-GEMM inference.

Integrate via ``pip install -e `` on this repo, or prepend ``<repo>/src`` to ``PYTHONPATH``.
Driver scripts remain under ``scripts/`` (e.g. ``18_*``, ``19_*``, ``21_*``).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from attention_compression.model_hub import get_self_attn


class MultiHeadQKLowRankProjection(torch.nn.Module):
    """Drop-in Q/K projection made from one low-rank branch per attention head."""

    def __init__(
        self,
        *,
        branch_states: list[dict[str, torch.Tensor]],
        branch_name: str,
        input_dim: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.down = torch.nn.ParameterList()
        self.up = torch.nn.ParameterList()
        self.bias = torch.nn.ParameterList()
        for head_index, state in enumerate(branch_states):
            prefix = f"{branch_name}."
            down = state[prefix + "down"].to(device=device, dtype=dtype)
            up = state[prefix + "up"].to(device=device, dtype=dtype)
            bias = state[prefix + "bias"].to(device=device, dtype=dtype)
            if down.shape[0] != input_dim or up.shape[1] != head_dim or bias.shape[0] != head_dim:
                raise ValueError(
                    f"Bad {branch_name} head {head_index} branch shape: "
                    f"down={tuple(down.shape)} up={tuple(up.shape)} bias={tuple(bias.shape)}"
                )
            self.down.append(torch.nn.Parameter(down, requires_grad=False))
            self.up.append(torch.nn.Parameter(up, requires_grad=False))
            self.bias.append(torch.nn.Parameter(bias, requires_grad=False))

        self.head_dim = int(head_dim)

    def _uniform_rank(self) -> bool:
        if not self.down:
            return False
        r = int(self.down[0].shape[1])
        for i in range(len(self.down)):
            if int(self.down[i].shape[1]) != r or int(self.up[i].shape[0]) != r:
                return False
        return True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The branches may be stored at a different precision than the model
        # (e.g. fp32 trainable params inside a bf16 model), so coerce
        # hidden_states to the weight dtype for the matmul and cast back out.
        weight_dtype = self.down[0].dtype
        in_dtype = hidden_states.dtype
        x = hidden_states if in_dtype == weight_dtype else hidden_states.to(weight_dtype)
        if getattr(self, "_force_naive_forward", False) or not self._uniform_rank():
            pieces = [
                x @ self.down[i] @ self.up[i] + self.bias[i]
                for i in range(len(self.down))
            ]
            out = torch.cat(pieces, dim=-1)
            return out if in_dtype == weight_dtype else out.to(in_dtype)

        # Fused path: one `[*, input_dim] @ [input_dim, H*r]` then batched head ups.
        # Avoids 16 small GEMMs + Python overhead per projection.
        h = len(self.down)
        r = int(self.down[0].shape[1])
        down_cat = torch.cat(list(self.down), dim=1)
        z = x @ down_cat
        bsz, seq = z.shape[0], z.shape[1]
        z = z.reshape(bsz, seq, h, r)
        up_stack = torch.stack(list(self.up), dim=0)
        o = torch.einsum("bshr,hrd->bshd", z, up_stack)
        bias_stack = torch.stack(list(self.bias), dim=0)
        o = o + bias_stack.unsqueeze(0).unsqueeze(0)
        out = o.reshape(bsz, seq, h * self.head_dim)
        return out if in_dtype == weight_dtype else out.to(in_dtype)


def materialize_dense_linear_from_branch_states(
    branch_states: list[dict[str, torch.Tensor]],
    *,
    branch_name: str,
    input_dim: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Linear:
    """One ``nn.Linear`` equal to concatenating per-head maps ``x @ down_h @ up_h + b_h``.

    Uses the same fused GEMM path as vanilla Hugging Face ``Linear``, giving inference
    parity with dense Q/K at the cost of **materializing** full ``[out, in]`` weights
    in VRAM (still load low-rank factors from disk).
    """
    cols: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    prefix = f"{branch_name}."
    for state in branch_states:
        down = state[prefix + "down"].to(device=device, dtype=dtype)
        up = state[prefix + "up"].to(device=device, dtype=dtype)
        b = state[prefix + "bias"].to(device=device, dtype=dtype)
        if down.shape[0] != input_dim or up.shape[1] != head_dim or b.shape[0] != head_dim:
            raise ValueError(
                f"Bad {branch_name} branch shape for materialize: "
                f"down={tuple(down.shape)} up={tuple(up.shape)} bias={tuple(b.shape)}"
            )
        w_h = down @ up
        cols.append(w_h)
        biases.append(b)
    w_full = torch.cat(cols, dim=1)
    out_features = int(w_full.shape[1])
    bias_full = torch.cat(biases, dim=0)
    lin = torch.nn.Linear(input_dim, out_features, bias=True, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(w_full.T.contiguous())
        lin.bias.copy_(bias_full.contiguous())
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin


def qk_checkpoint_dir(root: Path, *, layer: int, head: int) -> Path:
    path = root / f"layer_{layer:02d}_head_{head:02d}_q64_k48_densev"
    if (path / "qk_dense_v_model.pt").exists():
        return path
    raise FileNotFoundError(f"Missing QK+dense-V checkpoint for layer={layer} head={head}: {path}")


def joint_qkv_checkpoint_dir(root: Path, *, layer: int, head: int, config_name: str) -> Path:
    candidates = [
        root / f"layer_{layer:02d}_head_{head:02d}_{config_name}_large",
        root / f"layer_{layer:02d}_head_{head:02d}_{config_name}",
        root / f"head_{head:02d}_{config_name}",
    ]
    for path in candidates:
        if (path / "joint_qkv_model.pt").exists():
            return path
    raise FileNotFoundError(
        "Missing fallback joint-QKV checkpoint for "
        f"layer={layer} head={head}; tried {[str(p) for p in candidates]}"
    )


def load_layer_qk_states(
    root: Path,
    *,
    layer: int,
    num_heads: int,
    fallback_joint_root: Path | None,
    fallback_joint_config: str,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, object]], str]:
    states: list[dict[str, torch.Tensor]] = []
    reports: list[dict[str, object]] = []
    source = "qk_dense_v"
    for head in range(num_heads):
        try:
            ckpt_dir = qk_checkpoint_dir(root, layer=layer, head=head)
            model_file = ckpt_dir / "qk_dense_v_model.pt"
            report_file = ckpt_dir / "qk_dense_v_report.json"
            report_source = "qk_dense_v"
        except FileNotFoundError:
            if fallback_joint_root is None:
                raise
            ckpt_dir = joint_qkv_checkpoint_dir(
                fallback_joint_root,
                layer=layer,
                head=head,
                config_name=fallback_joint_config,
            )
            model_file = ckpt_dir / "joint_qkv_model.pt"
            report_file = ckpt_dir / "joint_qkv_report.json"
            report_source = "joint_qkv_fallback"
            source = "joint_qkv_fallback"
        states.append(torch.load(model_file, map_location="cpu"))
        report_path = report_file
        if report_path.exists():
            with report_path.open("r", encoding="utf-8") as f:
                report = json.load(f)
        else:
            report = {"target_layer": layer, "head_index": head}
        report["checkpoint_dir"] = str(ckpt_dir)
        report["checkpoint_source"] = report_source
        reports.append(report)
    return states, reports, source


def layer_is_complete(
    root: Path,
    *,
    layer: int,
    num_heads: int,
    fallback_joint_root: Path | None,
    fallback_joint_config: str,
) -> bool:
    for head in range(num_heads):
        qk_file = root / f"layer_{layer:02d}_head_{head:02d}_q64_k48_densev" / "qk_dense_v_model.pt"
        if qk_file.exists():
            continue
        if fallback_joint_root is not None:
            try:
                joint_qkv_checkpoint_dir(
                    fallback_joint_root,
                    layer=layer,
                    head=head,
                    config_name=fallback_joint_config,
                )
                continue
            except FileNotFoundError:
                pass
        return False
    return True


def patch_layer_qk_dense_v(
    model: torch.nn.Module,
    *,
    layer_index: int,
    branch_states: list[dict[str, torch.Tensor]],
    dtype: torch.dtype,
    device: torch.device,
    materialize_dense: bool = False,
) -> None:
    attn = get_self_attn(model, layer_index)
    input_dim = int(attn.q_proj.in_features)
    head_dim = int(attn.head_dim)
    if materialize_dense:
        attn.q_proj = materialize_dense_linear_from_branch_states(
            branch_states,
            branch_name="q",
            input_dim=input_dim,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        attn.k_proj = materialize_dense_linear_from_branch_states(
            branch_states,
            branch_name="k",
            input_dim=input_dim,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        return
    attn.q_proj = MultiHeadQKLowRankProjection(
        branch_states=branch_states,
        branch_name="q",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )
    attn.k_proj = MultiHeadQKLowRankProjection(
        branch_states=branch_states,
        branch_name="k",
        input_dim=input_dim,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
    )


def iter_batches(input_ids: torch.Tensor, batch_size: int):
    for start in range(0, input_ids.shape[0], batch_size):
        yield input_ids[start : start + batch_size]


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    losses = []
    token_counts = []
    for batch in iter_batches(input_ids, batch_size):
        batch = batch.to(device=device)
        out = model(input_ids=batch, labels=batch, use_cache=False)
        tokens = batch.numel() - batch.shape[0]
        losses.append(float(out.loss.detach().cpu()) * tokens)
        token_counts.append(tokens)
    total_tokens = sum(token_counts)
    loss = sum(losses) / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": total_tokens}


def state_parameter_counts(state: dict[str, torch.Tensor], *, head_dim: int) -> dict[str, int]:
    input_dim = int(state["q.down"].shape[0])
    q = int(state["q.down"].numel() + state["q.up"].numel() + state["q.bias"].numel())
    k = int(state["k.down"].numel() + state["k.up"].numel() + state["k.bias"].numel())
    dense_v = int(input_dim * head_dim + head_dim)
    dense_qkv = int(3 * (input_dim * head_dim + head_dim))
    return {
        "dense_qkv": dense_qkv,
        "qk_low_rank_dense_v": q + k + dense_v,
    }
