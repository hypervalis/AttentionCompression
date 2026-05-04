#!/usr/bin/env python3
"""Train a sealed compressed layer replacement from captured layer activations.

The replacement keeps the same residual interface ``[B, S, D] -> [B, S, D]``,
but routes attention through per-head subspaces and uses a smaller FFN:

    D -> H*r -> attention in r-space -> per-head lifts -> D
    D -> ffn_dim -> D

This is a prototype for testing whether heads/FFNs can be hermetically
compressed while preserving the surrounding model width.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention_compression.attention_metrics import cosine_similarity_mean, relative_mse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill a compressed D->D layer replacement.")
    parser.add_argument("--capture-dir", required=True, help="Directory from scripts/08_capture_layer_activations.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--target-layer", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--subspace-rank", type=int, default=64)
    parser.add_argument("--value-rank", type=int, default=None, help="Defaults to --subspace-rank")
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument(
        "--init-attn-from-supervised-pca",
        action="store_true",
        help="Initialize head subspaces/cores from activation PCA of teacher Q/K/V outputs.",
    )
    parser.add_argument("--q-init-rank", type=int, default=None)
    parser.add_argument("--k-init-rank", type=int, default=None)
    parser.add_argument("--v-init-rank", type=int, default=None)
    parser.add_argument("--train-windows-per-bin", type=int, default=32)
    parser.add_argument("--eval-windows-per-bin", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--head-contribution-loss-weight",
        type=float,
        default=0.0,
        help="Weight for per-head lifted contribution loss against teacher attention head contributions.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def refs_by_path(refs: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in refs:
        out[path_i].append(sample_i)
    return out


def group_rows_by_bin(paths: list[Path]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        for sample_i, row in enumerate(shard["rows"]):
            groups[row["rarity_bin"]].append((path_i, sample_i))
    return groups


def choose_train_eval(
    groups: dict[str, list[tuple[int, int]]],
    train_n: int,
    eval_n: int,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    gen = torch.Generator().manual_seed(seed)
    train: list[tuple[int, int]] = []
    eval_: list[tuple[int, int]] = []
    for refs in groups.values():
        if len(refs) < train_n + eval_n:
            raise ValueError(
                f"Need at least {train_n + eval_n} rows per bin; got bin with {len(refs)} rows"
            )
        perm = torch.randperm(len(refs), generator=gen).tolist()
        train.extend(refs[i] for i in perm[:train_n])
        eval_.extend(refs[i] for i in perm[train_n : train_n + eval_n])
    return train, eval_


def batch_refs(
    paths: list[Path],
    refs: list[tuple[int, int]],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
):
    grouped = refs_by_path(refs)
    path_items = list(grouped.items())
    gen = torch.Generator().manual_seed(seed)
    if shuffle and path_items:
        order = torch.randperm(len(path_items), generator=gen).tolist()
        path_items = [path_items[i] for i in order]
    for path_i, sample_ids in path_items:
        ids = list(sample_ids)
        if shuffle and ids:
            order = torch.randperm(len(ids), generator=gen).tolist()
            ids = [ids[i] for i in order]
        shard = torch.load(paths[path_i], map_location="cpu")
        for start in range(0, len(ids), batch_size):
            yield shard, ids[start : start + batch_size]


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SealedCompressedAttention(torch.nn.Module):
    """One fused down-projection into all head subspaces, then small per-head attention."""

    def __init__(self, hidden_size: int, num_heads: int, subspace_rank: int, value_rank: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.subspace_rank = int(subspace_rank)
        self.value_rank = int(value_rank)
        self.down = torch.nn.Linear(hidden_size, num_heads * subspace_rank, bias=False)
        self.q_core = torch.nn.Parameter(torch.empty(num_heads, subspace_rank, subspace_rank))
        self.k_core = torch.nn.Parameter(torch.empty(num_heads, subspace_rank, subspace_rank))
        self.v_core = torch.nn.Parameter(torch.empty(num_heads, subspace_rank, value_rank))
        self.q_bias = torch.nn.Parameter(torch.zeros(num_heads, subspace_rank))
        self.k_bias = torch.nn.Parameter(torch.zeros(num_heads, subspace_rank))
        self.v_bias = torch.nn.Parameter(torch.zeros(num_heads, value_rank))
        self.out_lift = torch.nn.Parameter(torch.empty(num_heads, value_rank, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.down.weight)
        torch.nn.init.xavier_uniform_(self.q_core)
        torch.nn.init.xavier_uniform_(self.k_core)
        torch.nn.init.xavier_uniform_(self.v_core)
        torch.nn.init.normal_(self.out_lift, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor, *, return_contributions: bool = False):
        bsz, seq_len, _ = x.shape
        z = self.down(x).view(bsz, seq_len, self.num_heads, self.subspace_rank)
        q = torch.einsum("bshr,hrd->bhsd", z, self.q_core) + self.q_bias.unsqueeze(0).unsqueeze(2)
        k = torch.einsum("bshr,hrd->bhsd", z, self.k_core) + self.k_bias.unsqueeze(0).unsqueeze(2)
        v = torch.einsum("bshr,hrv->bhsv", z, self.v_core) + self.v_bias.unsqueeze(0).unsqueeze(2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.subspace_rank)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).transpose(1, 2).contiguous()
        contributions = torch.einsum("bshv,hvd->bshd", ctx, self.out_lift)
        out = contributions.sum(dim=2)
        if return_contributions:
            return out, contributions
        return out


class SealedCompressedBlock(torch.nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, subspace_rank: int, value_rank: int, ffn_dim: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size)
        self.attn = SealedCompressedAttention(hidden_size, num_heads, subspace_rank, value_rank)
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, ffn_dim, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(ffn_dim, hidden_size, bias=False),
        )
        torch.nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-3)

    def forward(self, x: torch.Tensor, *, return_attn_contributions: bool = False):
        attn_in = self.attn_norm(x)
        if return_attn_contributions:
            attn_out, contributions = self.attn(attn_in, return_contributions=True)
            h = x + attn_out
            y = h + self.ffn(self.ffn_norm(h))
            return y, contributions
        h = x + self.attn(attn_in)
        return h + self.ffn(self.ffn_norm(h))


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def split_init_ranks(subspace_rank: int, q_rank: int | None, k_rank: int | None, v_rank: int | None) -> tuple[int, int, int]:
    if q_rank is None and k_rank is None and v_rank is None:
        q = subspace_rank // 3
        k = subspace_rank // 3
        return q, k, subspace_rank - q - k
    q = q_rank if q_rank is not None else subspace_rank // 3
    k = k_rank if k_rank is not None else subspace_rank // 3
    v = v_rank if v_rank is not None else subspace_rank - q - k
    if q <= 0 or k <= 0 or v <= 0 or q + k + v > subspace_rank:
        raise ValueError(f"Bad init rank split q={q} k={k} v={v} for subspace_rank={subspace_rank}")
    return q, k, v


@torch.no_grad()
def fit_qkv_output_pca(
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    input_norm_weight: torch.Tensor,
    norm_eps: float,
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    stats = {
        name: {
            "n": torch.zeros(num_heads, device=device),
            "sum": torch.zeros(num_heads, head_dim, device=device),
            "xtx": torch.zeros(num_heads, head_dim, head_dim, device=device),
        }
        for name in ("q", "k", "v")
    }
    weights = {"q": q_weight, "k": k_weight, "v": v_weight}
    for path_i, sample_ids in refs_by_path(train_refs).items():
        shard = torch.load(paths[path_i], map_location="cpu")
        x = shard["x"][sample_ids].to(device=device, dtype=torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm_eps) * input_norm_weight
        flat = x.reshape(-1, x.shape[-1])
        for name, weight in weights.items():
            out = (flat @ weight.T).view(flat.shape[0], num_heads, head_dim)
            stats[name]["n"] += out.shape[0]
            stats[name]["sum"] += out.sum(dim=0)
            stats[name]["xtx"] += torch.einsum("nhd,nhe->hde", out, out)

    pca: dict[str, dict[str, torch.Tensor]] = {}
    for name, st in stats.items():
        n = st["n"].view(num_heads, 1)
        mean = st["sum"] / n
        cov = (st["xtx"] - n.view(num_heads, 1, 1) * torch.einsum("hd,he->hde", mean, mean)) / (
            n.view(num_heads, 1, 1) - 1
        ).clamp_min(1)
        vals, vecs = torch.linalg.eigh(cov.cpu())
        order = torch.argsort(vals, descending=True)
        basis = torch.stack([vecs[h, :, order[h]] for h in range(num_heads)], dim=0).to(
            device=device, dtype=torch.float32
        )
        pca[name] = {"mean": mean, "basis": basis}
    return pca


@torch.no_grad()
def init_attention_from_supervised_pca(
    attn: SealedCompressedAttention,
    *,
    attn_norm: RMSNorm,
    ffn_norm: RMSNorm,
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    model_name: str,
    target_layer: int,
    q_rank: int,
    k_rank: int,
    v_rank: int,
    device: str,
) -> dict[str, object]:
    """Initialize the sealed head using the same activation-PCA recipe as QKV branches."""
    from transformers import AutoModelForCausalLM

    teacher = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, trust_remote_code=True)
    layer = teacher.model.layers[target_layer]
    teacher_attn = layer.self_attn
    head_dim = int(teacher_attn.head_dim)
    teacher_num_heads = int(teacher_attn.q_proj.out_features) // head_dim
    if teacher_num_heads != attn.num_heads:
        raise ValueError(f"teacher num_heads={teacher_num_heads} != compressed num_heads={attn.num_heads}")
    if attn.value_rank < head_dim:
        raise ValueError(
            f"value_rank={attn.value_rank} must be >= teacher head_dim={head_dim} for teacher QKV init"
        )

    q_weight = teacher_attn.q_proj.weight.detach().float().to(device)
    k_weight = teacher_attn.k_proj.weight.detach().float().to(device)
    v_weight = teacher_attn.v_proj.weight.detach().float().to(device)
    o_weight = teacher_attn.o_proj.weight.detach().float().cpu()
    input_weight_param = getattr(layer.input_layernorm, "weight", None)
    post_weight_param = getattr(layer.post_attention_layernorm, "weight", None)
    input_norm_weight = (
        input_weight_param.detach().float().to(device)
        if input_weight_param is not None
        else torch.ones(q_weight.shape[1], device=device)
    )
    post_norm_weight = (
        post_weight_param.detach().float().to(device)
        if post_weight_param is not None
        else torch.ones(q_weight.shape[1], device=device)
    )
    norm_eps = float(getattr(layer.input_layernorm, "variance_epsilon", getattr(layer.input_layernorm, "eps", 1e-6)))
    attn_norm.weight.copy_(input_norm_weight.to(device=device, dtype=attn_norm.weight.dtype))
    ffn_norm.weight.copy_(post_norm_weight.to(device=device, dtype=ffn_norm.weight.dtype))
    pca = fit_qkv_output_pca(
        paths,
        train_refs,
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        num_heads=attn.num_heads,
        head_dim=head_dim,
        input_norm_weight=input_norm_weight,
        norm_eps=norm_eps,
        device=device,
    )

    down_w = torch.zeros_like(attn.down.weight.detach().float().cpu())
    q_core = torch.zeros_like(attn.q_core.detach().float().cpu())
    k_core = torch.zeros_like(attn.k_core.detach().float().cpu())
    v_core = torch.zeros_like(attn.v_core.detach().float().cpu())
    q_bias = torch.zeros_like(attn.q_bias.detach().float().cpu())
    k_bias = torch.zeros_like(attn.k_bias.detach().float().cpu())
    v_bias = torch.zeros_like(attn.v_bias.detach().float().cpu())
    out_lift = torch.zeros_like(attn.out_lift.detach().float().cpu())
    for head in range(attn.num_heads):
        hs = head * head_dim
        he = hs + head_dim
        wq = q_weight[hs:he].T.contiguous().cpu()
        wk = k_weight[hs:he].T.contiguous().cpu()
        wv = v_weight[hs:he].T.contiguous().cpu()
        row_start = head * attn.subspace_rank
        q_slice = slice(0, q_rank)
        k_slice = slice(q_rank, q_rank + k_rank)
        v_slice = slice(q_rank + k_rank, q_rank + k_rank + v_rank)
        q_basis = pca["q"]["basis"][head, :, :q_rank].cpu()
        k_basis = pca["k"]["basis"][head, :, :k_rank].cpu()
        v_basis = pca["v"]["basis"][head, :, :v_rank].cpu()
        q_mean = pca["q"]["mean"][head].cpu()
        k_mean = pca["k"]["mean"][head].cpu()
        v_mean = pca["v"]["mean"][head].cpu()

        # Same as init_branch_from_pca: down = W U, up = U^T, bias = mean - mean U U^T.
        down_w[row_start + q_slice.start : row_start + q_slice.stop] = (wq @ q_basis).T
        down_w[row_start + k_slice.start : row_start + k_slice.stop] = (wk @ k_basis).T
        down_w[row_start + v_slice.start : row_start + v_slice.stop] = (wv @ v_basis).T
        q_core[head, q_slice, :head_dim] = q_basis.T
        k_core[head, k_slice, :head_dim] = k_basis.T
        v_core[head, v_slice, :head_dim] = v_basis.T
        q_bias[head, :head_dim] = q_mean - q_mean @ q_basis @ q_basis.T
        k_bias[head, :head_dim] = k_mean - k_mean @ k_basis @ k_basis.T
        v_bias[head, :head_dim] = v_mean - v_mean @ v_basis @ v_basis.T
        out_lift[head, :head_dim, :] = o_weight[:, hs:he].T

    attn.down.weight.copy_(down_w.to(device=device, dtype=attn.down.weight.dtype))
    attn.q_core.copy_(q_core.to(device=device, dtype=attn.q_core.dtype))
    attn.k_core.copy_(k_core.to(device=device, dtype=attn.k_core.dtype))
    attn.v_core.copy_(v_core.to(device=device, dtype=attn.v_core.dtype))
    attn.q_bias.copy_(q_bias.to(device=device, dtype=attn.q_bias.dtype))
    attn.k_bias.copy_(k_bias.to(device=device, dtype=attn.k_bias.dtype))
    attn.v_bias.copy_(v_bias.to(device=device, dtype=attn.v_bias.dtype))
    attn.out_lift.copy_(out_lift.to(device=device, dtype=attn.out_lift.dtype))
    del teacher
    return {
        "type": "supervised_output_pca",
        "model_name": model_name,
        "target_layer": target_layer,
        "teacher_head_dim": head_dim,
        "q_rank": q_rank,
        "k_rank": k_rank,
        "v_rank": v_rank,
        "copied_o_proj": True,
        "copied_layernorms": True,
    }


def rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = torch.mean((pred.float() - target.float()) ** 2)
    denom = torch.var(target.float()).clamp_min(1e-6)
    return err / denom


@torch.no_grad()
def load_teacher_contribution_state(model_name: str, target_layer: int, device: str) -> dict[str, object]:
    from transformers import AutoModelForCausalLM
    from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb

    teacher = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, trust_remote_code=True)
    layer = teacher.model.layers[target_layer]
    attn = layer.self_attn
    head_dim = int(attn.head_dim)
    num_heads = int(attn.q_proj.out_features) // head_dim
    input_weight_param = getattr(layer.input_layernorm, "weight", None)
    norm_weight = (
        input_weight_param.detach().float().to(device)
        if input_weight_param is not None
        else torch.ones(attn.q_proj.in_features, device=device)
    )
    norm_eps = float(getattr(layer.input_layernorm, "variance_epsilon", getattr(layer.input_layernorm, "eps", 1e-6)))
    state = {
        "q_weight": attn.q_proj.weight.detach().float().to(device),
        "k_weight": attn.k_proj.weight.detach().float().to(device),
        "v_weight": attn.v_proj.weight.detach().float().to(device),
        "o_weight": attn.o_proj.weight.detach().float().to(device),
        "norm_weight": norm_weight,
        "norm_eps": norm_eps,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "rotary_emb": teacher.model.rotary_emb.to(device),
        "apply_rotary_pos_emb": apply_rotary_pos_emb,
        "teacher": teacher,
    }
    teacher.to(device)
    teacher.eval()
    return state


@torch.no_grad()
def teacher_head_contributions(x: torch.Tensor, state: dict[str, object]) -> torch.Tensor:
    num_heads = int(state["num_heads"])
    head_dim = int(state["head_dim"])
    norm_weight = state["norm_weight"]
    norm_eps = float(state["norm_eps"])
    x_norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm_eps) * norm_weight
    bsz, seq_len, hidden_size = x_norm.shape
    flat = x_norm.reshape(-1, hidden_size)
    q = (flat @ state["q_weight"].T).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = (flat @ state["k_weight"].T).view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    v = (flat @ state["v_weight"].T).view(bsz, seq_len, num_heads, head_dim)
    position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
    cos, sin = state["rotary_emb"](q, position_ids)
    q, k = state["apply_rotary_pos_emb"](q, k, cos, sin, unsqueeze_dim=1)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
    mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(probs, v.transpose(1, 2)).transpose(1, 2).contiguous()
    o_lift = state["o_weight"].T.view(num_heads, head_dim, hidden_size)
    return torch.einsum("bshd,hdo->bsho", ctx, o_lift)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    value_rank = args.value_rank if args.value_rank is not None else args.subspace_rank
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No activation shards found in {args.capture_dir}")
    first = torch.load(paths[0], map_location="cpu")
    hidden_size = int(first["x"].shape[-1])
    seq_len = int(first["x"].shape[1])
    target_layer = first.get("target_layer", args.target_layer)
    if args.target_layer is not None and target_layer != args.target_layer:
        raise ValueError(f"capture target_layer={target_layer} does not match --target-layer={args.target_layer}")

    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(
        groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed
    )
    model = SealedCompressedBlock(
        hidden_size=hidden_size,
        num_heads=args.num_heads,
        subspace_rank=args.subspace_rank,
        value_rank=value_rank,
        ffn_dim=args.ffn_dim,
    ).to(device=device, dtype=torch.float32)
    init_report: dict[str, object] | None = None
    if args.init_attn_from_supervised_pca:
        if target_layer is None:
            raise ValueError("--init-attn-from-supervised-pca requires target_layer in capture or --target-layer")
        q_init, k_init, v_init = split_init_ranks(
            args.subspace_rank, args.q_init_rank, args.k_init_rank, args.v_init_rank
        )
        init_report = init_attention_from_supervised_pca(
            model.attn,
            attn_norm=model.attn_norm,
            ffn_norm=model.ffn_norm,
            paths=paths,
            train_refs=train_refs,
            model_name=args.model_name,
            target_layer=int(target_layer),
            q_rank=q_init,
            k_rank=k_init,
            v_rank=v_init,
            device=device,
        )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    teacher_contrib_state = None
    if args.head_contribution_loss_weight > 0:
        if target_layer is None:
            raise ValueError("--head-contribution-loss-weight requires target_layer in capture or --target-layer")
        teacher_contrib_state = load_teacher_contribution_state(args.model_name, int(target_layer), device)
        if int(teacher_contrib_state["num_heads"]) != args.num_heads:
            raise ValueError(
                f"teacher num_heads={teacher_contrib_state['num_heads']} != --num-heads={args.num_heads}"
            )

    def run_eval() -> dict[str, float]:
        model.eval()
        acc = defaultdict(float)
        count = 0
        with torch.no_grad():
            for shard, ids in batch_refs(paths, eval_refs, args.batch_size):
                x = shard["x"][ids].to(device=device, dtype=torch.float32)
                y = shard["y"][ids].to(device=device, dtype=torch.float32)
                if teacher_contrib_state is not None:
                    yhat, contrib = model(x, return_attn_contributions=True)
                    teacher_contrib = teacher_head_contributions(x, teacher_contrib_state)
                    head_contrib_mse = relative_mse(contrib, teacher_contrib)
                else:
                    yhat = model(x)
                    head_contrib_mse = 0.0
                b = x.shape[0]
                metrics = {
                    "relative_mse": relative_mse(yhat, y),
                    "delta_relative_mse": relative_mse(yhat - x, y - x),
                    "cosine": cosine_similarity_mean(yhat, y),
                    "head_contribution_relative_mse": head_contrib_mse,
                }
                for key, val in metrics.items():
                    acc[key] += val * b
                count += b
        return {key: val / count for key, val in acc.items()}

    history = [{"epoch": 0, "eval": run_eval()}]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = shard["x"][ids].to(device=device, dtype=torch.float32)
            y = shard["y"][ids].to(device=device, dtype=torch.float32)
            if teacher_contrib_state is not None:
                yhat, contrib = model(x, return_attn_contributions=True)
            else:
                yhat = model(x)
            loss = rel_loss(yhat, y) + 0.5 * rel_loss(yhat - x, y - x)
            if teacher_contrib_state is not None:
                with torch.no_grad():
                    teacher_contrib = teacher_head_contributions(x, teacher_contrib_state)
                loss = loss + args.head_contribution_loss_weight * rel_loss(contrib, teacher_contrib)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = run_eval()
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps, 1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "sealed_compressed_block.pt")
    report = {
        "capture_dir": args.capture_dir,
        "target_layer": target_layer,
        "hidden_size": hidden_size,
        "seq_len": seq_len,
        "num_heads": args.num_heads,
        "subspace_rank": args.subspace_rank,
        "value_rank": value_rank,
        "ffn_dim": args.ffn_dim,
        "attention_output": "sum_per_head_lifts",
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "head_contribution_loss_weight": args.head_contribution_loss_weight,
        "parameter_count": count_parameters(model),
        "attention_init": init_report,
        "history": history,
    }
    with (out / "sealed_compressed_block_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
