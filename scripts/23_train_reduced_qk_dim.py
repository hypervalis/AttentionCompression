#!/usr/bin/env python3
"""Train dense smaller Q/K head dimensions, then prune gated dimensions.

This tests the "same operation shape, fewer output dims" recipe:

1. Initialize Q/K as dense ``D -> start_qk_dim`` linears from supervised PCA of
   teacher Q/K outputs for one head.
2. Train with learnable per-dimension gates and an L1 sparsity penalty.
3. Prune to ``target_qk_dim`` dimensions.
4. Fine-tune the pruned dense Q/K linears against teacher attention behavior.

V stays dense/full head_dim for this head, so the attention payload is preserved
while Q/K score vectors become true smaller dense projections.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention_compression.attention_metrics import (
    attention_kl,
    causal_attention,
    causal_logit_relative_mse,
    cosine_similarity_mean,
    relative_mse,
    topk_overlap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/prune dense reduced-dim Q/K for one head.")
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--head-index", type=int, default=0)
    p.add_argument("--start-qk-dim", type=int, default=96)
    p.add_argument("--target-qk-dim", type=int, default=64)
    p.add_argument("--train-windows-per-bin", type=int, default=48)
    p.add_argument("--eval-windows-per-bin", type=int, default=16)
    p.add_argument("--sparse-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sparsity-weight", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


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


def choose_train_eval(groups, train_n: int, eval_n: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    train, eval_ = [], []
    for refs in groups.values():
        if len(refs) < train_n + eval_n:
            raise ValueError(f"Need {train_n + eval_n} rows per bin; got {len(refs)}")
        perm = torch.randperm(len(refs), generator=gen).tolist()
        train.extend(refs[i] for i in perm[:train_n])
        eval_.extend(refs[i] for i in perm[train_n : train_n + eval_n])
    return train, eval_


def batch_refs(paths: list[Path], refs: list[tuple[int, int]], batch_size: int, *, shuffle: bool = False, seed: int = 0):
    grouped = refs_by_path(refs)
    items = list(grouped.items())
    gen = torch.Generator().manual_seed(seed)
    if shuffle and items:
        order = torch.randperm(len(items), generator=gen).tolist()
        items = [items[i] for i in order]
    for path_i, ids in items:
        ids = list(ids)
        if shuffle and ids:
            order = torch.randperm(len(ids), generator=gen).tolist()
            ids = [ids[i] for i in order]
        shard = torch.load(paths[path_i], map_location="cpu")
        for start in range(0, len(ids), batch_size):
            yield shard, ids[start : start + batch_size]


class GatedReducedQK(torch.nn.Module):
    def __init__(self, input_dim: int, qk_dim: int) -> None:
        super().__init__()
        self.q_weight = torch.nn.Parameter(torch.empty(input_dim, qk_dim))
        self.k_weight = torch.nn.Parameter(torch.empty(input_dim, qk_dim))
        self.q_bias = torch.nn.Parameter(torch.zeros(qk_dim))
        self.k_bias = torch.nn.Parameter(torch.zeros(qk_dim))
        self.gate_logits = torch.nn.Parameter(torch.full((qk_dim,), 2.0))
        torch.nn.init.normal_(self.q_weight, std=0.01)
        torch.nn.init.normal_(self.k_weight, std=0.01)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.gates()
        return (x @ self.q_weight + self.q_bias) * g, (x @ self.k_weight + self.k_bias) * g


class ReducedQK(torch.nn.Module):
    def __init__(self, q_weight: torch.Tensor, k_weight: torch.Tensor, q_bias: torch.Tensor, k_bias: torch.Tensor) -> None:
        super().__init__()
        self.q_weight = torch.nn.Parameter(q_weight.clone())
        self.k_weight = torch.nn.Parameter(k_weight.clone())
        self.q_bias = torch.nn.Parameter(q_bias.clone())
        self.k_bias = torch.nn.Parameter(k_bias.clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x @ self.q_weight + self.q_bias, x @ self.k_weight + self.k_bias


@torch.no_grad()
def fit_output_pca(paths, train_refs, *, projection: torch.Tensor, out_dim: int, device: str):
    n = 0
    s = torch.zeros(out_dim, device=device)
    xtx = torch.zeros(out_dim, out_dim, device=device)
    for path_i, ids in refs_by_path(train_refs).items():
        shard = torch.load(paths[path_i], map_location="cpu")
        x = shard["x_attn"][ids].to(device=device, dtype=torch.float32).reshape(-1, projection.shape[0])
        y = x @ projection
        n += y.shape[0]
        s += y.sum(dim=0)
        xtx += y.T @ y
    mean = s / n
    cov = (xtx - n * torch.outer(mean, mean)) / max(n - 1, 1)
    vals, vecs = torch.linalg.eigh(cov.cpu())
    order = torch.argsort(vals, descending=True)
    return mean, vecs[:, order].to(device=device, dtype=torch.float32)


def apply_rope(q: torch.Tensor, k: torch.Tensor, *, teacher, apply_rotary_pos_emb) -> tuple[torch.Tensor, torch.Tensor]:
    # q/k: [B, T, D]. OLMo rotary returns full head_dim tables; slice to reduced D.
    pos = torch.arange(q.shape[1], device=q.device).unsqueeze(0).expand(q.shape[0], -1)
    cos, sin = teacher.model.rotary_emb(q.unsqueeze(1), pos)
    cos = cos[..., : q.shape[-1]]
    sin = sin[..., : q.shape[-1]]
    qr, kr = apply_rotary_pos_emb(q.unsqueeze(1), k.unsqueeze(1), cos, sin, unsqueeze_dim=1)
    return qr.squeeze(1), kr.squeeze(1)


def rel_loss_tensor(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.float() - target.float()) ** 2) / torch.var(target.float()).clamp_min(1e-6)


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM
    from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb

    if args.start_qk_dim <= args.target_qk_dim:
        raise ValueError("--start-qk-dim must be > --target-qk-dim for prune test")
    if args.target_qk_dim % 2 or args.start_qk_dim % 2:
        raise ValueError("Q/K dims must be even for RoPE")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No activation shards in {args.capture_dir}")
    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32, trust_remote_code=True)
    teacher.to(device)
    teacher.eval()
    attn = teacher.model.layers[args.target_layer].self_attn
    head_dim = int(attn.head_dim)
    hs, he = args.head_index * head_dim, (args.head_index + 1) * head_dim
    q_full = attn.q_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous()
    k_full = attn.k_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous()
    v_full = attn.v_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous()

    q_mean, q_basis = fit_output_pca(paths, train_refs, projection=q_full, out_dim=head_dim, device=device)
    k_mean, k_basis = fit_output_pca(paths, train_refs, projection=k_full, out_dim=head_dim, device=device)

    model = GatedReducedQK(q_full.shape[0], args.start_qk_dim).to(device)
    with torch.no_grad():
        uq = q_basis[:, : args.start_qk_dim]
        uk = k_basis[:, : args.start_qk_dim]
        model.q_weight.copy_(q_full @ uq)
        model.k_weight.copy_(k_full @ uk)
        model.q_bias.copy_(q_mean @ uq)
        model.k_bias.copy_(k_mean @ uk)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def run_batch(x: torch.Tensor, h: torch.Tensor, module: torch.nn.Module):
        with torch.no_grad():
            q_t = x @ q_full
            k_t = x @ k_full
            v = x @ v_full
            q_tr, k_tr = apply_rope(q_t, k_t, teacher=teacher, apply_rotary_pos_emb=apply_rotary_pos_emb)
            logits_t, probs_t, head_t = causal_attention(q_tr, k_tr, v)
        q_s, k_s = module(x)
        q_sr, k_sr = apply_rope(q_s, k_s, teacher=teacher, apply_rotary_pos_emb=apply_rotary_pos_emb)
        logits_s, probs_s, head_s = causal_attention(q_sr, k_sr, v)
        return logits_t, probs_t, head_t, logits_s, probs_s, head_s

    def evaluate(module: torch.nn.Module):
        module.eval()
        acc = defaultdict(float)
        count = 0
        with torch.no_grad():
            for shard, ids in batch_refs(paths, eval_refs, args.batch_size):
                x = shard["x_attn"][ids].to(device=device, dtype=torch.float32)
                h = shard["head_context"][ids].to(device=device, dtype=torch.float32)
                logits_t, probs_t, _head_t, logits_s, probs_s, head_s = run_batch(x, h, module)
                b = x.shape[0]
                metrics = {
                    "logit_relative_mse": causal_logit_relative_mse(logits_s, logits_t),
                    "attention_kl": attention_kl(probs_t, probs_s),
                    "attention_top5_overlap": topk_overlap(probs_t, probs_s, 5),
                    "head_context_relative_mse": relative_mse(head_s, h),
                    "head_context_cosine": cosine_similarity_mean(head_s, h),
                }
                for key, val in metrics.items():
                    acc[key] += val * b
                count += b
        return {k: v / count for k, v in acc.items()}

    history = [{"phase": "sparse", "epoch": 0, "eval": evaluate(model)}]
    print("phase=sparse epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)
    causal_mask = None
    for epoch in range(1, args.sparse_epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = shard["x_attn"][ids].to(device=device, dtype=torch.float32)
            h = shard["head_context"][ids].to(device=device, dtype=torch.float32)
            logits_t, probs_t, _head_t, logits_s, probs_s, head_s = run_batch(x, h, model)
            if causal_mask is None:
                causal_mask = torch.tril(torch.ones(logits_t.shape[-1], logits_t.shape[-1], device=device, dtype=torch.bool))
            loss = (
                0.5 * rel_loss_tensor(logits_s[..., causal_mask], logits_t[..., causal_mask])
                + 0.1 * torch.nn.functional.kl_div(probs_s.clamp_min(1e-12).log(), probs_t, reduction="batchmean")
                + rel_loss_tensor(head_s, h)
                + args.sparsity_weight * model.gates().mean()
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(model)
        history.append({"phase": "sparse", "epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"phase=sparse epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)

    with torch.no_grad():
        score = model.gates() * (model.q_weight.norm(dim=0) + model.k_weight.norm(dim=0))
        keep = torch.topk(score, k=args.target_qk_dim).indices.sort().values
        gates = model.gates()[keep]
        pruned = ReducedQK(
            model.q_weight[:, keep] * gates.unsqueeze(0),
            model.k_weight[:, keep] * gates.unsqueeze(0),
            model.q_bias[keep] * gates,
            model.k_bias[keep] * gates,
        ).to(device)
    opt = torch.optim.AdamW(pruned.parameters(), lr=args.lr, weight_decay=1e-4)
    metrics = evaluate(pruned)
    history.append({"phase": "pruned", "epoch": 0, "kept_indices": keep.detach().cpu().tolist(), "eval": metrics})
    print("phase=pruned epoch=0", json.dumps(metrics, sort_keys=True), flush=True)

    for epoch in range(1, args.finetune_epochs + 1):
        pruned.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + 100 + epoch):
            x = shard["x_attn"][ids].to(device=device, dtype=torch.float32)
            h = shard["head_context"][ids].to(device=device, dtype=torch.float32)
            logits_t, probs_t, _head_t, logits_s, probs_s, head_s = run_batch(x, h, pruned)
            loss = (
                0.5 * rel_loss_tensor(logits_s[..., causal_mask], logits_t[..., causal_mask])
                + 0.1 * torch.nn.functional.kl_div(probs_s.clamp_min(1e-12).log(), probs_t, reduction="batchmean")
                + rel_loss_tensor(head_s, h)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(pruned)
        history.append({"phase": "finetune", "epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"phase=finetune epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q_weight": pruned.q_weight.detach().cpu(),
            "k_weight": pruned.k_weight.detach().cpu(),
            "q_bias": pruned.q_bias.detach().cpu(),
            "k_bias": pruned.k_bias.detach().cpu(),
            "kept_indices": keep.detach().cpu(),
        },
        out / "reduced_qk_dim_model.pt",
    )
    report = {
        "capture_dir": args.capture_dir,
        "target_layer": args.target_layer,
        "head_index": args.head_index,
        "head_dim": head_dim,
        "start_qk_dim": args.start_qk_dim,
        "target_qk_dim": args.target_qk_dim,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "sparse_epochs": args.sparse_epochs,
        "finetune_epochs": args.finetune_epochs,
        "sparsity_weight": args.sparsity_weight,
        "history": history,
    }
    with (out / "reduced_qk_dim_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
