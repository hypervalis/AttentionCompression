#!/usr/bin/env python3
"""Train independent autoencoders for each attention head output."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-head autoencoders for captured head_contexts.")
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-windows-per-bin", type=int, default=48)
    p.add_argument("--eval-windows-per-bin", type=int, default=16)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--autoencoder-kind", default="linear", choices=["linear", "mlp", "decoder_residual_mlp"])
    p.add_argument("--init-linear-state", default=None, help="Optional PerHeadLinearAutoencoder state dict to initialize from.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
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


class PerHeadLinearAutoencoder(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder_weight = torch.nn.Parameter(torch.empty(num_heads, head_dim, bottleneck_dim))
        self.encoder_bias = torch.nn.Parameter(torch.zeros(num_heads, bottleneck_dim))
        self.decoder_weight = torch.nn.Parameter(torch.empty(num_heads, bottleneck_dim, head_dim))
        self.decoder_bias = torch.nn.Parameter(torch.zeros(num_heads, head_dim))
        torch.nn.init.xavier_uniform_(self.encoder_weight)
        torch.nn.init.xavier_uniform_(self.decoder_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bshd,hdr->bshr", x, self.encoder_weight) + self.encoder_bias
        return torch.einsum("bshr,hrd->bshd", z, self.decoder_weight) + self.decoder_bias


class PerHeadMlpAutoencoder(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, bottleneck_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.b1 = torch.nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.w2 = torch.nn.Parameter(torch.empty(num_heads, hidden_dim, bottleneck_dim))
        self.b2 = torch.nn.Parameter(torch.zeros(num_heads, bottleneck_dim))
        self.w3 = torch.nn.Parameter(torch.empty(num_heads, bottleneck_dim, hidden_dim))
        self.b3 = torch.nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.w4 = torch.nn.Parameter(torch.empty(num_heads, hidden_dim, head_dim))
        self.b4 = torch.nn.Parameter(torch.zeros(num_heads, head_dim))
        for weight in (self.w1, self.w2, self.w3, self.w4):
            torch.nn.init.xavier_uniform_(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bshd,hdr->bshr", x, self.w1) + self.b1
        z = torch.nn.functional.gelu(z)
        z = torch.einsum("bshr,hrk->bshk", z, self.w2) + self.b2
        z = torch.nn.functional.gelu(z)
        z = torch.einsum("bshk,hkr->bshr", z, self.w3) + self.b3
        z = torch.nn.functional.gelu(z)
        return torch.einsum("bshr,hrd->bshd", z, self.w4) + self.b4


class PerHeadDecoderResidualMlpAutoencoder(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, bottleneck_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder_weight = torch.nn.Parameter(torch.empty(num_heads, head_dim, bottleneck_dim))
        self.encoder_bias = torch.nn.Parameter(torch.zeros(num_heads, bottleneck_dim))
        self.decoder_weight = torch.nn.Parameter(torch.empty(num_heads, bottleneck_dim, head_dim))
        self.decoder_bias = torch.nn.Parameter(torch.zeros(num_heads, head_dim))
        self.res1_weight = torch.nn.Parameter(torch.empty(num_heads, bottleneck_dim, hidden_dim))
        self.res1_bias = torch.nn.Parameter(torch.zeros(num_heads, hidden_dim))
        self.res2_weight = torch.nn.Parameter(torch.zeros(num_heads, hidden_dim, head_dim))
        self.res2_bias = torch.nn.Parameter(torch.zeros(num_heads, head_dim))
        torch.nn.init.xavier_uniform_(self.encoder_weight)
        torch.nn.init.xavier_uniform_(self.decoder_weight)
        torch.nn.init.xavier_uniform_(self.res1_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.einsum("bshd,hdr->bshr", x, self.encoder_weight) + self.encoder_bias
        linear = torch.einsum("bshr,hrd->bshd", z, self.decoder_weight) + self.decoder_bias
        residual = torch.einsum("bshr,hrk->bshk", z, self.res1_weight) + self.res1_bias
        residual = torch.nn.functional.gelu(residual)
        residual = torch.einsum("bshk,hkd->bshd", residual, self.res2_weight) + self.res2_bias
        return linear + residual


def per_head_relative_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred.float() - target.float()).pow(2).mean(dim=(0, 1, 3))
    var = target.float().var(dim=(0, 1, 3)).clamp_min(1e-6)
    return mse / var


@torch.no_grad()
def evaluate(model, paths, eval_refs, batch_size, device):
    model.eval()
    rel_mse_sum = None
    cosine_sum = None
    count = 0
    for shard, ids in batch_refs(paths, eval_refs, batch_size):
        x = shard["head_contexts"][ids].to(device=device, dtype=torch.float32)
        y = model(x)
        rel = per_head_relative_mse(y, x)
        cos = torch.nn.functional.cosine_similarity(y.float(), x.float(), dim=-1).mean(dim=(0, 1))
        b = x.shape[0]
        rel_mse_sum = rel * b if rel_mse_sum is None else rel_mse_sum + rel * b
        cosine_sum = cos * b if cosine_sum is None else cosine_sum + cos * b
        count += b
    rel_avg = (rel_mse_sum / count).detach().cpu()
    cos_avg = (cosine_sum / count).detach().cpu()
    return {
        "mean_relative_mse": float(rel_avg.mean().item()),
        "mean_cosine": float(cos_avg.mean().item()),
        "per_head": [
            {"head": i, "relative_mse": float(rel_avg[i].item()), "cosine": float(cos_avg[i].item())}
            for i in range(rel_avg.numel())
        ],
    }


def train_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return per_head_relative_mse(pred, target).mean()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards in {args.capture_dir}")
    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed)
    first = torch.load(paths[0], map_location="cpu")
    num_heads = int(first["head_contexts"].shape[2])
    head_dim = int(first["head_contexts"].shape[3])

    if args.autoencoder_kind == "linear":
        hidden_dim = None
        model = PerHeadLinearAutoencoder(num_heads, head_dim, args.bottleneck_dim).to(device)
    elif args.autoencoder_kind == "mlp":
        hidden_dim = args.hidden_dim or max(head_dim // 2, args.bottleneck_dim)
        model = PerHeadMlpAutoencoder(num_heads, head_dim, args.bottleneck_dim, hidden_dim).to(device)
    else:
        hidden_dim = args.hidden_dim or max(head_dim // 2, args.bottleneck_dim)
        model = PerHeadDecoderResidualMlpAutoencoder(num_heads, head_dim, args.bottleneck_dim, hidden_dim).to(device)
    if args.init_linear_state:
        state = torch.load(args.init_linear_state, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded init_linear_state missing={missing} unexpected={unexpected}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = [{"epoch": 0, "eval": evaluate(model, paths, eval_refs, args.batch_size, device)}]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = shard["head_contexts"][ids].to(device=device, dtype=torch.float32)
            y = model(x)
            loss = train_loss(y, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(model, paths, eval_refs, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(
            f"epoch={epoch} train_loss={total / max(steps, 1):.6f} "
            f"mean_relative_mse={metrics['mean_relative_mse']:.6f} mean_cosine={metrics['mean_cosine']:.6f}",
            flush=True,
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "per_head_output_autoencoders.pt")
    report = {
        "capture_dir": args.capture_dir,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "num_heads": num_heads,
        "head_dim": head_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "autoencoder_kind": args.autoencoder_kind,
        "hidden_dim": hidden_dim,
        "epochs": args.epochs,
        "history": history,
    }
    with (out / "per_head_output_autoencoder_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
