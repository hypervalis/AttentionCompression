#!/usr/bin/env python3
"""Analyze head-context PCA and train activation autoencoders."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention_compression.attention_metrics import cosine_similarity_mean, relative_mse
from attention_compression.activations import find_transformer_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head PCA spectra + activation autoencoder.")
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--train-windows-per-bin", type=int, default=48)
    p.add_argument("--eval-windows-per-bin", type=int, default=16)
    p.add_argument("--autoencoder-dim", type=int, default=1024)
    p.add_argument("--autoencoder-hidden-dim", type=int, default=None)
    p.add_argument("--autoencoder-kind", default="linear", choices=["linear", "mlp", "decoder_residual_mlp"])
    p.add_argument(
        "--autoencoder-target",
        default="ffn_input",
        choices=["ffn_input", "head_context_concat"],
        help="Activation to reconstruct. head_context_concat flattens all heads before o_proj.",
    )
    p.add_argument(
        "--loss-space",
        default="activation",
        choices=["activation", "o_proj"],
        help="Optimize reconstruction before o_proj, or after the frozen attention o_proj.",
    )
    p.add_argument("--init-linear-state", default=None, help="Optional LinearAutoencoder state dict to initialize from.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
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


def rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.float() - target.float()) ** 2) / torch.var(target.float()).clamp_min(1e-6)


def apply_o_proj(x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return x
    return torch.nn.functional.linear(x, weight, bias)


def get_autoencoder_target(shard: dict, ids: list[int], target_name: str, *, device: str) -> torch.Tensor:
    if target_name == "ffn_input":
        return shard["ffn_input"][ids].to(device=device, dtype=torch.float32)
    if target_name == "head_context_concat":
        head_contexts = shard["head_contexts"][ids].to(device=device, dtype=torch.float32)
        return head_contexts.flatten(start_dim=2)
    raise ValueError(f"Unsupported autoencoder target: {target_name}")


class LinearAutoencoder(torch.nn.Module):
    def __init__(self, dim: int, bottleneck: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(dim, bottleneck, bias=True)
        self.decoder = torch.nn.Linear(bottleneck, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class MlpAutoencoder(torch.nn.Module):
    def __init__(self, dim: int, bottleneck: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, bottleneck),
            torch.nn.GELU(),
            torch.nn.Linear(bottleneck, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderResidualMlpAutoencoder(torch.nn.Module):
    def __init__(self, dim: int, bottleneck: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(dim, bottleneck, bias=True)
        self.decoder = torch.nn.Linear(bottleneck, dim, bias=True)
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(bottleneck, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z) + self.residual(z)


@torch.no_grad()
def head_pca_spectra(paths, train_refs, *, device: str):
    first = torch.load(paths[0], map_location="cpu")
    num_heads = int(first["head_contexts"].shape[2])
    head_dim = int(first["head_contexts"].shape[3])
    stats = {
        "n": torch.zeros(num_heads, device=device),
        "sum": torch.zeros(num_heads, head_dim, device=device),
        "xtx": torch.zeros(num_heads, head_dim, head_dim, device=device),
    }
    for path_i, ids in refs_by_path(train_refs).items():
        shard = torch.load(paths[path_i], map_location="cpu")
        h = shard["head_contexts"][ids].to(device=device, dtype=torch.float32)
        flat = h.reshape(-1, num_heads, head_dim)
        stats["n"] += flat.shape[0]
        stats["sum"] += flat.sum(dim=0)
        stats["xtx"] += torch.einsum("nhd,nhe->hde", flat, flat)
    n = stats["n"].view(num_heads, 1)
    mean = stats["sum"] / n
    cov = (stats["xtx"] - n.view(num_heads, 1, 1) * torch.einsum("hd,he->hde", mean, mean)) / (
        n.view(num_heads, 1, 1) - 1
    ).clamp_min(1)
    spectra = []
    for head in range(num_heads):
        vals = torch.linalg.eigvalsh(cov[head].cpu()).clamp_min(0).flip(0)
        total = vals.sum().clamp_min(1e-12)
        csum = torch.cumsum(vals, dim=0) / total
        ranks = {}
        for frac in (0.9, 0.95, 0.99):
            ranks[str(frac)] = int(torch.searchsorted(csum, torch.tensor(frac)).item() + 1)
        spectra.append(
            {
                "head": head,
                "top10_explained": [float(x) for x in csum[:10].tolist()],
                "rank_for_variance": ranks,
            }
        )
    return spectra


def evaluate_autoencoder(model, paths, eval_refs, batch_size, device, target_name: str, o_proj_weight=None, o_proj_bias=None):
    model.eval()
    acc = defaultdict(float)
    count = 0
    with torch.no_grad():
        for shard, ids in batch_refs(paths, eval_refs, batch_size):
            x = get_autoencoder_target(shard, ids, target_name, device=device)
            y = model(x)
            b = x.shape[0]
            metrics = {
                "relative_mse": relative_mse(y, x),
                "cosine": cosine_similarity_mean(y, x),
            }
            if o_proj_weight is not None:
                y_proj = apply_o_proj(y, o_proj_weight, o_proj_bias)
                x_proj = apply_o_proj(x, o_proj_weight, o_proj_bias)
                metrics["o_proj_relative_mse"] = relative_mse(y_proj, x_proj)
                metrics["o_proj_cosine"] = cosine_similarity_mean(y_proj, x_proj)
            for key, val in metrics.items():
                acc[key] += val * b
            count += b
    return {k: v / count for k, v in acc.items()}


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
    if args.autoencoder_target == "ffn_input":
        hidden_size = int(first["ffn_input"].shape[-1])
    else:
        hidden_size = int(first["head_contexts"].shape[2] * first["head_contexts"].shape[3])

    o_proj_weight = None
    o_proj_bias = None
    if args.loss_space == "o_proj":
        if args.autoencoder_target != "head_context_concat":
            raise ValueError("--loss-space o_proj requires --autoencoder-target head_context_concat")
        from transformers import AutoModelForCausalLM

        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
        teacher.eval().to(device)
        _layer_path, layers = find_transformer_layers(teacher)
        o_proj = layers[args.target_layer].self_attn.o_proj
        o_proj_weight = o_proj.weight.detach().to(device=device, dtype=torch.float32)
        o_proj_bias = (
            None if o_proj.bias is None else o_proj.bias.detach().to(device=device, dtype=torch.float32)
        )
        del teacher

    spectra = head_pca_spectra(paths, train_refs, device=device)
    if args.autoencoder_kind == "linear":
        model = LinearAutoencoder(hidden_size, args.autoencoder_dim).to(device)
        hidden_dim = None
    elif args.autoencoder_kind == "mlp":
        hidden_dim = args.autoencoder_hidden_dim or max(hidden_size // 2, args.autoencoder_dim)
        model = MlpAutoencoder(hidden_size, args.autoencoder_dim, hidden_dim).to(device)
    else:
        hidden_dim = args.autoencoder_hidden_dim or max(hidden_size // 2, args.autoencoder_dim)
        model = DecoderResidualMlpAutoencoder(hidden_size, args.autoencoder_dim, hidden_dim).to(device)
    if args.init_linear_state:
        state = torch.load(args.init_linear_state, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded init_linear_state missing={missing} unexpected={unexpected}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = [
        {
            "epoch": 0,
            "eval": evaluate_autoencoder(
                model, paths, eval_refs, args.batch_size, device, args.autoencoder_target, o_proj_weight, o_proj_bias
            ),
        }
    ]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = get_autoencoder_target(shard, ids, args.autoencoder_target, device=device)
            y = model(x)
            if args.loss_space == "o_proj":
                loss = rel_loss(apply_o_proj(y, o_proj_weight, o_proj_bias), apply_o_proj(x, o_proj_weight, o_proj_bias))
            else:
                loss = rel_loss(y, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate_autoencoder(
            model, paths, eval_refs, args.batch_size, device, args.autoencoder_target, o_proj_weight, o_proj_bias
        )
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / f"{args.autoencoder_target}_autoencoder.pt")
    report = {
        "capture_dir": args.capture_dir,
        "autoencoder_target": args.autoencoder_target,
        "autoencoder_kind": args.autoencoder_kind,
        "loss_space": args.loss_space,
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "hidden_size": hidden_size,
        "autoencoder_dim": args.autoencoder_dim,
        "autoencoder_hidden_dim": hidden_dim,
        "epochs": args.epochs,
        "head_pca_spectra": spectra,
        "autoencoder_history": history,
    }
    with (out / "head_pca_activation_autoencoder_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
