#!/usr/bin/env python3
"""Train an o_proj replacement to mimic the autoencoded o_proj target.

Given captured concatenated head outputs ``x`` and a trained autoencoder
``AE(x)``, learn a new projection ``P(x)`` to match
``teacher_o_proj(AE(x))``. This tests whether the autoencoder correction can be
distilled into ``o_proj`` itself, optionally using a low-rank replacement.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from attention_compression.activations import find_transformer_layers
from attention_compression.attention_metrics import cosine_similarity_mean, relative_mse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train o_proj replacement to match o_proj(AE(x)).")
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ae-state", required=True)
    p.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    p.add_argument("--target-layer", type=int, default=0)
    p.add_argument("--train-windows-per-bin", type=int, default=48)
    p.add_argument("--eval-windows-per-bin", type=int, default=16)
    p.add_argument("--bottleneck-dim", type=int, default=1024)
    p.add_argument("--ae-hidden-dim", type=int, default=1536)
    p.add_argument(
        "--ae-kind",
        default="decoder_residual_mlp",
        choices=["linear", "decoder_residual_mlp"],
    )
    p.add_argument("--projection-kind", default="dense", choices=["dense", "lowrank", "pca_lowrank"])
    p.add_argument("--projection-rank", type=int, default=512)
    p.add_argument("--pca-ridge", type=float, default=1e-3)
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


class FrozenAutoencoder(torch.nn.Module):
    def __init__(self, dim: int, bottleneck: int, hidden_dim: int, kind: str, state_path: str) -> None:
        super().__init__()
        state = torch.load(state_path, map_location="cpu")
        self.kind = kind
        self.encoder = torch.nn.Linear(dim, bottleneck)
        self.decoder = torch.nn.Linear(bottleneck, dim)
        self.encoder.weight.data.copy_(state["encoder.weight"].float())
        self.encoder.bias.data.copy_(state["encoder.bias"].float())
        self.decoder.weight.data.copy_(state["decoder.weight"].float())
        self.decoder.bias.data.copy_(state["decoder.bias"].float())
        if kind == "decoder_residual_mlp":
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(bottleneck, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, dim),
            )
            self.residual[0].weight.data.copy_(state["residual.0.weight"].float())
            self.residual[0].bias.data.copy_(state["residual.0.bias"].float())
            self.residual[2].weight.data.copy_(state["residual.2.weight"].float())
            self.residual[2].bias.data.copy_(state["residual.2.bias"].float())
        else:
            self.residual = None
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        y = self.decoder(z)
        if self.residual is not None:
            y = y + self.residual(z)
        return y

    def linearized_weight_bias(self, teacher_weight: torch.Tensor, teacher_bias: torch.Tensor | None):
        weight = teacher_weight @ self.decoder.weight.detach().cpu() @ self.encoder.weight.detach().cpu()
        decoded_bias = torch.nn.functional.linear(
            self.encoder.bias.detach().cpu(), self.decoder.weight.detach().cpu(), self.decoder.bias.detach().cpu()
        )
        bias = torch.nn.functional.linear(decoded_bias, teacher_weight, teacher_bias)
        return weight, bias


class LowRankProjection(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, init_dense: torch.Tensor, init_bias: torch.Tensor | None):
        super().__init__()
        u, s, vh = torch.linalg.svd(init_dense.float(), full_matrices=False)
        rank = min(rank, s.numel())
        self.down = torch.nn.Linear(in_dim, rank, bias=False)
        self.up = torch.nn.Linear(rank, out_dim, bias=init_bias is not None)
        # init_dense maps x -> y as y = x @ init_dense.T. Factor init_dense ~= up @ down.
        self.up.weight.data.copy_(u[:, :rank] * s[:rank].sqrt().unsqueeze(0))
        self.down.weight.data.copy_(s[:rank].sqrt().unsqueeze(1) * vh[:rank, :])
        if init_bias is not None:
            self.up.bias.data.copy_(init_bias.float())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(z))


def init_lowrank_from_supervised_pca(
    autoencoder,
    o_proj,
    paths,
    train_refs,
    batch_size: int,
    rank: int,
    device: str,
    ridge: float,
):
    first = torch.load(paths[0], map_location="cpu")
    dim = int(first["head_contexts"].shape[2] * first["head_contexts"].shape[3])
    n = 0
    sum_x = torch.zeros(dim, device=device)
    sum_y = torch.zeros(dim, device=device)
    xtx = torch.zeros(dim, dim, device=device)
    yty = torch.zeros(dim, dim, device=device)
    xty = torch.zeros(dim, dim, device=device)
    autoencoder.eval()
    o_proj.eval()
    with torch.no_grad():
        for shard, ids in batch_refs(paths, train_refs, batch_size):
            x = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(0, 1).flatten(start_dim=1)
            y = o_proj(autoencoder(x))
            n += x.shape[0]
            sum_x += x.sum(dim=0)
            sum_y += y.sum(dim=0)
            xtx += x.T @ x
            yty += y.T @ y
            xty += x.T @ y
    mean_x = sum_x / n
    mean_y = sum_y / n
    x_center = xtx - n * torch.outer(mean_x, mean_x)
    y_center = yty - n * torch.outer(mean_y, mean_y)
    xy_center = xty - n * torch.outer(mean_x, mean_y)

    # CPU eigensolve avoids CUDA/cuSolver version issues seen on this host.
    x_center_cpu = x_center.cpu()
    xy_center_cpu = xy_center.cpu()
    vals, vecs = torch.linalg.eigh((y_center / max(n - 1, 1)).cpu())
    order = torch.argsort(vals, descending=True)[:rank]
    basis = vecs[:, order].to(dtype=torch.float32).contiguous()
    xz = xy_center_cpu @ basis
    scale = torch.trace(x_center_cpu) / max(dim, 1)
    reg = ridge * scale.clamp_min(1e-6)
    eye = torch.eye(dim)
    coef = torch.linalg.solve(x_center_cpu + reg.cpu() * eye, xz)
    bias = mean_y.cpu() - (mean_x.cpu() @ coef) @ basis.T
    return coef.T.detach(), basis.detach(), bias.detach()


def rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.float() - target.float()) ** 2) / torch.var(target.float()).clamp_min(1e-6)


@torch.no_grad()
def evaluate(autoencoder, projector, o_proj, paths, eval_refs, batch_size, device):
    autoencoder.eval()
    projector.eval()
    acc = defaultdict(float)
    count = 0
    for shard, ids in batch_refs(paths, eval_refs, batch_size):
        x = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(start_dim=2)
        pred = projector(x)
        target = o_proj(autoencoder(x))
        teacher = o_proj(x)
        b = x.shape[0]
        metrics = {
            "target_relative_mse": relative_mse(pred, target),
            "target_cosine": cosine_similarity_mean(pred, target),
            "teacher_relative_mse": relative_mse(pred, teacher),
            "teacher_cosine": cosine_similarity_mean(pred, teacher),
            "ae_target_vs_teacher_relative_mse": relative_mse(target, teacher),
            "ae_target_vs_teacher_cosine": cosine_similarity_mean(target, teacher),
        }
        for key, val in metrics.items():
            acc[key] += val * b
        count += b
    return {key: val / count for key, val in acc.items()}


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
    in_dim = int(first["head_contexts"].shape[2] * first["head_contexts"].shape[3])

    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=dtype, trust_remote_code=True)
    teacher.eval().to(device)
    _layer_path, layers = find_transformer_layers(teacher)
    o_proj = layers[args.target_layer].self_attn.o_proj.to(device=device, dtype=torch.float32)
    teacher_weight = o_proj.weight.detach().float().cpu()
    teacher_bias = None if o_proj.bias is None else o_proj.bias.detach().float().cpu()

    autoencoder = FrozenAutoencoder(in_dim, args.bottleneck_dim, args.ae_hidden_dim, args.ae_kind, args.ae_state).to(device)
    init_weight, init_bias = autoencoder.linearized_weight_bias(teacher_weight, teacher_bias)

    if args.projection_kind == "dense":
        projector = torch.nn.Linear(in_dim, in_dim, bias=True)
        projector.weight.data.copy_(init_weight)
        projector.bias.data.copy_(init_bias)
    elif args.projection_kind == "lowrank":
        projector = LowRankProjection(in_dim, in_dim, args.projection_rank, init_weight, init_bias)
    else:
        projector = LowRankProjection(in_dim, in_dim, args.projection_rank, init_weight, init_bias)
        down_w, up_w, bias = init_lowrank_from_supervised_pca(
            autoencoder,
            o_proj,
            paths,
            train_refs,
            args.batch_size,
            args.projection_rank,
            device,
            args.pca_ridge,
        )
        projector.down.weight.data.copy_(down_w)
        projector.up.weight.data.copy_(up_w)
        projector.up.bias.data.copy_(bias)
    projector = projector.to(device)
    opt = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=1e-4)

    history = [{"epoch": 0, "eval": evaluate(autoencoder, projector, o_proj, paths, eval_refs, args.batch_size, device)}]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)
    for epoch in range(1, args.epochs + 1):
        projector.train()
        total = 0.0
        steps = 0
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = shard["head_contexts"][ids].to(device=device, dtype=torch.float32).flatten(start_dim=2)
            with torch.no_grad():
                target = o_proj(autoencoder(x))
            pred = projector(x)
            loss = rel_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        metrics = evaluate(autoencoder, projector, o_proj, paths, eval_refs, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": total / max(steps, 1), "eval": metrics})
        print(f"epoch={epoch} train_loss={total / max(steps,1):.6f} {json.dumps(metrics, sort_keys=True)}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"autoencoder": autoencoder.state_dict(), "projector": projector.state_dict()}, out / "compressed_oproj.pt")
    trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    report = {
        "capture_dir": args.capture_dir,
        "ae_state": args.ae_state,
        "model_name": args.model_name,
        "target_layer": args.target_layer,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "in_dim": in_dim,
        "bottleneck_dim": args.bottleneck_dim,
        "ae_kind": args.ae_kind,
        "ae_hidden_dim": args.ae_hidden_dim,
        "projection_kind": args.projection_kind,
        "projection_rank": args.projection_rank if args.projection_kind in {"lowrank", "pca_lowrank"} else None,
        "pca_ridge": args.pca_ridge if args.projection_kind == "pca_lowrank" else None,
        "trainable_projection_params": trainable_params,
        "frozen_autoencoder_params": sum(p.numel() for p in autoencoder.parameters()),
        "history": history,
    }
    with (out / "compressed_oproj_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
