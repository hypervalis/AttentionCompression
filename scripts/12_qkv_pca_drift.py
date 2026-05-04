#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from attention_compression.attention_metrics import (
    attention_kl,
    attention_logits,
    causal_attention,
    causal_logit_relative_mse,
    cosine_similarity_mean,
    relative_mse,
    topk_overlap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PCA-reconstructed Q/K/V through attention drift.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--head-index", type=int, default=0)
    parser.add_argument("--configs", default="64,64,128;96,64,128;64,48,128;96,64,96;128,96,96;64,64,96")
    parser.add_argument("--train-windows-per-bin", type=int, default=64)
    parser.add_argument("--eval-windows-per-bin", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def parse_configs(text: str) -> list[tuple[int, int, int]]:
    configs = []
    for part in text.split(";"):
        if not part.strip():
            continue
        q, k, v = [int(x) for x in part.split(",")]
        configs.append((q, k, v))
    return configs


def group_rows_by_bin(paths: list[Path]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        for sample_i, row in enumerate(shard["rows"]):
            groups[row["rarity_bin"]].append((path_i, sample_i))
    return groups


def choose_train_eval(groups: dict[str, list[tuple[int, int]]], train_n: int, eval_n: int, seed: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    train: set[tuple[int, int]] = set()
    eval_: set[tuple[int, int]] = set()
    for bin_name, refs in groups.items():
        idx = np.arange(len(refs))
        rng.shuffle(idx)
        for i in idx[:train_n]:
            train.add(refs[int(i)])
        for i in idx[train_n : train_n + eval_n]:
            eval_.add(refs[int(i)])
    return train, eval_


def fit_pca_from_qkv_samples(
    *,
    paths: list[Path],
    train_refs: set[tuple[int, int]],
    projs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    stats = {
        name: {"n": 0, "sum": torch.zeros(128, device=device), "xtx": torch.zeros(128, 128, device=device)}
        for name in projs
    }
    refs_by_path: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in train_refs:
        refs_by_path[path_i].append(sample_i)
    for path_i, sample_ids in refs_by_path.items():
        shard = torch.load(paths[path_i], map_location="cpu")
        x = shard["x_attn"][sample_ids].to(device=device, dtype=torch.float32)
        x_flat = x.reshape(-1, x.shape[-1])
        for name, w in projs.items():
            z = x_flat @ w
            stats[name]["sum"] += z.sum(dim=0)
            stats[name]["xtx"] += z.T @ z
            stats[name]["n"] += int(z.shape[0])
        print(f"fit shard={path_i} samples={len(sample_ids)}", flush=True)

    bases: dict[str, dict[str, torch.Tensor]] = {}
    for name, st in stats.items():
        mean = st["sum"] / st["n"]
        cov = (st["xtx"] - st["n"] * torch.outer(mean, mean)) / (st["n"] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov.cpu())
        order = torch.argsort(eigvals, descending=True)
        bases[name] = {"mean": mean, "basis": eigvecs[:, order].to(device), "eigvals": eigvals[order]}
    return bases


def reconstruct(z: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, rank: int) -> torch.Tensor:
    b = basis[:, :rank]
    centered = z - mean
    return centered @ b @ b.T + mean


def add_metric(acc: dict[str, float], name: str, value: float, weight: int = 1) -> None:
    acc[f"{name}_sum"] = acc.get(f"{name}_sum", 0.0) + value * weight
    acc[f"{name}_weight"] = acc.get(f"{name}_weight", 0.0) + weight


def finalize(acc: dict[str, float]) -> dict[str, float]:
    out = {}
    for key, value in acc.items():
        if not key.endswith("_sum"):
            continue
        name = key[:-4]
        out[name] = value / max(acc.get(f"{name}_weight", 0.0), 1e-12)
    return out


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM
    from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    configs = parse_configs(args.configs)
    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(args.capture_dir)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, trust_remote_code=True)
    attn = model.model.layers[args.target_layer].self_attn
    head_dim = int(attn.head_dim)
    hs = args.head_index * head_dim
    he = hs + head_dim
    projs = {
        "q": attn.q_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous(),
        "k": attn.k_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous(),
        "v": attn.v_proj.weight[hs:he].detach().to(device=device, dtype=torch.float32).T.contiguous(),
    }
    seq_len = torch.load(paths[0], map_location="cpu")["x_attn"].shape[1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids.expand(q.shape[0], -1)
        cos, sin = model.model.rotary_emb(q, pos)
        q_rot, k_rot = apply_rotary_pos_emb(q.unsqueeze(1), k.unsqueeze(1), cos, sin, unsqueeze_dim=1)
        return q_rot.squeeze(1), k_rot.squeeze(1)

    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed)
    bases = fit_pca_from_qkv_samples(paths=paths, train_refs=train_refs, projs=projs, device=device)

    eval_by_path: dict[int, list[int]] = defaultdict(list)
    bins_by_ref = {}
    for path_i, sample_i in eval_refs:
        eval_by_path[path_i].append(sample_i)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        for sample_i, row in enumerate(shard["rows"]):
            bins_by_ref[(path_i, sample_i)] = row["rarity_bin"]

    results: dict[str, dict[str, dict[str, float]]] = {}
    for q_rank, k_rank, v_rank in configs:
        label = f"q{q_rank}_k{k_rank}_v{v_rank}"
        global_acc: dict[str, float] = {}
        by_bin: dict[str, dict[str, float]] = defaultdict(dict)
        for path_i, sample_ids in eval_by_path.items():
            shard = torch.load(paths[path_i], map_location="cpu")
            for start in range(0, len(sample_ids), args.batch_size):
                ids = sample_ids[start : start + args.batch_size]
                x = shard["x_attn"][ids].to(device=device, dtype=torch.float32)
                teacher_head = shard["head_context"][ids].to(device=device, dtype=torch.float32)
                q = x @ projs["q"]
                k = x @ projs["k"]
                v = x @ projs["v"]
                qh = reconstruct(q, bases["q"]["mean"], bases["q"]["basis"], q_rank)
                kh = reconstruct(k, bases["k"]["mean"], bases["k"]["basis"], k_rank)
                vh = reconstruct(v, bases["v"]["mean"], bases["v"]["basis"], v_rank)
                q_rot, k_rot = apply_rope(q, k)
                qh_rot, kh_rot = apply_rope(qh, kh)
                raw_logits = attention_logits(q_rot, k_rot)
                raw_logits_h = attention_logits(qh_rot, kh_rot)
                _, probs, _ = causal_attention(q_rot, k_rot, v)
                _, probs_h, head_h = causal_attention(qh_rot, kh_rot, vh)
                metrics = {
                    "q_relative_mse": relative_mse(qh, q),
                    "k_relative_mse": relative_mse(kh, k),
                    "v_relative_mse": relative_mse(vh, v),
                    "logit_relative_mse": causal_logit_relative_mse(raw_logits_h, raw_logits),
                    "attention_kl": attention_kl(probs, probs_h),
                    "attention_top1_overlap": topk_overlap(probs, probs_h, 1),
                    "attention_top5_overlap": topk_overlap(probs, probs_h, 5),
                    "attention_top10_overlap": topk_overlap(probs, probs_h, 10),
                    "head_context_relative_mse": relative_mse(head_h, teacher_head),
                    "head_context_cosine": cosine_similarity_mean(head_h, teacher_head),
                }
                for name, value in metrics.items():
                    add_metric(global_acc, name, value, weight=len(ids))
                batch_bins = [bins_by_ref[(path_i, sample_i)] for sample_i in ids]
                for bin_name in sorted(set(batch_bins)):
                    count = batch_bins.count(bin_name)
                    for name, value in metrics.items():
                        add_metric(by_bin[bin_name], name, value, weight=count)
        results[label] = {"global": finalize(global_acc), "by_bin": {k: finalize(v) for k, v in by_bin.items()}}
        print(label, json.dumps(results[label]["global"], sort_keys=True), flush=True)

    report = {
        "capture_dir": args.capture_dir,
        "target_layer": args.target_layer,
        "head_index": args.head_index,
        "train_windows_per_bin": args.train_windows_per_bin,
        "eval_windows_per_bin": args.eval_windows_per_bin,
        "configs": [f"q{q}_k{k}_v{v}" for q, k, v in configs],
        "results": results,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "qkv_pca_drift.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
