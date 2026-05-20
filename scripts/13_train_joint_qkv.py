#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from attention_compression.attention_metrics import attention_kl, causal_attention, causal_logit_relative_mse, cosine_similarity_mean, relative_mse, topk_overlap
from attention_compression.joint_qkv import JointQKVBranches, init_branch_from_pca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Co-train low-rank Q/K/V branches under attention losses.")
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="allenai/OLMo-1B-0724-hf")
    parser.add_argument("--target-layer", type=int, default=8)
    parser.add_argument("--head-index", type=int, default=0)
    parser.add_argument("--q-rank", type=int, default=64)
    parser.add_argument("--k-rank", type=int, default=48)
    parser.add_argument("--v-rank", type=int, default=128)
    parser.add_argument("--train-windows-per-bin", type=int, default=32)
    parser.add_argument("--eval-windows-per-bin", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def group_rows_by_bin(paths: list[Path]) -> dict[str, list[tuple[int, int]]]:
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for path_i, path in enumerate(paths):
        shard = torch.load(path, map_location="cpu")
        for sample_i, row in enumerate(shard["rows"]):
            groups[row["rarity_bin"]].append((path_i, sample_i))
    return groups


def choose_train_eval(groups: dict[str, list[tuple[int, int]]], train_n: int, eval_n: int, seed: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    gen = torch.Generator().manual_seed(seed)
    train: list[tuple[int, int]] = []
    eval_: list[tuple[int, int]] = []
    for refs in groups.values():
        perm = torch.randperm(len(refs), generator=gen).tolist()
        train.extend(refs[i] for i in perm[:train_n])
        eval_.extend(refs[i] for i in perm[train_n : train_n + eval_n])
    return train, eval_


def refs_by_path(refs: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = defaultdict(list)
    for path_i, sample_i in refs:
        out[path_i].append(sample_i)
    return out


def fit_output_pca(
    *,
    paths: list[Path],
    train_refs: list[tuple[int, int]],
    projs: dict[str, torch.Tensor],
    device: str,
) -> dict[str, dict[str, torch.Tensor]]:
    stats = {
        name: {"n": 0, "sum": torch.zeros(128, device=device), "xtx": torch.zeros(128, 128, device=device)}
        for name in projs
    }
    for path_i, sample_ids in refs_by_path(train_refs).items():
        shard = torch.load(paths[path_i], map_location="cpu")
        x = shard["x_attn"][sample_ids].to(device=device, dtype=torch.float32)
        flat = x.reshape(-1, x.shape[-1])
        for name, w in projs.items():
            z = flat @ w
            stats[name]["sum"] += z.sum(0)
            stats[name]["xtx"] += z.T @ z
            stats[name]["n"] += z.shape[0]
    pca = {}
    for name, st in stats.items():
        mean = st["sum"] / st["n"]
        cov = (st["xtx"] - st["n"] * torch.outer(mean, mean)) / (st["n"] - 1)
        vals, vecs = torch.linalg.eigh(cov.cpu())
        order = torch.argsort(vals, descending=True)
        pca[name] = {"mean": mean, "basis": vecs[:, order].to(device=device, dtype=torch.float32)}
    return pca


def batch_refs(paths: list[Path], refs: list[tuple[int, int]], batch_size: int, *, shuffle: bool = False, seed: int = 0):
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


def rel_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    err = torch.mean((pred.float() - target.float()) ** 2)
    denom = torch.var(target.float()).clamp_min(1e-6)
    return err / denom


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM
    from transformers.models.olmo.modeling_olmo import apply_rotary_pos_emb

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(Path(args.capture_dir).glob("*.pt"))
    groups = group_rows_by_bin(paths)
    train_refs, eval_refs = choose_train_eval(groups, args.train_windows_per_bin, args.eval_windows_per_bin, args.seed)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, trust_remote_code=True)
    attn = teacher.model.layers[args.target_layer].self_attn
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

    def rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids.expand(q.shape[0], -1)
        cos, sin = teacher.model.rotary_emb(q, pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(1), k.unsqueeze(1), cos, sin, unsqueeze_dim=1)
        return qr.squeeze(1), kr.squeeze(1)

    pca = fit_output_pca(paths=paths, train_refs=train_refs, projs=projs, device=device)
    model = JointQKVBranches(2048, head_dim, args.q_rank, args.k_rank, args.v_rank).to(device)
    init_branch_from_pca(model.q, projection=projs["q"], mean=pca["q"]["mean"], basis=pca["q"]["basis"])
    init_branch_from_pca(model.k, projection=projs["k"], mean=pca["k"]["mean"], basis=pca["k"]["basis"])
    init_branch_from_pca(model.v, projection=projs["v"], mean=pca["v"]["mean"], basis=pca["v"]["basis"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def run_eval() -> dict[str, float]:
        acc = defaultdict(float)
        count = 0
        model.eval()
        with torch.no_grad():
            for shard, ids in batch_refs(paths, eval_refs, args.batch_size):
                x = shard["x_attn"][ids].to(device=device, dtype=torch.float32)
                h = shard["head_context"][ids].to(device=device, dtype=torch.float32)
                q = x @ projs["q"]
                k = x @ projs["k"]
                v = x @ projs["v"]
                qh, kh, vh = model(x)
                qr, kr = rope(q, k)
                qhr, khr = rope(qh, kh)
                logits, probs, _ = causal_attention(qr, kr, v)
                logits_h, probs_h, head_h = causal_attention(qhr, khr, vh)
                b = x.shape[0]
                metrics = {
                    "q_relative_mse": relative_mse(qh, q),
                    "k_relative_mse": relative_mse(kh, k),
                    "v_relative_mse": relative_mse(vh, v),
                    "logit_relative_mse": causal_logit_relative_mse(logits_h, logits),
                    "attention_kl": attention_kl(probs, probs_h),
                    "attention_top5_overlap": topk_overlap(probs, probs_h, 5),
                    "head_context_relative_mse": relative_mse(head_h, h),
                    "head_context_cosine": cosine_similarity_mean(head_h, h),
                }
                for key, val in metrics.items():
                    acc[key] += val * b
                count += b
        return {key: val / count for key, val in acc.items()}

    history = [{"epoch": 0, "eval": run_eval()}]
    print("epoch=0", json.dumps(history[-1]["eval"], sort_keys=True), flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        for shard, ids in batch_refs(paths, train_refs, args.batch_size, shuffle=True, seed=args.seed + epoch):
            x = shard["x_attn"][ids].to(device=device, dtype=torch.float32)
            h = shard["head_context"][ids].to(device=device, dtype=torch.float32)
            q = x @ projs["q"]
            k = x @ projs["k"]
            v = x @ projs["v"]
            with torch.no_grad():
                qr, kr = rope(q, k)
                logits, probs, _ = causal_attention(qr, kr, v)
            qh, kh, vh = model(x)
            qhr, khr = rope(qh, kh)
            logits_h, probs_h, head_h = causal_attention(qhr, khr, vh)
            loss = (
                0.05 * rel_loss(qh, q)
                + 0.05 * rel_loss(kh, k)
                + 0.10 * rel_loss(vh, v)
                + 0.50 * rel_loss(logits_h[..., causal_mask], logits[..., causal_mask])
                + 0.10 * torch.nn.functional.kl_div(probs_h.clamp_min(1e-12).log(), probs, reduction="batchmean")
                + 1.00 * rel_loss(head_h, h)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
        eval_metrics = run_eval()
        history.append({"epoch": epoch, "train_loss": total_loss / max(steps, 1), "eval": eval_metrics})
        print(f"epoch={epoch} train_loss={total_loss / max(steps, 1):.6f} {json.dumps(eval_metrics, sort_keys=True)}", flush=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "joint_qkv_model.pt")
    report = {
        "capture_dir": args.capture_dir,
        "target_layer": args.target_layer,
        "head_index": args.head_index,
        "q_rank": args.q_rank,
        "k_rank": args.k_rank,
        "v_rank": args.v_rank,
        "train_windows": len(train_refs),
        "eval_windows": len(eval_refs),
        "epochs": args.epochs,
        "lr": args.lr,
        "history": history,
    }
    with (out / "joint_qkv_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
