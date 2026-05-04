from __future__ import annotations

import torch


def relative_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    err = torch.sum((pred.float() - target.float()) ** 2)
    denom = torch.sum((target.float() - target.float().mean()) ** 2).clamp_min(1e-12)
    return float((err / denom).item())


def cosine_similarity_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_f = pred.float().reshape(-1, pred.shape[-1])
    target_f = target.float().reshape(-1, target.shape[-1])
    return float(torch.nn.functional.cosine_similarity(pred_f, target_f, dim=-1).mean().item())


def attention_kl(teacher: torch.Tensor, student: torch.Tensor) -> float:
    teacher_f = teacher.float().clamp_min(1e-12)
    student_f = student.float().clamp_min(1e-12)
    kl = teacher_f * (teacher_f.log() - student_f.log())
    return float(kl.sum(dim=-1).mean().item())


def topk_overlap(teacher: torch.Tensor, student: torch.Tensor, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, teacher.shape[-1])
    teacher_idx = torch.topk(teacher, k=k, dim=-1).indices
    student_idx = torch.topk(student, k=k, dim=-1).indices
    matches = teacher_idx.unsqueeze(-1) == student_idx.unsqueeze(-2)
    overlap = matches.any(dim=-1).float().mean()
    return float(overlap.item())


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute single-head causal attention for tensors `[B, T, D]`."""
    logits = attention_logits(q, k)
    t = logits.shape[-1]
    mask = torch.triu(torch.ones((t, t), dtype=torch.bool, device=logits.device), diagonal=1)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(logits, dim=-1)
    context = torch.matmul(probs.to(v.dtype), v)
    return logits, probs, context


def attention_logits(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    return torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale


def causal_logit_relative_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative MSE over valid causal positions only."""
    t = target.shape[-1]
    mask = torch.tril(torch.ones((t, t), dtype=torch.bool, device=target.device))
    err = pred.float() - target.float()
    err = err[..., mask]
    tgt = target.float()[..., mask]
    denom = torch.sum((tgt - tgt.mean()) ** 2).clamp_min(1e-12)
    return float((torch.sum(err**2) / denom).item())
