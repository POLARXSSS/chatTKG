from __future__ import annotations

import torch
from torch import nn


class ConservativeReranker(nn.Module):
    """A bounded query-gated residual over the frozen TLogic score."""

    def __init__(
        self,
        candidate_features: int,
        query_features: int,
        hidden_size: int = 32,
        max_residual: float = 0.5,
    ) -> None:
        super().__init__()
        if candidate_features < 1 or query_features < 1:
            raise ValueError("feature dimensions must be positive")
        if hidden_size < 1 or max_residual <= 0:
            raise ValueError("hidden_size and max_residual must be positive")
        self.candidate_net = nn.Sequential(
            nn.Linear(candidate_features + query_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(query_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.max_residual = float(max_residual)

        # Start close to the frozen base model. The model must earn permission
        # to modify scores through the ranking objective.
        nn.init.zeros_(self.candidate_net[-1].weight)
        nn.init.zeros_(self.candidate_net[-1].bias)
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, -2.0)

    def forward(
        self,
        candidate_features: torch.Tensor,
        query_features: torch.Tensor,
        base_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if candidate_features.ndim != 3 or query_features.ndim != 2:
            raise ValueError("expected [batch,candidates,features] and [batch,features]")
        expanded_query = query_features[:, None, :].expand(
            -1, candidate_features.shape[1], -1
        )
        raw_delta = self.candidate_net(
            torch.cat((candidate_features, expanded_query), dim=-1)
        ).squeeze(-1)
        gate = torch.sigmoid(self.gate_net(query_features)).squeeze(-1)
        delta = self.max_residual * torch.tanh(raw_delta)
        final = base_score + gate[:, None] * delta
        return final, gate, delta


def pairwise_ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    hard_negatives: int,
) -> tuple[torch.Tensor, int]:
    positive = mask & (labels > 0.5)
    negative = mask & ~positive
    usable = positive.any(dim=1) & negative.any(dim=1)
    if not bool(usable.any()):
        return scores.sum() * 0.0, 0

    best_positive = scores.masked_fill(~positive, float("-inf")).max(dim=1).values
    negative_scores = scores.masked_fill(~negative, float("-inf"))
    width = negative_scores.shape[1]
    count = width if hard_negatives <= 0 else min(width, hard_negatives)
    hardest = torch.topk(negative_scores, k=count, dim=1).values
    valid_negative = torch.isfinite(hardest)
    losses = torch.nn.functional.softplus(
        hardest - best_positive[:, None] + float(margin)
    )
    losses = torch.where(valid_negative, losses, torch.zeros_like(losses))
    per_query = losses.sum(dim=1) / valid_negative.sum(dim=1).clamp_min(1)
    return per_query[usable].mean(), int(usable.sum().item())


def preservation_loss(
    final_scores: torch.Tensor,
    base_scores: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    base_candidate_mask: torch.Tensor,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, int]:
    available = candidate_mask & base_candidate_mask
    base_top_index = base_scores.masked_fill(~available, float("-inf")).argmax(dim=1)
    row = torch.arange(base_scores.shape[0], device=base_scores.device)
    has_base = available.any(dim=1)
    base_top_correct = has_base & (labels[row, base_top_index] > 0.5)
    if not bool(base_top_correct.any()):
        return final_scores.sum() * 0.0, 0

    safe_temperature = max(float(temperature), 1e-4)
    base_logits = (base_scores / safe_temperature).masked_fill(~available, -1e9)
    final_logits = (final_scores / safe_temperature).masked_fill(~available, -1e9)
    target = torch.softmax(base_logits.detach(), dim=1)
    log_prediction = torch.log_softmax(final_logits, dim=1)
    per_query = torch.nn.functional.kl_div(
        log_prediction, target, reduction="none"
    ).sum(dim=1)
    return per_query[base_top_correct].mean(), int(base_top_correct.sum().item())
