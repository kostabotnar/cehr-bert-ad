"""Attention-based attribution for the fine-tuned CEHR-BERT classifier.

The classifier's prediction is driven by the ``[CLS]`` token at position 0 (the pooler
reads ``last_hidden_state[:, 0]``; see ``CehrBertForClassification.forward``). So "which
clinical concepts drove this prediction" maps to "how much signal flows into ``[CLS]`` from
each concept token". This module computes three per-token attribution scores from a single
example's attention stack, each a vector over sequence positions:

  raw      -- last-layer, head-averaged ``[CLS]`` attention row. Cheapest; a single-layer
              view that ignores how information mixes through depth.
  rollout  -- attention rollout (Abnar & Zuidema, 2020): multiply head-averaged,
              residual-corrected attention across all layers, then read the ``[CLS]`` row.
              Accounts for multi-hop information flow.
  gradxatt -- gradient-weighted attention rollout (Chefer et al., 2021): weight each layer's
              attention by the gradient of the target logit w.r.t. that attention, ReLU,
              then roll out. The most faithful of the three for "drivers of the prediction",
              because raw attention alone is a known-unreliable explanation
              (Jain & Wallace, 2019, "Attention is not Explanation").

All functions operate on detached CPU tensors and return a 1-D tensor of length ``seq_len``.
"""

from __future__ import annotations

import torch

CLS_INDEX = 0  # the collator prepends [CLS] at position 0 for non-packed sequences


def stack_example_attentions(attentions: tuple[torch.Tensor, ...], batch_index: int = 0) -> torch.Tensor:
    """Stack a HF ``output.attentions`` tuple into ``(num_layers, num_heads, seq, seq)``.

    ``attentions`` is a tuple of length ``num_layers``, each ``(batch, heads, seq, seq)``.
    Returns the slice for one example, detached on CPU.
    """
    return torch.stack([a[batch_index].detach().cpu() for a in attentions], dim=0)


def raw_cls_attention(attn_lhss: torch.Tensor, cls_index: int = CLS_INDEX) -> torch.Tensor:
    """Last-layer, head-averaged attention *from* ``[CLS]`` to every token -> ``(seq,)``."""
    last_layer = attn_lhss[-1].mean(dim=0)  # head-average -> (seq, seq)
    return last_layer[cls_index]


def attention_rollout(attn_lhss: torch.Tensor, cls_index: int = CLS_INDEX) -> torch.Tensor:
    """Attention rollout across all layers; returns the ``[CLS]`` row -> ``(seq,)``.

    Per Abnar & Zuidema (2020): add the identity to model residual connections, renormalize
    each row to a distribution, and multiply the per-layer matrices.
    """
    num_layers, _, seq, _ = attn_lhss.shape
    eye = torch.eye(seq, dtype=attn_lhss.dtype)
    rolled = eye
    for layer in range(num_layers):
        a = attn_lhss[layer].mean(dim=0) + eye  # head-average + residual
        a = a / a.sum(dim=-1, keepdim=True)
        rolled = a @ rolled
    return rolled[cls_index]


def grad_attention_rollout(
    attn_lhss: torch.Tensor, grad_lhss: torch.Tensor, cls_index: int = CLS_INDEX
) -> torch.Tensor:
    """Gradient-weighted attention rollout; returns the ``[CLS]`` row -> ``(seq,)``.

    Per Chefer et al. (2021): per layer, form ``relu(mean_heads(grad * attention))`` as a
    relevance map, add the identity residual, renormalize, and roll out across layers.
    ``grad_lhss`` is the gradient of the target logit w.r.t. each layer's attention,
    same shape as ``attn_lhss``.
    """
    num_layers, _, seq, _ = attn_lhss.shape
    eye = torch.eye(seq, dtype=attn_lhss.dtype)
    rolled = eye
    for layer in range(num_layers):
        cam = (grad_lhss[layer] * attn_lhss[layer]).clamp(min=0).mean(dim=0)  # (seq, seq)
        cam = cam + eye
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-12)
        rolled = cam @ rolled
    return rolled[cls_index]


def normalize_over_real_tokens(scores: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
    """Zero out non-kept positions and renormalize the rest to sum to 1.

    ``keep_mask`` is a boolean tensor (True = a real concept token to attribute). Returns a
    distribution over the kept positions so each patient contributes equal total mass to the
    cohort aggregation. Returns all-zeros if nothing is kept.
    """
    masked = torch.where(keep_mask, scores, torch.zeros_like(scores))
    total = masked.sum()
    if total <= 0:
        return torch.zeros_like(scores)
    return masked / total


METHODS = ("raw", "rollout", "gradxatt")
