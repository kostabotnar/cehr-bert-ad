#!/usr/bin/env python
"""Step 02: Attention-weight analysis -- which clinical concepts drive predictions.

Runs the fine-tuned CEHR-BERT classifier over the held-out test cohort with attention
output enabled, and attributes each prediction back to the input concept tokens using three
methods (see ``pipeline.attention``):

  raw       last-layer head-averaged [CLS] attention
  rollout   attention rollout across all layers (Abnar & Zuidema, 2020)
  gradxatt  gradient-weighted attention rollout (Chefer et al., 2021)

The attribution target is the positive-class logit, so a high score means the concept pushes
the prediction *toward AD*. Per-patient token scores are normalized to a distribution over
real concept tokens (specials/visit markers excluded), then aggregated per concept and split
by the model's predicted class (prob >= 0.5). Outputs, per run label, under
``build/feature_importance/<label>/attention/``:

  concept_attribution_<group>.csv   ranked concepts per method (group = pred_pos / pred_neg)
  method_rank_correlation.json      Spearman rho between the three methods' rankings
  ../figures/attention_top_concepts.png  top concepts driving positive predictions (3 methods)

Two codebase-specific requirements are enforced here: the model is loaded with
``attn_implementation="eager"`` (flash attention returns no weights), and sample packing is
left off so each row is one patient with [CLS] at position 0.

Patients are processed ``--batch-size`` at a time (default 32): one padded forward and one
backward per batch, then each example's attention and gradient tensors are sliced to its real
length before attribution, so scores match a one-at-a-time run. To limit padding, patients are
sorted by length within chunks of 16 batches; the cohort sums do not depend on that order
beyond floating-point rounding.

Usage
  python feature_importance/02_attention_analysis.py
  python feature_importance/02_attention_analysis.py --batch-size 64
  python feature_importance/02_attention_analysis.py --max-patients 300 --top-k 20
  python feature_importance/02_attention_analysis.py --device cpu --window-days 365 --ctx 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import attention as attn  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.config_validation import load_yaml  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# Special / structural tokens excluded from concept attribution by default. The five reserved
# tokens never carry clinical meaning; VS/VE (and the bracketed variants) are visit boundary
# markers. Artificial time-interval tokens are kept -- a gap between visits can be informative.
EXCLUDED_TOKENS = frozenset({"[PAD]", "[CLS]", "[MASK]", "[UNUSED]", "[OOV]", "VS", "VE", "[VS]", "[VE]"})

DATASET_FEATURE_KEYS = (
    "input_ids",
    "ages",
    "dates",
    "visit_concept_orders",
    "concept_values",
    "concept_value_masks",
    "visit_segments",
    "age_at_index",
    "classifier_label",
    "person_id",
)


def _find_prepared_dataset(prepared_dir: Path):
    """Load the predict-prepared HF dataset's 'test' split from its hashed subfolder."""
    from datasets import load_from_disk

    # The runner writes a DatasetDict into a hashed subfolder (e.g. test_<hash>/).
    candidates = [prepared_dir] + sorted(p for p in prepared_dir.glob("*") if p.is_dir())
    for cand in candidates:
        try:
            ds = load_from_disk(str(cand))
        except (FileNotFoundError, ValueError):
            continue
        if hasattr(ds, "keys"):
            ds = ds["test"] if "test" in ds else ds[next(iter(ds.keys()))]
        return ds
    raise FileNotFoundError(f"No loadable HF dataset under {prepared_dir}")


def _build_collator(tokenizer, max_length: int):
    from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator

    collator = CehrBertDataCollator(tokenizer, max_length=max_length, is_pretraining=False)
    collator.sample_packing = False  # one patient per row; [CLS] prepended at position 0
    return collator


def _example_for_collator(record: dict) -> dict:
    """Keep only the fields the collator reads, as a single-example list input."""
    return {k: record[k] for k in DATASET_FEATURE_KEYS if k in record}


# Patients are read SORT_CHUNK_BATCHES batches at a time and sorted by length inside the chunk.
SORT_CHUNK_BATCHES = 16

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "ages",
    "dates",
    "visit_concept_orders",
    "concept_values",
    "concept_value_masks",
    "visit_segments",
    "age_at_index",
)


def _truncate_with_concepts(collator, record: dict) -> dict | None:
    """Apply the collator's truncation; return ``None`` when no concept tokens remain.

    Such an example would collate to a width-1 ``[CLS]``-only row that carries no clinical
    signal to attribute, and its ragged time-series features can break the embedding concat
    when collated alongside others, so it is filtered out before batching. The collator
    re-applies truncation to the returned record as a no-op.
    """
    truncated = collator.generate_start_end_index(_example_for_collator(record))
    if len(truncated["input_ids"]) == 0:
        return None
    return truncated


def _attribute_batch(model, batch, device) -> list[tuple[dict[str, torch.Tensor], float, int]]:
    """Forward+backward one padded batch; return ``(scores, prob, real_length)`` per example.

    Batching relies on examples not interacting in this model in eval mode: eager attention is
    masked per row, LayerNorm is per token, dropout is off, the pooler reads position 0 of each
    row, and the age BatchNorm uses running statistics. So the backward of the summed logits
    gives each example's attention exactly the gradient of its own logit.
    """
    inputs = {k: v.to(device) for k, v in batch.items() if k in MODEL_INPUT_KEYS}
    # PositionalEncodingLayer subtracts each row's minimum positive visit_concept_order. The
    # collator pads that field with max_length - 1, which becomes the minimum (shifting every
    # real position) when a row's real orders all exceed it, as in long tail-truncated
    # histories. Orders <= 0 are excluded from the minimum, so pad with 0 instead.
    inputs["visit_concept_orders"] = torch.where(
        inputs["attention_mask"] > 0, inputs["visit_concept_orders"], torch.zeros_like(inputs["visit_concept_orders"])
    )
    # CehrBertForClassification._apply_age_norm puts age_batch_norm back in train mode after any
    # batch-size-1 forward; a later multi-example batch would then normalize age with batch
    # statistics (mixing examples). Re-enter eval mode before every forward.
    model.eval()
    model.zero_grad(set_to_none=True)
    out = model(**inputs, output_attentions=True)
    if out.attentions is None:
        # Should not happen with attn_implementation="eager", but sdpa/flash return no
        # weights; fail loud rather than crash with an opaque NoneType error downstream.
        raise RuntimeError("model returned no attention weights; load with attn_implementation='eager'")
    # Retain grad on the attention probs so backward populates .grad on each layer.
    for a in out.attentions:
        a.retain_grad()
    logits = out.logits[:, 0]
    probs = torch.sigmoid(logits).detach().cpu().tolist()
    logits.sum().backward()

    attentions = out.attentions
    # A layer's attention probs may not receive a gradient (a.grad is None) if that layer
    # does not contribute to the logit; substitute zeros so the stack stays well-formed and
    # that layer simply gets zero relevance in grad-weighted rollout.
    grads = tuple((a.grad if a.grad is not None else torch.zeros_like(a)) for a in attentions)
    # The collator pads on the right and prepends [CLS] at column 0, so each row's real tokens
    # are its leading attention_mask.sum() columns. Slicing to that length gives the same
    # (seq, seq) matrices an unbatched forward produces.
    lengths = batch["attention_mask"].sum(dim=1).to(torch.long).tolist()

    # Rollouts run on the model's device: copying the padded (batch, heads, seq, seq) stacks to
    # the CPU and rolling out there costs far more than the forward and backward. Only the
    # per-token score vectors are moved, in one transfer per batch.
    flat = []
    for i, length in enumerate(lengths):
        attn_lhss = attn.slice_example_attentions(attentions, i, length)
        grad_lhss = attn.slice_example_attentions(grads, i, length)
        scores = {
            "raw": attn.raw_cls_attention(attn_lhss),
            "rollout": attn.attention_rollout(attn_lhss),
            "gradxatt": attn.grad_attention_rollout(attn_lhss, grad_lhss),
        }
        flat += [scores[m] for m in attn.METHODS]
    k = len(attn.METHODS)
    split = torch.cat(flat).cpu().split([n for n in lengths for _ in range(k)])
    return [(dict(zip(attn.METHODS, split[k * i : k * (i + 1)])), probs[i], length) for i, length in enumerate(lengths)]


class _TokenLookup:
    """Cached token-id to concept-id lookup; the tokenizer call per token is slow."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._cache: dict[int, str] = {}

    def __call__(self, ids: list[int]) -> list[str]:
        cache = self._cache
        out = []
        for i in ids:
            tok = cache.get(i)
            if tok is None:
                tok = cache[i] = self._tokenizer.convert_id_to_token(i)
            out.append(tok)
        return out


def _accumulate(scores, prob, ids, lookup, acc, present, group_counts):
    """Fold one patient's attribution into the cohort accumulators (in place).

    ``ids`` are the patient's real (unpadded) token ids; ``lookup`` maps them to concept ids.
    """
    group = "pred_pos" if prob >= 0.5 else "pred_neg"
    group_counts[group] += 1

    concept_ids = lookup(ids)
    keep = torch.tensor([c not in EXCLUDED_TOKENS for c in concept_ids], dtype=torch.bool)
    if not bool(keep.any()):
        return False

    seen_this_patient = set()
    for method, vec in scores.items():
        dist = attn.normalize_over_real_tokens(vec, keep)
        # Positions in increasing order, as a full scan visits them, so the float sums match.
        positive = torch.nonzero(dist > 0, as_tuple=True)[0].tolist()
        weights = dist[positive].tolist()
        acc_method = acc[group][method]
        for pos, w in zip(positive, weights):
            cid = concept_ids[pos]
            acc_method[cid] += w
            if method == "raw" and cid not in seen_this_patient:
                present[group][cid] += 1
                seen_this_patient.add(cid)
    return True


def _run_attribution(model, collator, dataset, n_run, batch_size, device, tokenizer):
    """Attribute the first ``n_run`` patients in batches; return the cohort accumulators.

    Returns ``(acc, present, group_counts, n_attributed, n_skipped)``. Patients with no concept
    tokens are counted as skipped and never enter a batch.
    """
    acc = {g: {m: defaultdict(float) for m in attn.METHODS} for g in ("pred_pos", "pred_neg")}
    present = {g: defaultdict(int) for g in ("pred_pos", "pred_neg")}
    group_counts = defaultdict(int)
    n_attributed = n_skipped = n_reported = 0
    lookup = _TokenLookup(tokenizer)
    t_start = time.perf_counter()
    chunk_size = batch_size * SORT_CHUNK_BATCHES

    for start in range(0, n_run, chunk_size):
        chunk = []
        for idx in range(start, min(start + chunk_size, n_run)):
            record = _truncate_with_concepts(collator, dataset[idx])
            if record is None:
                n_skipped += 1
            else:
                chunk.append(record)
        if batch_size > 1:
            # Grouping similar lengths keeps padding, and so forward/backward cost, small. Batch
            # size 1 keeps dataset order so its float accumulation order matches the original.
            chunk.sort(key=lambda r: len(r["input_ids"]))
        for b in range(0, len(chunk), batch_size):
            batch = collator(chunk[b : b + batch_size])
            input_ids = batch["input_ids"].tolist()
            for row, (scores, prob, length) in zip(input_ids, _attribute_batch(model, batch, device)):
                if _accumulate(scores, prob, row[:length], lookup, acc, present, group_counts):
                    n_attributed += 1
                else:
                    n_skipped += 1
            n_reported = _report_progress(n_attributed, n_skipped, n_reported, n_run, t_start)
        n_reported = _report_progress(n_attributed, n_skipped, n_reported, n_run, t_start)
    return acc, present, group_counts, n_attributed, n_skipped


def _report_progress(n_attributed, n_skipped, n_reported, n_run, t_start) -> int:
    """Print progress when the processed count crosses a multiple of 100 or reaches the end."""
    n_done = n_attributed + n_skipped
    if n_done // 100 > n_reported // 100 or (n_done == n_run and n_reported < n_run):
        rate = 60.0 * n_done / max(time.perf_counter() - t_start, 1e-9)
        print(
            f"  processed {n_done}/{n_run} (attributed={n_attributed}, skipped={n_skipped}, "
            f"{rate:.0f} patients/min)",
            flush=True,
        )
        return n_done
    return n_reported


def _concept_name(tokenizer, concept_id: str) -> str:
    mapping = getattr(tokenizer, "_concept_name_mapping", {}) or {}
    return mapping.get(concept_id, concept_id)


def _write_group_csv(out_dir, group, acc_group, present_group, n_group, tokenizer, top_k):
    """Write a ranked per-method concept table for one predicted-class group; return top lists."""
    import pandas as pd

    concept_ids = sorted({c for m in acc_group for c in acc_group[m]})
    rows = []
    for cid in concept_ids:
        row = {
            "concept_id": cid,
            "concept_name": _concept_name(tokenizer, cid),
            "n_patients_present": present_group.get(cid, 0),
            "pct_patients_present": round(100.0 * present_group.get(cid, 0) / max(n_group, 1), 2),
        }
        for method in attn.METHODS:
            # Mean attribution mass per patient in this group (0 where the concept is absent).
            row[f"{method}_mean"] = acc_group[method].get(cid, 0.0) / max(n_group, 1)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("gradxatt_mean", ascending=False)
    path = Path(out_dir) / f"concept_attribution_{group}.csv"
    df.to_csv(path, index=False)

    tops = {m: df.sort_values(f"{m}_mean", ascending=False).head(top_k) for m in attn.METHODS}
    return path, df, tops


def _rank_correlation(df) -> dict:
    """Spearman rho between the three methods' per-concept mean scores."""
    from scipy.stats import spearmanr

    out = {}
    for i, a in enumerate(attn.METHODS):
        for b in attn.METHODS[i + 1 :]:
            rho, p = spearmanr(df[f"{a}_mean"], df[f"{b}_mean"])
            out[f"{a}_vs_{b}"] = {"spearman_rho": round(float(rho), 4), "p_value": float(p)}
    return out


def _plot_top_concepts(tops, n_pos, fig_path, top_k):
    """3-panel horizontal bar chart: top concepts driving positive predictions per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 0.45 * top_k + 1.5))
    titles = {"raw": "Raw [CLS] attention", "rollout": "Attention rollout", "gradxatt": "Gradient x attention"}
    colors = {"raw": "#4C72B0", "rollout": "#55A868", "gradxatt": "#C44E52"}
    for ax, method in zip(axes, attn.METHODS):
        d = tops[method].iloc[::-1]  # largest at top
        labels = [f"{n[:38]} ({c})" for n, c in zip(d["concept_name"], d["concept_id"])]
        ax.barh(range(len(d)), d[f"{method}_mean"], color=colors[method])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(titles[method], fontsize=11)
        ax.set_xlabel("mean attribution per patient")
    fig.suptitle(
        f"Top {top_k} concepts driving positive AD predictions " f"(predicted-positive patients, n={n_pos})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attention-weight analysis of CEHR-BERT predictions")
    parser.add_argument("--finetune-config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument("--model-dir", default=str(paths.FINETUNE_RESULTS_DIR))
    parser.add_argument("--tokenizer-dir", default=str(paths.PRETRAIN_RESULTS_DIR))
    parser.add_argument("--prepared-dir", default=str(paths.PREDICT_PREPARED_DIR))
    parser.add_argument("--window-days", type=int, default=paths.DEFAULT_WINDOW_DAYS)
    parser.add_argument("--ctx", type=int, default=paths.DEFAULT_CTX)
    parser.add_argument(
        "--max-patients", type=int, default=None, help="Cap the number of test patients processed (default: all)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Patients per forward/backward pass (1 = one at a time)"
    )
    parser.add_argument("--top-k", type=int, default=20, help="Concepts shown per method in the figure")
    parser.add_argument("--device", default=None, help="torch device (default: cuda when available, otherwise cpu)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    label = paths.run_label(args.window_days, args.ctx)
    output_dir = Path(args.output_dir) if args.output_dir else paths.feature_importance_dir(label) / "attention"
    fig_dir = paths.feature_importance_figures_dir(label)
    report_dir = (
        Path(args.report_dir)
        if args.report_dir
        else paths.feature_importance_report_dir("02_attention_analysis", label)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = StepReport("02_attention_analysis", title="Step 02 - Attention-weight analysis")
    report.add_metric("run_label", label)
    report.add_metric("model_dir", args.model_dir)
    report.add_metric("device", args.device)
    report.add_metric("batch_size", args.batch_size)

    # --- preflight --------------------------------------------------------
    model_ok = report.add_check(
        "fine-tuned model exists", Path(args.model_dir, "config.json").is_file(), detail=args.model_dir
    )
    tok_ok = report.add_check(
        "tokenizer exists", Path(args.tokenizer_dir, "tokenizer.json").is_file(), detail=args.tokenizer_dir
    )
    prep_ok = report.add_check("prepared test data exists", Path(args.prepared_dir).is_dir(), detail=args.prepared_dir)
    if not (model_ok and tok_ok and prep_ok):
        report.note("Preflight failed. Run cohort step 01 and cehrbert_model steps 09 and 10 first.")
        return _finalize(report, report_dir)

    max_length = int(load_yaml(Path(args.finetune_config)).get("max_position_embeddings", 512))
    report.add_metric("max_length", max_length)

    # --- load model + tokenizer (eager attention is mandatory) ------------
    from cehrbert.models.hf_models.hf_cehrbert import CehrBertForClassification
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer

    tokenizer = CehrBertTokenizer.from_pretrained(args.tokenizer_dir)
    model = CehrBertForClassification.from_pretrained(
        args.model_dir, attn_implementation="eager", output_attentions=True
    )
    model.to(args.device).eval()
    attn_impl = getattr(model.config, "_attn_implementation", "eager")
    if not report.add_check(
        "eager attention active (flash/sdpa return no weights)", attn_impl == "eager", detail=attn_impl
    ):
        report.note("Attention weights are only emitted by the eager implementation.")
        return _finalize(report, report_dir)

    dataset = _find_prepared_dataset(Path(args.prepared_dir))
    n_total = len(dataset)
    n_run = min(n_total, args.max_patients) if args.max_patients else n_total
    report.add_metric("test_patients_total", n_total)
    report.add_metric("test_patients_processed", n_run)

    # age_at_index is a required positional argument of the classifier's forward; without it
    # every forward would fail with an opaque TypeError. Verify it is present up front.
    if not report.add_check(
        "dataset has age_at_index (required by classifier forward)",
        n_total > 0 and "age_at_index" in dataset[0],
        detail=f"columns={list(dataset[0].keys())}" if n_total > 0 else "empty dataset",
    ):
        return _finalize(report, report_dir)

    collator = _build_collator(tokenizer, max_length)

    # --- accumulate per-concept attribution by predicted class ------------
    acc, present, group_counts, n_attributed, n_skipped = _run_attribution(
        model, collator, dataset, n_run, args.batch_size, args.device, tokenizer
    )

    report.add_metric("patients_attributed", n_attributed)
    report.add_metric("patients_skipped_no_concepts", n_skipped)
    report.add_metric("pred_positive", group_counts["pred_pos"])
    report.add_metric("pred_negative", group_counts["pred_neg"])
    if not report.add_check("attributed at least one patient", n_attributed > 0):
        return _finalize(report, report_dir)

    # --- write tables + correlations + figure -----------------------------
    pos_tops = None
    for group in ("pred_pos", "pred_neg"):
        n_group = group_counts[group]
        if n_group == 0:
            report.warn(f"{group} group non-empty", ok=False, detail="no patients in this group")
            continue
        csv_path, df, tops = _write_group_csv(
            output_dir, group, acc[group], present[group], n_group, tokenizer, args.top_k
        )
        report.add_artifact(csv_path)
        if group == "pred_pos":
            pos_tops = tops
            corr = _rank_correlation(df)
            corr_path = output_dir / "method_rank_correlation.json"
            corr_path.write_text(json.dumps(corr, indent=2), encoding="utf-8")
            report.add_artifact(corr_path)
            for pair, vals in corr.items():
                report.add_metric(f"spearman_{pair}", vals["spearman_rho"])
            top_concept = df.sort_values("gradxatt_mean", ascending=False).iloc[0]
            report.add_metric("top_positive_concept", f"{top_concept['concept_name']} ({top_concept['concept_id']})")

    if pos_tops is not None:
        fig_path = fig_dir / "attention_top_concepts.png"
        _plot_top_concepts(pos_tops, group_counts["pred_pos"], fig_path, args.top_k)
        report.add_artifact(fig_path)

    return _finalize(report, report_dir)


def _finalize(report: StepReport, report_dir) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
