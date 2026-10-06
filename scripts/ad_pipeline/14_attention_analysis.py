#!/usr/bin/env python
"""Step 14: Attention-weight analysis -- which clinical concepts drive predictions.

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
``build/<label>/attention/``:

  concept_attribution_<group>.csv   ranked concepts per method (group = pred_pos / pred_neg)
  method_rank_correlation.json      Spearman rho between the three methods' rankings
  figures/attention_top_concepts.png  top concepts driving positive predictions (3 methods)

Two codebase-specific requirements are enforced here: the model is loaded with
``attn_implementation="eager"`` (flash attention returns no weights), and sample packing is
left off so each row is one patient with [CLS] at position 0.

Usage
  python 14_attention_analysis.py
  python 14_attention_analysis.py --max-patients 300 --top-k 20
  python 14_attention_analysis.py --device cuda --window-days 365 --ctx 512
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import attention as attn  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.config_validation import load_yaml  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402

# Special / structural tokens excluded from concept attribution by default. The five reserved
# tokens never carry clinical meaning; VS/VE (and the bracketed variants) are visit boundary
# markers. Artificial time-interval tokens are kept -- a gap between visits can be informative.
EXCLUDED_TOKENS = frozenset({"[PAD]", "[CLS]", "[MASK]", "[UNUSED]", "[OOV]", "VS", "VE", "[VS]", "[VE]"})

DATASET_FEATURE_KEYS = (
    "input_ids", "ages", "dates", "visit_concept_orders",
    "concept_values", "concept_value_masks", "visit_segments",
    "age_at_index", "classifier_label", "person_id",
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


def _attribute_one(model, batch, device) -> tuple[dict[str, torch.Tensor], float] | None:
    """Forward+backward one single-patient batch; return per-method token scores and prob.

    Returns ``None`` for sequences with no real concept tokens after [CLS] -- such an example
    collates to a width-1 ``[CLS]``-only row that carries no clinical signal to attribute (and
    whose ragged time-series features can break the embedding concat), so it is skipped before
    the forward.
    """
    if batch["input_ids"].shape[1] <= 1:
        return None
    inputs = {k: v.to(device) for k, v in batch.items() if k in (
        "input_ids", "attention_mask", "ages", "dates", "visit_concept_orders",
        "concept_values", "concept_value_masks", "visit_segments", "age_at_index",
    )}
    model.zero_grad(set_to_none=True)
    out = model(**inputs, output_attentions=True)
    if out.attentions is None:
        # Should not happen with attn_implementation="eager", but sdpa/flash return no
        # weights -- fail loud rather than crash with an opaque NoneType error downstream.
        raise RuntimeError(
            "model returned no attention weights; load with attn_implementation='eager'")
    # Retain grad on the attention probs so backward populates .grad on each layer.
    for a in out.attentions:
        a.retain_grad()
    logit = out.logits[0, 0]
    prob = float(torch.sigmoid(logit).detach().cpu())
    logit.backward()

    attn_lhss = attn.stack_example_attentions(out.attentions, batch_index=0)
    # A layer's attention probs may not receive a gradient (a.grad is None) if that layer
    # does not contribute to the logit; substitute zeros so the stack stays well-formed and
    # that layer simply gets zero relevance in grad-weighted rollout.
    grad_lhss = torch.stack(
        [
            (a.grad[0].detach().cpu() if a.grad is not None else torch.zeros_like(a[0]).cpu())
            for a in out.attentions
        ],
        dim=0,
    )

    scores = {
        "raw": attn.raw_cls_attention(attn_lhss),
        "rollout": attn.attention_rollout(attn_lhss),
        "gradxatt": attn.grad_attention_rollout(attn_lhss, grad_lhss),
    }
    return scores, prob


def _accumulate(scores, prob, input_ids, tokenizer, acc, present, group_counts):
    """Fold one patient's attribution into the cohort accumulators (in place)."""
    group = "pred_pos" if prob >= 0.5 else "pred_neg"
    group_counts[group] += 1

    ids = input_ids[0].tolist()
    concept_ids = [tokenizer.convert_id_to_token(i) for i in ids]
    keep = torch.tensor([c not in EXCLUDED_TOKENS for c in concept_ids], dtype=torch.bool)
    if not bool(keep.any()):
        return False

    seen_this_patient = set()
    for method, vec in scores.items():
        dist = attn.normalize_over_real_tokens(vec, keep).tolist()
        for pos, w in enumerate(dist):
            if w <= 0:
                continue
            cid = concept_ids[pos]
            acc[group][method][cid] += w
            if method == "raw" and cid not in seen_this_patient:
                present[group][cid] += 1
                seen_this_patient.add(cid)
    return True


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
        for b in attn.METHODS[i + 1:]:
            rho, p = spearmanr(df[f"{a}_mean"], df[f"{b}_mean"])
            out[f"{a}_vs_{b}"] = {"spearman_rho": round(float(rho), 4), "p_value": float(p)}
    return out


def _plot_top_concepts(tops, n_pos, fig_path, top_k):
    """3-panel horizontal bar chart: top concepts driving positive predictions per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 0.45 * top_k + 1.5))
    titles = {"raw": "Raw [CLS] attention", "rollout": "Attention rollout",
              "gradxatt": "Gradient x attention"}
    colors = {"raw": "#4C72B0", "rollout": "#55A868", "gradxatt": "#C44E52"}
    for ax, method in zip(axes, attn.METHODS):
        d = tops[method].iloc[::-1]  # largest at top
        labels = [f"{n[:38]} ({c})" for n, c in zip(d["concept_name"], d["concept_id"])]
        ax.barh(range(len(d)), d[f"{method}_mean"], color=colors[method])
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(titles[method], fontsize=11)
        ax.set_xlabel("mean attribution per patient")
    fig.suptitle(f"Top {top_k} concepts driving positive AD predictions "
                 f"(predicted-positive patients, n={n_pos})", fontsize=13)
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
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Cap the number of test patients processed (default: all)")
    parser.add_argument("--top-k", type=int, default=20, help="Concepts shown per method in the figure")
    parser.add_argument("--device", default="cpu", help="torch device (cpu / cuda)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    output_dir = Path(args.output_dir) if args.output_dir else paths.run_dir(label) / "attention"
    fig_dir = paths.figures_dir(label)
    report_dir = (Path(args.report_dir) if args.report_dir
                  else paths.build_report_dir_for("14_attention_analysis", label))
    output_dir.mkdir(parents=True, exist_ok=True)

    report = StepReport("14_attention_analysis", title="Step 14 - Attention-weight analysis")
    report.add_metric("run_label", label)
    report.add_metric("model_dir", args.model_dir)
    report.add_metric("device", args.device)

    # --- preflight --------------------------------------------------------
    model_ok = report.add_check(
        "fine-tuned model exists",
        Path(args.model_dir, "config.json").is_file(), detail=args.model_dir)
    tok_ok = report.add_check(
        "tokenizer exists",
        Path(args.tokenizer_dir, "tokenizer.json").is_file(), detail=args.tokenizer_dir)
    prep_ok = report.add_check(
        "prepared test data exists", Path(args.prepared_dir).is_dir(), detail=args.prepared_dir)
    if not (model_ok and tok_ok and prep_ok):
        report.note("Preflight failed. Run steps 6, 8 and 9 first.")
        return _finalize(report, report_dir)

    max_length = int(load_yaml(Path(args.finetune_config)).get("max_position_embeddings", 512))
    report.add_metric("max_length", max_length)

    # --- load model + tokenizer (eager attention is mandatory) ------------
    from cehrbert.models.hf_models.hf_cehrbert import CehrBertForClassification
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer

    tokenizer = CehrBertTokenizer.from_pretrained(args.tokenizer_dir)
    model = CehrBertForClassification.from_pretrained(
        args.model_dir, attn_implementation="eager", output_attentions=True)
    model.to(args.device).eval()
    attn_impl = getattr(model.config, "_attn_implementation", "eager")
    if not report.add_check(
        "eager attention active (flash/sdpa return no weights)",
        attn_impl == "eager", detail=attn_impl):
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
        detail=f"columns={list(dataset[0].keys())}" if n_total > 0 else "empty dataset"):
        return _finalize(report, report_dir)

    collator = _build_collator(tokenizer, max_length)

    # --- accumulate per-concept attribution by predicted class ------------
    acc = {g: {m: defaultdict(float) for m in attn.METHODS} for g in ("pred_pos", "pred_neg")}
    present = {g: defaultdict(int) for g in ("pred_pos", "pred_neg")}
    group_counts = defaultdict(int)
    n_attributed = n_skipped = 0

    for idx in range(n_run):
        batch = collator([_example_for_collator(dataset[idx])])
        result = _attribute_one(model, batch, args.device)
        if result is None:
            n_skipped += 1
            continue
        scores, prob = result
        ok = _accumulate(scores, prob, batch["input_ids"].cpu(), tokenizer, acc, present, group_counts)
        if ok:
            n_attributed += 1
        else:
            n_skipped += 1
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{n_run} (attributed={n_attributed}, skipped={n_skipped})")

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
            output_dir, group, acc[group], present[group], n_group, tokenizer, args.top_k)
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
            report.add_metric("top_positive_concept",
                              f"{top_concept['concept_name']} ({top_concept['concept_id']})")

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
