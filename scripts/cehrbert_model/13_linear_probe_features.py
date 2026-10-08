#!/usr/bin/env python
"""Step 13: Extract frozen CEHR-BERT features for the linear probe.

For each canonical split (``train`` / ``val`` / ``test`` from ``splits.parquet``, cohort step 01) the
patient's pre-index history inside the look-back window is cut out of the full tokenized
dataset (step 07) with cehrbert's ``extract_cohort_sequences``, fed through the frozen
pretrained encoder (no fine-tuning), and pooled into two feature sets (see
``pipeline.linear_probe``): ``cls`` (the [CLS] hidden state) and ``mean`` (mean over real
tokens). Step 14 trains a logistic regression on them.

Context longer than ``--ctx`` tokens is cut to the most recent tokens by the collator; the
number and share of such patients is reported per split. With ``keep_demographic_prompt`` in
the fine-tune config, the four leading demographic tokens (year, age, gender, race) are kept
for every patient whatever the window and survive truncation. When the first visit is after the
index date only the gender and race tokens are kept (2 tokens). "Empty window" still means no
in-window event tokens, not counting those four.

Outputs (under build/cehrbert_model/<run_label>/linear_probe/features/<split>/)
  features_cls.npy, features_mean.npy    float32, N x hidden size
  meta.parquet                           person_id, index_date, label,
                                         n_tokens_before_truncation, truncated,
                                         n_event_tokens, demographic_prompt_tokens
  build/cehrbert_model/<run_label>/reports/13_linear_probe_features/{status.json,report.md}

Usage
  python cehrbert_model/13_linear_probe_features.py
  python cehrbert_model/13_linear_probe_features.py --batch-size 256 --device cuda
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import baselines as bl  # noqa: E402
from pipeline import linear_probe as lp  # noqa: E402
from pipeline import paths  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def _tokenized_person_ids(tokenized_path: str) -> set[int]:
    from cehrbert.runners.data_utils import resolve_tokenized_dataset_path
    from datasets import load_from_disk

    dataset = load_from_disk(resolve_tokenized_dataset_path(tokenized_path))
    ids: set[int] = set()
    for split in dataset.values():
        ids.update(split["person_id"])
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract frozen CEHR-BERT features for the AD linear probe")
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE))
    parser.add_argument(
        "--tokenized-dataset",
        default=str(paths.FULL_TOKENIZED_DIR),
        help="Tokenized dataset with every patient (step 07), or a folder holding exactly one such DatasetDict",
    )
    parser.add_argument("--finetune-config", default=str(paths.FINETUNE_CONFIG))
    parser.add_argument("--model-dir", default=str(paths.PRETRAIN_RESULTS_DIR))
    parser.add_argument("--output-dir", default=None, help="Features folder (default: per-run linear_probe/features)")
    parser.add_argument(
        "--window-days",
        type=int,
        default=paths.DEFAULT_WINDOW_DAYS,
        help="Look-back window in days; negative => unbounded (all history)",
    )
    parser.add_argument(
        "--ctx", type=int, default=paths.DEFAULT_CTX, help="Token cap per patient (collator max_length)"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=list(lp.SPLIT_NAMES), choices=list(lp.SPLIT_NAMES))
    parser.add_argument("--report-dir", default=None, help="Report dir (default: per-run report dir)")
    args = parser.parse_args(argv)

    label = paths.run_label(args.window_days, args.ctx)
    output_dir = Path(args.output_dir) if args.output_dir else paths.cehrbert_linear_probe_features_dir(label)
    report_dir = (
        Path(args.report_dir) if args.report_dir else paths.cehrbert_report_dir("13_linear_probe_features", label)
    )

    report = StepReport("13_linear_probe_features", title="Step 13 - Frozen CEHR-BERT features")
    for name, value in (
        ("run_label", label),
        ("window_days", args.window_days),
        ("ctx", args.ctx),
        ("device", args.device),
        ("batch_size", args.batch_size),
        ("output_dir", str(output_dir)),
    ):
        report.add_metric(name, value)

    if not report.add_check("splits file exists", Path(args.splits_file).is_file(), detail=args.splits_file):
        return _finalize(report, report_dir)
    if not report.add_check("model dir exists", Path(args.model_dir).is_dir(), detail=args.model_dir):
        return _finalize(report, report_dir)

    splits = bl.load_canonical_splits(args.splits_file)
    frames = {"train": splits.train, "val": splits.val, "test": splits.test}

    tokenized_ids = _tokenized_person_ids(args.tokenized_dataset)
    missing = {
        name: sum(1 for p in frames[name]["person_id"].to_list() if p not in tokenized_ids) for name in args.splits
    }
    for name, n_missing in missing.items():
        report.add_metric(f"{name}_missing_from_tokenized", n_missing)
    if not report.add_check(
        "no patients missing from the tokenized dataset",
        sum(missing.values()) == 0,
        detail=str(missing),
    ):
        return _finalize(report, report_dir)

    import datasets
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer

    # Keep the filter/map results of extract_cohort_sequences out of the source dataset folder.
    datasets.disable_caching()
    tokenizer = CehrBertTokenizer.from_pretrained(args.model_dir)
    model = lp.load_frozen_model(args.model_dir, device=args.device)
    keep_prompt = lp.keep_demographic_prompt_from_config(args.finetune_config)
    report.add_metric("keep_demographic_prompt", keep_prompt)
    report.add_metric("hidden_size", int(model.config.hidden_size))

    work_root = output_dir / "_work"
    for name in args.splits:
        frame = frames[name]
        start = time.time()
        dataset = lp.build_split_dataset(
            frame, work_root / name, args.finetune_config, args.tokenized_dataset, args.window_days
        )
        feats = lp.extract_features(
            model,
            dataset,
            tokenizer,
            max_length=args.ctx,
            batch_size=args.batch_size,
            device=args.device,
            progress_every=200,
            keep_demographic_prompt=keep_prompt,
        )

        expected_ids = set(frame["person_id"].to_list())
        got_ids = set(feats.person_id.tolist())
        report.add_check(
            f"{name}: rows equal the split's patients, once each",
            len(feats.person_id) == frame.height and got_ids == expected_ids,
            detail=f"rows={len(feats.person_id)} split_size={frame.height} "
            f"missing={len(expected_ids - got_ids)} unexpected={len(got_ids - expected_ids)}",
        )
        label_map = dict(zip(frame["person_id"].to_list(), frame["label"].to_list()))
        date_map = dict(zip(frame["person_id"].to_list(), frame["index_date"].to_list()))
        report.add_check(
            f"{name}: labels match the splits file",
            all(label_map.get(int(p)) == int(y) for p, y in zip(feats.person_id, feats.label)),
        )
        index_dates = np.asarray([np.datetime64(date_map[int(p)]) for p in feats.person_id])
        written = lp.save_split_features(output_dir / name, feats, index_dates)
        for path in written:
            report.add_artifact(path)

        n_trunc = int(feats.truncated.sum())
        report.add_metric(f"{name}_rows", int(len(feats.person_id)))
        report.add_metric(f"{name}_truncated", n_trunc)
        report.add_metric(f"{name}_truncated_share", round(n_trunc / max(len(feats.person_id), 1), 4))
        report.add_metric(f"{name}_empty_windows", int((feats.n_event_tokens == 0).sum()))
        for n_prompt in (0, 2, 4):
            report.add_metric(
                f"{name}_prompt_tokens_{n_prompt}", int((feats.demographic_prompt_tokens == n_prompt).sum())
            )
        report.add_metric(f"{name}_seconds", round(time.time() - start, 1))
        report.add_check(
            f"{name}: features are finite",
            bool(np.isfinite(feats.cls).all() and np.isfinite(feats.mean).all()),
        )

    shutil.rmtree(work_root, ignore_errors=True)
    return _finalize(report, report_dir)


def _finalize(report: StepReport, report_dir: Path) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python cehrbert_model/14_train_linear_probe.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
