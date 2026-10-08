#!/usr/bin/env python
"""Step 07: Tokenize every patient's sequence with the pretrained tokenizer.

Pretraining (step 05) only sees ``patient_sequence_pretrain`` (no test patients), so its prepared
dataset cannot serve fine-tuning, prediction or the linear probe, which need every cohort
patient. This step tokenizes the full ``patient_sequence`` (step 01) with the tokenizer that step 05
trained (``tokenizer_name_or_path`` of the pretrain config) and the pretrain runner's own
processing (``load_parquet_as_dataset`` + ``create_cehrbert_pretraining_dataset``), and saves a
DatasetDict to ``data/full_tokenized`` (``tokenized_full_dataset_path`` of the fine-tune config).

Splits: ``validation`` holds the persons of the canonical ``val`` split (cohort step 01) and ``train``
everyone else, test patients included. ``extract_cohort_sequences`` keeps this assignment, so
step 09 trains on canonical train and validates on canonical val; steps 10 and 13 pool the splits.

The step fails if any cohort patient (any split of ``splits.parquet``) is missing from the result.

Usage:
    python cehrbert_model/07_tokenize_full_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import paths  # noqa: E402
from pipeline import pretrain_data as pd_  # noqa: E402
from pipeline.config_validation import load_yaml  # noqa: E402
from pipeline.reporting import StepReport  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tokenize all patient sequences with the pretrained tokenizer")
    parser.add_argument("--config", default=str(paths.PRETRAIN_CONFIG), help="Pretrain config (data arguments)")
    parser.add_argument("--patient-sequence-dir", default=str(paths.PATIENT_SEQUENCE_DIR))
    parser.add_argument("--splits-file", default=str(paths.COHORT_SPLITS_FILE))
    parser.add_argument("--output-dir", default=str(paths.FULL_TOKENIZED_DIR))
    parser.add_argument("--report-dir", default=str(paths.report_dir_for("cehrbert_model", "07_tokenize_full_dataset")))
    args = parser.parse_args(argv)

    report = StepReport("07_tokenize_full_dataset", title="Step 07: Tokenize the full patient_sequence")
    seq_dir, out_dir = Path(args.patient_sequence_dir), Path(args.output_dir)
    for name, value in (("config", args.config), ("patient_sequence_dir", seq_dir), ("output_dir", out_dir)):
        report.add_metric(name, str(value))

    if not report.add_check("patient_sequence has parquet files", bool(pd_.sequence_files(seq_dir)), str(seq_dir)):
        return _finalize(report, args.report_dir)
    if not report.add_check("splits file exists", Path(args.splits_file).is_file(), detail=args.splits_file):
        return _finalize(report, args.report_dir)
    if not report.add_check("pretrain config exists", Path(args.config).is_file(), detail=args.config):
        return _finalize(report, args.report_dir)
    tokenizer_dir = paths.resolve_under_root(load_yaml(Path(args.config))["tokenizer_name_or_path"])
    if not report.add_check(
        "pretrained tokenizer exists", (tokenizer_dir / "tokenizer.json").is_file(), detail=str(tokenizer_dir)
    ):
        report.note("Run step 05 first; this step never trains a tokenizer.")
        return _finalize(report, args.report_dir)

    val_ids = pd_.split_person_ids(args.splits_file, "val")
    dataset = pd_.tokenize_full_dataset(args.config, seq_dir, out_dir, val_ids, paths.resolve_under_root)
    report.add_artifact(out_dir)

    ids = {name: set(split["person_id"]) for name, split in dataset.items()}
    all_ids = set().union(*ids.values())
    for name, split in dataset.items():
        report.add_metric(f"{name}_rows", split.num_rows)
    report.add_check(
        "splits are train and validation", set(dataset.keys()) == {"train", "validation"}, ",".join(dataset.keys())
    )
    report.add_check("splits are person-disjoint", not (ids.get("train", set()) & ids.get("validation", set())))

    expected = pd_.sequence_person_ids(seq_dir, require_concepts=True)
    report.add_metric("sequences_with_events", len(expected))
    report.add_check(
        "every sequence with events is tokenized",
        all_ids == expected,
        detail=f"tokenized={len(all_ids)} expected={len(expected)}",
    )

    for split in ("train", "val", "test"):
        split_ids = pd_.split_person_ids(args.splits_file, split)
        missing = len(split_ids - all_ids)
        report.add_metric(f"cohort_{split}_missing", missing)
        report.add_check(f"every cohort {split} person is present", missing == 0, detail=f"{missing} missing")
    report.add_check(
        "validation split equals the canonical val persons",
        ids.get("validation", set()) == (val_ids & all_ids),
        detail=f"validation={len(ids.get('validation', set()))} val={len(val_ids)}",
    )
    return _finalize(report, args.report_dir)


def _finalize(report: StepReport, report_dir: str) -> int:
    report.write(report_dir)
    report.print_summary()
    print(f"\nReport written to {report_dir}")
    if report.passed:
        print("Next: python cehrbert_model/08_validate_finetune_config.py")
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
