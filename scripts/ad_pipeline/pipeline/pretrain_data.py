"""Pretraining data that excludes the step 6 test patients, and the full tokenized dataset.

The pretraining input (``patient_sequence_pretrain``, step 3b) is ``patient_sequence`` without
the test split of the canonical splits file, so the pretrained encoder never sees a test
patient. Fine-tuning and the linear probe still need every cohort patient, so step 5b
tokenizes the full ``patient_sequence`` with the trained tokenizer and the pretrain runner's
own processing into ``data/full_tokenized``.

``datasets`` and ``cehrbert`` are imported lazily, so the parquet helpers work without them.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PERSON_ID = "person_id"
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


# --------------------------------------------------------------------------- #
# Splits file and raw patient sequences
# --------------------------------------------------------------------------- #
def split_person_ids(splits_file: str | Path, split: str) -> set[int]:
    """person_ids of one split (``train`` / ``val`` / ``test``) of the canonical splits file."""
    df = pl.read_parquet(splits_file, columns=[PERSON_ID, "split"])
    return set(df.filter(pl.col("split") == split)[PERSON_ID].to_list())


def sequence_files(seq_dir: str | Path) -> List[Path]:
    """The parquet files the pretrain runner reads from ``seq_dir`` (``*.parquet``, no recursion)."""
    return sorted(Path(seq_dir).glob("*.parquet"))


def sequence_person_ids(seq_dir: str | Path, require_concepts: bool = True) -> set[int]:
    """person_ids with a sequence in ``seq_dir``.

    With ``require_concepts`` only sequences with ``num_of_concepts > 0`` count: the tokenization
    (``create_cehrbert_pretraining_dataset``) drops the others, so they never reach step 8.
    """
    files = sequence_files(seq_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files in {seq_dir}")
    lf = pl.scan_parquet([str(f) for f in files])
    if require_concepts and "num_of_concepts" in lf.collect_schema().names():
        lf = lf.filter(pl.col("num_of_concepts") > 0)
    return set(lf.select(PERSON_ID).unique().collect()[PERSON_ID].to_list())


def newest_mtime(seq_dir: str | Path) -> Optional[float]:
    files = sequence_files(seq_dir)
    return max(f.stat().st_mtime for f in files) if files else None


def write_sequences_without(src_dir: str | Path, dst_dir: str | Path, exclude_ids: Iterable[int]) -> Dict[str, int]:
    """Copy the patient sequences in ``src_dir`` to ``dst_dir`` without the persons in ``exclude_ids``.

    Keeps the Spark layout: one output file per input file, same name, schema (with its
    metadata) and snappy compression. A file left with no rows is not written, because the
    ``datasets`` parquet reader fails on it ("BatchSize must be greater than 0"). The output is built in a sibling folder and swapped in at the
    end, so an interrupted run never leaves a partial ``dst_dir``.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    files = sequence_files(src_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files in {src_dir}")
    exclude = pa.array(sorted({int(p) for p in exclude_ids}), type=pa.int64())

    tmp_dir = dst_dir.with_name(dst_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    counts = {"files": 0, "files_written": 0, "rows_total": 0, "rows_removed": 0, "rows_remaining": 0}
    for path in files:
        table = pq.read_table(path)
        ids = table.column(PERSON_ID).cast(pa.int64())
        keep = pc.invert(pc.is_in(ids, value_set=exclude))
        kept = table.filter(keep)
        if kept.num_rows:
            pq.write_table(kept, tmp_dir / path.name, compression="snappy")
            counts["files_written"] += 1
        counts["files"] += 1
        counts["rows_total"] += table.num_rows
        counts["rows_remaining"] += kept.num_rows
        counts["rows_removed"] += table.num_rows - kept.num_rows

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    tmp_dir.rename(dst_dir)
    return counts


# --------------------------------------------------------------------------- #
# Pretrain config: the runner's arguments and its prepared-dataset path
# --------------------------------------------------------------------------- #
def parse_pretrain_args(config_path: str | Path):
    """``(DataTrainingArguments, ModelArguments)`` parsed from the pretrain YAML, as the runner parses them."""
    import yaml
    from transformers import HfArgumentParser

    from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments, ModelArguments

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    parser = HfArgumentParser((DataTrainingArguments, ModelArguments))
    data_args, model_args = parser.parse_dict(cfg, allow_extra_keys=True)
    return data_args, model_args


def pretrain_prepared_path(config_path: str | Path, resolve: Callable[[str], Path]) -> Path:
    """The ``dataset_prepared_path/<name>_<hash>`` folder the pretrain runner reads and writes.

    The hash is computed from the config values exactly as written, because the runner is
    launched from the project root with those same (relative) strings; only the result is
    resolved with ``resolve``.
    """
    from cehrbert.runners.runner_util import generate_prepared_ds_path

    data_args, model_args = parse_pretrain_args(config_path)
    return Path(resolve(str(generate_prepared_ds_path(data_args, model_args))))


def stale_pretrain_artifacts(config_path: str | Path, resolve: Callable[[str], Path]) -> List[Path]:
    """Pretraining artifacts older than the newest parquet in the configured ``data_folder``.

    The runner reuses each of these without checking what they were built from: the prepared
    dataset (matched by a hash of paths and settings, not content), the tokenizer and model
    config in ``tokenizer_name_or_path`` / ``model_name_or_path``, and ``checkpoint-*`` folders
    in ``output_dir`` (training resumes from the last one). Artifacts newer than the input are
    taken to belong to this input, which keeps resuming an interrupted run possible.
    """
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    input_mtime = newest_mtime(resolve(cfg["data_folder"]))
    if input_mtime is None:
        return []
    candidates: List[Path] = []
    prepared = pretrain_prepared_path(config_path, resolve)
    if prepared.exists():
        candidates.append(prepared)
    for key, names in (
        ("tokenizer_name_or_path", ("tokenizer.json", "cehrbert_lab_stats.json", "concept_name_mapping.json")),
        ("model_name_or_path", ("config.json",)),
    ):
        if cfg.get(key):
            candidates += [Path(resolve(cfg[key])) / n for n in names if (Path(resolve(cfg[key])) / n).exists()]
    if cfg.get("output_dir"):
        candidates += sorted(Path(resolve(cfg["output_dir"])).glob("checkpoint-*"))
    stale: List[Path] = []
    for path in candidates:
        if path not in stale and path.stat().st_mtime < input_mtime:
            stale.append(path)
    return stale


# --------------------------------------------------------------------------- #
# Tokenized datasets
# --------------------------------------------------------------------------- #
def dataset_split_person_ids(dataset_dir: str | Path) -> Dict[str, set]:
    """person_ids per split of a saved DatasetDict (or ``{"": ids}`` for a single Dataset)."""
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(str(dataset_dir))
    if isinstance(dataset, DatasetDict):
        return {name: set(split[PERSON_ID]) for name, split in dataset.items()}
    return {"": set(dataset[PERSON_ID])}


def leaked_person_ids(dataset_dir: str | Path, person_ids: Iterable[int]) -> Dict[str, int]:
    """Per split, how many of ``person_ids`` appear in the saved dataset (all zero when none leak)."""
    wanted = set(person_ids)
    return {name: len(ids & wanted) for name, ids in dataset_split_person_ids(dataset_dir).items()}


def add_test_exclusion_check(report, dataset_dir: str | Path, splits_file: str | Path) -> bool:
    """Fail ``report`` when a test-split person appears in any split of the saved dataset."""
    if not report.add_check("splits file exists", Path(splits_file).is_file(), detail=str(splits_file)):
        return False
    test_ids = split_person_ids(splits_file, "test")
    leaked = leaked_person_ids(dataset_dir, test_ids)
    for name, count in leaked.items():
        report.add_metric(f"test_persons_in_{name or 'dataset'}", count)
    return report.add_check(
        "no test person in the pretraining dataset",
        sum(leaked.values()) == 0,
        detail=f"{len(test_ids)} test persons; found per split: {leaked}",
    )


def tokenize_full_dataset(
    config_path: str | Path,
    data_folder: str | Path,
    output_dir: str | Path,
    validation_person_ids: Iterable[int],
    resolve: Callable[[str], Path],
):
    """Tokenize every sequence in ``data_folder`` as the pretrain runner does and save it to ``output_dir``.

    Uses the runner's ``load_parquet_as_dataset`` and ``create_cehrbert_pretraining_dataset`` with
    the pretrain config's data arguments and the tokenizer already trained in
    ``tokenizer_name_or_path`` (it is never retrained here). The result is a DatasetDict with
    ``validation`` = the persons in ``validation_person_ids`` and ``train`` = everyone else, because
    ``extract_cohort_sequences`` keeps that assignment and step 8 uses it as its train/validation
    split. Returns the saved DatasetDict.
    """
    from datasets import DatasetDict, load_from_disk

    from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
    from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_pretraining_dataset
    from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
    from cehrbert.runners.runner_util import load_parquet_as_dataset

    data_args, model_args = parse_pretrain_args(config_path)
    data_args.streaming = False
    tokenizer = CehrBertTokenizer.from_pretrained(str(resolve(model_args.tokenizer_name_or_path)))

    dataset = load_parquet_as_dataset(str(data_folder), split="train", streaming=False)
    val_ids = {int(p) for p in validation_person_ids}
    filter_args = {
        "batched": True,
        "batch_size": data_args.preprocessing_batch_size,
        "num_proc": data_args.preprocessing_num_workers,
    }
    dataset = DatasetDict(
        {
            TRAIN_SPLIT: dataset.filter(lambda b: [p not in val_ids for p in b[PERSON_ID]], **filter_args),
            VALIDATION_SPLIT: dataset.filter(lambda b: [p in val_ids for p in b[PERSON_ID]], **filter_args),
        }
    )
    cache_file_collector = CacheFileCollector()
    processed = create_cehrbert_pretraining_dataset(
        dataset=dataset,
        concept_tokenizer=tokenizer,
        data_args=data_args,
        cache_file_collector=cache_file_collector,
    )

    output_dir = Path(output_dir)
    tmp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    processed.save_to_disk(str(tmp_dir))
    processed.cleanup_cache_files()
    cache_file_collector.remove_cache_files()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.replace(tmp_dir, output_dir)
    return load_from_disk(str(output_dir))
